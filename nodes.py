"""ComfyUI nodes for CLSS (Closed-Loop Streaming Synthesis) on MiniMax H3.

Port of the LTX-2 CLSS node layer (ComfyUI-LTX2.3-CLSS) to the MiniMax H3
packed audio-video DiT.  The model-agnostic algorithm core (CLSSConfig /
CLSSState: SLB bookkeeping, §2.3 EMA-AdaIN drift correction) lives in the
vendored `clss.py`; this file is the H3-specific orchestration.

Key mechanism differences from the LTX version:

- Latent: one dict {"samples": NestedTensor((video [B,24,T,H/16,W/16],
  audio [B,32,2,Ta]))}.  Video grid: px frames snap to 17k+5 ⇔ latent tokens
  5k+2 (`video_latent_t`); token k covers FRAME_PER_TOKEN[k%5] = (1,4,4,4,4)
  px frames.  Audio: Ta = round(px × 5/3) at 24 fps (40 latent fps) — the same
  math as `temporal_shape` in comfy_extras/nodes_minimax_h3.py.
- §2.1 SLB overlap (video only): the previous chunk's corrected tail is
  written INTO the chunk's initial latent (video tokens [0:F_ol]) and a
  per-stream denoise mask rides in the latent dict ("noise_mask",
  NestedTensor).  H3 turns mask value m into a per-row sigma = m·σ
  (model.py:587-609) and re-blends preserved rows toward the cond-strength
  injection every step (model_base.py:2248-2272 scale_latent_inpaint, called
  from KSamplerX0Inpaint in comfy/samplers.py:634-643).
- Audio seam (H3-Motion-Context's recipe): the audio overlap rows carry NO
  previous-tail content — they are 100% fresh noise under an all-ones mask
  (measured on H3: masked-in audio rows are treated as output, not context).
  The only audio context is the previous tail pinned as a cond_audio guide
  keyframe whose window ENDS at the join (fractional/negative
  resolved_frame_index), and the rendered overlap is trimmed with a plain
  cut (no crossfade).
- The i2v guide image rides as a `minimax_keyframes` conditioning row
  ({"resolved_frame_index", "latent"}) — the H3-native pinning mechanism
  (model.py:340-361, pinned near-clean, re-injected every step).
- No hard RoPE wall on H3 (no max_pos); the trained range is ~124-362 px
  frames (~5-15 s, per the EmptyMiniMaxH3LatentAV tooltip).  The LTX
  RoPE-wall auto-split is kept in shape with _WINDOW_CAP_S = 12.0.
- Split video/audio CFG lives in CLSSH3Guider (a CFGGuider subclass doing
  unpack → per-stream CFG → rescale → repack).  No STG, no modality_scale, no
  per-modality sigma logic — H3's ModelSamplingAV owns the audio schedule and
  the shift knobs live on the stock MiniMaxH3SigmaShift node.
"""

from __future__ import annotations

import copy
import dataclasses
import math
import os

# Long chunked runs alloc/free large transient tensors every chunk; without
# expandable_segments the CUDA caching allocator fragments and a 6-chunk run
# OOMs on a 16 GB card with GiBs still reserved-but-unusable (measured:
# 12.4 GiB allocated, 1.1 GiB request failed, 11 MiB free).
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_minimax_h3 import AUDIO_LATENT_FPS, FPS as _NATIVE_FPS

try:
    from .clss import CLSSConfig, CLSSState
except ImportError:  # direct `import nodes` (smoke tests, scripts)
    from clss import CLSSConfig, CLSSState


# ---------------------------------------------------------------------------
# §-constants carried over from the LTX port (validated there, revalidated on
# H3 only by live run — see AGENTS.md conventions).
# ---------------------------------------------------------------------------

# §2.1 per-chunk tau_c schedule rises from the configured base toward this
# ceiling with a 5-chunk half-life (late chunks drift more, get more repair).
_VIDEO_TAU_C_CEILING = 0.10

# H3 has no hard RoPE wall, but the model is trained on ~124-362 px frames
# (~5-15 s at 24 fps).  Keep each sampling window (overlap + new) at or under
# this many seconds; longer chunks are auto-split into uniform sub-chunks.
_WINDOW_CAP_S = 12.0

# Minimum SLB overlap in latent tokens on the 5k+2 grid (2 tokens = 5 px ≈
# 0.21 s).  Overlaps are always 5m+2 tokens so that window token phases line
# up with the absolute decode grid (see _px_of_tokens).
_MIN_OVERLAP_TOKENS = 2

_SCENE_BLEND_W = 0.5


# ---------------------------------------------------------------------------
# Small grid helpers — H3's latent↔pixel↔audio time mapping.
# ---------------------------------------------------------------------------


def _px_of_tokens(n_tokens: int, start_phase: int = 0) -> int:
    """Exact px-frame span of n_tokens consecutive video latent tokens.

    Token k covers FRAME_PER_TOKEN[k % 5] = (1,4,4,4,4) px frames; the phase
    is the token's index mod 5 in its home sequence (window for conditioning,
    absolute video for decode).  Phase matters: 7 tokens span 22 px at phase 0
    but 25 px at phase 2.
    """
    return sum(FRAME_PER_TOKEN[(start_phase + j) % 5] for j in range(n_tokens))


def _af_of_px(px: int) -> int:
    """Audio latent frames covering px pixel frames (temporal_shape math;
    FRAME_RESCALE = 5/3 = AUDIO_LATENT_FPS / 24 exactly)."""
    return round(px * FRAME_RESCALE)


def _snap_overlap(overlap: int) -> int:
    """Snap the SLB size to the 5m+2 token grid.

    Chunk 0 contributes 5k+2 tokens and every continuation chunk a multiple of
    5, so every chunk starts at absolute token phase 2; an overlap of 5m+2
    tokens then starts at phase (2 − (5m+2)) % 5 = 0, which makes the overlap's
    window-relative token spans equal its absolute spans and keeps the audio
    overlap length exact.
    """
    return max(_MIN_OVERLAP_TOKENS, 5 * round((overlap - 2) / 5) + 2)


def _split_run(n_tokens: int, max_new: int, plus2: bool) -> list[int]:
    """Split a chunk's new-token run into grid-aligned parts ≤ max_new.

    n_tokens is 5k+2 (plus2=True, the run starting at absolute phase 0 — i.e.
    chunk 0) or 5k (plus2=False).  The first part keeps the +2 so downstream
    parts stay multiples of 5 and every part's start phase stays on the
    absolute grid.
    """
    plus = 2 if plus2 else 0
    groups = (n_tokens - plus) // 5
    cap = max(1, (max_new - plus) // 5)
    s = min(groups, max(1, math.ceil(groups / cap)))
    base, extra = divmod(groups, s)
    gs = [base + (1 if j < extra else 0) for j in range(s)]
    return [5 * gs[0] + plus] + [5 * g for g in gs[1:]]


# ---------------------------------------------------------------------------
# Scene crossfade (ported as-is from the LTX node layer, plus token-tag
# alignment for the qwen3vl conditioning).
# ---------------------------------------------------------------------------


def _blend_scene_cond(prev: dict, new: dict, w: float = _SCENE_BLEND_W) -> dict:
    """Weighted cross-attn embedding blend for scene-transition chunks.

    The scene hand-off swaps the whole window's text conditioning at a chunk
    boundary, which reads as a hard cut: the frozen SLB overlap is the only
    visual bridge, and every new frame is denoised under the incoming scene
    alone at full CFG.  Blending the outgoing scene's embedding into the
    boundary chunks (25%-incoming on the outgoing block's last chunk,
    75%-incoming on the incoming block's first) lets the tail action finish
    on screen while the new scene's content takes over.  Scenes tokenize to
    different lengths, so the shorter sequence is edge-padded (EOS-repeat)
    before blending; structurally incompatible entries fall back to the new
    scene unblended.  ``w`` is the incoming-scene weight.

    H3 addition: `minimax_token_tags` (per-token modality tags consumed as tag
    runs in model.py:624-634) must cover the padded length too — they are
    edge-padded in lockstep with cross_attn; if the two scenes' tags can't be
    aligned (different width or missing), the blended entry falls back to the
    incoming scene's tags, which match its (padded) embedding by construction.
    """
    pe, ne = prev.get("cross_attn"), new.get("cross_attn")
    if pe is None or ne is None or pe.shape[-1] != ne.shape[-1]:
        return new
    t = max(pe.shape[1], ne.shape[1])
    if pe.shape[1] < t:
        pe = torch.cat([pe, pe[:, -1:].expand(-1, t - pe.shape[1], -1)], dim=1)
    if ne.shape[1] < t:
        ne = torch.cat([ne, ne[:, -1:].expand(-1, t - ne.shape[1], -1)], dim=1)
    blended = dict(new)
    blended["cross_attn"] = ((1.0 - w) * pe.float() + w * ne.float()).to(ne.dtype)
    pt, nt = prev.get("minimax_token_tags"), new.get("minimax_token_tags")
    if pt is not None and nt is not None and pt.shape[-1] in (prev["cross_attn"].shape[1], t) \
            and nt.shape[-1] in (new["cross_attn"].shape[1], t):
        def _pad_tags(tags, src_len):
            tags = tags.reshape(-1)[:src_len]
            if tags.shape[0] < t:
                tags = torch.cat([tags, tags[-1:].expand(t - tags.shape[0])])
            return tags
        pt, nt = _pad_tags(pt, prev["cross_attn"].shape[1]), _pad_tags(nt, new["cross_attn"].shape[1])
        if pt.shape[0] == t and nt.shape[0] == t and bool((pt == nt).all()):
            blended["minimax_token_tags"] = nt.reshape(new["minimax_token_tags"].shape[:-1] + (t,)) \
                if new["minimax_token_tags"].ndim > 1 else nt
        else:
            # tags disagree between scenes (e.g. one has vision tokens): the
            # blend is only meaningful over the shared text span then — keep
            # the incoming scene's tags, padded to the blend length.
            blended["minimax_token_tags"] = nt
    return blended


# ---------------------------------------------------------------------------
# Telemetry helpers (structure metrics only — they localize failures, they
# never prove a quality win; the user's eyes/ears on a live decode are the
# only ground truth).  Audio latents are [B, 32, 2, Ta] here — the time axis
# is LAST (the LTX port had [B, C, T, freq]).
# ---------------------------------------------------------------------------


def _frame_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two video frames ([B, C, H, W] each)."""
    with torch.no_grad():
        fa = F.normalize(a.float().reshape(a.shape[0], a.shape[1], -1).mean(-1), dim=1)
        fb = F.normalize(b.float().reshape(b.shape[0], b.shape[1], -1).mean(-1), dim=1)
        return (fa * fb).sum(dim=1).mean().item()


def _aud_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    with torch.no_grad():
        min_t = min(a.shape[-1], b.shape[-1])
        fa = F.normalize(a[..., :min_t].float().reshape(a.shape[0], -1), dim=1)
        fb = F.normalize(b[..., :min_t].float().reshape(b.shape[0], -1), dim=1)
        return (fa * fb).sum(dim=1).mean().item()


# ---------------------------------------------------------------------------
# Delivered-domain seam telemetry (patch #10).  aud_bnd/aud_dlv are
# single-frame latent cosines — a 40 Hz audio latent frame is 25 ms and NOT
# frame-to-frame redundant, so they cannot see clicks, phase offsets, or
# loops.  These four measure what the listener actually receives, on the
# post-crossfade delivered audio.  Read-only: they never touch the tensors.
# ---------------------------------------------------------------------------
# Loop detection searches ALL shifts via FFT — window-cosine approaches miss
# repeats that don't align to the window grid even when the repeat is
# bit-exact (verified in the harness: 1-frame offset -> cosine collapses).


def _aud_flat(t: torch.Tensor) -> torch.Tensor:
    """[B,C,L,T] audio latent -> [B, C*L, T] float32 for telemetry."""
    return t.float().reshape(t.shape[0], -1, t.shape[-1])


def _aud_seam_step(prev_tail: torch.Tensor, new_head: torch.Tensor,
                   ctx: int = 40) -> float:
    """Frame-to-frame jump ACROSS the join divided by the median jump in the
    ±ctx frames around it.  ~1.0 = the join is as smooth as ordinary content;
    >>1 = a click/edge exactly at the seam."""
    with torch.no_grad():
        n = min(ctx, prev_tail.shape[-1], new_head.shape[-1])
        if n < 3:
            return float("nan")
        cat = torch.cat([_aud_flat(prev_tail)[..., -n:],
                         _aud_flat(new_head)[..., :n]], dim=-1)
        d = (cat[..., 1:] - cat[..., :-1]).pow(2).mean(dim=1).sqrt()[0]
        return float(d[n - 1]) / max(float(d.median()), 1e-8)


def _aud_best_lag(prev_tail: torch.Tensor, new_head: torch.Tensor,
                  win: int = 24, max_lag: int = 6) -> tuple[float, int]:
    """Best-lag cosine across the seam: last `win` frames before the join vs
    a lagged window after it.  (high cos, lag 0) = phase-aligned; a nonzero
    best lag = a residual px<->audio rounding offset (25 ms per frame — the
    bug family patch #8 fixed; this is the regression alarm for it)."""
    with torch.no_grad():
        pa, na = _aud_flat(prev_tail), _aud_flat(new_head)
        if pa.shape[-1] < win or na.shape[-1] < win + max_lag:
            return float("nan"), 0
        ref = F.normalize(pa[..., -win:].reshape(1, -1), dim=1)
        best, best_lag = -2.0, 0
        for lag in range(max_lag + 1):
            w = F.normalize(na[..., lag:lag + win].reshape(1, -1), dim=1)
            c = float((ref * w).sum())
            if c > best:
                best, best_lag = c, lag
        return best, best_lag


def _aud_loop_ncc(history: torch.Tensor, new_chunk: torch.Tensor,
                  ) -> tuple[float, float]:
    """Loop-lock detector on DELIVERED audio: normalized cross-correlation of
    the new chunk against ALL previously delivered audio of the scene,
    searched over every time shift (FFT convolution).  A chunk that replays
    earlier material — the loop failure — peaks near 1.0 at the shift where
    the copy lives; fresh content stays near 0.  Returns (ncc, seconds)."""
    with torch.no_grad():
        f = _aud_flat(history)[0]      # [D, Th]
        g = _aud_flat(new_chunk)[0]    # [D, Tn]
        Th, Tn = f.shape[-1], g.shape[-1]
        if Th < Tn or Tn < 8:
            return float("nan"), float("nan")
        nfft = 1 << (Th + Tn - 1).bit_length()
        # cc[k] = sum_d sum_t f_d[t+k] * g_d[t]  ->  new[0] aligned at hist k
        cc = torch.fft.irfft(torch.fft.rfft(f, n=nfft)
                             * torch.fft.rfft(g, n=nfft).conj(),
                             n=nfft)[..., :Th].sum(0)
        e = f.pow(2).sum(0).cumsum(0)
        win_e = e[Tn - 1:] - torch.cat([e.new_zeros(1), e[:-Tn]])
        ncc = cc[:win_e.numel()] / (win_e * g.pow(2).sum()).clamp(min=1e-12).sqrt()
        k = int(ncc.argmax())
        return float(ncc[k]), k / AUDIO_LATENT_FPS


def _aud_within_chunk_sims(new_aud: torch.Tensor, n_seg: int = 3) -> list[float]:
    """Cosine sims between consecutive thirds of a chunk's new audio.

    Detects the chunked-audio "metronome" fixed-point (within-chunk segment
    repetition reads as rising similarity).
    """
    T = new_aud.shape[-1]
    if T < n_seg * 2:
        return []
    seg_len = T // n_seg
    sims: list[float] = []
    with torch.no_grad():
        for i in range(n_seg - 1):
            s1 = new_aud[..., i * seg_len:(i + 1) * seg_len].float().mean(dim=-1)
            s2 = new_aud[..., (i + 1) * seg_len:(i + 2) * seg_len].float().mean(dim=-1)
            f1 = F.normalize(s1.reshape(new_aud.shape[0], -1), dim=1)
            f2 = F.normalize(s2.reshape(new_aud.shape[0], -1), dim=1)
            sims.append((f1 * f2).sum(dim=1).mean().item())
    return sims


def _post_process_audio_latent(
    audio_lat: torch.Tensor,
    chunk_ends: list[int],
    smooth_half: int = 2,
    energy_beta: float = 0.0,
    label: str = "",
) -> torch.Tensor:
    """Cross-chunk audio hygiene on the concatenated [B, 32, 2, Ta] latent.

    energy_beta > 0 soft-matches each chunk's RMS toward the median chunk
    (disabled at the call site — kept for experiments).  The always-on part
    smooths a few frames across each chunk boundary to hide seam clicks.
    """
    if not chunk_ends:
        return audio_lat
    audio_lat = audio_lat.clone()
    T = audio_lat.shape[-1]
    boundaries = [0] + list(chunk_ends)
    n = len(chunk_ends)
    if n >= 2 and energy_beta > 0.0:
        chunk_rms = []
        for i in range(n):
            seg = audio_lat[..., boundaries[i]:boundaries[i + 1]].float()
            chunk_rms.append(seg.pow(2).mean().sqrt().item())
        median_rms = sorted(chunk_rms)[n // 2]
        if median_rms > 1e-6:
            for i in range(n):
                if chunk_rms[i] < 1e-6:
                    continue
                raw_gain = median_rms / chunk_rms[i]
                soft_gain = 1.0 + energy_beta * (raw_gain - 1.0)
                if abs(soft_gain - 1.0) > 0.005:
                    audio_lat[..., boundaries[i]:boundaries[i + 1]] = (
                        audio_lat[..., boundaries[i]:boundaries[i + 1]] * soft_gain
                    )
    for boundary in chunk_ends[:-1]:
        b = boundary
        if b < smooth_half or b + smooth_half > T:
            continue
        for i in range(1, smooth_half + 1):
            alpha = i / (smooth_half + 1)
            prev = b - i
            nxt = b + i - 1
            audio_lat[..., prev] = (
                (1.0 - alpha) * audio_lat[..., prev] + alpha * audio_lat[..., b]
            )
            audio_lat[..., nxt] = (
                (1.0 - alpha) * audio_lat[..., nxt] + alpha * audio_lat[..., b - 1]
            )
    return audio_lat


def _tau_c_eff(base: float, ceiling: float, chunk_idx: int, half_life: float = 5.0) -> float:
    """§2.1 schedule: rise from base toward ceiling with a 5-chunk half-life."""
    if base <= 0.0:
        return 0.0
    decay = 0.5 ** (chunk_idx / half_life)
    return ceiling - (ceiling - base) * decay


_GUIDE_LAYOUT_CHECKED = False


def _check_guide_layout() -> None:
    """The end-aligned audio guide needs a fractional, NEGATIVE
    resolved_frame_index to be taken literally by PackedLayout
    (cond_t = cursor + FRAME_RESCALE * index).  ComfyUI < 0.34's layout
    (constructor still takes frame_count) rejected any keyframe anchor
    other than the first/last frame, so on it the guide would land at the
    wrong instant or fail deep inside the model.  Fail loudly with the
    fix instead (the H3-Motion-Context pack's layout_contract does the
    full behavioural proof; the signature check covers the one breaking
    upstream change known to matter here)."""
    global _GUIDE_LAYOUT_CHECKED
    if _GUIDE_LAYOUT_CHECKED:
        return
    import inspect
    from comfy.ldm.minimax.model import PackedLayout
    try:
        params = inspect.signature(PackedLayout.__init__).parameters
    except (TypeError, ValueError):
        params = {}
    if "frame_count" in params:
        raise RuntimeError(
            "[CLSS] audio_guide_seconds > 0 needs ComfyUI 0.34.0 or newer: "
            "this ComfyUI's PackedLayout still takes frame_count and only "
            "accepts first/last-frame keyframe anchors, so the guide's "
            "fractional negative index would not be honoured. Update "
            "ComfyUI, or set audio_guide_seconds to 0.")
    _GUIDE_LAYOUT_CHECKED = True


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------


class CLSSH3Config:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tau_c":   ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5,  "step": 0.01,
                                      "tooltip": "§2.1 calibrated context re-noising: per-token sigma level the SLB overlap is re-noised to (H3 turns mask m into per-row sigma = m·σ). 0 = fully frozen overlap (maximal continuity, maximal drift accumulation); higher = more distributional repair at the cost of softer motion lock. The per-chunk schedule rises from this base toward a 0.10 ceiling with a 5-chunk half-life.",
                                      }),
                "beta":    ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0,  "step": 0.05,
                                      "tooltip": "§2.3 drift correction: blend factor of the EMA-tracked per-channel AdaIN renormalisation applied to every new chunk. 0 = no correction, 1 = full replacement with the EMA reference statistics. The EMA reference resets at every scene change — the first chunk of a new scene is uncorrected and re-anchors it.",
                                      }),
                "overlap": ("INT",   {"default": 7,    "min": 2,   "max": 32,
                                      "tooltip": "SLB size in video latent tokens shared between consecutive chunks, snapped to the 5k+2 grid (2, 7, 12, …; 7 tokens ≈ 22 px frames ≈ 0.9 s at 24 fps) — the hard temporal context the model sees from the previous chunk. Auto-clamped down (in steps of 5) at runtime so overlap+new stays under the 12 s window cap.",
                                      }),
            },
        }
    RETURN_TYPES = ("CLSS_CONFIG",)
    RETURN_NAMES = ("clss_config",)
    FUNCTION = "build"
    CATEGORY = "MiniMaxH3-CLSS"

    def build(self, tau_c, beta, overlap):
        # Validated-production values carried over from the LTX node layer;
        # the dataclass defaults in clss.py are the paper's, not ours.
        return (CLSSConfig(
            tau_c=tau_c,
            beta=beta,
            ema_lambda=0.10,
            ema_sigma_max_drift=0.05,
            overlap_latent_frames=_snap_overlap(overlap),
            adain_max_amplification=1.2,
        ),)


class CLSSH3ScenePrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip":    ("CLIP",   {"tooltip": "CLIP (qwen3vl-32B) text encoder. Each scene's raw text is encoded as its own CONDITIONING entry — no system prompt / chat template (that was LTX-specific; H3's RoPE t-origin sits after the text span, so a scene's text must stay identical across its chunks, which raw reuse guarantees)."}),
                "prompts": ("STRING", {"multiline": True, "dynamicPrompts": False,
                                       "default": "Scene 1 description\n---\nScene 2 description",
                                       "tooltip": "One scene per block, separated by a line containing only '---'. Each scene is encoded as its own CONDITIONING entry (minimax_token_tags preserved); with N entries the sampler assigns one scene per chunk proportionally across num_chunks.",
                                       }),
            },
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "generate"
    CATEGORY = "MiniMaxH3-CLSS"

    def generate(self, clip, prompts: str):
        scenes = [s.strip() for s in prompts.split("\n---\n") if s.strip()]
        if not scenes:
            scenes = [prompts.strip()]
        flat_conditioning = []
        for scene in scenes:
            # raw text, no chat template; encode_token_weights attaches
            # minimax_token_tags to the conditioning entry automatically.
            flat_conditioning.extend(
                clip.encode_from_tokens_scheduled(clip.tokenize(scene))
            )
        return (flat_conditioning,)


# ---------------------------------------------------------------------------
# Split-CFG guider: unpack the packed AV output, CFG per stream, repack.
# ---------------------------------------------------------------------------


class _GuiderCLSSH3(comfy.samplers.CFGGuider):
    """CFGGuider with independent video/audio CFG over the packed AV latent.

    calc_cond_batch returns the packed [B, 1, N] denoised predictions
    (model_base.py:250-253 packs the model's [video, audio] list output).
    We unpack with the latent shapes recorded in sample(), apply
    uncond + cfg·(cond − uncond) per stream, rescale each stream toward its
    conditional prediction's std, and repack.  This is the minimal port of
    the LTX `_GuiderCLSSAV`: no STG, no modality_scale, no per-modality sigma
    re-derivation — H3's ModelSamplingAV owns the audio schedule internally.
    """

    # H3 is CFG-distilled: the stock graph runs cfg=1.0 (BasicGuider), and a
    # live A/B measured cfg 4.0/7.0 corrupting BOTH streams into oversaturated
    # glitch (latent std 0.93->1.13).  These class attributes are only a
    # fallback — CLSSH3Guider.get_guider always overrides them through
    # set_av_params — so keep them at the safe values.
    _video_cfg = 1.0
    _audio_cfg = 1.0
    _rescale = 0.7
    _av_latent_shapes = None

    def set_av_params(self, video_cfg, audio_cfg, rescale):
        self._video_cfg = video_cfg
        self._audio_cfg = audio_cfg
        self._rescale = rescale
        self.set_cfg(video_cfg)      # feeds any non-AV fallback path
        self.audio_cfg = audio_cfg   # telemetry introspection by the sampler

    @staticmethod
    def _rescale_pred(pred: torch.Tensor, cond: torch.Tensor, r: float) -> torch.Tensor:
        if r <= 0.0:
            return pred
        ratio = cond.float().std() / pred.float().std().clamp(min=1e-8)
        factor = (r * ratio + (1.0 - r)).clamp(0.5, 2.0)
        return pred * factor.to(pred.dtype)

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None,
               callback=None, disable_pbar=False, seed=None):
        if getattr(latent_image, "is_nested", False):
            self._av_latent_shapes = [t.shape for t in latent_image.unbind()]
        else:
            self._av_latent_shapes = None
        return super().sample(noise, latent_image, sampler, sigmas,
                              denoise_mask=denoise_mask, callback=callback,
                              disable_pbar=disable_pbar, seed=seed)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        positive = self.conds.get("positive", None)
        negative = self.conds.get("negative", None)
        is_nested = isinstance(x, comfy.nested_tensor.NestedTensor)
        shapes = self._av_latent_shapes
        is_packed_av = (not is_nested and shapes is not None
                        and len(shapes) == 2 and getattr(x, "ndim", 0) == 3)
        if (not is_nested and not is_packed_av) or negative is None:
            return super().predict_noise(x, timestep, model_options, seed)
        if self._video_cfg == 1.0 and self._audio_cfg == 1.0:
            # cfg=1 on both streams is exactly a conditional-only pass
            # (uncond + 1·(cond − uncond) ≡ cond, rescale ratio 1) — skip the
            # uncond eval, halving model evals per step.
            # calc_cond_batch ALWAYS returns a list (one entry per cond) —
            # unwrap it, the sampler loop expects a bare tensor.
            return comfy.samplers.calc_cond_batch(
                self.inner_model, [positive], x, timestep, model_options)[0]

        def _split(t):
            if isinstance(t, comfy.nested_tensor.NestedTensor):
                return t.unbind()
            return comfy.utils.unpack_latents(t, shapes)

        def _join(v, a):
            if is_nested:
                return comfy.nested_tensor.NestedTensor((v, a))
            return comfy.utils.pack_latents([v, a])[0]

        out_cond, out_uncond = comfy.samplers.calc_cond_batch(
            self.inner_model, [positive, negative], x, timestep, model_options
        )
        vid_c, aud_c = _split(out_cond)
        vid_u, aud_u = _split(out_uncond)
        pred_v = vid_u + self._video_cfg * (vid_c - vid_u)
        pred_a = aud_u + self._audio_cfg * (aud_c - aud_u)
        pred_v = self._rescale_pred(pred_v, vid_c, self._rescale)
        pred_a = self._rescale_pred(pred_a, aud_c, self._rescale)
        return _join(pred_v, pred_a)


class CLSSH3Guider:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":    ("MODEL",        {"tooltip": "MODEL (MiniMax H3) the guider is built on."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive CONDITIONING. One entry per scene (from CLSSH3ScenePrompts) enables per-scene chunk guidance in the sampler."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative CONDITIONING, required for split CFG; without it the guider falls back to a plain conditional pass."}),
                "video_cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.5,
                                        "tooltip": "Video CFG scale. DEFAULT 1.0 = off: H3 is CFG-distilled (the official graph uses BasicGuider with no CFG) and a live A/B measured cfg 4.0 producing corrupted, oversaturated glitch frames (latent std 0.93→1.13). Raise only as an experiment."}),
                "audio_cfg": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Audio CFG scale, independent of video_cfg. DEFAULT 1.0 = off (H3 is CFG-distilled; the LTX-derived 7.0 was measured corrupting output on a live run). With video_cfg==audio_cfg==1.0 the guider skips the uncond eval entirely — same cost as stock BasicGuider.",
                }),
                "rescale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Per-stream CFG rescale toward the conditional prediction's std (factor = r·std_ratio + 1−r, clamped to [0.5, 2.0]). 0 = off. Counteracts CFG oversaturation.",
                                      }),
            },
        }
    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "MiniMaxH3-CLSS"

    def get_guider(self, model, positive, negative, video_cfg, audio_cfg, rescale):
        if video_cfg != 1.0 or audio_cfg != 1.0:
            # H3 is CFG-distilled: cfg 4/7 was measured live to corrupt frames
            # into oversaturated glitch (latent std 0.93→1.13). Warn loudly —
            # a stale browser tab can silently keep pre-fix widget values.
            print(f"[CLSS] WARNING: video_cfg={video_cfg} audio_cfg={audio_cfg} on "
                  f"CFG-distilled H3 — measured to corrupt output; use 1.0/1.0 unless "
                  f"you are deliberately experimenting.")
        guider = _GuiderCLSSH3(model)
        guider.set_conds(positive, negative)
        guider.set_av_params(video_cfg, audio_cfg, rescale)
        return (guider,)


# ---------------------------------------------------------------------------
# Sliced noise: each chunk's initial noise is cut from one run-constant
# full-length noise field so the new region's noise is continuous across the
# seam (the overlap region gets fresh noise — it is re-noised by the mask
# anyway).  H3 audio latents are [B, 32, 2, Ta]: time on the LAST axis.
# ---------------------------------------------------------------------------


class _SlicedNoise:
    def __init__(self, full_noise_vid: torch.Tensor, pos: int, chunk_overlap: int, seed: int = 0,
                 full_noise_aud: torch.Tensor | None = None, a_pos: int = 0, a_overlap: int = 0):
        self._full = full_noise_vid
        self._pos = pos
        self._chunk_overlap = chunk_overlap
        self._full_aud = full_noise_aud
        self._a_pos = a_pos
        self._a_overlap = a_overlap
        self.seed = seed

    def generate_noise(self, input_latent: dict):
        samples = input_latent["samples"]
        is_av = isinstance(samples, comfy.nested_tensor.NestedTensor)
        vid = samples.unbind()[0] if is_av else samples
        _g = torch.Generator(device="cpu").manual_seed(
            (int(self.seed) % (2 ** 31)) * 1_000_003
            + self._pos * 7_919 + self._a_pos * 104_729)
        noise_vid = torch.randn(vid.shape, generator=_g, dtype=vid.dtype).to(vid.device)
        n_new = vid.shape[2] - self._chunk_overlap
        src_end = min(self._pos + n_new, self._full.shape[2])
        src_n = src_end - self._pos
        if src_n > 0:
            noise_vid[:, :, self._chunk_overlap:self._chunk_overlap + src_n] = \
                self._full[:, :, self._pos:src_end].to(vid.device)
        if is_av:
            aud = samples.unbind()[1]
            noise_aud = torch.randn(aud.shape, generator=_g, dtype=aud.dtype).to(aud.device)
            if self._full_aud is not None:
                a_new = aud.shape[-1] - self._a_overlap
                a_end = min(self._a_pos + a_new, self._full_aud.shape[-1])
                a_n = a_end - self._a_pos
                if a_n > 0:
                    noise_aud[..., self._a_overlap:self._a_overlap + a_n] = \
                        self._full_aud[..., self._a_pos:a_end].to(aud.device)
            return comfy.nested_tensor.NestedTensor((noise_vid, noise_aud))
        return noise_vid


# ---------------------------------------------------------------------------
# The streaming sampler
# ---------------------------------------------------------------------------


class CLSSH3StreamingSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider":      ("GUIDER",      {"tooltip": "GUIDER from CLSSH3Guider. When its positive conditioning holds N scene entries, one scene is unpacked per chunk proportionally across num_chunks."}),
                "sampler":     ("SAMPLER",     {"tooltip": "SAMPLER for the per-chunk denoise (KSamplerSelect). The audio stream's own shifted schedule is handled inside the model (ModelSamplingAV); set shifts on the stock MiniMaxH3SigmaShift node."}),
                "sigmas":      ("SIGMAS",      {"tooltip": "SIGMAS schedule (e.g. BasicScheduler). This is the video schedule; the audio schedule is derived from it inside the model."}),
                "noise":       ("NOISE",       {"tooltip": "NOISE source (RandomNoise). Its seed drives the run-constant full-length noise tensors that each chunk's initial noise is sliced from."}),
                "latent":      ("LATENT",      {"tooltip": "Per-chunk AV latent template from EmptyMiniMaxH3LatentAV (NestedTensor video [B,24,T,H/16,W/16] + audio [B,32,2,Ta]). Its frame count sets the per-chunk length; total length = num_chunks × chunk. T must be on the 5k+2 grid (the stock node guarantees this)."}),
                "clss_config": ("CLSS_CONFIG", {"tooltip": "CLSS_CONFIG from the CLSSH3Config node (tau_c, beta, overlap)."}),
                "num_chunks":  ("INT",         {"default": 10, "min": 1, "max": 500,
                                                "tooltip": "Number of streaming chunks; total video length = num_chunks × chunk length (chunk 0 covers 17k+5 px, each continuation 17k px). A chunk whose window (overlap+new) would exceed the 12 s cap is auto-split into uniform grid-aligned sub-chunks. With scene_handoff=transition_chunk every scene block needs ≥ 2 chunks, i.e. num_chunks ≥ 2×scenes.",
                                                }),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional i2v guide image; VAE-encoded and pinned as a minimax_keyframes row at frame 0 of chunk 0 (the H3-native first-frame conditioning). Requires vae."}),
                "vae":   ("VAE",   {"tooltip": "Video VAE, only needed together with image for the i2v guide encode."}),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Frames per second of the output. H3 is 24 fps native — the px↔audio time mapping is fixed (40 audio latent fps, temporal_shape), so any other value only triggers a warning and 24 is used.",
                    }),
                "detail_anchor": (["on", "off"], {
                    "default": "on",
                    "tooltip": "Two-band spatial detail anchor: each chunk's low/high-frequency band energies are re-scaled toward the scene's first-chunk reference (gains clamped to [0.90, 1.10] low / [0.90, 1.12] high) to fight the long-run detail fade. Off = uncorrected.",
                    }),
                "video_slb_tau_mult": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.25,
                    "tooltip": "Scales the video overlap re-noise: effective tau_c × this, still rising toward the 0.10 ceiling with the 5-chunk half-life. 0 = frozen clean seam (no re-noising of the video overlap).",
                    }),
                "audio_guide_seconds": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.25,
                    "tooltip": "Audio seam guide: pins the last N seconds of the previous chunk's audio as a cond_audio guide keyframe whose window ENDS exactly at the join and reaches BACKWARD (fractional/negative anchor index — the H3-Motion-Context pack's measured mechanism: seam correlation 0.45 -> 0.95+ vs reference placement; a forward/overlap-aligned guide makes the model loop the motif instead). The guide is the ONLY audio context: the overlap rows are fresh noise and the join is a plain cut. 1.0 s = 40 latent steps, the pack's validated default. 0 = off. Requires ComfyUI 0.34.0+ (fractional/negative keyframe anchors).",
                    }),
                "audio_cfg_cont": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 30.0, "step": 0.5,
                    "tooltip": "Audio CFG for CONTINUATION chunks (chunk 2+). Chunk 1 keeps the guider's audio_cfg and establishes the sound; every later chunk drops to this. DEFAULT 1.0 = off, and it is the measured fix for chunk-boundary section changes: the SLB overlap sits in the shared latent, so it is present in BOTH cond and uncond passes and cancels out of the CFG direction — at audio_cfg 4 the re-applied text prompt is amplified 4x while the tail context contributes nothing to guidance, so the model opens a NEW musical section at every join. With video_cfg 1.0 + this at 1.0 the guider also skips the uncond pass entirely (half the model evals per step on continuation chunks).",
                    }),
                # How the text conditioning changes at a scene boundary:
                # "transition_chunk" — two-step crossfade straddling the boundary: the
                #   outgoing scene block's last chunk is guided by a 25%-incoming blend,
                #   the incoming scene block's first chunk by 75%-incoming; needs every
                #   scene block >= 2 chunks, i.e. num_chunks >= 2*scenes (3 scenes -> 6).
                # "blend" — the first chunk of each new scene gets a single 50/50 blend.
                # "hard"  — plain text swap (pre-crossfade baseline).
                "scene_handoff": (["transition_chunk", "blend", "hard"], {
                    "default": "transition_chunk",
                    "tooltip": "How text conditioning changes at a scene boundary. transition_chunk: two-step crossfade straddling the boundary — the outgoing scene's last chunk is guided by a 25%-incoming embedding blend, the incoming scene's first chunk by 75%-incoming (on LTX a single 50/50 chunk between far-apart scenes measured off-manifold and poisoned the next scene's SLB); needs every scene block ≥ 2 chunks. blend: single 50/50 blend on the first chunk of the new scene. hard: plain text swap.",
                    }),
            },
        }
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "generate"
    CATEGORY = "MiniMaxH3-CLSS"

    @torch.inference_mode()
    def generate(
        self,
        guider,
        sampler,
        sigmas,
        noise,
        latent,
        clss_config: CLSSConfig,
        num_chunks: int,
        image=None,
        vae=None,
        fps: float = 24.0,
        detail_anchor: str = "on",
        video_slb_tau_mult: float = 1.0,
        audio_guide_seconds: float = 1.0,
        audio_cfg_cont: float = 1.0,
        scene_handoff: str = "transition_chunk",
    ):
        if fps != _NATIVE_FPS:
            # The px↔audio time mapping is hard-wired (40 audio latent fps at
            # 24 px fps — temporal_shape); a non-24 fps would silently desync
            # the audio seam math, so refuse it rather than approximate.
            print(f"[CLSS] WARNING: fps={fps} requested but MiniMax H3 is "
                  f"{_NATIVE_FPS} fps native; using {_NATIVE_FPS}.")
        fps = float(_NATIVE_FPS)

        # Guard the schedule: it is reused for EVERY chunk, each window starting
        # at sigmas[0] and ending at sigmas[-1].  H3 is a flow model (CONST
        # sampling, sigma in [0, 1]); BasicScheduler on the shifted model yields
        # exactly [1.0 ... 0.0].  A foreign schedule (karras/exponential/beta
        # from a non-H3 workflow) starts every chunk at sigma_max * noise with
        # sigma_max >> 1 and drives the DiT's timestep labels out of range, and
        # a schedule not terminating at 0 leaves the final latent noisy — in
        # both cases the run COMPLETES but video AND audio decode as pure noise.
        # Fail loudly instead of silently returning noise.
        _s = sigmas.flatten().float().cpu()
        if (_s.numel() < 2 or not (0.98 <= float(_s[0]) <= 1.02)
                or float(_s[-1]) > 1e-4 or not bool((_s[:-1] >= _s[1:]).all())):
            raise ValueError(
                "[CLSS] sigmas must be a monotonically decreasing flow schedule "
                "from 1.0 to 0.0 (use BasicScheduler on the MiniMaxH3SigmaShift-"
                f"patched model); got [0]={float(_s[0]):.6g} "
                f"[-1]={float(_s[-1]):.6g} len={_s.numel()}")

        samples = latent["samples"]
        if not (getattr(samples, "is_nested", False) and len(samples.unbind()) == 2):
            raise ValueError("CLSSH3StreamingSampler expects a MiniMax H3 AV latent "
                             "(NestedTensor video+audio) from EmptyMiniMaxH3LatentAV")
        vid_tmpl, aud_tmpl = samples.unbind()
        B, C_v, new_lf0, H, W = vid_tmpl.shape
        B_a, C_a, lanes_a, _Ta0 = aud_tmpl.shape
        device = vid_tmpl.device
        if new_lf0 % 5 != 2 or new_lf0 < 7:
            raise ValueError(f"chunk template video latent T={new_lf0} is not on the "
                             f"5k+2 grid (k>=1) — use EmptyMiniMaxH3LatentAV, which snaps "
                             f"the frame count to 17k+5 px")
        overlap = _snap_overlap(clss_config.overlap_latent_frames)
        # chunk 0 contributes 5k+2 tokens (px 17k+5); every continuation chunk
        # a multiple of 5 (px 17k) so chunk starts stay at absolute phase 2.
        new_cont = new_lf0 - 2

        # ---- i2v guide image → keyframe row (H3-native first-frame pin) ----
        img_guide_latent: torch.Tensor | None = None
        if image is not None and vae is not None:
            img = image[:1, ..., :3].movedim(-1, 1)
            img = comfy.utils.common_upscale(img, W * 16, H * 16, "lanczos", "disabled")
            img_guide_latent = vae.encode(img.movedim(1, -1))  # [1, 24, 1, H, W]

        # ---- window budget: 12 s soft cap (H3 trained range ~5-15 s) ----
        px0 = _px_of_tokens(new_lf0, 0)          # 17k+5
        pxc = _px_of_tokens(new_cont, 2)         # 17k
        cap_px = int(_WINDOW_CAP_S * fps)
        _eff_overlap = overlap
        _win_px = px0 if num_chunks == 1 else _px_of_tokens(_eff_overlap, 0) + max(px0, pxc)
        # overlap clamp only matters when an SLB actually exists (chunk ≥ 1)
        while num_chunks > 1 and _win_px > cap_px and _eff_overlap > _MIN_OVERLAP_TOKENS:
            _eff_overlap -= 5
            _win_px = _px_of_tokens(_eff_overlap, 0) + max(px0, pxc)
        px_ol = _px_of_tokens(_eff_overlap, 0)
        plan_tokens: list[int] = []
        if _win_px <= cap_px:
            plan_tokens = [new_lf0] + [new_cont] * (num_chunks - 1)
        elif px_ol + max(px0, pxc) > cap_px:
            # the chunk alone would exceed the cap even at minimum overlap:
            # split into the fewest uniform grid-aligned sub-chunks that fit.
            rem = max(6, cap_px - px_ol)
            max_new0 = 5 * max(1, (rem - 5) // 17) + 2   # px(5a+2 @phase0) = 17a+5 ≤ rem
            max_newc = 5 * max(1, rem // 17)             # px(5b @phase2)  = 17b   ≤ rem
            for _ci in range(num_chunks):
                if _ci == 0:
                    plan_tokens.extend(_split_run(new_lf0, max_new0, plus2=True))
                else:
                    plan_tokens.extend(_split_run(new_cont, max_newc, plus2=False))
            print(f"[CLSS] chunk window exceeds the {_WINDOW_CAP_S:.0f} s cap — "
                  f"auto-split into {len(plan_tokens)} sub-chunks "
                  f"(overlap clamped to {_eff_overlap} tokens).")
        else:
            plan_tokens = [new_lf0] + [new_cont] * (num_chunks - 1)
        if _eff_overlap != overlap:
            print(f"[CLSS] overlap clamped {overlap} -> {_eff_overlap} tokens "
                  f"to keep windows under the {_WINDOW_CAP_S:.0f} s cap.")
            clss_config = dataclasses.replace(clss_config, overlap_latent_frames=_eff_overlap)

        # per-chunk new audio frames from CUMULATIVE absolute px positions, so
        # the concatenated audio matches round(total_px × 5/3) exactly
        # (temporal_shape of the full video) with no per-chunk rounding drift.
        chunk_plan: list[tuple[int, int, int]] = []   # (new video tokens, new audio frames, new px frames)
        _p_acc = _a_acc = _t_acc = 0
        for _n in plan_tokens:
            _px_new = _px_of_tokens(_n, _t_acc % 5)
            _p_acc += _px_new
            _a_end = _af_of_px(_p_acc)
            chunk_plan.append((_n, _a_end - _a_acc, _px_new))
            _a_acc = _a_end
            _t_acc += _n
        _eff_num_chunks = len(chunk_plan)
        T_total, Ta_total = _t_acc, _a_acc
        Ta_ol = _af_of_px(px_ol)

        # ---- scene hand-off plan (_cond_plan), one entry per chunk:
        #   int               → chunk guided by that scene's prompt alone
        #   (int, int, float) → crossfade chunk guided by the embedding blend of
        #                       (outgoing, incoming) scene with incoming weight w
        #                       (_blend_scene_cond)
        # "transition_chunk": TWO-STEP crossfade straddling each boundary — the
        # outgoing scene's last chunk gets w=0.25 (mostly outgoing), the incoming
        # scene's first chunk gets w=0.75 (mostly incoming).  A single 50/50
        # chunk between far-apart scenes is off-manifold guidance: measured live
        # on LTX, the 50/50 transition chunk drifted to anchor-sim 0.24 and
        # poisoned the next scene's SLB.  A scene block needs >=2 chunks to host
        # its half of the crossfade; 1-chunk blocks fall through to hard swaps.
        pos_conds = guider.original_conds.get("positive", [])
        num_scenes = len(pos_conds)
        _scene_of = [min(int(_i * num_scenes / _eff_num_chunks), num_scenes - 1)
                     if num_scenes > 1 else 0
                     for _i in range(_eff_num_chunks)]
        # ---- run banner: every knob that shapes the run, printed once, so a
        # telemetry dump is self-describing (which mult/guide produced which
        # trend line is never in question again). ----
        _bn_px0 = chunk_plan[0][2]
        _bn_pxc = chunk_plan[1][2] if _eff_num_chunks > 1 else 0
        _bn_af0 = chunk_plan[0][1]
        _bn_afc = chunk_plan[1][1] if _eff_num_chunks > 1 else 0
        _tv0 = _tau_c_eff(clss_config.tau_c * video_slb_tau_mult,
                          _VIDEO_TAU_C_CEILING, 0)
        _tvN = _tau_c_eff(clss_config.tau_c * video_slb_tau_mult,
                          _VIDEO_TAU_C_CEILING, max(0, _eff_num_chunks - 2))
        print("[CLSS] ================ run settings ================")
        print(f"[CLSS] chunks {_eff_num_chunks} (requested {num_chunks}) | "
              f"scenes {num_scenes} handoff={scene_handoff} | "
              f"seed {getattr(noise, 'seed', '?')}")
        print(f"[CLSS] chunk0 {_bn_px0}px->{_bn_af0}af ({_bn_px0 / fps:.2f}s) | "
              f"cont {_bn_pxc}px->{_bn_afc}af | "
              f"overlap {_eff_overlap}tok={px_ol}px/{Ta_ol}af "
              f"({px_ol / fps:.2f}s) | window {_win_px}px "
              f"({_win_px / fps:.2f}s, cap {cap_px}px) | "
              f"total ~{Ta_total / AUDIO_LATENT_FPS:.1f}s audio")
        print(f"[CLSS] video SLB tau_v {_tv0:.3f}->{_tvN:.3f} "
              f"(ceiling {_VIDEO_TAU_C_CEILING}) | audio: free overlap + "
              f"end-aligned guide, plain cut (no audio SLB — H3-MC mode)")
        print(f"[CLSS] guide {audio_guide_seconds:.2f}s = "
              f"{round(audio_guide_seconds * AUDIO_LATENT_FPS)}af end-aligned at "
              f"join | detail_anchor {detail_anchor} | clss tau_c "
              f"{clss_config.tau_c} beta {getattr(clss_config, 'beta', '?')} "
              f"overlap {clss_config.overlap_latent_frames}tok")
        print(f"[CLSS] sigmas {_s.numel() - 1} steps "
              f"[{float(_s[0]):.3f}..{float(_s[-1]):.3f}] | cfg v="
              f"{getattr(guider, '_video_cfg', '?')} a="
              f"{getattr(guider, 'audio_cfg', '?')} rescale="
              f"{getattr(guider, '_rescale', '?')} | audio cfg "
              f"continuation chunks={audio_cfg_cont}")
        print("[CLSS] ================================================")
        _cond_plan: list = list(_scene_of)
        if num_scenes > 1 and scene_handoff != "hard":
            for _i in range(_eff_num_chunks):
                _s = _scene_of[_i]
                _prv = _scene_of[_i - 1] if _i > 0 else None
                _nxt = _scene_of[_i + 1] if _i + 1 < _eff_num_chunks else None
                if scene_handoff == "blend":
                    if _prv is not None and _s != _prv:
                        _cond_plan[_i] = (_prv, _s, 0.5)
                elif _nxt is not None and _nxt != _s and _scene_of.count(_s) >= 2:
                    _cond_plan[_i] = (_s, _nxt, 0.25)
                elif _prv is not None and _prv != _s and _scene_of.count(_s) >= 2:
                    _cond_plan[_i] = (_prv, _s, 0.75)
            if (scene_handoff == "transition_chunk"
                    and not any(isinstance(_e, tuple) for _e in _cond_plan)):
                print(f"[CLSS] scene_handoff=transition_chunk but every scene has a "
                      f"single chunk ({num_scenes} scenes / {_eff_num_chunks} chunks) — "
                      f"no crossfade inserted; use num_chunks >= 2*scenes "
                      f"(e.g. 6 for 3 scenes).")

        _max_win_px = max(
            [_px_of_tokens(chunk_plan[0][0], 0)]
            + [px_ol + _px_of_tokens(_p, _eff_overlap % 5) for _p, _a, _pxn in chunk_plan[1:]]
        )
        print(f"[CLSS] plan: {_eff_num_chunks} chunk(s) of {plan_tokens[0]}"
              f"{'+' + str(plan_tokens[1]) if _eff_num_chunks > 1 else ''} tokens, "
              f"overlap={_eff_overlap} tokens ({px_ol} px / {Ta_ol} af), "
              f"window ≤ {_max_win_px / fps:.1f} s, "
              f"total {T_total} tokens / {_p_acc} px / {Ta_total} af "
              f"({_p_acc / fps:.1f} s), scenes={num_scenes}")

        # ---- run-constant full-length noise fields ----
        _noise_seed = getattr(noise, "seed", 0)
        # CPU template: generate_noise only reads shape/dtype, and the full-length
        # field is ~0.4 GB fp32 at long totals — no reason to touch VRAM for it.
        _noise_tmpl = torch.zeros(B, C_v, T_total, H, W)
        _full_noise_vid: torch.Tensor = noise.generate_noise({"samples": _noise_tmpl})
        del _noise_tmpl
        _g_aud = torch.Generator(device="cpu").manual_seed(
            (int(_noise_seed) + 1) % (2 ** 63))
        _full_noise_aud = torch.randn(B_a, C_a, lanes_a, Ta_total,
                                      generator=_g_aud, dtype=aud_tmpl.dtype)

        clss_state = CLSSState(clss_config)
        acc_video: list[torch.Tensor] = []
        acc_audio: list[torch.Tensor] = []
        audio_chunk_ends: list[int] = []
        # rolling kept-audio history (CPU, ends at the current join) — the
        # source of the end-aligned guide window (see audio_guide_seconds)
        _audio_tail: torch.Tensor | None = None
        _s1_prev_last: torch.Tensor | None = None
        _s1_aud_rms_ref: float | None = None
        _s1_vid_std_ref: float | None = None
        _prev_scene_idx: int | None = None
        _s1_band_ref: tuple[float, float] | None = None
        _origin_ref: torch.Tensor | None = None
        _origin_layout: torch.Tensor | None = None
        _prev_aud_env: torch.Tensor | None = None
        _s1_aud_prev_last: torch.Tensor | None = None
        _s1_prev_vfeat: torch.Tensor | None = None
        _hist_scene_start = 0
        _s1_audio_freq_ref: list[float] | None = None
        _trend = {
            "vid_std": [], "vid_intra": [], "vid_bnd": [],
            "vid_hf": [], "vid_origin": [],
            "aud_env": [], "aud_rms": [], "aud_bnd": [],
            "aud_dlv": [], "aud_lvl": [],
            "aud_wc": [], "aud_hf": [], "aud_hf_raw": [],
            "aud_peak": [], "aud_step": [], "aud_lag": [], "aud_lagf": [],
            "aud_loop": [], "aud_loopt": [], "vid_prev": [],
        }

        _vid_pos = 0
        _aud_pos = 0
        for chunk_idx in range(_eff_num_chunks):
            is_first = chunk_idx == 0
            _cur_new_lf, cur_new_af, _cur_new_px = chunk_plan[chunk_idx]
            _cfg_v, _cfg_a, _cfg_g = "tau_v=-", "aud=-", "guide=off"
            chunk_overlap = 0 if is_first else _eff_overlap
            total_lf = chunk_overlap + _cur_new_lf
            _plan_entry = _cond_plan[chunk_idx]
            _is_transition = isinstance(_plan_entry, tuple)
            # A crossfade chunk statistically belongs to the scene its text leans
            # toward (w < 0.5 → outgoing, w >= 0.5 → incoming): the per-scene ref
            # resets (incl. the §2.3 EMA) fire on the first incoming-leaning chunk.
            scene_idx = (_plan_entry[1] if _plan_entry[2] >= 0.5 else _plan_entry[0]) \
                if _is_transition else _plan_entry
            _scene_switch = (num_scenes > 1 and chunk_idx > 0
                             and _prev_scene_idx is not None
                             and scene_idx != _prev_scene_idx)
            if _scene_switch:
                _s1_vid_std_ref = None
                _s1_band_ref = None
                _origin_ref = None
                _origin_layout = None
                _s1_aud_rms_ref = None
                _s1_audio_freq_ref = None
                _s1_prev_vfeat = None
                _hist_scene_start = len(acc_audio)
                # §2.3: drop the old scene's EMA reference — the first chunk of the
                # new scene is uncorrected and re-anchors the EMA (incl. _init_std).
                clss_state.reset_drift_refs()
            _prev_scene_idx = scene_idx
            has_slb = not is_first and clss_state.overlap_latent is not None

            # ---- conditioning: scene (+ crossfade blend) + keyframe rows ----
            keyframes: list[dict] = []
            if is_first and img_guide_latent is not None:
                keyframes.append({"resolved_frame_index": 0, "latent": img_guide_latent})
            # ---- audio seam guide: cond_audio keyframe, END-aligned at the
            # join and reaching BACKWARD (H3-Motion-Context's measured
            # mechanism: seam correlation 0.45 -> 0.95+ vs reference
            # placement).  The pinned window must end at the join and reach
            # backward into audio that already played — a fractional,
            # negative resolved_frame_index, legal PackedLayout arithmetic
            # that no stock node produces.  The join is already an integer
            # on the 40 Hz grid, so no end snapping is needed. ----
            if (not is_first and audio_guide_seconds > 0.0
                    and _audio_tail is not None):
                _join_af = _af_of_px(px_ol + _cur_new_px) - cur_new_af
                _g = min(round(audio_guide_seconds * AUDIO_LATENT_FPS),
                         _audio_tail.shape[-1])
                if _g > 0 and _join_af > 0:
                    _check_guide_layout()
                    keyframes.append({
                        # start coord = join - g  =>  index = (join - g)/RESCALE;
                        # negative/fractional whenever g > overlap — intended.
                        "resolved_frame_index": (_join_af - _g) / FRAME_RESCALE,
                        "audio_latent": _audio_tail[..., -_g:],
                    })
                _cfg_g = (f"guide={_g}af@idx"
                          f"{((_join_af - _g) / FRAME_RESCALE):+.2f}")
            guider_chunk = copy.copy(guider)
            # Per-chunk audio CFG: chunk 0 runs the guider's audio_cfg (it
            # establishes the sound); every continuation chunk drops to
            # audio_cfg_cont (default 1.0 = off).  Measured on live runs:
            # the SLB overlap lives in the shared latent, so it is present
            # in BOTH cond and uncond passes and cancels out of the CFG
            # direction — at audio_cfg=4 the re-applied text prompt is
            # amplified 4x while the tail context contributes nothing to
            # guidance, and the model opens a NEW musical section at every
            # join.  copy.copy is shallow and floats are immutable: this
            # touches only the per-chunk copy, the user's guider keeps its
            # values.  With video_cfg 1.0 + audio 1.0 _GuiderCLSSH3 skips
            # the uncond pass entirely — half the model evals per step.
            if not is_first:
                guider_chunk._audio_cfg = float(audio_cfg_cont)
                guider_chunk.audio_cfg = float(audio_cfg_cont)  # telemetry attr
            _cfg_s = f"acfg={getattr(guider_chunk, '_audio_cfg', '?')}"
            if num_scenes > 1 or keyframes:
                _pos_entry = (_blend_scene_cond(pos_conds[_plan_entry[0]],
                                                pos_conds[_plan_entry[1]],
                                                _plan_entry[2])
                              if _is_transition else pos_conds[scene_idx])
                if keyframes:
                    _pos_entry = {**_pos_entry, "minimax_keyframes": keyframes}
                guider_chunk.original_conds = {
                    **guider.original_conds,
                    "positive": [_pos_entry],
                }

            # ---- chunk latent + per-stream denoise masks (§2.1) ----
            # Video mask [1,1,T,1,1] / audio mask [1,1,2,Ta]: prepare_mask
            # (comfy.utils.reshape_mask) interpolates them to the full latent
            # grid; with T/Ta already exact the temporal values survive
            # bit-exact, then _pool_masks_to_token_grid amaxes onto the 2x2
            # patch / per-frame token grid and ceil-quantizes to 1/256 steps
            # (model_base.py:2215-2232).
            lat_vid = torch.zeros(B, C_v, total_lf, H, W, device=device)
            mask_vid = torch.ones(1, 1, total_lf, 1, 1, device=device)
            if has_slb:
                _tau_c_v = _tau_c_eff(clss_config.tau_c * video_slb_tau_mult,
                                      _VIDEO_TAU_C_CEILING, chunk_idx - 1)
                _slb_v = clss_state.overlap_latent.to(device)
                _n_v = min(chunk_overlap, _slb_v.shape[2])
                lat_vid[:, :, :_n_v] = _slb_v[:, :, :_n_v]
                mask_vid[:, :, :_n_v] = _tau_c_v
                _cfg_v = f"tau_v={_tau_c_v:.3f}"
                # mask=0 verified to preserve: KSamplerX0Inpaint forces the
                # region's x0 output to latent_image (out = out·m + lat·(1−m))
                # and re-injects the cond-strength latent every step; the only
                # impurity is scale_latent_inpaint's 0.1% cond-noise-aug
                # (VISUAL_COND_TIMESTEP=0.999), which decays across steps as
                # the pinned x0 pulls the row back onto the clean latent.
                # Audio rows are even stricter (AUDIO_COND_TIMESTEP=1.0 — zero
                # aug): mask=0 audio is exactly frozen.
            # window audio length follows temporal_shape over the WHOLE window
            # (round(window_px * 5/3)) — Ta_ol + cur_new_af double-rounds and can
            # land ±1 audio frame off the model's px↔audio time map.
            chunk_af = _af_of_px((0 if is_first else px_ol) + _cur_new_px)
            _Ta_ol_w = chunk_af - cur_new_af  # window-local audio overlap (≈Ta_ol)
            lat_aud = torch.zeros(B_a, C_a, lanes_a, chunk_af, device=device)
            mask_aud = torch.ones(1, 1, lanes_a, chunk_af, device=device)
            # Audio rows stay 100% fresh noise under an all-ones mask: the
            # previous tail's only presence is the cond_audio guide keyframe
            # above, and the rendered overlap is trimmed with a plain cut
            # (H3-Motion-Context's recipe — no audio SLB, no crossfade).
            if not is_first:
                _cfg_a = "aud=free"
            chunk_latent = {
                "samples": comfy.nested_tensor.NestedTensor((lat_vid, lat_aud)),
                "noise_mask": comfy.nested_tensor.NestedTensor((mask_vid, mask_aud)),
            }

            if is_first:
                print(f"[CLSS] chunk {chunk_idx + 1}/{_eff_num_chunks} "
                      f"scene {scene_idx}: win {_cur_new_px}px/{chunk_af}af "
                      f"{_cfg_v} {_cfg_a} {_cfg_g} {_cfg_s}")
            else:
                print(f"[CLSS] chunk {chunk_idx + 1}/{_eff_num_chunks} "
                      f"scene {scene_idx}"
                      f"{' (transition)' if _is_transition else ''}: "
                      f"win {px_ol + _cur_new_px}px/{chunk_af}af "
                      f"join_af={_Ta_ol_w} (round {_Ta_ol_w - Ta_ol:+d}) "
                      f"{_cfg_v} {_cfg_a} {_cfg_g} {_cfg_s}")
            _chunk_noise = _SlicedNoise(
                _full_noise_vid, _vid_pos, chunk_overlap, seed=_noise_seed,
                full_noise_aud=_full_noise_aud,
                a_pos=_aud_pos,
                a_overlap=_Ta_ol_w,
            )
            _, denoised = SamplerCustomAdvanced().sample(
                noise=_chunk_noise,
                guider=guider_chunk,
                sampler=sampler,
                sigmas=sigmas,
                latent_image=chunk_latent,
            )
            vid_out, aud_out = denoised["samples"].unbind()

            # ---- §2.3 corrections on the new video frames ----
            new_vid = vid_out[:, :, chunk_overlap:]
            corrected = clss_state.post_process(new_vid)

            # two-band spatial detail anchor (fights the long-run detail fade)
            _da_x = corrected.float()
            _da_b, _da_c, _da_t, _da_h, _da_w = _da_x.shape
            _da_flat = _da_x.permute(0, 2, 1, 3, 4).contiguous().reshape(
                _da_b * _da_t, _da_c, _da_h, _da_w)
            _da_low = F.avg_pool2d(_da_flat, 3, stride=1, padding=1)
            _da_high = _da_flat - _da_low
            _e_low = float(_da_low.pow(2).mean())
            _e_high = float(_da_high.pow(2).mean())
            _hf_share = _e_high / max(_e_low + _e_high, 1e-12)
            if detail_anchor == "on":
                if _s1_band_ref is None:
                    _s1_band_ref = (_e_low, _e_high)
                else:
                    _g_lo = min(1.10, max(0.90, (_s1_band_ref[0] / max(_e_low, 1e-12)) ** 0.5))
                    _g_hi = min(1.12, max(0.90, (_s1_band_ref[1] / max(_e_high, 1e-12)) ** 0.5))
                    if abs(_g_lo - 1.0) > 0.005 or abs(_g_hi - 1.0) > 0.005:
                        corrected = (_da_low * _g_lo + _da_high * _g_hi).reshape(
                            _da_b, _da_t, _da_c, _da_h, _da_w
                        ).permute(0, 2, 1, 3, 4).contiguous().to(corrected.dtype)
                        _e_low_p, _e_high_p = _e_low * _g_lo ** 2, _e_high * _g_hi ** 2
                        _hf_share = _e_high_p / max(_e_low_p + _e_high_p, 1e-12)
            _trend["vid_hf"].append(_hf_share)

            # per-scene origin telemetry (min cosine distance of the chunk's
            # frames to the scene's first corrected frame + its coarse layout)
            if _origin_ref is None:
                _origin_ref = corrected[:, :, -1:].detach().float().cpu()
                _origin_layout = F.avg_pool2d(
                    _origin_ref[0].mean(0).square(), 3, stride=3).flatten()
            _oc = corrected.detach().float().cpu()
            _o_flat = _origin_ref.flatten()
            _osims, _lsims = [], []
            for _fi in range(_oc.shape[2]):
                _fr = _oc[:, :, _fi:_fi + 1]
                _osims.append(float(F.cosine_similarity(_fr.flatten(), _o_flat, dim=0)))
                _fl = F.avg_pool2d(_fr[0].mean(0).square(), 3, stride=3).flatten()
                _lsims.append(float(F.cosine_similarity(_fl, _origin_layout, dim=0)))
            _trend["vid_origin"].append(min(_osims))

            # per-scene video-std anchor: soft gain toward the first chunk's std
            if _s1_vid_std_ref is None:
                _s1_vid_std_ref = corrected.float().std().item()
            else:
                _cur_vstd = corrected.float().std().item()
                _ratio = _s1_vid_std_ref / max(_cur_vstd, 1e-6)
                if _ratio < 0.96 or _ratio > 1.04:
                    _g_v = 1.0 + 0.5 * (_ratio - 1.0)
                    _m = corrected.float().mean()
                    corrected = ((corrected.float() - _m) * _g_v + _m).to(corrected.dtype)
            _trend["vid_std"].append(corrected.float().std().item())

            clss_state.update_buffer(corrected)
            acc_video.append(corrected.cpu())
            _trend["vid_intra"].append(_frame_cos(corrected[:, :, 0], corrected[:, :, -1]))
            if _s1_prev_last is not None:
                _trend["vid_bnd"].append(_frame_cos(_s1_prev_last.to(device), corrected[:, :, 0]))
            # coarse scene signature (channel means over space+time): catches
            # the "morphs to a different scene halfway" failure that vid_bnd
            # (single boundary frame) and vid_origin (vs chunk 0) both miss.
            _cur_vfeat = F.normalize(
                corrected.float().mean(dim=(3, 4)).mean(dim=2), dim=1)
            if _s1_prev_vfeat is not None:
                _trend["vid_prev"].append(float(F.cosine_similarity(
                    _cur_vfeat, _s1_prev_vfeat.to(device), dim=1).mean()))
            _s1_prev_vfeat = _cur_vfeat.detach().cpu()
            _s1_prev_last = corrected[:, :, -1].cpu()

            # ---- audio: keep the new frames, update the audio SLB ----
            aud_drop = _Ta_ol_w
            if aud_drop > 0 and aud_out.shape[-1] < aud_drop:
                aud_drop = 0
            new_aud = aud_out[..., aud_drop:]
            _env = new_aud.detach().float().pow(2).mean(dim=(0, 1, 2)).cpu()
            if _prev_aud_env is not None and len(_prev_aud_env) > 8:
                _L = min(len(_env), len(_prev_aud_env))
                _ea = _env[:_L] - _env[:_L].mean()
                _eb = _prev_aud_env[:_L] - _prev_aud_env[:_L].mean()
                _trend["aud_env"].append(float((_ea * _eb).sum() /
                                               (_ea.norm() * _eb.norm() + 1e-8)))
            _prev_aud_env = _env
            if is_first:
                # fade-in on the very first chunk only (a ramp at a chunk
                # join would read as a level dip at the seam).
                _n_fade = min(8, new_aud.shape[-1])
                if _n_fade >= 2:
                    _ramp = torch.linspace(0.125, 1.0, _n_fade, device=device)
                    new_aud = new_aud.clone()
                    new_aud[..., :_n_fade] = new_aud[..., :_n_fade] * _ramp
            # soft clip EVERY chunk, not just the first: each chunk's opening
            # frames (right after the pinned overlap) overshoot the audio
            # VAE's calibrated range the same way the run's opening does —
            # delivered unclipped they land as transient "noise" starting
            # exactly at every chunk join (seed-independent, rms-neutral,
            # spectrogram-visible).  Gentle tanh knee above 3.5 sigma of the
            # chunk's own level; transparent for in-range content.
            _fa = new_aud.float()
            _sig = _fa.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            _over = (_fa.abs() - _sig * 3.5).clamp(min=0)
            # gentle knee, 5-sigma ceiling (was a hard 4-sigma limiter): a
            # limiter at 4 sigma sits ON percussive attacks — natural musical
            # peaks run 4-6 sigma — and flattened them every chunk (measured:
            # aud_peak pinned at ~4.0 every chunk, percussion audibly
            # squashed).  The 1.5-sigma tanh wing preserves real transients
            # while still taming pathological overshoot spikes to 5 sigma.
            new_aud = (_fa - torch.sign(_fa)
                       * (_over - 1.5 * _sig
                          * torch.tanh(_over / (1.5 * _sig)))).to(aud_out.dtype)
            # overshoot alarm: peak/sigma of the delivered chunk.  A chunk
            # arriving hot (>>7) means transients survived the clipper.
            _fa2 = new_aud.float()
            _trend["aud_peak"].append(float(
                _fa2.abs().max() / _fa2.std().clamp(min=1e-8)))
            _aud_sims = _aud_within_chunk_sims(new_aud)
            if _aud_sims:
                _trend["aud_wc"].append(_aud_sims[-1])
            if _s1_aud_prev_last is not None:
                _trend["aud_bnd"].append(_aud_cos(_s1_aud_prev_last.to(device), new_aud[..., :1]))
            # §2.3 audio counterpart: per-scene RMS anchor (deadband, then a
            # half-strength pull toward the scene's first chunk — same shape
            # as the video std anchor above).  Without it the re-rendered
            # chunks creep hot (+2-4 %/chunk measured: 0.45 -> 0.61 over a
            # 10-chunk run) and the end-aligned guide pins the hot tail as
            # the next chunk's clean reference — a compounding noise spiral.
            # Applied BEFORE the SLB/tail bookkeeping so the guide and the
            # next overlap carry the corrected audio.
            _cur_arms = new_aud.float().pow(2).mean().sqrt().item()
            if _s1_aud_rms_ref is None:
                _s1_aud_rms_ref = _cur_arms
            else:
                _ratio_a = _s1_aud_rms_ref / max(_cur_arms, 1e-6)
                if _ratio_a < 0.94 or _ratio_a > 1.06:
                    _g_a = 1.0 + 0.5 * (_ratio_a - 1.0)
                    new_aud = (new_aud.float() * _g_a).to(new_aud.dtype)
                    _cur_arms *= _g_a
            _trend["aud_rms"].append(_cur_arms)
            with torch.no_grad():
                _freq_e = new_aud.float().abs().mean(dim=(0, 2, 3)).tolist()
            if _s1_audio_freq_ref is None:
                _s1_audio_freq_ref = _freq_e
            else:
                # raw (pre-correction) per-channel decay — what the anchor had
                # to fight this chunk:
                _freq_raw = [e / r if r > 1e-6 else 0.0
                             for e, r in zip(_freq_e, _s1_audio_freq_ref)]
                if len(_freq_raw) >= 4:
                    _trend["aud_hf_raw"].append(sum(_freq_raw[-4:]) / 4.0)
                # §2.3b audio spectral anchor.  The model's own output is
                # slightly HF-deficient vs its training data, and each chunk's
                # tail is fed back as guide/pin/SLB context — so every
                # generation renders a little duller than its context and the
                # decay COMPOUNDS (measured: aud_hf 1.00 -> 0.89 -> 0.80 over
                # 3 chunks, 0.75 by chunk 10 — "quality drops with every
                # chunk").  Video has the two-band detail anchor for exactly
                # this and its vid_hf is dead flat; audio only had the scalar
                # RMS anchor, which preserves total level but not spectral
                # balance.  Mirror the video recipe per channel: gain =
                # ref/cur (mean-abs amplitudes — no sqrt), clamped to
                # [0.90, 1.12] per chunk, 0.5% deadband, anchored to the
                # FIXED scene reference so it converges instead of ratcheting.
                # TWO WIDE BANDS, not 32 narrow channels (v2 of this anchor):
                # per-channel gains fought GENUINE content variation — a chunk
                # whose instrumentation differs from chunk 0 got equalized
                # toward chunk 0's exact spectrum ("instruments flatten out"),
                # and with heterogeneous channels the post mean could land
                # BELOW raw (bright channels clamp-cut at 0.90 while dull ones
                # boosted only 1.12).  Wide bands average content variation
                # out and correct only systematic tilt.  Bands mirror the
                # aud_hf telemetry: hi = last 4 channels (where the measured
                # decay lives), lo = the rest.  Hi clamp 1.20: the observed
                # HF decay runs ~10-20%/chunk, past the video-derived 1.12.
                _n_ch = len(_freq_e)
                _lo_e = sum(_freq_e[:-4]) / max(1, _n_ch - 4)
                _hi_e = sum(_freq_e[-4:]) / 4.0
                _lo_r = sum(_s1_audio_freq_ref[:-4]) / max(1, _n_ch - 4)
                _hi_r = sum(_s1_audio_freq_ref[-4:]) / 4.0
                _g_lo = min(1.12, max(0.90, _lo_r / max(_lo_e, 1e-6)))
                _g_hi = min(1.20, max(0.90, _hi_r / max(_hi_e, 1e-6)))
                if abs(_g_lo - 1.0) > 0.005 or abs(_g_hi - 1.0) > 0.005:
                    _gt = torch.ones(1, _n_ch, 1, 1, dtype=new_aud.dtype,
                                     device=new_aud.device)
                    _gt[0, :-4] = _g_lo
                    _gt[0, -4:] = _g_hi
                    new_aud = (new_aud * _gt).to(new_aud.dtype)
                    _freq_e = [_e * (_g_hi if _i >= _n_ch - 4 else _g_lo)
                               for _i, _e in enumerate(_freq_e)]
                # delivered (post-correction) ratio — ≈1.0 means the anchor
                # fully absorbed this chunk's decay; <1.0 means the clamp
                # saturated and the decay outran it.
                _freq_ratio = [e / r if r > 1e-6 else 0.0
                               for e, r in zip(_freq_e, _s1_audio_freq_ref)]
                if len(_freq_ratio) >= 4:
                    _trend["aud_hf"].append(sum(_freq_ratio[-4:]) / 4.0)
            # roll the audio history forward (final kept audio, post fade/clip);
            # bounded to the longest guide window the widget allows.
            _audio_tail = (new_aud.cpu() if _audio_tail is None
                           else torch.cat([_audio_tail, new_aud.cpu()], dim=-1))
            _tail_keep = max(1, round(audio_guide_seconds * AUDIO_LATENT_FPS))
            if _audio_tail.shape[-1] > _tail_keep:
                _audio_tail = _audio_tail[..., -_tail_keep:]
            if not is_first and acc_audio and new_aud.shape[-1] > 0:
                # DELIVERED seam — what the listener actually gets.  aud_bnd
                # above measures the raw edge of the model's continuation
                # before the level/spectral anchors; aud_dlv measures the
                # delivered seam after them, aud_lvl the 1 s RMS level step
                # across it in dB (0 = no step; the audible "the room tone
                # jumped" failure).  Concept stolen from H3-Motion-Context's
                # Seam Probe (correlation / RMS step / floor step across a
                # join).
                _trend["aud_dlv"].append(_aud_cos(
                    acc_audio[-1][..., -1:].to(device), new_aud[..., :1]))
                _Lw = min(round(AUDIO_LATENT_FPS), acc_audio[-1].shape[-1],
                          new_aud.shape[-1])
                if _Lw > 0:
                    _lv_prev = acc_audio[-1][..., -_Lw:].float().pow(2).mean().sqrt()
                    _lv_new = new_aud[..., :_Lw].float().pow(2).mean().sqrt()
                    _trend["aud_lvl"].append(
                        20.0 * math.log10(float(_lv_new)
                                          / max(float(_lv_prev), 1e-8)))
                # delivered-domain seam forensics (patch #10):
                #   aud_step  — join jump / median jump (~1 = invisible seam)
                #   aud_lag/@aud_lagf — best-lag corr across the seam; a
                #     nonzero best lag = px<->audio rounding regression (25 ms)
                #   aud_loop/@aud_loopt — max cosine of any 1 s window of the
                #     new chunk against ALL previously delivered audio of this
                #     scene, and where the match lives (seconds).  High =
                #     the chunk repeats earlier material = loop-lock, measured
                #     on delivered audio (aud_wc only sees WITHIN a chunk).
                _trend["aud_step"].append(_aud_seam_step(
                    acc_audio[-1], new_aud.cpu()))
                _lag_c, _lag_f = _aud_best_lag(acc_audio[-1], new_aud.cpu())
                _trend["aud_lag"].append(_lag_c)
                _trend["aud_lagf"].append(float(_lag_f))
                _hist = torch.cat(acc_audio[_hist_scene_start:], dim=-1)
                _loop_c, _loop_t = _aud_loop_ncc(_hist, new_aud.cpu())
                _trend["aud_loop"].append(_loop_c)
                _trend["aud_loopt"].append(
                    _loop_t + sum(a.shape[-1]
                                  for a in acc_audio[:_hist_scene_start])
                    / AUDIO_LATENT_FPS)
            acc_audio.append(new_aud.cpu())
            audio_chunk_ends.append(sum(a.shape[-1] for a in acc_audio))
            _s1_aud_prev_last = new_aud[..., -1:].cpu()

            _vid_pos += _cur_new_lf
            _aud_pos += cur_new_af

        full_vid = torch.cat(acc_video, dim=2)
        # End-of-run trend dump (structure metrics — they localize failures,
        # they never prove a quality win).
        for _k, _v in _trend.items():
            if _v:
                print(f"[CLSS] trend {_k}: " + " ".join(f"{_x:.3f}" for _x in _v))
        full_aud = torch.cat(acc_audio, dim=-1)
        full_aud = _post_process_audio_latent(full_aud, audio_chunk_ends,
                                              energy_beta=0.0, label=" S1")
        output_samples = comfy.nested_tensor.NestedTensor((full_vid, full_aud))
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return ({"samples": output_samples},)


# ---------------------------------------------------------------------------
# Streaming decode + save
# ---------------------------------------------------------------------------


class CLSSH3VideoDecodeSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae":       ("VAE",    {"tooltip": "Video VAE. Decoding is temporally sliced: each slice is decoded standalone (the H3 VAE further chunks it internally with bounded VRAM) and its PNG frames are written to disk before the next slice decodes — the whole decoded video never sits in RAM."}),
                "audio_vae": ("VAE",    {"tooltip": "Audio VAE (MiniMaxH3AudioVAE) for the audio stream; decoded in one shot (the audio latent is small) via the same logic as the stock VAEDecodeAudio node."}),
                "latent":    ("LATENT", {"tooltip": "Full AV latent from CLSSH3StreamingSampler (NestedTensor video+audio)."}),
                "filename_prefix": ("STRING", {"default": "clss_h3/CLSSH3_frame_",
                                               "tooltip": "Output filename prefix; frames are written as <prefix>_NNNNN.png under the ComfyUI output directory.",
                                               }),
                "frames_per_slice": ("INT", {"default": 27, "min": 5, "max": 502,
                                             "tooltip": "Video latent TOKENS decoded per VAE call, snapped to a multiple of 5 (the 17k+5 grid's group size; 27 → 25 tokens ≈ 85 px frames ≈ 3.5 s). Slice boundaries stay on the absolute 5-token grid so per-slice px counts tile the true timeline exactly.",
                                             }),
                "context_frames": ("INT", {"default": 0, "min": 0, "max": 16,
                                           "tooltip": "Extra latent tokens of temporal context prepended to each non-first slice and dropped after decode. MEASURED WORSE THAN 0 on H3: the causal VAE's first tokens cover 1+4 px instead of 4+4, so prepended context misaligns the slice phase (seam diff max 0.90 vs 0.23 with ctx=0). Keep 0.",}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0,
                                  "tooltip": "Informational only (duration logging). H3 is 24 fps native; frames are written in timeline order regardless."}),
            },
        }
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "decode_save"
    OUTPUT_NODE = True
    CATEGORY = "MiniMaxH3-CLSS"

    @torch.inference_mode()
    def decode_save(self, vae, audio_vae, latent, filename_prefix,
                    frames_per_slice=27, context_frames=0, fps=24.0):
        import folder_paths
        import numpy as np
        from PIL import Image
        samples = latent["samples"]
        if not (getattr(samples, "is_nested", False) and len(samples.unbind()) == 2):
            raise ValueError("CLSSH3VideoDecodeSave expects a MiniMax H3 AV latent")
        vid, aud = samples.unbind()
        T = vid.shape[2]

        fsm = getattr(vae, "first_stage_model", None)

        def _px_for_tokens(n: int) -> int:
            # exact px count of a standalone decode of n tokens
            if fsm is not None and hasattr(fsm, "decode_output_shape"):
                return fsm.decode_output_shape((1, vid.shape[1], n, vid.shape[3], vid.shape[4]))[2]
            return _px_of_tokens(n, 0)  # exact (1,4,4,4,4) token→px span, phase-0 anchored

        output_dir = folder_paths.get_output_directory()
        full_folder, filename, _, _, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir)
        os.makedirs(full_folder, exist_ok=True)

        # Slice boundaries on the absolute 5-token grid (positions ≡ 0 mod 5):
        # a standalone decode of a 5s-token slice yields exactly 17s px frames
        # (and 17k+5 for the 5k+2 tail), so the slices tile the true px
        # timeline with no gaps or duplicates — AV sync is preserved.
        step = 5 * max(1, round(frames_per_slice / 5))
        ctx = max(0, int(context_frames))
        frame_idx = 0
        pos = 0
        while pos < T:
            end = min(pos + step, T)
            n_tok = end - pos
            c = 0 if pos == 0 else min(ctx, pos)
            px = vae.decode(vid[:, :, pos - c:end])   # [B, T_px, H, W, 3] in [0,1]
            # the prepended context occupies the FIRST c tokens of the decoded
            # slice, whose px span is _px_for_tokens(c) — the old expression
            # (px(c+n) − px(n)) is only exact for n ≡ 0 mod 5 and over-dropped
            # up to 3 px on the 17k+2 (5k+2) tail slice, leaving a mid-video gap.
            drop = _px_for_tokens(c) if c else 0
            arr = (px[0, drop:].float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            for f in range(arr.shape[0]):
                Image.fromarray(arr[f]).save(
                    os.path.join(full_folder, f"{filename}_{frame_idx:05d}.png"),
                    compress_level=4)
                frame_idx += 1
            del px, arr
            pos = end

        # Stock VAEDecodeAudio logic (comfy_extras/nodes_audio.py:100-112):
        # decode, move channels to dim 1, soft-normalize to ≤ 5 std.
        audio = audio_vae.decode(aud).movedim(-1, 1)
        std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio = audio / std
        vae_sr = getattr(audio_vae, "audio_sample_rate_output",
                         getattr(audio_vae, "audio_sample_rate", 32000))
        print(f"[CLSS] decode_save: {frame_idx} frames ({frame_idx / fps:.1f} s @ "
              f"{fps:g} fps) -> {full_folder}/{filename}_*.png; "
              f"audio {audio.shape[-1] / vae_sr:.1f} s @ {vae_sr} Hz")
        return ({"waveform": audio, "sample_rate": vae_sr},)


NODE_CLASS_MAPPINGS = {
    "CLSSH3Config":           CLSSH3Config,
    "CLSSH3ScenePrompts":     CLSSH3ScenePrompts,
    "CLSSH3StreamingSampler": CLSSH3StreamingSampler,
    "CLSSH3Guider":           CLSSH3Guider,
    "CLSSH3VideoDecodeSave":  CLSSH3VideoDecodeSave,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLSSH3Config":           "CLSS H3 Config",
    "CLSSH3ScenePrompts":     "CLSS H3 Scene Prompts",
    "CLSSH3StreamingSampler": "CLSS H3 Streaming Sampler",
    "CLSSH3Guider":           "CLSS H3 Guider (Split AV CFG)",
    "CLSSH3VideoDecodeSave":  "CLSS H3 Video Decode+Save (streaming)",
}
