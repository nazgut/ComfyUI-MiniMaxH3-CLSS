"""ComfyUI nodes for CLSS (Closed-Loop Streaming Synthesis) on MiniMax H3.

Port of the LTX-2 CLSS node layer (ComfyUI-LTX2.3-CLSS) to the MiniMax H3
packed audio-video DiT.  The model-agnostic algorithm core (CLSSConfig /
CLSSState: SLB bookkeeping, §2.3 EMA-AdaIN drift correction, §2.5 anchor bank)
lives in the vendored `clss.py`; this file is the H3-specific orchestration.

Key mechanism differences from the LTX version (see also the plan / AGENTS.md):

- Latent: one dict {"samples": NestedTensor((video [B,24,T,H/16,W/16],
  audio [B,32,2,Ta]))}.  Video grid: px frames snap to 17k+5 ⇔ latent tokens
  5k+2 (`video_latent_t`); token k covers FRAME_PER_TOKEN[k%5] = (1,4,4,4,4)
  px frames.  Audio: Ta = round(px × 5/3) at 24 fps (40 latent fps) — the same
  math as `temporal_shape` in comfy_extras/nodes_minimax_h3.py.
- §2.1 SLB overlap: the previous chunk's corrected tail is written INTO the
  chunk's initial latent (video tokens [0:F_ol], audio frames [0:Ta_ol]) and a
  per-stream denoise mask rides in the latent dict ("noise_mask",
  NestedTensor).  H3 turns mask value m into a per-row sigma = m·σ
  (model.py:587-609) and re-blends preserved rows toward the cond-strength
  injection every step (model_base.py:2248-2272 scale_latent_inpaint, called
  from KSamplerX0Inpaint in comfy/samplers.py:634-643).
- §2.5 anchors and the i2v guide image ride as `minimax_keyframes`
  conditioning rows ({"resolved_frame_index", "latent"}) — the H3-native
  pinning mechanism (model.py:340-361, pinned near-clean, re-injected every
  step).  Anchors anchor at resolved_frame_index=0 of the chunk window.  NOTE:
  the LTX node layer only tracked the anchor bank for telemetry; H3's
  keyframe rows make §2.5 conditioning expressible, so it is wired up here.
  The overlap is deliberately NOT also pinned as keyframe rows (double-pinning
  conflict — the overlap's job is the mask's).
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
# Hot-audio re-noise ceiling: audio SLB re-noise never exceeds this even for
# large audio_slb_tau_mult (the chunked-audio "metronome" fixed-point needs
# repair but collapses if the seam is re-noised too hot).
_AUDIO_TAU_C_CEILING = 0.35

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
            "optional": {
                "noise_temporal_corr": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.8, "step": 0.05,
                                      "tooltip": "Temporally-correlated video noise: mixes a run-constant shared frame into every noise frame, n_t = sqrt(1-a)·eps_t + sqrt(a)·eps_shared, keeping each frame's marginal exactly N(0,1) while raising frame-to-frame noise correlation (FreeNoise/PYoCo family). Video stream only. Targets the measured ~4 s layout oscillation. 0 = off.",
                                      }),
            },
        }
    RETURN_TYPES = ("CLSS_CONFIG",)
    RETURN_NAMES = ("clss_config",)
    FUNCTION = "build"
    CATEGORY = "MiniMaxH3-CLSS"

    def build(self, tau_c, beta, overlap, noise_temporal_corr=0.3):
        # Validated-production values carried over from the LTX node layer;
        # the dataclass defaults in clss.py are the paper's, not ours.
        return (CLSSConfig(
            tau_c=tau_c,
            beta=beta,
            ema_lambda=0.10,
            ema_sigma_max_drift=0.05,
            anchor_force_every=0,  # 0 = auto: the sampler derives ceil(chunks/4) clamped to [2, 5]
            overlap_latent_frames=_snap_overlap(overlap),
            adain_max_amplification=1.2,
            noise_temporal_corr=noise_temporal_corr,
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
                "clss_config": ("CLSS_CONFIG", {"tooltip": "CLSS_CONFIG from the CLSSH3Config node (tau_c, beta, overlap, noise_temporal_corr)."}),
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
                # < 0: the audio overlap regenerates freely (best musical continuity),
                # but the last |value| SECONDS of the previous tail stay frozen at the
                # end of the overlap — without that pin the vocal phrase restarts at
                # every chunk boundary and words are cut mid-phoneme.
                # 0 = SLB placed fully frozen, > 0 = SLB re-noised with tau_c*mult.
                "audio_slb_tau_mult": ("FLOAT", {
                    "default": 0.0, "min": -4.0, "max": 6.0, "step": 0.5,
                    "tooltip": "Audio SLB handling. 0 = SLB placed fully frozen (no tau_c on audio). > 0 = SLB re-noised with tau_c×mult (ceiling 0.35). < 0 = overlap regenerates freely (best musical continuity) but the last |value| SECONDS of the previous tail stay pinned frozen at the end of the overlap, keeping the vocal phrase glued across the seam instead of restarting mid-phoneme.",
                    }),
                "audio_guide_seconds": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.25,
                    "tooltip": "Audio seam guide: pins the last N seconds of the previous chunk's audio as a cond_audio guide keyframe whose window ENDS exactly at the join and reaches BACKWARD (fractional/negative anchor index — the H3-Motion-Context pack's measured mechanism: seam correlation 0.45 -> 0.95+ vs reference placement; a forward/overlap-aligned guide makes the model loop the motif instead). Pairs with audio_slb_tau_mult > 0 (re-noised, re-rendered overlap) — do not freeze the overlap under the guide. 1.0 s = 40 latent steps, the pack's validated default. 0 = off.",
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
        audio_slb_tau_mult: float = 0.0,
        audio_guide_seconds: float = 3.0,
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

        if clss_config.anchor_force_every <= 0:
            _auto_anchor = max(2, min(5, math.ceil(_eff_num_chunks / 4)))
            clss_config = dataclasses.replace(clss_config, anchor_force_every=_auto_anchor)

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
        _ntc = float(getattr(clss_config, "noise_temporal_corr", 0.0))
        if _ntc > 0.0:
            # §-noise: mix a run-constant shared frame into every video noise
            # frame, n_t = sqrt(1-a)·eps_t + sqrt(a)·eps_shared — each frame's
            # marginal stays exactly N(0,1) while frame-to-frame correlation
            # rises to a at all lags (FreeNoise/PYoCo family).  Video only.
            _g_shared = torch.Generator(device="cpu").manual_seed(
                (int(_noise_seed) + 2) % (2 ** 63))
            _eps_shared = torch.randn(
                _full_noise_vid.shape[0], _full_noise_vid.shape[1], 1,
                _full_noise_vid.shape[3], _full_noise_vid.shape[4],
                generator=_g_shared, dtype=_full_noise_vid.dtype,
            ).to(_full_noise_vid.device)
            _full_noise_vid = (
                math.sqrt(1.0 - _ntc) * _full_noise_vid
                + math.sqrt(_ntc) * _eps_shared
            )
        _g_aud = torch.Generator(device="cpu").manual_seed(
            (int(_noise_seed) + 1) % (2 ** 63))
        _full_noise_aud = torch.randn(B_a, C_a, lanes_a, Ta_total,
                                      generator=_g_aud, dtype=aud_tmpl.dtype)

        clss_state = CLSSState(clss_config)
        acc_video: list[torch.Tensor] = []
        acc_audio: list[torch.Tensor] = []
        audio_chunk_ends: list[int] = []
        audio_slb_latent: torch.Tensor | None = None
        # rolling kept-audio history (CPU, ends at the current join) — the
        # source of the end-aligned guide window (see audio_guide_seconds)
        _audio_tail: torch.Tensor | None = None
        _s1_prev_last: torch.Tensor | None = None
        _s1_vid_std_ref: float | None = None
        _prev_scene_idx: int | None = None
        _s1_band_ref: tuple[float, float] | None = None
        _origin_ref: torch.Tensor | None = None
        _origin_layout: torch.Tensor | None = None
        _prev_aud_env: torch.Tensor | None = None
        _s1_aud_prev_last: torch.Tensor | None = None
        _s1_audio_freq_ref: list[float] | None = None
        _trend = {
            "vid_std": [], "vid_ident": [], "vid_intra": [], "vid_bnd": [],
            "vid_hf": [], "vid_origin": [],
            "aud_env": [], "aud_rms": [], "aud_bnd": [], "aud_slb": [],
            "aud_wc": [], "aud_hf": [],
        }

        _vid_pos = 0
        _aud_pos = 0
        for chunk_idx in range(_eff_num_chunks):
            is_first = chunk_idx == 0
            _cur_new_lf, cur_new_af, _cur_new_px = chunk_plan[chunk_idx]
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
                # §2.3: drop the old scene's EMA reference — the first chunk of the
                # new scene is uncorrected and re-anchors the EMA (incl. _init_std).
                clss_state.reset_drift_refs()
            _prev_scene_idx = scene_idx
            has_slb = not is_first and clss_state.overlap_latent is not None
            has_aud_slb = not is_first and audio_slb_latent is not None

            # ---- conditioning: scene (+ crossfade blend) + keyframe rows ----
            keyframes: list[dict] = []
            if is_first and img_guide_latent is not None:
                keyframes.append({"resolved_frame_index": 0, "latent": img_guide_latent})
            if not is_first and clss_config.anchor_top_m > 0:
                # §2.5: top-m anchors pinned as keyframe rows at the window's
                # frame 0 (near-clean, re-injected every step).  NOT also pinned
                # in-latent — the overlap's job is the mask's (double-pinning
                # conflict).
                for _a in clss_state.top_anchors():
                    keyframes.append({"resolved_frame_index": 0, "latent": _a.latent})
            # ---- audio seam guide: cond_audio keyframe, END-aligned at the
            # join and reaching BACKWARD (H3-Motion-Context's measured
            # mechanism) ----
            # What was tried and measured on H3 before this:
            #  - bare frozen in-latent SLB: aud_bnd ≈ 0.2-0.35 — the model
            #    treats masked-in audio rows as output, not context;
            #  - minimax_refs pre-window block: "cover band" effect (their
            #    words — same groove, never the same recording; aud_bnd
            #    ≈ 0.3) and the ref span shifts the whole target RoPE
            #    origin, which measurably hurt video;
            #  - guide keyframe OVERLAID on the frozen overlap: the model
            #    loops the pinned motif across every following chunk.
            # What works (their pack, seam correlation 0.45 -> 0.95+): the
            # pinned audio window must END at the join and reach backward
            # into audio that already played — a fractional, negative
            # resolved_frame_index, legal PackedLayout arithmetic that no
            # stock node produces.  The overlap itself must be RE-RENDERED
            # (re-noised SLB, audio_slb_tau_mult > 0), never frozen under
            # the guide.  The join sits at window audio position _Ta_ol_w —
            # already an integer on the 40 Hz grid, so no end snapping is
            # needed (their overhang dance exists only because they align
            # to a px-frame end instead).
            if (not is_first and audio_guide_seconds > 0.0
                    and _audio_tail is not None):
                _join_af = _af_of_px(px_ol + _cur_new_px) - cur_new_af
                _g = min(round(audio_guide_seconds * AUDIO_LATENT_FPS),
                         _audio_tail.shape[-1])
                if _g > 0 and _join_af > 0:
                    keyframes.append({
                        # start coord = join - g  =>  index = (join - g)/RESCALE;
                        # negative/fractional whenever g > overlap — intended.
                        "resolved_frame_index": (_join_af - _g) / FRAME_RESCALE,
                        "audio_latent": _audio_tail[..., -_g:],
                    })
            guider_chunk = copy.copy(guider)
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
            _slb_ctx_used: torch.Tensor | None = None
            _slb_ctx_pos = 0
            if has_aud_slb and audio_slb_tau_mult >= 0.0:
                slb = audio_slb_latent.to(device)
                n = min(_Ta_ol_w, slb.shape[-1], chunk_af)
                _slb_ctx = slb[..., :n]
                lat_aud[..., :n] = _slb_ctx
                if audio_slb_tau_mult > 0.0:
                    _tau_c_a = _tau_c_eff(clss_config.tau_c * audio_slb_tau_mult,
                                          _AUDIO_TAU_C_CEILING, chunk_idx - 1)
                else:
                    _tau_c_a = 0.0
                mask_aud[..., :n] = _tau_c_a
                _slb_ctx_used = _slb_ctx.detach().cpu()
            elif has_aud_slb:
                # audio_slb_tau_mult < 0: regenerate the overlap freely (best
                # musical continuity — a fully frozen SLB drags the whole window
                # toward an exact repeat of the tail), but pin the last |mult|
                # SECONDS of the previous tail frozen at the END of the overlap.
                # Without the pin the vocal phrase restarts at every chunk
                # boundary: words are cut mid-phoneme and the singing goes dead.
                # The pinned frames sit immediately before the kept region, so
                # they glue the seam without constraining the rest of the window.
                slb = audio_slb_latent.to(device)
                n = min(_Ta_ol_w, slb.shape[-1], chunk_af)
                _pin_af = min(round(-audio_slb_tau_mult * AUDIO_LATENT_FPS), n)
                if _pin_af > 0:
                    _slb_ctx_pos = n - _pin_af
                    _slb_ctx = slb[..., _slb_ctx_pos:n]
                    lat_aud[..., _slb_ctx_pos:n] = _slb_ctx
                    mask_aud[..., _slb_ctx_pos:n] = 0.0
                    _slb_ctx_used = _slb_ctx.detach().cpu()
            chunk_latent = {
                "samples": comfy.nested_tensor.NestedTensor((lat_vid, lat_aud)),
                "noise_mask": comfy.nested_tensor.NestedTensor((mask_vid, mask_aud)),
            }

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
            if not is_first:
                _cur_feat = F.normalize(corrected[:, :, 0].float().reshape(B, C_v, -1).mean(-1), dim=1)
                _bank = clss_state._anchor_bank
                if _bank.anchors:
                    _trend["vid_ident"].append(max(
                        F.cosine_similarity(
                            _cur_feat,
                            F.normalize(a.feature.unsqueeze(0).to(device), dim=1),
                        ).item()
                        for a in _bank.anchors
                    ))
            _s1_prev_last = corrected[:, :, -1].cpu()

            # ---- audio: keep the new frames, update the audio SLB ----
            aud_drop = _Ta_ol_w
            if aud_drop > 0 and aud_out.shape[-1] < aud_drop:
                aud_drop = 0
            new_aud = aud_out[..., aud_drop:]
            if not is_first and _slb_ctx_used is not None and Ta_ol > 0:
                _trend["aud_slb"].append(_aud_cos(
                    _slb_ctx_used.to(device),
                    aud_out[..., _slb_ctx_pos:_slb_ctx_pos + _slb_ctx_used.shape[-1]]))
            _env = new_aud.detach().float().pow(2).mean(dim=(0, 1, 2)).cpu()
            if _prev_aud_env is not None and len(_prev_aud_env) > 8:
                _L = min(len(_env), len(_prev_aud_env))
                _ea = _env[:_L] - _env[:_L].mean()
                _eb = _prev_aud_env[:_L] - _prev_aud_env[:_L].mean()
                _trend["aud_env"].append(float((_ea * _eb).sum() /
                                               (_ea.norm() * _eb.norm() + 1e-8)))
            _prev_aud_env = _env
            if is_first:
                # fade-in + soft clip on the very first chunk: the model's
                # opening frames tend to overshoot the audio VAE's calibrated
                # range; carried over from the LTX port (revalidate on H3).
                _n_fade = min(8, new_aud.shape[-1])
                if _n_fade >= 2:
                    _ramp = torch.linspace(0.125, 1.0, _n_fade, device=device)
                    new_aud = new_aud.clone()
                    new_aud[..., :_n_fade] = new_aud[..., :_n_fade] * _ramp
                _fa = new_aud.float()
                _sig = _fa.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
                _over = (_fa.abs() - _sig * 3.5).clamp(min=0)
                new_aud = (_fa - torch.sign(_fa)
                           * (_over - 0.5 * _sig * torch.tanh(_over / _sig))).to(aud_out.dtype)
            _aud_sims = _aud_within_chunk_sims(new_aud)
            if _aud_sims:
                _trend["aud_wc"].append(_aud_sims[-1])
            if _s1_aud_prev_last is not None:
                _trend["aud_bnd"].append(_aud_cos(_s1_aud_prev_last.to(device), new_aud[..., :1]))
            _trend["aud_rms"].append(new_aud.float().pow(2).mean().sqrt().item())
            with torch.no_grad():
                _freq_e = new_aud.float().abs().mean(dim=(0, 2, 3)).tolist()
            if _s1_audio_freq_ref is None:
                _s1_audio_freq_ref = _freq_e
            else:
                _freq_ratio = [e / r if r > 1e-6 else 0.0
                               for e, r in zip(_freq_e, _s1_audio_freq_ref)]
                if len(_freq_ratio) >= 4:
                    _trend["aud_hf"].append(sum(_freq_ratio[-4:]) / 4.0)
            if Ta_ol > 0:
                audio_slb_latent = (new_aud[..., -Ta_ol:] if new_aud.shape[-1] >= Ta_ol
                                    else new_aud).cpu()
            # roll the audio history forward (final kept audio, post fade/clip);
            # bounded to the SLB + the longest guide window the widget allows.
            _audio_tail = (new_aud.cpu() if _audio_tail is None
                           else torch.cat([_audio_tail, new_aud.cpu()], dim=-1))
            _tail_keep = Ta_ol + max(0, round(audio_guide_seconds * AUDIO_LATENT_FPS))
            if _tail_keep > 0 and _audio_tail.shape[-1] > _tail_keep:
                _audio_tail = _audio_tail[..., -_tail_keep:]
            # ---- soft seam: blend the previous tail with the window's
            # RE-RENDERED overlap instead of hard-cutting to the new frames ----
            # Every hard-edged audio config measured discontinuous on H3
            # (aud_bnd ≈ 0.2-0.35): in-stream frozen audio rows are an
            # untrained configuration — the model cannot join new content to
            # their end.  With a re-noised SLB (audio_slb_tau_mult > 0, the
            # LTX/CLSS soft boundary — what the working VIDEO seam uses) the
            # model re-renders the overlap so it FLOWS into the new content,
            # and this linear blend of the old tail with that re-rendered span
            # removes the residual edge at the join.  With a frozen SLB
            # (mult = 0) both versions are identical (aud_slb = 1.0) and the
            # blend is an exact no-op.
            if aud_drop > 0 and acc_audio:
                _n_xf = min(aud_drop, acc_audio[-1].shape[-1])
                if audio_slb_tau_mult == 0.0 and _slb_ctx_used is not None:
                    # frozen SLB: only the pinned span holds previous-tail
                    # content.  The window rounding (_Ta_ol_w vs Ta_ol, ±1 af)
                    # can drop one frame BEYOND the pinned span — that frame
                    # is the window's own fresh render of an already-delivered
                    # position, and blending toward it (w=1.0 at the join)
                    # would splice the preserved tail.  Cap the blend to the
                    # pinned span so frozen stays an exact no-op.
                    _n_xf = min(_n_xf, _slb_ctx_used.shape[-1])
                _w_xf = torch.linspace(0.0, 1.0, _n_xf,
                                       dtype=acc_audio[-1].dtype).view(1, 1, 1, -1)
                _prev_kept = acc_audio[-1]
                # no in-place write: on CPU-only runs new_aud.cpu() aliases and
                # an in-place blend would retroactively rewrite audio_slb_latent
                # and _audio_tail (the guide's source) with delivered values.
                acc_audio[-1] = torch.cat([
                    _prev_kept[..., :-_n_xf],
                    _prev_kept[..., -_n_xf:] * (1.0 - _w_xf)
                    + aud_out[..., :_n_xf].cpu().to(_prev_kept.dtype) * _w_xf,
                ], dim=-1)
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
