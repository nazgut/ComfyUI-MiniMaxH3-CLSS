from __future__ import annotations

import copy
import dataclasses
import math
import os
import time

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced
from comfy_extras.nodes_minimax_h3 import AUDIO_LATENT_FPS, FPS as _NATIVE_FPS

try:
    from .clss import CLSSConfig, CLSSState
except ImportError:
    from clss import CLSSConfig, CLSSState


_WINDOW_CAP_S = 12.0

_VIDEO_TAU_C_CEILING = 0.10

_MIN_OVERLAP_TOKENS = 2

_SCENE_BLEND_W = 0.5

_REF_IMAGE_SHORT_EDGE = 2048

_TAIL_ANCHOR_S = 1.0

_REF_DECODE_MARGIN_AF = 40


_NOISE_FIELD_CAP_TOK = 40000
_NOISE_FIELD_CAP_AF = 220000


def _px_of_tokens(n_tokens: int, start_phase: int = 0) -> int:
    return sum(FRAME_PER_TOKEN[(start_phase + j) % 5] for j in range(n_tokens))


def _af_of_px(px: int) -> int:
    return round(px * FRAME_RESCALE)


def _snap_overlap(overlap: int) -> int:
    return max(_MIN_OVERLAP_TOKENS, 5 * round((overlap - 2) / 5) + 2)


def _tau_c_eff(base: float, ceiling: float, chunk_idx: int,
               half_life: float = 5.0) -> float:
    if base <= 0.0:
        return 0.0
    decay = 0.5 ** (chunk_idx / half_life)
    return ceiling - (ceiling - base) * decay


def _unload_before_sampling() -> float | None:
    mm = comfy.model_management
    try:
        dev = mm.get_torch_device()
        before = int(mm.get_free_memory(dev))
    except Exception:
        dev = None
        before = None
    mm.unload_all_models()
    mm.soft_empty_cache()
    if dev is None or before is None:
        return None
    try:
        return max(0.0, int(mm.get_free_memory(dev)) - before) / 1024.0 ** 3
    except Exception:
        return None


def _enable_expandable_segments() -> str | None:
    if not torch.cuda.is_available():
        return None
    try:
        conf = str(os.environ.get("PYTORCH_CUDA_ALLOC_CONF", ""))
        if "expandable_segments" in conf:
            return "env"
        torch.cuda.memory._set_allocator_settings(
            "expandable_segments:True")
        return "runtime"
    except Exception:
        return "off"


def _split_run(n_tokens: int, max_new: int, plus2: bool) -> list[int]:
    plus = 2 if plus2 else 0
    groups = (n_tokens - plus) // 5
    cap = max(1, (max_new - plus) // 5)
    s = min(groups, max(1, math.ceil(groups / cap)))
    base, extra = divmod(groups, s)
    gs = [base + (1 if j < extra else 0) for j in range(s)]
    return [5 * gs[0] + plus] + [5 * g for g in gs[1:]]


def _tail_margin_tokens(tail_margin_px: int) -> tuple[int, int]:
    _in = max(0, min(48, int(tail_margin_px)))
    lf = 0
    while lf < 16 and _px_of_tokens(lf, 0) < _in:
        lf += 1
    return lf, _px_of_tokens(lf, 0)


def _plan_chunk_tokens(new_lf0: int, num_chunks: int, overlap: int,
                       tail_margin_px: int = 12, ctx_mode: str = "none"
                       ) -> tuple[list[int], int, int, bool]:
    new_cont = new_lf0 - 2
    px0 = _px_of_tokens(new_lf0, 0)
    pxc = _px_of_tokens(new_cont, 2)
    ctx = ctx_mode == "cont"
    _tm_lf, _tm_px = _tail_margin_tokens(tail_margin_px)
    cap_px = int(_WINDOW_CAP_S * _NATIVE_FPS)
    eff = _snap_overlap(overlap)

    def _win(e: int) -> int:
        if ctx:
            return _px_of_tokens(e, 0) + pxc + _tm_px
        if num_chunks == 1:
            return px0 + _tm_px
        return _px_of_tokens(e, 0) + max(px0, pxc) + _tm_px

    win_px = _win(eff)
    while ((num_chunks > 1 or ctx) and win_px > cap_px
           and eff > _MIN_OVERLAP_TOKENS):
        eff -= 5
        win_px = _win(eff)
    px_ol = _px_of_tokens(eff, 0)
    if win_px <= cap_px or px_ol + max(px0, pxc) <= cap_px:
        return ([new_lf0] + [new_cont] * (num_chunks - 1), eff, win_px, False)
    rem = max(6, cap_px - px_ol)
    max_new0 = 5 * max(1, (rem - 5) // 17) + 2
    max_newc = 5 * max(1, rem // 17)
    tokens: list[int] = []
    for _ci in range(num_chunks):
        if _ci == 0:
            tokens.extend(_split_run(new_lf0, max_new0, plus2=True))
        else:
            tokens.extend(_split_run(new_cont, max_newc, plus2=False))
    return tokens, eff, win_px, True


def _blend_scene_cond(prev: dict, new: dict, w: float = _SCENE_BLEND_W) -> dict:
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
            blended["minimax_token_tags"] = nt
    _ref_src = new if w >= 0.5 else prev
    if _ref_src.get("minimax_refs") is not None:
        blended["minimax_refs"] = _ref_src["minimax_refs"]
    else:
        blended.pop("minimax_refs", None)
    if _ref_src.get("clss_cont") is not None:
        blended["clss_cont"] = _ref_src["clss_cont"]
        blended["clss_cont_text"] = _ref_src.get("clss_cont_text")
    else:
        blended.pop("clss_cont", None)
        blended.pop("clss_cont_text", None)
    return blended


def _apply_scene_cont(entry: dict) -> dict:
    cont = entry.get("clss_cont")
    if not cont:
        return entry
    out = {**entry, **cont}
    if "minimax_token_tags" not in cont:
        out["minimax_token_tags"] = None
    return out


def _frame_cos(a: torch.Tensor, b: torch.Tensor) -> float:
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


def _aud_flat(t: torch.Tensor) -> torch.Tensor:
    return t.float().reshape(t.shape[0], -1, t.shape[-1])


def _aud_seam_step(prev_tail: torch.Tensor, new_head: torch.Tensor,
                   ctx: int = 40) -> float:
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
    with torch.no_grad():
        f = _aud_flat(history)[0]
        g = _aud_flat(new_chunk)[0]
        Th, Tn = f.shape[-1], g.shape[-1]
        if Th < Tn or Tn < 8:
            return float("nan"), float("nan")
        nfft = 1 << (Th + Tn - 1).bit_length()
        cc = torch.fft.irfft(torch.fft.rfft(f, n=nfft)
                             * torch.fft.rfft(g, n=nfft).conj(),
                             n=nfft)[..., :Th].sum(0)
        e = f.pow(2).sum(0).cumsum(0)
        win_e = e[Tn - 1:] - torch.cat([e.new_zeros(1), e[:-Tn]])
        ncc = cc[:win_e.numel()] / (win_e * g.pow(2).sum()).clamp(min=1e-12).sqrt()
        k = int(ncc.argmax())
        return float(ncc[k]), k / AUDIO_LATENT_FPS


def _aud_within_chunk_sims(new_aud: torch.Tensor, n_seg: int = 3) -> list[float]:
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


def _take_loop_metrics(hist: torch.Tensor | None,
                       take: torch.Tensor) -> tuple[float, float]:
    wc = float("nan")
    sims = _aud_within_chunk_sims(take)
    if sims:
        wc = float(sims[-1])
    loop = float("nan")
    if hist is not None:
        loop, _ = _aud_loop_ncc(hist, take)
    return wc, loop


def _loop_guard_bad(wc: float, loop: float, thr_wc: float,
                    thr_loop: float) -> bool:
    return ((wc == wc and wc >= thr_wc)
            or (loop == loop and loop >= thr_loop))


def _loop_guard_pick(tries: list) -> int:
    return min(range(len(tries)), key=lambda i: (
        tries[i][1] if tries[i][1] == tries[i][1] else 1.0,
        tries[i][2] if tries[i][2] == tries[i][2] else -1.0))


def _rc_seed_for(audio_recompose_seed: int, noise_seed: int, chunk_idx: int,
                 attempt: int = 0) -> int:
    if int(audio_recompose_seed) != 0:
        base = int(audio_recompose_seed) + 7919 * int(chunk_idx)
    else:
        base = int(noise_seed) + 424_243 + 1_000_003 * int(chunk_idx)
    return (base + int(attempt)) % (2 ** 63)


_AUDIO_CFG_DEFAULTS = {
    "audio_recompose_steps": 0,
    "audio_recompose_sigma": 1.0,
    "audio_arc_margin_ms": 4000,
    "audio_recompose_pool": 2,
    "audio_recompose_stride": 2,
    "audio_recompose_seed": 0,
    "audio_recompose_ref_ms": 0,
    "audio_head_discard_ms": 0,
    "loop_guard_rerolls": 2,
    "loop_guard_wc": 0.99,
    "loop_guard_loop": 0.70,
    "loop_guard_retry_ref_ms": 2000,
}
_AUDIO_CFG_KEYS = tuple(_AUDIO_CFG_DEFAULTS)


def _resolve_audio_settings(audio_config) -> dict:
    out = dict(_AUDIO_CFG_DEFAULTS)
    if audio_config:
        for _k in _AUDIO_CFG_KEYS:
            if _k in audio_config:
                out[_k] = audio_config[_k]
    return out


def _post_process_audio_latent(
    audio_lat: torch.Tensor,
    chunk_ends: list[int],
    smooth_half: int = 2,
    energy_beta: float = 0.0,
    label: str = "",
) -> torch.Tensor:
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


_MC_AUDIO_KEY = "motion_context_audio_end_frame"
_MC_VIDEO_KEY = "motion_context_video_stride"

_MC_PATCHED = False
_MC_FAILED = None


def _mc_target_origin(layout) -> float:
    if not layout.segments:
        raise RuntimeError("PackedLayout has no segments")
    a, b, kind = layout.segments[-1]
    if kind != "video" or b <= a:
        raise RuntimeError("PackedLayout target video is not the final segment")
    return float(layout.position_ids[a, 0])


def _mc_ref_map(layout, refs):
    def emitted(block):
        kind = block.get("kind")
        rt = int(block.get("ref_audio_t", 0))
        if kind == "image":
            return ("ref_img",)
        if kind == "audio":
            return ("ref_audio",) if rt > 0 else ()
        if kind in ("video", "video_audio"):
            return (("ref_audio",) if rt > 0 else ()) + ("ref_img",)
        raise RuntimeError("unknown MiniMax H3 reference kind %r" % (kind,))

    actual = [(a, b, k) for a, b, k in layout.segments
              if k in ("ref_img", "ref_audio")]
    expected = [(i, k) for i, ref in enumerate(refs or [])
                for k in emitted(ref)]
    if len(actual) != len(expected):
        raise RuntimeError("MiniMax H3 reference layout segment count changed")
    out = {}
    for (index, wanted), (a, b, got) in zip(expected, actual):
        if wanted != got:
            raise RuntimeError("MiniMax H3 reference layout order changed")
        out.setdefault(index, {})[wanted] = (a, b)
    return out


def _mc_fixup_audio(layout, refs) -> None:
    marked = [i for i, r in enumerate(refs or [])
              if r.get(_MC_AUDIO_KEY) is not None]
    if len(marked) != 1:
        raise RuntimeError("expected exactly one marked Motion Audio Context ref")
    index = marked[0]
    ref = refs[index]
    if ref.get("kind") != "audio":
        raise RuntimeError("Motion Audio Context marker must be on an audio ref")
    steps = int(ref.get("ref_audio_t", 0))
    segment = _mc_ref_map(layout, refs).get(index, {}).get("ref_audio")
    if steps <= 0 or segment is None:
        raise RuntimeError("Motion Audio Context emitted no audio rows")
    a, b = segment
    if b - a != steps * 2:
        raise RuntimeError("Motion Audio Context row count changed")
    span_px = float(ref[_MC_AUDIO_KEY])
    if span_px < 0 or span_px > 1e6:
        raise RuntimeError(
            f"Motion Audio Context span out of range: {span_px} px")
    desired = (_mc_target_origin(layout)
               + FRAME_RESCALE * span_px - steps)
    current = float(layout.position_ids[a, 0])
    new_end = desired + steps
    origin = _mc_target_origin(layout)
    if new_end > origin + FRAME_RESCALE * span_px + 1e-6:
        raise RuntimeError(
            f"Motion Audio Context block would end past its span "
            f"(end {new_end:.3f} > {origin + FRAME_RESCALE * span_px:.3f})")
    layout.position_ids[a:b, 0] += desired - current


def _mc_fixup_video(layout, refs) -> None:
    marked = [i for i, r in enumerate(refs or ())
              if r.get(_MC_VIDEO_KEY) is not None]
    if not marked:
        return
    if len(marked) != 1:
        raise RuntimeError(
            "expected exactly one marked Motion Video Context ref")
    index = marked[0]
    ref = refs[index]
    if ref.get("kind") not in ("video", "video_audio"):
        raise RuntimeError(
            "Motion Video Context marker must be on a video ref")
    stride = int(ref[_MC_VIDEO_KEY])
    if stride <= 1:
        return
    vt = int(ref.get("latent_t", 0))
    segment = _mc_ref_map(layout, refs).get(index, {}).get("ref_img")
    if vt <= 0 or segment is None:
        raise RuntimeError("Motion Video Context emitted no video rows")
    a, b = segment
    n = b - a
    if n % vt:
        raise RuntimeError("Motion Video Context row count changed")
    frame_rows = n // vt
    origin = _mc_target_origin(layout)
    off = 0.0
    f = 0
    for k in range(vt):
        desired = origin + FRAME_RESCALE * off
        current = float(layout.position_ids[a + k * frame_rows, 0])
        layout.position_ids[a + k * frame_rows: a + (k + 1) * frame_rows, 0] += (
            desired - current)
        for _ in range(stride):
            off += FRAME_PER_TOKEN[f % 5]
            f += 1


def _mc_apply_patch() -> bool:
    global _MC_PATCHED, _MC_FAILED
    if _MC_PATCHED:
        return True
    if _MC_FAILED is not None:
        return False
    import inspect
    from comfy.ldm.minimax import model as mm
    for name in ("PackedLayout", "FRAME_RESCALE", "FRAME_PER_TOKEN"):
        if not hasattr(mm, name):
            _MC_FAILED = "MiniMax H3 model module is missing %s" % name
            return False
    orig_init = mm.PackedLayout.__init__
    try:
        _has_fc = "frame_count" in inspect.signature(orig_init).parameters
    except (TypeError, ValueError):
        _has_fc = False

    def patched_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                     keyframes=None, refs=None, frame_count=None):
        if _has_fc:
            orig_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                      keyframes=keyframes, refs=refs, frame_count=frame_count)
        else:
            orig_init(self, text_len, latent_t, latent_h, latent_w, audio_t,
                      keyframes=keyframes, refs=refs)
        if refs and any(r.get(_MC_VIDEO_KEY) is not None for r in refs):
            _mc_fixup_video(self, refs)
        if refs and any(r.get(_MC_AUDIO_KEY) is not None for r in refs):
            _mc_fixup_audio(self, refs)

    try:
        probe = mm.PackedLayout.__new__(mm.PackedLayout)
        patched_init(probe, 7, 7, 22, 38, 16,
                     refs=[{"kind": "audio", "ref_audio_t": 8,
                            _MC_AUDIO_KEY: 4.0}])
        a, b = _mc_ref_map(probe, [{"kind": "audio", "ref_audio_t": 8}])[0]["ref_audio"]
        wanted = (_mc_target_origin(probe) + mm.FRAME_RESCALE * 4.0 - 8.0)
        if b - a != 16 or abs(float(probe.position_ids[a, 0]) - wanted) > 1e-6:
            raise RuntimeError("Motion Audio Context self-test position mismatch")
        probe = mm.PackedLayout.__new__(mm.PackedLayout)
        patched_init(probe, 7, 8, 2, 2, 16,
                     refs=[{"kind": "video", "latent_t": 4, "latent_h": 2,
                            "latent_w": 2, "ref_audio_t": 0,
                            _MC_VIDEO_KEY: 2}])
        a, b = _mc_ref_map(probe, [{"kind": "video", "latent_t": 4,
                                    "latent_h": 2, "latent_w": 2,
                                    "ref_audio_t": 0}])[0]["ref_img"]
        va, vb, _vk = probe.segments[-1]
        if b - a != 4 or vb - va != 8:
            raise RuntimeError("Motion Video Context self-test row count")
        for _k in range(4):
            if abs(float(probe.position_ids[a + _k, 0])
                   - float(probe.position_ids[va + 2 * _k, 0])) > 1e-6:
                raise RuntimeError(
                    "Motion Video Context self-test position mismatch")
    except Exception as exc:
        _MC_FAILED = str(exc)
        print(f"[CLSS] WARNING: Motion Context layout patch self-test failed "
              f"({exc!r}); the context blocks fall back to unpositioned "
              f"placement.")
        return False

    mm.PackedLayout.__init__ = patched_init
    _MC_PATCHED = True
    return True


class CLSSH3Config:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tau_c":   ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5,  "step": 0.01,
                                      "tooltip": 'Context re-noising level: per-row sigma the SLB overlap rows are re-noised to, on both streams (video rows and the target audio overlap rows). Video rows are additionally replayed as clean keyframes; the audio side seeds the delivered tail on the target rows. 0 = fully frozen video overlap and no audio seed. The per-chunk schedule rises from this base toward a 0.10 ceiling with a 5-chunk half-life.',
                                      }),
                "beta":    ("FLOAT", {"default": 0.40, "min": 0.0, "max": 1.0,  "step": 0.05,
                                      "tooltip": 'Drift correction: blend factor of the EMA-tracked per-channel AdaIN renormalisation applied to every new chunk. 0 = no correction, 1 = full replacement with the EMA reference statistics. The EMA reference resets at every scene change.',
                                      }),
                "overlap": ("INT",   {"default": 7,    "min": 2,   "max": 32,
                                      "tooltip": "Context span in video latent tokens, snapped to the 5k+2 grid (2, 7, 12, ...; 7 tokens = 22 px = 0.92 s at 24 fps). The previous chunk's last rows are replayed as keyframe conditioning rows at their pixel times at the start of every continuation window and trimmed from the output; the window's other first rows stay free. Auto-clamped down in steps of 5 so overlap+new stays under the 12 s window cap.",
                                      }),
            },
            "optional": {
                "ema_mean_max": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05,
                                      "tooltip": "Mean anchor: how far the per-channel EMA mean may drift from chunk 0, in units of that channel's chunk-0 std. 0.25 bounds the mean within a quarter-sigma of chunk 0 while slow intentional changes still pass; 0 = uncapped.",
                                      }),
            },
        }
    RETURN_TYPES = ("CLSS_CONFIG",)
    RETURN_NAMES = ("clss_config",)
    FUNCTION = "build"
    CATEGORY = "MiniMaxH3-CLSS"

    def build(self, tau_c, beta, overlap, ema_mean_max=0.25):
        return (CLSSConfig(
            tau_c=tau_c,
            beta=beta,
            ema_lambda=0.10,
            ema_sigma_max_drift=0.05,
            ema_mean_max_drift=ema_mean_max,
            overlap_latent_frames=_snap_overlap(overlap),
            adain_max_amplification=1.2,
        ),)


class CLSSH3AudioConfig:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_recompose_steps": ("INT", {
                    "default": 0, "min": 0, "max": 30,
                    "tooltip": "Recompose pass against the finished video: the chunk's audio is replaced by a take from the audio_refine_guider model (wire the BASE model, not the turbo LoRA). The take generates its window plus audio_arc_margin_ms of extra audio that is not delivered, and the finished chunk video rides as a downscaled frozen video reference, so the pack shrinks greatly and each step is fast. 0 = off.",
                    }),
                "audio_recompose_sigma": ("FLOAT", {
                    "default": 1.0, "min": 0.50, "max": 1.0, "step": 0.05,
                    "tooltip": "Start sigma of the recompose. 1.0 = full fresh take from pure noise. Below 1.0 it refines the joint-pass take instead, which carries the joint take's repetition into the delivered audio. Keep 1.0 unless deliberately experimenting.",
                    }),
                "audio_arc_margin_ms": ("INT", {
                    "default": 4000, "min": 0, "max": 8000, "step": 250,
                    "tooltip": "End-of-window arc margin: the recompose generates this many ms of audio past the delivered window and keeps only the window-length head, so each take's end-of-window wind-down stays outside the delivery. 0 = off. Costs ~10-15% extra recompose tokens.",
                    }),
                "audio_recompose_pool": ("INT", {
                    "default": 2, "min": 1, "max": 4, "step": 1,
                    "tooltip": 'Spatial downscale of the chunk-video reference before packing. 2 = quarter of the video ref tokens; 4 = 1/16 (fastest, coarsest motion cue); 1 = full-res ref (~4x the tokens of pool 2). The ref is cropped to a 2xpool multiple first so patchify never sees an odd dim.',
                    }),
                "audio_recompose_stride": ("INT", {
                    "default": 2, "min": 1, "max": 5, "step": 1,
                    "tooltip": "Temporal stride over the reference's latent frames. 2 halves the ref tokens (~12 latent fps, ~0.3 s resolution) with the kept frames re-spaced onto the target time grid, so the ref still covers the full window at true time. 1 = every latent frame (tightest sync cues, 2x the ref tokens).",
                    }),
                "audio_recompose_seed": ("INT", {
                    "default": 0, "min": 0, "max": 0xffffffffffffffff,
                    "tooltip": "Noise seed of the fresh take. 0 = derive per chunk from the run's noise seed (deterministic per run, different take per chunk). Any other value is used directly, plus a per-chunk offset. The exact per-chunk seed is printed in the 'recompose' log line.",
                    }),
                "audio_recompose_ref_ms": ("INT", {
                    "default": 0, "min": 0, "max": 8000, "step": 250,
                    "tooltip": 'Recompose audio-ref span in ms. 0 = the ref spans exactly the overlap context (0.92 s at the default overlap); a longer ref (4000 ms = 2 bars, 8000 ms = a phrase at 120 BPM) gives the take real musical context so it can continue the piece instead of restating a bar.',
                    }),
                "audio_head_discard_ms": ("INT", {
                    "default": 0, "min": 0, "max": 300, "step": 5,
                    "tooltip": "Legacy onset-skip: discards N ms of the incoming chunk's audio right after the context end, on top of the context head. Every discarded 10 ms walks the audio ~N ms behind the video at each join and shortens the total by N ms per join. Raise only if you hear an onset pop at a join, in 10 ms steps.",
                    }),
                "loop_guard_rerolls": ("INT", {
                    "default": 2, "min": 0, "max": 4, "step": 1,
                    "tooltip": "Loop guard: how many times a chunk's recompose take may be re-rolled when it measures as a self-repeating vamp (0 = off). Each attempt costs one extra recompose; the take with the lowest within-chunk stationarity (aud_wc) wins, aud_loop breaks ties. Skipped on scenes whose audio rides their own <Audio j> reference.",
                    }),
                "loop_guard_wc": ("FLOAT", {
                    "default": 0.99, "min": 0.90, "max": 1.0, "step": 0.005,
                    "tooltip": "Within-chunk stationarity threshold for the loop guard (aud_wc: cosine between the take's consecutive thirds - the end-of-run repetition warning uses the same metric). 0.99 fires only on near-literal repeats; lower = more aggressive re-rolling.",
                    }),
                "loop_guard_loop": ("FLOAT", {
                    "default": 0.70, "min": 0.20, "max": 0.90, "step": 0.02,
                    "tooltip": 'History-loop NCC threshold for the loop guard (aud_loop: best NCC of the take against all previously delivered audio). NaN on the first chunk (no history) never fires. Material that merely continues the piece usually measures ~0.3-0.5 - lower this only deliberately.',
                    }),
                "loop_guard_retry_ref_ms": ("INT", {
                    "default": 2000, "min": 0, "max": 8000, "step": 250,
                    "tooltip": 'Loop-guard rescue ref span: re-roll attempts rebuild the recompose audio ref at this span for that attempt only (the piece-level span stays untouched); 2000 ms is the take-class loop relaxer. 0 = off (seed-only re-rolls). The best-of pick is unchanged, so the original take survives unless the rescue measures better.',
                    }),
            },
        }
    RETURN_TYPES = ("CLSS_AUDIO_CONFIG",)
    RETURN_NAMES = ("audio_config",)
    FUNCTION = "build"
    CATEGORY = "MiniMaxH3-CLSS"

    def build(self, audio_recompose_steps=0, audio_recompose_sigma=1.0,
              audio_arc_margin_ms=4000, audio_recompose_pool=2,
              audio_recompose_stride=2, audio_recompose_seed=0,
              audio_recompose_ref_ms=0, audio_head_discard_ms=0,
              loop_guard_rerolls=2, loop_guard_wc=0.99,
              loop_guard_loop=0.70, loop_guard_retry_ref_ms=2000):
        return ({
            "audio_recompose_steps": audio_recompose_steps,
            "audio_recompose_sigma": audio_recompose_sigma,
            "audio_arc_margin_ms": audio_arc_margin_ms,
            "audio_recompose_pool": audio_recompose_pool,
            "audio_recompose_stride": audio_recompose_stride,
            "audio_recompose_seed": audio_recompose_seed,
            "audio_recompose_ref_ms": audio_recompose_ref_ms,
            "audio_head_discard_ms": audio_head_discard_ms,
            "loop_guard_rerolls": loop_guard_rerolls,
            "loop_guard_wc": loop_guard_wc,
            "loop_guard_loop": loop_guard_loop,
            "loop_guard_retry_ref_ms": loop_guard_retry_ref_ms,
        },)


class CLSSH3ScenePrompts:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip":    ("CLIP",   {"tooltip": "CLIP text encoder (qwen3vl-32B). Each scene's raw text is encoded as its own CONDITIONING entry - no system prompt / chat template. A scene's text must stay identical across its chunks."}),
                "prompts": ("STRING", {"multiline": True, "dynamicPrompts": False,
                                       "default": "Scene 1 description\n---\nScene 2 description",
                                       "tooltip": "One scene per block, separated by a line containing only '---'. Each scene is encoded as its own CONDITIONING entry (token tags preserved); with N entries the sampler assigns one scene per chunk proportionally across num_chunks. The global_text field (if filled) is prepended to every block before encoding.",
                                       }),
            },
            "optional": {
                "global_text": ("STRING", {"multiline": True, "dynamicPrompts": False,
                                           "default": "",
                                           "tooltip": "Text copied to the top of every scene block before encoding - write shared sections once instead of repeating them per block. Empty = off. The prefix is baked into each scene's text, so it stays identical across that scene's chunks and survives the ref nodes' re-tokenization."}),
                "audio_continuity_text": ("STRING", {"multiline": True, "dynamicPrompts": False,
                                           "default": "",
                                           "tooltip": "Prompt-level continuation instruction, appended to the text on chunks that actually carry the continuation tail ref (every chunk except the run's first, and except scenes with their own audio refs). Each scene is encoded twice at build time and the sampler swaps the variant in on those chunks. Example: 'continue audio the same piece of music from exactly where it left off: same beat, rhythm, tempo, key and timbre, one uninterrupted take with no cut, no gap and no restart.' Empty = off."}),
            },
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "generate"
    CATEGORY = "MiniMaxH3-CLSS"

    def generate(self, clip, prompts: str, global_text: str = "",
                 audio_continuity_text: str = ""):
        scenes = [s.strip() for s in prompts.split("\n---\n") if s.strip()]
        if not scenes:
            scenes = [prompts.strip()]
        prefix = (global_text or "").strip()
        if prefix:
            print(f"[CLSS] scene prompts: global text ({len(prefix)} chars) "
                  f"prepended to all {len(scenes)} scene block(s).")
        cont = (audio_continuity_text or "").strip()
        if cont:
            print(f"[CLSS] scene prompts: audio-continuity text "
                  f"({len(cont)} chars) encoded as a second text variant "
                  f"per scene (used on continuation chunks only).")
        flat_conditioning = []
        for scene in scenes:
            text = f"{prefix}\n{scene}" if prefix else scene
            encoded = clip.encode_from_tokens_scheduled(clip.tokenize(text))
            for entry in encoded:
                entry[1]["clss_scene_text"] = text
            if cont:
                text2 = f"{text}\n\n{cont}"
                encoded2 = clip.encode_from_tokens_scheduled(
                    clip.tokenize(text2))
                for entry, e2 in zip(encoded, encoded2):
                    d2 = {k: v for k, v in e2[1].items()
                          if not str(k).startswith("clss_")}
                    d2["cross_attn"] = e2[0]
                    entry[1]["clss_cont"] = d2
                    entry[1]["clss_cont_text"] = text2
            flat_conditioning.extend(encoded)
        return (flat_conditioning,)


def _encode_ref_image_pair(vae, image, ref_image_size, width, height):
    h, w = image.shape[1], image.shape[2]
    if ref_image_size == "match":
        scale = min(1.0, math.sqrt((width * height) / (w * h)))
    else:
        scale = min(1.0, _REF_IMAGE_SHORT_EDGE / min(w, h))
    tw = max(32, round(w * scale / 32) * 32)
    th = max(32, round(h * scale / 32) * 32)
    resized = comfy.utils.common_upscale(
        image[:1, ..., :3].movedim(-1, 1), tw, th, "lanczos", "disabled"
    ).movedim(1, -1)
    z = vae.encode(resized)
    return ({"type": "image", "data": resized},
            {"kind": "image", "latent_h": th // 16, "latent_w": tw // 16,
             "latent": z})


def _ref_audio_waveform(audio_vae, audio):
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    waveform = waveform[:1]
    if waveform.shape[1] == 1:
        waveform = waveform.expand(-1, 2, -1)
    elif waveform.shape[1] > 2:
        waveform = waveform[:, :2]
    return waveform.contiguous(), vae_sr


def _encode_ref_audio_slice(audio_vae, waveform):
    z = audio_vae.encode(waveform.movedim(1, -1))
    return ({"type": "audio"},
            {"kind": "audio", "ref_audio_t": z.shape[-1], "audio_latent": z})


def _encode_ref_audio_pair(audio_vae, audio):
    waveform, _ = _ref_audio_waveform(audio_vae, audio)
    return _encode_ref_audio_slice(audio_vae, waveform)


_SCENE_TRACK_KEY = "clss_ref_audio_track"
_SCENE_WINDOW_KEY = "clss_scene_window"
_SCENE_WIN0_KEY = "clss_scene_window_win0"

_SCENE_WINDOW_GUARD_S = 4.0


def _scene_grid_window_bounds(cum_px: list[int], scene_of: list[int],
                              n_scenes: int, total_samples: int,
                              samples_per_af: int,
                              px_ol: float = 0.0,
                              ) -> list[tuple[int, int] | None]:
    bounds: list[tuple[int, int] | None] = []
    for _si in range(n_scenes):
        _idx = [_ci for _ci in range(len(scene_of)) if scene_of[_ci] == _si]
        if not _idx:
            bounds.append(None)
            continue
        _lead = (float(FRAME_RESCALE) * float(px_ol)
                 if _idx[0] > 0 else 0.0)
        s = max(0, min(total_samples,
                       round(cum_px[_idx[0]] * float(FRAME_RESCALE) - _lead)
                       * samples_per_af))
        e = min(total_samples,
                round(cum_px[_idx[-1] + 1] * float(FRAME_RESCALE))
                * samples_per_af)
        bounds.append((s, e))
    return bounds


def _recut_scene_audio_windows(pos_conds: list, cum_px: list[int],
                               scene_of: list[int], n_scenes: int,
                               audio_vae, px_ol: float = 0.0
                               ) -> tuple[int, str]:
    stash = None
    for _pc in pos_conds:
        if isinstance(_pc, dict) and _pc.get(_SCENE_TRACK_KEY):
            stash = _pc[_SCENE_TRACK_KEY]
            break
    if stash is None:
        return 0, ""
    marked = [i for i, _pc in enumerate(pos_conds)
              if isinstance(_pc, dict)
              and any(_r.get(_SCENE_WINDOW_KEY)
                      for _r in (_pc.get("minimax_refs") or []))]
    if not marked:
        return 0, ""
    track = stash.get("waveform")
    total = (int(track.shape[-1])
             if torch.is_tensor(track) and track.ndim == 3 else 0)
    vae_sr = int(stash.get("sample_rate")
                 or getattr(audio_vae, "audio_sample_rate", 32000) or 32000)
    spf = max(1, round(vae_sr / AUDIO_LATENT_FPS))
    bounds = _scene_grid_window_bounds(
        cum_px, scene_of, n_scenes, max(total, 1), spf, px_ol=px_ol)
    srcs = list(stash.get("windows") or [])
    recut = crop_n = reenc_n = 0
    _lens: list[str] = []
    _warn: list[str] = []
    for _si, _pc in enumerate(pos_conds):
        if not isinstance(_pc, dict) or _si >= len(bounds):
            continue
        refs = _pc.get("minimax_refs") or []
        idx = [i for i, _r in enumerate(refs) if _r.get(_SCENE_WINDOW_KEY)]
        if not idx:
            continue
        _b = bounds[_si]
        if _b is None or _b[1] - _b[0] < spf:
            continue
        a, b = _b
        blk = None
        src = srcs[_si] if _si < len(srcs) else None
        if (isinstance(src, dict)
                and src.get(_SCENE_WIN0_KEY) is not None):
            win0 = int(src[_SCENE_WIN0_KEY])
            rows = int(src.get("ref_audio_t") or 0)
            z = src.get("audio_latent")
            c0 = (a - win0) // spf
            c1 = (b - win0) // spf
            if (torch.is_tensor(z) and a >= win0 and 0 <= c0 < c1
                    and c1 <= rows and int(z.shape[-1]) >= c1):
                blk = {"kind": "audio", "ref_audio_t": int(c1 - c0),
                       "audio_latent": z[..., c0:c1].contiguous(),
                       _SCENE_WINDOW_KEY: True, _SCENE_WIN0_KEY: win0}
                crop_n += 1
        if blk is None and audio_vae is not None and total > 0:
            s = max(0, min(total, a))
            e = max(0, min(total, b))
            if e - s >= spf:
                try:
                    _item, blk = _encode_ref_audio_slice(
                        audio_vae, track[..., s:e].contiguous())
                    blk[_SCENE_WINDOW_KEY] = True
                    blk[_SCENE_WIN0_KEY] = int(s)
                    reenc_n += 1
                except Exception as exc:
                    print(f"[CLSS] WARNING: scene-grid ref-window re-cut "
                          f"failed for scene {_si + 1} ({exc!r}); keeping "
                          f"the build-time window.")
                    blk = None
        if blk is None:
            _warn.append(f"scene {_si + 1} (span {a // spf}-{b // spf} af "
                         f"outside its guarded window)")
            continue
        i0 = idx[0]
        _pc["minimax_refs"] = list(refs[:i0]) + [blk] + list(refs[i0 + 1:])
        recut += 1
        _lens.append(str(int(blk["ref_audio_t"])))
    if not recut:
        return 0, (
            "WARNING: could not re-cut " + "; ".join(_warn)
            + " — set audio_seconds_per_scene closer to the scene duration, "
            "or wire an audio_vae into this sampler." if _warn else "")
    line = (f"{recut}/{n_scenes} window(s) re-cut to the scene spans + "
            "window lead-in — " + "/".join(_lens) + f" af ({crop_n} cropped"
            + (f", {reenc_n} re-encoded" if reenc_n else "") + ")")
    if _warn:
        line += ("; WARNING: " + "; ".join(_warn)
                 + " — set audio_seconds_per_scene closer to the scene "
                 "duration, or wire an audio_vae into this sampler")
    return recut, line


def _build_audio_ref_block(audio_tail: torch.Tensor, span_px: float = 0.0,
                           end_px: float | None = None,
                           audio_vae=None,
                           device=None) -> tuple[dict | None, str]:
    if audio_tail is None or audio_tail.shape[-1] <= 0 or span_px <= 0:
        return None, ""
    wanted = int(round(span_px / float(_NATIVE_FPS) * AUDIO_LATENT_FPS))
    if wanted <= 0:
        return None, ""
    src = "lat"
    z = None
    if audio_vae is not None:
        try:
            z_in = audio_tail.to(device) if device is not None else audio_tail
            wav = audio_vae.decode(z_in)
            std = torch.std(wav, dim=[1, 2], keepdim=True) * 5.0
            std[std < 1.0] = 1.0
            wav = wav / std
            vae_sr = int(getattr(audio_vae, "audio_sample_rate", 32000))
            wanted_smp = int(round(span_px / float(_NATIVE_FPS) * vae_sr))
            have = int(wav.shape[1])
            enc = None
            if wanted_smp > 0 and have + 1 >= wanted_smp:
                enc = audio_vae.encode(wav[:, have - wanted_smp:, :])
            if enc is not None and enc.ndim == 4 and enc.shape[-1] > 0:
                z = enc
                src = "wav"
        except Exception as exc:
            print(f"[CLSS] WARNING: audio-ref waveform refresh failed "
                  f"({exc!r}); falling back to the delivered latent.")
            z = None
    if z is None:
        src = "lat"
        z = (audio_tail[..., -wanted:] if audio_tail.shape[-1] > wanted
             else audio_tail)
        if z.ndim != 4 or z.shape[-1] <= 0:
            return None, ""
    blk = {"kind": "audio", "ref_audio_t": int(z.shape[-1]),
           "audio_latent": z}
    if _mc_apply_patch():
        blk[_MC_AUDIO_KEY] = float(span_px if end_px is None else end_px)
    return blk, src


def _build_video_context_keyframes(context_latent: torch.Tensor | None,
                                   rows: int) -> list[dict]:
    if context_latent is None or rows <= 0:
        return []
    n = min(int(rows), int(context_latent.shape[2]))
    if n <= 0:
        return []
    tail = context_latent[:, :, -n:]
    out: list[dict] = []
    cursor = 0
    for i in range(n):
        out.append({"resolved_frame_index": int(cursor),
                    "latent": tail[:, :, i:i + 1].clone()})
        cursor += FRAME_PER_TOKEN[i % len(FRAME_PER_TOKEN)]
    return out


def _attach_scene_refs(clip, conditioning, scene_index, new_pairs):
    if not new_pairs:
        raise ValueError("connect at least one image and/or audio reference")
    idx = scene_index - 1
    if not 0 <= idx < len(conditioning):
        raise ValueError(f"scene_index {scene_index} is out of range — the "
                         f"conditioning holds {len(conditioning)} scene(s)")
    prev = conditioning[idx][1]
    text = prev.get("clss_scene_text")
    if text is None:
        raise ValueError("scene references can only attach to conditioning "
                         "from CLSSH3ScenePrompts (the scene's raw text is "
                         "needed to re-tokenize it with the reference "
                         "presentation).")
    prev_items = prev.get("clss_ref_items", [])
    prev_blocks = prev.get("minimax_refs", [])
    if len(prev_items) != len(prev_blocks):
        raise ValueError("the scene already carries minimax_refs from a "
                         "foreign node without the matching tokenizer "
                         "presentation — attach refs with the CLSS ref "
                         "nodes only")
    pairs = list(zip(prev_items, prev_blocks)) + new_pairs
    pairs.sort(key=lambda p: 0 if p[1]["kind"] == "image" else 1)
    items = [p[0] for p in pairs]
    blocks = [p[1] for p in pairs]

    tokens = clip.tokenize(text, minimax_ref_items=items)
    encoded = clip.encode_from_tokens_scheduled(tokens)
    nt, nd = encoded[0]
    nd = dict(nd)
    nd["minimax_refs"] = blocks
    nd["clss_ref_items"] = items
    nd["clss_scene_text"] = text
    cont_text = prev.get("clss_cont_text")
    if cont_text:
        enc2 = clip.encode_from_tokens_scheduled(
            clip.tokenize(cont_text, minimax_ref_items=items))
        nt2, nd2 = enc2[0]
        d2 = {k: v for k, v in dict(nd2).items()
              if not str(k).startswith("clss_")}
        d2["cross_attn"] = nt2
        nd["clss_cont"] = d2
        nd["clss_cont_text"] = cont_text
    out = list(conditioning)
    out[idx] = [nt, nd]
    ni = sum(1 for b in blocks if b["kind"] == "image")
    na = sum(1 for b in blocks if b["kind"] == "audio")
    _labels = ([f"<Picture 1..{ni}>"] if ni else []) + \
              ([f"<Audio 1..{na}>"] if na else [])
    print(f"[CLSS] scene {scene_index}: R2V refs = {ni} image(s) + "
          f"{na} audio(s) — labels {' / '.join(_labels) or 'none'}")
    return out


class CLSSH3SceneReference:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING", {"tooltip": 'Per-scene CONDITIONING from CLSSH3ScenePrompts (directly, or through previous CLSSH3SceneReference nodes).'}),
                "clip":    ("CLIP",   {"tooltip": "CLIP (qwen3vl-32B). The scene's text is re-tokenized with the reference presentation so the <Picture N>/<Audio N> labels bind to the ref media."}),
                "scene_index": ("INT", {"default": 2, "min": 1, "max": 64,
                                        "tooltip": "1-based scene this reference belongs to (scene 2 = the second '---' block). Only that scene's chunks carry the ref."}),
            },
            "optional": {
                "image":   ("IMAGE", {"tooltip": 'Reference image (identity/style/composition anchor - <Picture N> in the scene text). Requires vae.'}),
                "audio":   ("AUDIO", {"tooltip": 'Reference audio (voice/beat/texture anchor - <Audio N> in the scene text). Requires audio_vae.'}),
                "vae":     ("VAE",   {"tooltip": 'Video VAE, needed to encode the reference image.'}),
                "audio_vae": ("VAE", {"tooltip": 'Audio VAE (MiniMaxH3AudioVAE), needed to encode the reference audio.'}),
                "ref_image_size": (["match", "max"], {"default": "match",
                    "tooltip": "Reference image sizing. match: aspect-preserving downscale (never upscale) to the generation's pixel area - set width/height to the generation canvas. max: 2048 px short edge, best identity fidelity, but ref tokens ride every chunk of the scene and can be several times slower."}),
                "width":  ("INT", {"default": 1344, "min": 32, "max": 8192, "step": 32,
                                   "tooltip": 'Generation canvas width - only used by ref_image_size=match.'}),
                "height": ("INT", {"default": 768, "min": 32, "max": 8192, "step": 32,
                                   "tooltip": 'Generation canvas height - only used by ref_image_size=match.'}),
            },
        }
    RETURN_TYPES = ("CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "generate"
    CATEGORY = "MiniMaxH3-CLSS"

    @torch.inference_mode()
    def generate(self, clip, conditioning, scene_index, image=None, audio=None,
                 vae=None, audio_vae=None, ref_image_size="match",
                 width=1344, height=768):
        if image is not None and vae is None:
            raise ValueError("encoding a reference image needs the vae input")
        if audio is not None and audio_vae is None:
            raise ValueError("encoding a reference audio needs the audio_vae input")
        new_pairs = []
        if image is not None:
            new_pairs.append(_encode_ref_image_pair(
                vae, image, ref_image_size, width, height))
        if audio is not None:
            new_pairs.append(_encode_ref_audio_pair(audio_vae, audio))
        return (_attach_scene_refs(clip, conditioning, scene_index, new_pairs),)


class CLSSH3SceneReferences(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLSSH3SceneReferences",
            display_name="CLSS H3 Scene References (R2V multi)",
            category="MiniMaxH3-CLSS",
            description="Attach multiple R2V reference images/audios to ONE scene's conditioning. ref_image_1..N -> <Picture 1..N>, ref_audio_1..M -> <Audio 1..M> in the scene's prompt text.",
            inputs=[
                io.Conditioning.Input("conditioning", tooltip='Per-scene CONDITIONING from CLSSH3ScenePrompts (directly or through other CLSS ref nodes).'),
                io.Clip.Input("clip", tooltip="CLIP (qwen3vl-32B). The scene's text is re-tokenized with the reference presentation so the <Picture N>/<Audio N> labels bind."),
                io.Int.Input("scene_index", default=2, min=1, max=64,
                             tooltip="1-based scene these references belong to (scene 2 = the second '---' block). Only that scene's chunks carry the refs."),
                io.Vae.Input("vae", optional=True, tooltip='Video VAE, needed when any ref_image is connected.'),
                io.Vae.Input("audio_vae", optional=True, tooltip='Audio VAE (MiniMaxH3AudioVAE), needed when any ref_audio is connected.'),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                    tooltip="Reference image sizing. match: aspect-preserving downscale (never upscale) to the generation's pixel area - set width/height to the generation canvas. max: 2048 px short edge, best identity fidelity, but ref tokens ride every chunk of the scene and can be several times slower."),
                io.Int.Input("width", default=1344, min=32, max=8192, step=32, optional=True,
                             tooltip='Generation canvas width - only used by ref_image_size=match.'),
                io.Int.Input("height", default=768, min=32, max=8192, step=32, optional=True,
                             tooltip='Generation canvas height - only used by ref_image_size=match.'),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip='Reference image (identity/style/composition anchor). Socket order = <Picture N> order.'),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", tooltip='Reference audio (voice/beat/texture anchor). Socket order = <Audio N> order.'),
                        prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    @torch.inference_mode()
    def execute(cls, conditioning, clip, scene_index, vae=None, audio_vae=None,
                ref_image_size="match", width=1344, height=768,
                ref_images=None, ref_audios=None):
        ref_images = {k: v for k, v in (ref_images or {}).items()
                      if v is not None}
        ref_audios = {k: v for k, v in (ref_audios or {}).items()
                      if v is not None}
        if ref_images and vae is None:
            raise ValueError("encoding reference images needs the vae input")
        if ref_audios and audio_vae is None:
            raise ValueError("encoding reference audios needs the audio_vae input")
        new_pairs = []
        for name in sorted(ref_images,
                           key=lambda n: int(n.rsplit("_", 1)[-1])):
            new_pairs.append(_encode_ref_image_pair(
                vae, ref_images[name], ref_image_size, width, height))
        for name in sorted(ref_audios,
                           key=lambda n: int(n.rsplit("_", 1)[-1])):
            new_pairs.append(_encode_ref_audio_pair(audio_vae, ref_audios[name]))
        return io.NodeOutput(
            _attach_scene_refs(clip, conditioning, scene_index, new_pairs))


class CLSSH3SceneReferencesAll(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLSSH3SceneReferencesAll",
            display_name="CLSS H3 Scene References (R2V all scenes)",
            category="MiniMaxH3-CLSS",
            description="All-scene R2V refs in one node: every ref_image attaches to ALL scenes ('---' blocks), and the ref audio is encoded into one GUARDED window per scene (T ± 4 s). The sampler re-cuts every window automatically to the piece's exact scene span at run start by cropping the guard — no geometry to enter anywhere, and no drift against the delivered timeline. Replaces chaining one CLSSH3SceneReferences per scene.",
            inputs=[
                io.Conditioning.Input("conditioning", tooltip="Per-scene CONDITIONING from CLSSH3ScenePrompts - one entry per '---' block. Every scene is re-tokenized with its own reference presentation, so refs bind per scene."),
                io.Clip.Input("clip", tooltip="CLIP (qwen3vl-32B / ClipProj). Each scene's raw text is re-tokenized with that scene's <Picture N>/<Audio N> presentation; the text itself stays byte-identical."),
                io.Vae.Input("vae", optional=True, tooltip="Video VAE, needed when any ref_image is connected. Images are encoded once and the same latent is shared by every scene's block."),
                io.Vae.Input("audio_vae", optional=True, tooltip='Audio VAE (MiniMaxH3AudioVAE), needed when any ref_audio is connected. Connected files are resampled to the VAE rate, concatenated in socket order, and encoded into one guarded window per scene. The sampler crops each window to the exact scene span automatically; its own audio_vae is only the fallback if a span falls outside the guard.'),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                    tooltip="Reference image sizing, applied to every scene's copy. match: aspect-preserving downscale (never upscale) to the generation's pixel area - set width/height to the generation canvas. max: 2048 px short edge, best identity fidelity, but ref tokens ride every chunk of every scene and can be many times slower."),
                io.Int.Input("width", default=1344, min=32, max=8192, step=32, optional=True,
                             tooltip='Generation canvas width - only used by ref_image_size=match.'),
                io.Int.Input("height", default=768, min=32, max=8192, step=32, optional=True,
                             tooltip='Generation canvas height - only used by ref_image_size=match.'),
                io.Float.Input("audio_seconds_per_scene", default=10.0, min=1.0, max=60.0, step=0.5,
                    tooltip="Scene window pitch T: scene i's window covers seconds [i*T, (i+1)*T) of the concatenated ref_audio track, with a guard band around it. Set it to the time one scene actually generates; the sampler crops each window to the scene's exact delivered span automatically. A final partial window is kept; scenes past the end of the track get image refs only."),
                io.Autogrow.Input("ref_images", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_image", tooltip="Reference image (identity/style/composition anchor) - attached to every scene; reference it as <Picture N> in each scene's text. Socket order = <Picture N> order."),
                        prefix="ref_image_", min=0, max=9)),
                io.Autogrow.Input("ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio", tooltip="Reference audio (voice/beat/texture anchor). All connected files are concatenated in socket order, then cut into per-scene windows - each scene's window is its own <Audio 1>."),
                        prefix="ref_audio_", min=0, max=3)),
            ],
            outputs=[io.Conditioning.Output(display_name="conditioning")],
        )

    @classmethod
    @torch.inference_mode()
    def execute(cls, conditioning, clip, vae=None, audio_vae=None,
                ref_image_size="match", width=1344, height=768,
                audio_seconds_per_scene=10.0,
                ref_images=None, ref_audios=None):
        ref_images = {k: v for k, v in (ref_images or {}).items()
                      if v is not None}
        ref_audios = {k: v for k, v in (ref_audios or {}).items()
                      if v is not None}
        if not ref_images and not ref_audios:
            raise ValueError("connect at least one reference image and/or audio")
        if ref_images and vae is None:
            raise ValueError("encoding reference images needs the vae input")
        if ref_audios and audio_vae is None:
            raise ValueError("encoding reference audios needs the audio_vae input")
        n_scenes = len(conditioning)
        if n_scenes < 1:
            raise ValueError("the conditioning holds no scenes — feed it from "
                             "CLSSH3ScenePrompts (one entry per '---' block)")

        img_pairs = []
        for name in sorted(ref_images,
                           key=lambda n: int(n.rsplit("_", 1)[-1])):
            img_pairs.append(_encode_ref_image_pair(
                vae, ref_images[name], ref_image_size, width, height))

        aud_pairs: list[list] = [[] for _ in range(n_scenes)]
        _win_blocks: list[dict | None] = [None] * n_scenes
        _covered = 0
        _win = _total = vae_sr = 0
        _used = 0
        _track: torch.Tensor | None = None
        if ref_audios:
            track = None
            for name in sorted(ref_audios,
                               key=lambda n: int(n.rsplit("_", 1)[-1])):
                w, _ = _ref_audio_waveform(audio_vae, ref_audios[name])
                track = w if track is None else torch.cat([track, w], dim=-1)
            vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
            _track = track
            _spf = max(1, round(vae_sr / AUDIO_LATENT_FPS))
            _total = int(track.shape[-1])
            _win = max(1, round(float(audio_seconds_per_scene)
                                * AUDIO_LATENT_FPS)) * _spf
            _guard = int(round(_SCENE_WINDOW_GUARD_S * vae_sr))
            for i in range(n_scenes):
                _w0 = (max(0, i * _win - _guard) // _spf) * _spf
                if _w0 >= _total:
                    break
                _w1 = min(_total, (i + 1) * _win + _guard)
                _w1 = ((_w1 + _spf - 1) // _spf) * _spf
                if _w1 - _w0 < _spf:
                    break
                _seg = track[..., _w0:_w1]
                if int(_seg.shape[-1]) < _w1 - _w0:
                    _seg = F.pad(_seg, (0, (_w1 - _w0) - int(_seg.shape[-1])))
                _item, _blk = _encode_ref_audio_slice(
                    audio_vae, _seg.contiguous())
                _blk[_SCENE_WINDOW_KEY] = True
                _blk[_SCENE_WIN0_KEY] = int(_w0)
                _win_blocks[i] = _blk
                aud_pairs[i] = [(_item, _blk)]
                _covered += 1
                _used = min(_total, (i + 1) * _win)

        if ref_audios:
            _aud_desc = (f"{_covered}/{n_scenes} window(s) of "
                         f"{_win / vae_sr:.1f}s (+{_SCENE_WINDOW_GUARD_S:g}s "
                         f"guard) from {_total / vae_sr:.1f}s of audio")
        else:
            _aud_desc = "none"
        print(f"[CLSS] all-scenes refs: {len(img_pairs)} image(s) -> ALL "
              f"{n_scenes} scene(s) | audio: {_aud_desc}")
        if ref_audios:
            if _covered == n_scenes and _used < _total:
                print(f"[CLSS] all-scenes refs: {(_total - _used) / vae_sr:.1f}s "
                      f"of the ref audio is unused past the last scene — "
                      f"shorten the track, or add scenes/chunks to use it.")
            if _covered < n_scenes:
                print(f"[CLSS] WARNING: ref audio covers {_covered}/{n_scenes} "
                      f"scene(s) — scenes {_covered + 1}..{n_scenes} get image "
                      f"refs only (shorten the track, or add scenes/chunks).")

        out = conditioning
        for i in range(n_scenes):
            pairs = img_pairs + aud_pairs[i]
            if not pairs:
                print(f"[CLSS] all-scenes refs: scene {i + 1} has no refs — "
                      f"left as plain text conditioning.")
                continue
            out = _attach_scene_refs(clip, out, i + 1, pairs)
        if _track is not None:
            _stash = {"waveform": _track, "sample_rate": vae_sr,
                      "windows": _win_blocks}
            for _entry in out:
                _d = None
                if isinstance(_entry, (list, tuple)) and len(_entry) > 1 \
                        and isinstance(_entry[1], dict):
                    _d = _entry[1]
                elif isinstance(_entry, dict):
                    _d = _entry
                if _d is not None:
                    _d[_SCENE_TRACK_KEY] = _stash
        return io.NodeOutput(out)


class _GuiderCLSSH3(comfy.samplers.CFGGuider):

    _video_cfg = 1.0
    _audio_cfg = 1.0
    _rescale = 0.7
    _av_latent_shapes = None

    def set_av_params(self, video_cfg, audio_cfg, rescale):
        self._video_cfg = video_cfg
        self._audio_cfg = audio_cfg
        self._rescale = rescale
        self.set_cfg(video_cfg)
        self.audio_cfg = audio_cfg

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
                "model":    ("MODEL",        {"tooltip": 'MODEL (MiniMax H3) the guider is built on.'}),
                "positive": ("CONDITIONING", {"tooltip": 'Positive CONDITIONING. One entry per scene (from CLSSH3ScenePrompts) enables per-scene chunk guidance in the sampler.'}),
                "negative": ("CONDITIONING", {"tooltip": 'Negative CONDITIONING, required for split CFG; without it the guider falls back to a plain conditional pass.'}),
                "video_cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 30.0, "step": 0.5,
                                        "tooltip": 'Video CFG scale. 1.0 = off: H3 is CFG-distilled (the official graph uses BasicGuider with no CFG), and higher values produce corrupted, oversaturated frames. Raise only as an experiment.'}),
                "audio_cfg": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 30.0, "step": 0.5,
                    "tooltip": 'Audio CFG scale, independent of video_cfg. 1.0 = off; the canonical audio recipe runs 4.0. With video_cfg == audio_cfg == 1.0 the guider skips the uncond eval entirely - same cost as stock BasicGuider.',
                }),
                "rescale": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.05,
                                      "tooltip": "Per-stream CFG rescale toward the conditional prediction's std (factor = r*std_ratio + 1-r, clamped to [0.5, 2.0]). 0 = off. Counteracts CFG oversaturation.",
                                      }),
            },
        }
    RETURN_TYPES = ("GUIDER",)
    RETURN_NAMES = ("guider",)
    FUNCTION = "get_guider"
    CATEGORY = "MiniMaxH3-CLSS"

    def get_guider(self, model, positive, negative, video_cfg, audio_cfg, rescale):
        guider = _GuiderCLSSH3(model)
        guider.set_conds(positive, negative)
        guider.set_av_params(video_cfg, audio_cfg, rescale)
        return (guider,)


_UPSCALE_MODEL_FOLDER = "latent_upscale_models"
_H3_UPSCALER_DTYPES = {"fp32": torch.float32, "fp16": torch.float16,
                       "bf16": torch.bfloat16}
_H3_UPSCALER_MODULE = None
_H3_UPSCALER_ERROR: str | None = None


def _list_upscale_models() -> list[str]:
    try:
        import folder_paths
        if _UPSCALE_MODEL_FOLDER not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(
                _UPSCALE_MODEL_FOLDER,
                os.path.join(folder_paths.models_dir, _UPSCALE_MODEL_FOLDER))
        names = [
            n for n in folder_paths.get_filename_list(_UPSCALE_MODEL_FOLDER)
            if os.path.splitext(n)[1].lower()
            in (".safetensors", ".pt", ".pth", ".ckpt")
        ]
    except Exception as exc:
        print(f"[CLSS] WARNING: cannot scan latent_upscale_models ({exc!r})")
        names = []
    return names or ["(place models in: models/latent_upscale_models)"]


def _h3_upscaler_module():
    global _H3_UPSCALER_MODULE, _H3_UPSCALER_ERROR
    if _H3_UPSCALER_MODULE is not None:
        return _H3_UPSCALER_MODULE
    if _H3_UPSCALER_ERROR is not None:
        raise RuntimeError(_H3_UPSCALER_ERROR)
    import glob
    import importlib.util
    import sys as _sys
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _cands: list[str] = []
    for _pat in ("*/nodes/minimax_h3_latent_upscaler_3d.py",
                 "*/minimax_h3_latent_upscaler_3d.py"):
        _cands += sorted(glob.glob(os.path.join(_root, _pat)))
    if not _cands:
        _H3_UPSCALER_ERROR = (
            "[CLSS] the per-chunk upscaler needs the "
            "'Comfyui_Minimax_h3_latent_Upscaler' custom node pack installed "
            "next to this one (it is soft-imported by path, never vendored); "
            f"no minimax_h3_latent_upscaler_3d.py found under {_root}/*/")
        raise RuntimeError(_H3_UPSCALER_ERROR)
    _spec = importlib.util.spec_from_file_location(
        "clss_minimax_h3_latent_upscaler_3d", _cands[0])
    _mod = importlib.util.module_from_spec(_spec)
    _sys.modules[_spec.name] = _mod
    _spec.loader.exec_module(_mod)
    if not (hasattr(_mod, "load_model") and hasattr(_mod, "_make_norm_tensors")):
        _H3_UPSCALER_ERROR = (
            f"[CLSS] {_cands[0]} does not expose load_model/_make_norm_tensors — "
            f"unsupported version of the upscaler pack")
        raise RuntimeError(_H3_UPSCALER_ERROR)
    print(f"[CLSS] latent upscaler: soft-imported {_cands[0]}")
    _H3_UPSCALER_MODULE = _mod
    return _mod


def _blend_upscaled_overlap(acc_hr: list, up_hr: torch.Tensor,
                            overlap: int) -> None:
    if not acc_hr or overlap <= 0:
        return
    _bl = min(overlap, acc_hr[-1].shape[2], up_hr.shape[2])
    if _bl <= 0:
        return
    _ramp = torch.linspace(0.0, 1.0, _bl + 2, device=up_hr.device,
                           dtype=torch.float32)[1:-1].view(1, 1, -1, 1, 1)
    _prev = acc_hr[-1][:, :, -_bl:].to(up_hr.device).float()
    _blend = _prev * (1.0 - _ramp) + up_hr[:, :, :_bl].float() * _ramp
    acc_hr[-1] = torch.cat([acc_hr[-1][:, :, :-_bl],
                            _blend.to(up_hr.dtype).cpu()], dim=2)


class _H3UpscalerHandle:

    def __init__(self, module, model, name, precision):
        self.module = module
        self.model = model
        self.name = name
        self.precision = precision

    def _model_device(self) -> torch.device:
        for p in self.model.parameters():
            return p.device
        return torch.device("cpu")

    def _ensure_device(self, device) -> None:
        dev = torch.device(device)
        if self._model_device() != dev:
            self.model.to(dev, non_blocking=True)

    def to_device(self, device) -> None:
        self._ensure_device(device)

    def offload(self) -> None:
        if self._model_device().type != "cpu":
            self.model.to("cpu", non_blocking=True)
            comfy.model_management.soft_empty_cache()
            print(f"[CLSS] upscaler: offloaded {self.name} to CPU (VRAM released)")

    def upscale(self, video: torch.Tensor, scale: float,
                device=None) -> torch.Tensor:
        mod = self.module
        dev = torch.device(device) if device is not None \
            else comfy.model_management.get_torch_device()
        _b, _c, t, h_in, w_in = video.shape
        w_out = max(2, 2 * round(w_in * float(scale) / 2))
        h_out = max(2, 2 * round(h_in * float(scale) / 2))
        if w_out == w_in and h_out == h_in:
            return video
        dtype = (_H3_UPSCALER_DTYPES.get(self.precision, torch.float16)
                 if dev.type != "cpu" else torch.float32)
        self._ensure_device(dev)
        if dev.type != "cpu":
            comfy.model_management.soft_empty_cache()
        mean, std = mod._make_norm_tensors(dev, dtype)
        s_norm = (video.to(device=dev, dtype=dtype) - mean) / std
        eff = (w_out * 16 / (w_in * 16) + h_out * 16 / (h_in * 16)) / 2.0
        out = self.model(s_norm, scale=eff, target_size=(t, h_out, w_out),
                         enable_chunking=True)
        out = out * std + mean
        return out.to(dtype=video.dtype, device=video.device)


class CLSSH3LoadLatentUpscaleModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_list_upscale_models(), {
                    "tooltip": 'Minimax H3 latent-upscaler checkpoint from models/latent_upscale_models (same files as the Comfyui_Minimax_h3_latent_Upscaler pack - its 3D module must be installed; nothing is vendored). Loaded on CPU; the sampler moves it to the compute device (CUDA) for the run and offloads it afterwards. Without CUDA the upscaler runs fp32 on CPU, which is very slow.',
                    }),
                "precision": (["fp16", "bf16", "fp32"], {
                    "default": "fp16",
                    "tooltip": 'Inference precision. fp16 matches the validated checkpoints; bf16 if you see fp16 overflow; fp32 for exact reference math (slow, 2x memory).',
                    }),
            },
        }
    RETURN_TYPES = ("LATENT_UPSCALER",)
    RETURN_NAMES = ("upscaler",)
    FUNCTION = "load"
    CATEGORY = "MiniMaxH3-CLSS"

    def load(self, model_name, precision="fp16"):
        if model_name.startswith("("):
            raise ValueError(
                "place a Minimax H3 latent-upscaler checkpoint in "
                "models/latent_upscale_models (the same files the "
                "Comfyui_Minimax_h3_latent_Upscaler pack scans)")
        mod = _h3_upscaler_module()
        model = mod.load_model(model_name, torch.device("cpu"), precision)
        print(f"[CLSS] latent upscaler ready: {model_name} ({precision})")
        return (_H3UpscalerHandle(mod, model, model_name, precision),)


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


class _FreshAVNoise:

    def __init__(self, vid_noise: torch.Tensor, aud_noise: torch.Tensor,
                 seed: int):
        self._vid = vid_noise
        self._aud = aud_noise
        self.seed = seed

    def generate_noise(self, input_latent: dict):
        samples = input_latent["samples"]
        if isinstance(samples, comfy.nested_tensor.NestedTensor):
            return comfy.nested_tensor.NestedTensor((self._vid, self._aud))
        return self._vid


class CLSSH3StreamingSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider":      ("GUIDER",      {"tooltip": 'GUIDER from CLSSH3Guider. When its positive conditioning holds N scene entries, one scene is unpacked per chunk proportionally across num_chunks.'}),
                "sampler":     ("SAMPLER",     {"tooltip": "SAMPLER for the per-chunk denoise (KSamplerSelect). The audio stream's own shifted schedule is handled inside the model; set shifts on the stock MiniMaxH3SigmaShift node."}),
                "sigmas":      ("SIGMAS",      {"tooltip": 'SIGMAS schedule (e.g. BasicScheduler, SplitSigmas, ManualSigmas). This is the video schedule; the audio schedule is derived from it inside the model. Any slice of the 1.0 -> 0.0 flow schedule is accepted: a low-res pass may end above 0; a schedule may also start below 1.0 (every chunk then starts from noise at sigma0).'}),
                "noise":       ("NOISE",       {"tooltip": "NOISE source (RandomNoise). Its seed drives the run-constant full-length noise tensors that each chunk's initial noise is sliced from."}),
                "latent":      ("LATENT",      {"tooltip": 'Per-chunk AV latent template from EmptyMiniMaxH3LatentAV (video [B,24,T,H/16,W/16] + audio [B,32,2,Ta]). Its frame count sets the per-chunk length; total length = num_chunks x chunk. T must be on the 5k+2 grid.'}),
                "clss_config": ("CLSS_CONFIG", {"tooltip": 'CLSS_CONFIG from the CLSSH3Config node (tau_c, beta, overlap).'}),
                "num_chunks":  ("INT",         {"default": 10, "min": 1, "max": 500,
                                                "tooltip": 'Number of streaming chunks; total video length = num_chunks x chunk length (chunk 0 covers 17k+5 px, each continuation 17k px). A chunk whose window would exceed the 12 s cap is auto-split into uniform grid-aligned sub-chunks. With scene_handoff=transition_chunk every scene block needs >= 2 chunks, i.e. num_chunks >= 2x scenes.',
                                                }),
            },
            "optional": {
                "audio_config": ("CLSS_AUDIO_CONFIG", {
                    "tooltip": 'CLSSH3AudioConfig - owns all audio-chain settings (recompose, arc margin, pool/stride/seed, ref span, head discard, loop guard). Wire it to configure audio; without it the audio chain runs on its defaults (recompose off, guard idle).',
                    }),
                "image": ("IMAGE", {"tooltip": 'Optional i2v guide image; VAE-encoded and pinned as a keyframe row at frame 0 of chunk 0. Requires vae.'}),
                "vae":   ("VAE",   {"tooltip": 'Video VAE, only needed together with image for the i2v guide encode.'}),
                "audio_vae": ("VAE", {"tooltip": 'Audio VAE (MiniMaxH3AudioVAE - the same checkpoint CLSSH3VideoDecodeSave uses; wire the same VAELoader). When wired, the cross-chunk audio continuity ref is refreshed from the audible waveform at every boundary: the delivered tail is decoded, normalized like the export chain, cut to the ref span and re-encoded, so each chunk continues from what the ears hear. It is also the fallback for scene ref-audio windows whose span falls outside the stashed guard band. Without it - or when the waveform cannot cover the span - the delivered latent is carried.'}),
                "upscaler": ("LATENT_UPSCALER", {"tooltip": "Optional neural latent upscaler (from CLSSH3LoadLatentUpscaleModel) applied inside the stream: every chunk is upscaled right after its SLB step, so the streaming state stays low-res and a long video never exists at high resolution all at once. The chunk's full window goes in for temporal context, and the overlap span is cross-faded over the previous delivered tail. The output latent is high-res video + unchanged audio."}),
                "upscale_scale": ("FLOAT", {
                    "default": 1.5, "min": 1.0, "max": 4.0, "step": 0.05,
                    "tooltip": 'Spatial upscale factor applied per chunk: output = chunk template x this, aligned to the 32-px canvas rule (e.g. a 832x480 template x 1.5 -> 1248x720 px). Set EmptyMiniMaxH3LatentAV to the low-res generation size. 1.0 = no upscaling (warns).',
                    }),
                "fps": ("FLOAT", {
                    "default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0,
                    "tooltip": 'Frames per second of the output. H3 is 24 fps native - the pixel/audio time mapping is fixed, so any other value only triggers a warning and 24 is used.',
                    }),
                "detail_anchor": (["on", "off"], {
                    "default": "on",
                    "tooltip": "Two-band spatial detail anchor: each chunk's low/high-frequency band energies are rescaled toward the scene's first-chunk reference (gains clamped to [0.90, 1.10] low / [0.90, 1.12] high) to fight the long-run detail fade. Off = uncorrected.",
                    }),
                "audio_xfade_ms": ("INT", {
                    "default": 0, "min": 0, "max": 500, "step": 10,
                    "tooltip": 'DEPRECATED and ignored; kept only so older workflow files keep loading. The delivered seam is the sample-exact splice only.',
                    }),
                "audio_join_lead_ms": ("INT", {
                    "default": 0, "min": 0, "max": 900, "step": 10,
                    "tooltip": 'DEPRECATED and ignored; kept only so older workflow files keep loading. The delivered seam is the sample-exact splice only.',
                    }),
                "audio_refine_guider": ("GUIDER", {
                    "tooltip": "Separate GUIDER for the audio recompose pass: wire a CLSSH3Guider built on the BASE model (no LoRA), with the same conditioning as the main guider. Only used when the audio config's recompose steps > 0.",
                    }),
                "scene_handoff": (["transition_chunk", "blend", "hard"], {
                    "default": "transition_chunk",
                    "tooltip": "How text conditioning changes at a scene boundary. transition_chunk: two-step crossfade straddling the boundary (outgoing scene's last chunk 25%-incoming, incoming scene's first chunk 75%-incoming; every scene block needs >= 2 chunks). blend: single 50/50 blend on the first chunk of the new scene. hard: plain text swap.",
                    }),
                "tail_margin_px": ("INT", {
                    "default": 12, "min": 0, "max": 48, "step": 4,
                    "tooltip": "Extra pixel frames generated at the end of every chunk but NOT delivered: each take's end-of-window wind-down lands there instead of in the seam. 0 = grid-exact windows. The soft 12 s window cap counts the margin.",
                    }),
                "continuation_context": ("CLSS_CONTEXT", {
                    "tooltip": "Continuation context from CLSSH3ContinueFromVideo or CLSSH3ReeditChunk. Wired: the run opens on a continuation window - the first chunk carries the context rows (keyframe replay + SLB seed re-noised at tau_v) and the context audio as its tail ref, and delivers the new span preceded by a 2-token head (the saved tail's last 5 px re-covered, the 17k+5 piece head the decoder expects - without it every 5th token would render at the wrong grid phase). A continue context (CLSSH3ContinueFromVideo) asks decode-save to drop that head from the saved frames and audio, so the take starts at the new span; a re-edit context (CLSSH3ReeditChunk) keeps it - the head is part of the span the re-edit replaces. Unwired: the sampler behaves exactly as before.",
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
        audio_config=None,
        image=None,
        vae=None,
        audio_vae=None,
        upscaler=None,
        upscale_scale: float = 1.5,
        fps: float = 24.0,
        detail_anchor: str = "on",
        audio_xfade_ms: int = 0,
        audio_join_lead_ms: int = 0,
        scene_handoff: str = "transition_chunk",
        audio_refine_guider=None,
        tail_margin_px: int = 12,
        continuation_context=None,
    ):
        _freed_gb = _unload_before_sampling()
        print("[CLSS] unloaded all models before sampling"
              + (f" ({_freed_gb:.2f} GB freed)" if _freed_gb is not None
                 else ""))
        _alloc_state = _enable_expandable_segments()
        if _alloc_state == "env":
            print("[CLSS] CUDA allocator: expandable_segments enabled "
                  "(from PYTORCH_CUDA_ALLOC_CONF)")
        elif _alloc_state == "runtime":
            print("[CLSS] CUDA allocator: expandable_segments enabled at "
                  "runtime")
        elif _alloc_state == "off":
            print("[CLSS] CUDA allocator: expandable_segments off — set "
                  "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True before "
                  "starting ComfyUI")
        if fps != _NATIVE_FPS:
            print(f"[CLSS] WARNING: fps={fps} ignored — H3 is {_NATIVE_FPS} fps.")
        fps = float(_NATIVE_FPS)
        _aud = _resolve_audio_settings(audio_config)
        audio_head_discard_ms = int(_aud["audio_head_discard_ms"])
        audio_recompose_steps = int(_aud["audio_recompose_steps"])
        audio_recompose_sigma = float(_aud["audio_recompose_sigma"])
        audio_arc_margin_ms = int(_aud["audio_arc_margin_ms"])
        audio_recompose_pool = int(_aud["audio_recompose_pool"])
        audio_recompose_stride = int(_aud["audio_recompose_stride"])
        audio_recompose_seed = int(_aud["audio_recompose_seed"])
        audio_recompose_ref_ms = int(_aud["audio_recompose_ref_ms"])
        loop_guard_rerolls = int(_aud["loop_guard_rerolls"])
        loop_guard_wc = float(_aud["loop_guard_wc"])
        loop_guard_loop = float(_aud["loop_guard_loop"])
        loop_guard_retry_ref_ms = int(_aud["loop_guard_retry_ref_ms"])
        _tm_px_in = max(0, min(48, int(tail_margin_px)))
        _tm_lf, _tm_px = _tail_margin_tokens(tail_margin_px)
        _tm_af = int(round(_tm_px * FRAME_RESCALE)) if _tm_px > 0 else 0
        if _tm_px > 0:
            print(f"[CLSS] tail margin {_tm_px}px ({_tm_lf} tokens, {_tm_af}af "
                  f"= {_tm_px / float(_NATIVE_FPS) * 1000.0:.0f} ms, "
                  f"generated but not delivered)")

        _s = sigmas.flatten().float().cpu()
        if (_s.numel() < 2 or not (0.0 < float(_s[0]) <= 1.02)
                or float(_s[-1]) < -1e-6
                or not bool((_s[:-1] >= _s[1:]).all())):
            raise ValueError(
                "[CLSS] sigmas must be a monotonically decreasing flow schedule "
                "within [0, 1] — ANY slice of the 1.0->0.0 schedule is accepted: "
                "a low-res pass may END above 0 (its x0 estimate is then "
                "upscaled) and a partial schedule may START below 1.0 (e.g. "
                "0.9035, 0.6316, 0.3158, 0.0 — every chunk then starts from "
                "noise at sigma0). "
                "Build it with BasicScheduler / SplitSigmas / ManualSigmas on the "
                f"MiniMaxH3SigmaShift-patched model; got [0]={float(_s[0]):.6g} "
                f"[-1]={float(_s[-1]):.6g} len={_s.numel()}")

        if audio_refine_guider is not None and audio_recompose_steps <= 0:
            print("[CLSS] WARNING: audio_refine_guider connected but the "
                  "recompose is off — the guider is unused.")
        if audio_recompose_steps > 0 and audio_refine_guider is None:
            print("[CLSS] WARNING: recompose is on without an "
                  "audio_refine_guider — using the MAIN guider.")

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

        _ctx = continuation_context if isinstance(continuation_context, dict) else None
        _ctx_video = _ctx.get("video") if _ctx is not None else None
        _ctx_aud = _ctx.get("aud_ref") if _ctx is not None else None
        _ctx_pins = list(_ctx.get("pins") or []) if _ctx is not None else []
        _ctx_vref = _ctx.get("vref") if _ctx is not None else None
        _ctx_mode = "cont" if _ctx_video is not None else "none"
        _ctx_trim = bool(_ctx.get("trim_head")) if _ctx is not None else False
        if _ctx_video is not None:
            if (int(_ctx_video.ndim) != 5 or int(_ctx_video.shape[3]) != H
                    or int(_ctx_video.shape[4]) != W):
                raise ValueError(
                    f"continuation context rows {tuple(_ctx_video.shape)} do "
                    f"not match the run's latent grid [*, *, *, {H}, {W}] — "
                    f"the context must come from the same template "
                    f"resolution")
            if int(_ctx_video.shape[2]) != overlap:
                raise ValueError(
                    f"continuation context carries {int(_ctx_video.shape[2])} "
                    f"row(s) but the config's overlap is {overlap} token(s) — "
                    f"rebuild the context node with the same clss_config")
        for _pin in _ctx_pins:
            _pl = _pin.get("latent")
            if (not torch.is_tensor(_pl) or int(_pl.ndim) != 5
                    or int(_pl.shape[2]) != 1
                    or int(_pl.shape[3]) != H or int(_pl.shape[4]) != W):
                raise ValueError(
                    "a pinned frame latent does not match the run's latent "
                    "grid — rebuild the context node from the same template")
        if _ctx_aud is not None or _ctx_vref is not None:
            _mc_apply_patch()

        _ups = upscaler
        if _ups is not None and not hasattr(_ups, "upscale"):
            raise ValueError(
                "[CLSS] upscaler must come from CLSSH3LoadLatentUpscaleModel "
                f"(got {type(_ups).__name__})")
        _up_scale = float(upscale_scale)
        _up_active = _ups is not None and _up_scale > 1.0001
        _up_w = _up_h = 0
        _up_dev = None
        if _up_active:
            _up_w = max(2, 2 * round(W * _up_scale / 2))
            _up_h = max(2, 2 * round(H * _up_scale / 2))
            _up_dev = comfy.model_management.get_torch_device()
            if _up_w == W and _up_h == H:
                _up_active = False
                print(f"[CLSS] WARNING: upscale_scale={_up_scale:g} does not change "
                      f"the {W}x{H} latent grid after 32-px alignment — upscaler "
                      f"skipped for this run.")
        elif _ups is not None:
            print("[CLSS] WARNING: upscaler connected but upscale_scale <= 1.0 — "
                  "no upscaling happens.")

        img_guide_latent: torch.Tensor | None = None
        if image is not None and vae is not None:
            if _ctx is not None:
                print("[CLSS] WARNING: the i2v guide image is ignored — a "
                      "continuation context is wired (its pins carry the "
                      "anchor frames).")
            img = image[:1, ..., :3].movedim(-1, 1)
            img = comfy.utils.common_upscale(img, W * 16, H * 16, "lanczos", "disabled")
            img_guide_latent = vae.encode(img.movedim(1, -1))

        cap_px = int(_WINDOW_CAP_S * fps)
        plan_tokens, _eff_overlap, _win_px, _auto_split = _plan_chunk_tokens(
            new_lf0, num_chunks, overlap, tail_margin_px, ctx_mode=_ctx_mode)
        px_ol = _px_of_tokens(_eff_overlap, 0)
        if _ctx_mode != "none" and _eff_overlap != overlap:
            raise ValueError(
                f"the {_WINDOW_CAP_S:.0f} s window cap clamped the overlap "
                f"{overlap} -> {_eff_overlap} token(s); the continuation "
                f"context was built for {overlap} — shorten the chunk "
                f"template or the tail_margin_px")
        if _auto_split:
            print(f"[CLSS] chunk window exceeds the {_WINDOW_CAP_S:.0f} s cap — "
                  f"auto-split into {len(plan_tokens)} sub-chunks "
                  f"(overlap clamped to {_eff_overlap} tokens).")
        if _eff_overlap != overlap:
            print(f"[CLSS] overlap clamped {overlap} -> {_eff_overlap} tokens "
                  f"to keep windows under the {_WINDOW_CAP_S:.0f} s cap.")
            clss_config = dataclasses.replace(clss_config, overlap_latent_frames=_eff_overlap)

        chunk_plan: list[tuple[int, int, int]] = []
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

        pos_conds = guider.original_conds.get("positive", [])
        num_scenes = len(pos_conds)
        _scene_of = [min(int(_i * num_scenes / _eff_num_chunks), num_scenes - 1)
                     if num_scenes > 1 else 0
                     for _i in range(_eff_num_chunks)]
        if _ctx is not None and int(_ctx.get("scene", 0)) >= 1:
            _cs = min(int(_ctx["scene"]) - 1, max(0, num_scenes - 1))
            if _cs != _scene_of[0]:
                print(f"[CLSS] continuation context: first window uses scene "
                      f"{_cs + 1} of {num_scenes} (scene override).")
            _scene_of = [_cs] + list(_scene_of[1:])
        _cumpx = [0]
        for _cf in chunk_plan:
            _cumpx.append(_cumpx[-1] + int(_cf[2]))
        _scene_span_af: list[float] = []
        _scene_lead_af: list[float] = []
        for _si in range(num_scenes):
            _idx = [_ci for _ci in range(_eff_num_chunks)
                    if _scene_of[_ci] == _si]
            if not _idx:
                _scene_span_af.append(0.0)
                _scene_lead_af.append(0.0)
                continue
            _span_px = _cumpx[_idx[-1] + 1] - _cumpx[_idx[0]]
            _scene_span_af.append(_span_px * float(FRAME_RESCALE))
            _scene_lead_af.append(
                float(FRAME_RESCALE) * float(px_ol) if _idx[0] > 0 else 0.0)
        _recut_n, _recut_line = _recut_scene_audio_windows(
            pos_conds, _cumpx, _scene_of, num_scenes, audio_vae, px_ol=px_ol)
        if _recut_line:
            print(f"[CLSS] ref-audio windows: {_recut_line}")
        _bn_px0 = chunk_plan[0][2]
        _bn_pxc = chunk_plan[1][2] if _eff_num_chunks > 1 else 0
        _bn_af0 = chunk_plan[0][1]
        _bn_afc = chunk_plan[1][1] if _eff_num_chunks > 1 else 0
        print("[CLSS] ================ run settings ================")
        print(f"[CLSS] chunks {_eff_num_chunks} (requested {num_chunks}) | "
              f"scenes {num_scenes} handoff={scene_handoff} | "
              f"seed {getattr(noise, 'seed', '?')}")
        _ref_desc = []
        _win_warn: list[str] = []
        for _si, _pc in enumerate(pos_conds):
            _rl = _pc.get("minimax_refs") or []
            if _rl:
                _ni = sum(1 for _r in _rl if _r.get("kind") == "image")
                _arts = [_r for _r in _rl if _r.get("kind") == "audio"]
                _ref_desc.append(f"scene{_si + 1}={_ni}i+{len(_arts)}a")
                if len(_arts) == 1 and _scene_span_af[_si] > 0:
                    _aw = int(_arts[0].get("ref_audio_t") or 0)
                    _want = round(_scene_span_af[_si] + _scene_lead_af[_si])
                    if _aw > 0 and abs(_aw - _want) > 2:
                        _win_warn.append(
                            f"scene {_si + 1}: ref-audio window {_aw}af vs "
                            f"scene span {_scene_span_af[_si]:.1f}af "
                            f"(expected {_want}af incl. lead-in)")
        print(f"[CLSS] R2V refs: {' | '.join(_ref_desc) if _ref_desc else 'none'}")
        if _win_warn:
            print("[CLSS] WARNING: ref-audio window length != the scene's "
                  "delivered span (" + "; ".join(_win_warn) + ") — not "
                  "re-cut automatically (re-run the ref node, or this is a "
                  "custom ref).")
        print(f"[CLSS] chunk0 {_bn_px0}px->{_bn_af0}af ({_bn_px0 / fps:.2f}s) | "
              f"cont {_bn_pxc}px->{_bn_afc}af | "
              f"overlap {_eff_overlap}tok={px_ol}px/{Ta_ol}af "
              f"({px_ol / fps:.2f}s) | window {_win_px}px "
              f"({_win_px / fps:.2f}s, cap {cap_px}px) | "
              f"total ~{Ta_total / AUDIO_LATENT_FPS:.1f}s audio")
        _tv0 = _tau_c_eff(clss_config.tau_c, _VIDEO_TAU_C_CEILING, 0)
        _tvN = _tau_c_eff(clss_config.tau_c, _VIDEO_TAU_C_CEILING,
                          max(0, _eff_num_chunks - 2))
        print(f"[CLSS] video continuity: {_eff_overlap}-token context at "
              f"0..{px_ol - 1} px — keyframe replay + SLB seed re-noised at "
              f"tau_v {_tv0:.3f}->{_tvN:.3f} (ceiling {_VIDEO_TAU_C_CEILING})")
        _aud_ref_desc = ("refreshed from the decoded waveform"
                         if audio_vae is not None
                         else "carried as the delivered latent")
        print(f"[CLSS] audio continuity: ref {Ta_ol}af "
              f"({Ta_ol / AUDIO_LATENT_FPS:.2f}s) ending at the join, "
              f"{_aud_ref_desc}; tail ref only on chunks whose scene has no "
              f"audio ref | head discard {audio_head_discard_ms}ms "
              + (f"| tail margin {_tm_px}px " if _tm_px > 0 else "")
              + f"| sample-exact seams "
              + f"| detail_anchor {detail_anchor} | clss "
              f"tau_c {getattr(clss_config, 'tau_c', '?')} "
              f"beta {getattr(clss_config, 'beta', '?')} "
              f"mean_anchor 0.9±{getattr(clss_config, 'ema_mean_max_drift', '?')}"
              f"σ0 "
              f"overlap {clss_config.overlap_latent_frames}tok"
              + (f" | audio RECOMPOSE {audio_recompose_steps} steps from "
                 f"sigma {audio_recompose_sigma:.2f} | arc margin "
                 f"{audio_arc_margin_ms}ms | ref pool x{audio_recompose_pool}"
                 f"/stride {audio_recompose_stride} | recompose-ref "
                 f"{('ctx' if int(audio_recompose_ref_ms) == 0 else str(int(audio_recompose_ref_ms)) + 'ms')})"
                 if audio_recompose_steps > 0
                 else ""))
        print(f"[CLSS] sigmas {_s.numel() - 1} steps "
              f"[{float(_s[0]):.3f}..{float(_s[-1]):.3f}] | cfg v="
              f"{getattr(guider, '_video_cfg', '?')} a="
              f"{getattr(guider, 'audio_cfg', '?')} rescale="
              f"{getattr(guider, '_rescale', '?')}")
        if _up_active:
            _up_note = (" [CPU — fp32 fallback, expect very slow]"
                        if _up_dev.type == "cpu" else "")
            print(f"[CLSS] upscaler: {getattr(_ups, 'name', '?')} x{_up_scale:g} "
                  f"on {_up_dev} -> {_up_w * 16}x{_up_h * 16} px output "
                  f"(per chunk){_up_note}")
            _ups.to_device(_up_dev)
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
            [_px_of_tokens(chunk_plan[0][0], 0)
             + (px_ol if _ctx_mode != "none" else 0)]
            + [px_ol + _px_of_tokens(_p, _eff_overlap % 5) for _p, _a, _pxn in chunk_plan[1:]]
        )
        if _ctx is not None:
            _cv_n = 0 if _ctx_video is None else int(_ctx_video.shape[2])
            print(f"[CLSS] continuation context: {_cv_n} row(s) "
                  f"({_px_of_tokens(_cv_n, 0)} px)"
                  + (f" + audio ref {int(_ctx_aud['ref_audio_t'])}af"
                     if _ctx_aud is not None else " + no audio ref")
                  + f" | pins {len(_ctx_pins)}"
                  + (f" | video ref {int(_ctx_vref['latent_t'])}tok"
                     if _ctx_vref is not None else "")
                  + " | first window "
                  + (f"re-render ({chunk_plan[0][2]} px = 2-token head "
                     f"({_px_of_tokens(2, (_eff_overlap - 2) % 5)} px) + "
                     f"{_px_of_tokens(chunk_plan[0][0] - 2, 2)} px new; head "
                     f"{'dropped' if _ctx_trim else 'kept'} at decode-save)"
                     if _ctx_mode == "cont"
                     else "generation + pins"))
        print(f"[CLSS] plan: {_eff_num_chunks} chunk(s) of {plan_tokens[0]}"
              f"{'+' + str(plan_tokens[1]) if _eff_num_chunks > 1 else ''} tokens, "
              f"overlap={_eff_overlap} tokens ({px_ol} px / {Ta_ol} af), "
              f"window ≤ {_max_win_px / fps:.1f} s, "
              f"total {T_total} tokens / {_p_acc} px / {Ta_total} af "
              f"({_p_acc / fps:.1f} s), scenes={num_scenes}")

        _noise_seed = getattr(noise, "seed", 0)
        _cap_v = max(T_total, _NOISE_FIELD_CAP_TOK)
        _cap_a = max(Ta_total, _NOISE_FIELD_CAP_AF)
        _noise_tmpl = torch.zeros(B, C_v, _cap_v, H, W)
        _full_noise_vid: torch.Tensor = noise.generate_noise({"samples": _noise_tmpl})
        del _noise_tmpl
        _g_aud = torch.Generator(device="cpu").manual_seed(
            (int(_noise_seed) + 1) % (2 ** 63))
        _full_noise_aud = torch.randn(B_a, C_a, lanes_a, _cap_a,
                                      generator=_g_aud, dtype=aud_tmpl.dtype)

        clss_state = CLSSState(clss_config)
        acc_video: list[torch.Tensor] = []
        acc_audio: list[torch.Tensor] = []
        acc_video_hr: list[torch.Tensor] = []
        _up_secs = 0.0
        audio_chunk_ends: list[int] = []
        _splice: dict = {"delivered_af": [], "head_af": [], "fill_af": [],
                         "bad": [], "join_af_exact": [],
                         "total_af_exact": 0.0}
        _audio_tail: torch.Tensor | None = None
        _ctx_trim_rec: dict | None = None
        _slb_std_base: float | None = None
        _s1_prev_last: torch.Tensor | None = None
        _s1_aud_rms_ref: float | None = None
        _s1_aud_level_ref: float | None = None
        _s1_vid_std_ref: float | None = None
        _prev_scene_idx: int | None = None
        _s1_band_ref: tuple[float, float] | None = None
        _origin_ref: torch.Tensor | None = None
        _origin_layout: torch.Tensor | None = None
        _prev_aud_env: torch.Tensor | None = None
        _s1_aud_prev_last: torch.Tensor | None = None
        _s1_prev_vfeat: torch.Tensor | None = None
        _hist_scene_start = 0
        _lg_fired_total = 0
        _lg_rerolls_total = 0
        _s1_audio_freq_ref: list[float] | None = None
        _trend = {
            "vid_std": [], "vid_intra": [], "vid_bnd": [],
            "vid_hf": [], "vid_origin": [],
            "aud_env": [], "aud_rms": [], "aud_bnd": [],
            "aud_dlv": [], "aud_lvl": [],
            "aud_wc": [], "aud_hf": [], "aud_hf_raw": [],
            "aud_peak": [], "aud_step": [], "aud_lag": [], "aud_lagf": [],
            "aud_loop": [], "aud_loopt": [], "vid_prev": [],
            "aud_rcm": [], "aud_rcm_rms": [], "aud_rcm_s": [],
        }

        _vid_pos = 0
        _aud_pos = 0
        for chunk_idx in range(_eff_num_chunks):
            is_first = chunk_idx == 0 and _ctx_mode == "none"
            _ext_v = _ctx_video if chunk_idx == 0 else None
            _ext_a = _ctx_aud if chunk_idx == 0 else None
            _ext_vref = _ctx_vref if chunk_idx == 0 else None
            _cur_new_lf, cur_new_af, _cur_new_px = chunk_plan[chunk_idx]
            _head_lf = 2 if (chunk_idx == 0 and _ctx_mode != "none") else 0
            _cfg_v, _cfg_a, _cfg_g = "kf=-", "aud=-", "guide=off"
            _cfg_lock = ""
            _l_band = _l_std = _l_arms = _l_tg = _l_sp = None
            _cfg_r = ""
            chunk_overlap = 0 if is_first else _eff_overlap
            total_lf = chunk_overlap + (_cur_new_lf - _head_lf) + _tm_lf
            _plan_entry = _cond_plan[chunk_idx]
            _is_transition = isinstance(_plan_entry, tuple)
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
                _s1_prev_vfeat = None
                _hist_scene_start = len(acc_audio)
                clss_state.reset_drift_refs()
            _prev_scene_idx = scene_idx
            _chunk_refs = pos_conds[scene_idx].get("minimax_refs") or []
            _scene_aud_n = sum(1 for _r in _chunk_refs
                               if _r.get("kind") == "audio")
            if _chunk_refs:
                _ni = sum(1 for _r in _chunk_refs if _r.get("kind") == "image")
                _na = _scene_aud_n
                _cfg_r = f" refs={_ni}i+{_na}a"

            keyframes: list[dict] = []
            if is_first and img_guide_latent is not None:
                keyframes.append({"resolved_frame_index": 0, "latent": img_guide_latent})
            has_slb = (not is_first
                       and (_ext_v is not None
                            or clss_state.overlap_latent is not None))
            _slb_ctx = None
            if has_slb:
                _slb_ctx = (_ext_v if _ext_v is not None
                            else clss_state.overlap_latent)
                _cur_std = float(_slb_ctx.float().std(unbiased=False))
                if _slb_std_base is None:
                    _slb_std_base = _cur_std
                elif _cur_std > 1e-6:
                    _lock = _slb_std_base / _cur_std
                    if abs(_lock - 1.0) > 0.005:
                        _m = _slb_ctx.float().mean()
                        _slb_ctx = ((_slb_ctx.float() - _m) * _lock
                                    + _m).to(_slb_ctx.dtype)
                        _cfg_lock = f" lock={_lock:.3f}"
            if not is_first and not _scene_switch:
                keyframes.extend(_build_video_context_keyframes(
                    _slb_ctx if _slb_ctx is not None
                    else clss_state.overlap_latent, chunk_overlap))
            if chunk_idx == 0 and _ctx_pins:
                for _pin in _ctx_pins:
                    keyframes.append({"resolved_frame_index": int(_pin["index"]),
                                      "latent": _pin["latent"]})
            _aud_ref_blk = None
            _aud_src = ""
            if not is_first and _ext_a is not None and _scene_aud_n == 0:
                _aud_ref_blk = _ext_a
                _aud_src = "ext"
                _cfg_g = (f"audref={int(_aud_ref_blk['ref_audio_t'])}af(ext)")
            elif (not is_first and _audio_tail is not None
                    and _scene_aud_n == 0):
                _span_px = float(px_ol)
                _aud_ref_blk, _aud_src = _build_audio_ref_block(
                    _audio_tail, span_px=_span_px,
                    end_px=float(px_ol),
                    audio_vae=audio_vae, device=device)
                _cfg_g = (f"audref={_aud_ref_blk['ref_audio_t'] if _aud_ref_blk else 0}"
                          f"af({_aud_src or 'lat'})")
            elif not is_first and _scene_aud_n > 0:
                _cfg_g = "audref=off(scene ref)"
            guider_chunk = copy.copy(guider)
            _cfg_s = f"acfg={getattr(guider_chunk, '_audio_cfg', '?')}"
            if keyframes or _aud_ref_blk is not None or num_scenes > 1:
                _pos_entry = (_blend_scene_cond(pos_conds[_plan_entry[0]],
                                                pos_conds[_plan_entry[1]],
                                                _plan_entry[2])
                              if _is_transition else pos_conds[scene_idx])
                if keyframes:
                    _pos_entry = {**_pos_entry, "minimax_keyframes": keyframes}
                if _ext_vref is not None or _aud_ref_blk is not None:
                    _new_refs = list(_pos_entry.get("minimax_refs") or [])
                    if _ext_vref is not None:
                        _new_refs = _new_refs + [_ext_vref]
                    if _aud_ref_blk is not None:
                        _new_refs = _new_refs + [_aud_ref_blk]
                    _pos_entry = {**_pos_entry, "minimax_refs": _new_refs}
                if _aud_ref_blk is not None:
                    _pos_entry = _apply_scene_cont(_pos_entry)
                guider_chunk.original_conds = {
                    **guider.original_conds,
                    "positive": [_pos_entry],
                }

            lat_vid = torch.zeros(B, C_v, total_lf, H, W, device=device)
            mask_vid = torch.ones(1, 1, total_lf, 1, 1, device=device)
            _tau_v = 0.0
            if has_slb:
                _tau_v = _tau_c_eff(clss_config.tau_c, _VIDEO_TAU_C_CEILING,
                                    chunk_idx - 1)
                _slb_v = (_slb_ctx if _slb_ctx is not None
                          else clss_state.overlap_latent).to(device)
                _n_v = min(chunk_overlap, _slb_v.shape[2])
                lat_vid[:, :, :_n_v] = _slb_v[:, :, :_n_v]
                mask_vid[:, :, :_n_v] = _tau_v
            _cfg_v = (f"kf={len(keyframes)}" if keyframes else "kf=-")
            if has_slb:
                _cfg_v += f" tau_v={_tau_v:.3f}{_cfg_lock}"
            _win_new_px = _px_of_tokens(_cur_new_lf - _head_lf,
                                        chunk_overlap % 5)
            _head_px = _px_of_tokens(_head_lf,
                                     (_eff_overlap - _head_lf) % 5)
            chunk_af = _af_of_px((0 if is_first else px_ol) + _win_new_px
                                 + _tm_px)
            _Ta_ol_w = chunk_af - cur_new_af - _tm_af
            _b_af = (0.0 if is_first
                     else float(FRAME_RESCALE) * float(px_ol - _head_px))
            _d_af = int(_b_af)
            _head_af = float(_b_af - _d_af)
            _fill_af = (_b_af + float(_cur_new_px) * float(FRAME_RESCALE)
                        - float(chunk_af - _tm_af))
            if _head_lf and _ctx_trim:
                _ctx_trim_rec = {
                    "px": int(_head_px),
                    "af": float(_head_af)
                    + float(_head_px) * float(FRAME_RESCALE)}
            lat_aud = torch.zeros(B_a, C_a, lanes_a, chunk_af, device=device)
            mask_aud = torch.ones(1, 1, lanes_a, chunk_af, device=device)
            if not is_first:
                _cfg_a = "aud=free"
            chunk_latent = {
                "samples": comfy.nested_tensor.NestedTensor((lat_vid, lat_aud)),
                "noise_mask": comfy.nested_tensor.NestedTensor((mask_vid, mask_aud)),
            }
            _rc_tag = (
                f" recompose={audio_recompose_steps}@{audio_recompose_sigma:.2f}"
                f"+m{audio_arc_margin_ms // 1000}s"
                f"/p{audio_recompose_pool}s{audio_recompose_stride}"
                if audio_recompose_steps > 0 else "")

            if is_first:
                print(f"[CLSS] chunk {chunk_idx + 1}/{_eff_num_chunks} "
                      f"scene {scene_idx}: win {_cur_new_px + _tm_px}px/"
                      f"{chunk_af}af tm={_tm_px}px "
                      f"{_cfg_v} {_cfg_a} {_cfg_g} {_cfg_s}{_cfg_r}{_rc_tag}")
            else:
                print(f"[CLSS] chunk {chunk_idx + 1}/{_eff_num_chunks} "
                      f"scene {scene_idx}"
                      f"{' (transition)' if _is_transition else ''}: "
                      f"win {px_ol + _win_new_px + _tm_px}px/{chunk_af}af "
                      f"tm={_tm_px}px "
                      + (f"head={_head_px}px " if _head_lf else "")
                      + f"join_af={_Ta_ol_w} cut={_d_af}af "
                      f"splice={_head_af:.2f}/{_fill_af:.2f}af "
                      f"{_cfg_v} {_cfg_a} {_cfg_g} {_cfg_s}{_cfg_r}{_rc_tag}")
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

            if audio_recompose_steps > 0:
                _rc_pool = max(1, int(audio_recompose_pool))
                _rc_stride = max(1, int(audio_recompose_stride))
                _vr = vid_out
                if _rc_pool > 1:
                    _m = 2 * _rc_pool
                    _vr = _vr[..., :(_vr.shape[-2] // _m) * _m,
                              :(_vr.shape[-1] // _m) * _m]
                    _vr = F.avg_pool3d(_vr, (1, _rc_pool, _rc_pool))
                if _rc_stride > 1:
                    _vr = _vr[:, :, ::_rc_stride]
                _rc_vid = {
                    "kind": "video",
                    "latent_t": int(_vr.shape[2]),
                    "latent_h": int(_vr.shape[3]),
                    "latent_w": int(_vr.shape[4]),
                    "ref_audio_t": 0,
                    "latent": _vr.contiguous(),
                    "audio_latent": None,
                }
                if _rc_stride > 1 and _mc_apply_patch():
                    _rc_vid[_MC_VIDEO_KEY] = int(_rc_stride)
                _rc_w = 2 * max(1, round(W / H))
                _margin_af = max(0, int(round(audio_arc_margin_ms
                                              * AUDIO_LATENT_FPS / 1000.0)))
                _win_af = int(aud_out.shape[-1])
                _rc_af = _win_af + _margin_af
                _margin_lf = 0
                _margin_px = round(_margin_af * 3 / 5)
                while (_margin_lf < 2 * total_lf
                       and _px_of_tokens(_margin_lf, 0) < _margin_px):
                    _margin_lf += 1
                _rc_lf = total_lf + (0 if _margin_af == 0 else _margin_lf)
                _dummy_vid = torch.zeros(1, C_v, _rc_lf, 2, _rc_w,
                                         device=device, dtype=vid_out.dtype)
                _rc_seed = _rc_seed_for(audio_recompose_seed, _noise_seed,
                                        chunk_idx)
                _rc_kf = ([{**kf, "latent": None} for kf in keyframes
                           if kf.get("audio_latent") is not None] or None)
                if audio_refine_guider is not None:
                    _rc_guider = copy.copy(audio_refine_guider)
                    _rc_base_conds = audio_refine_guider.original_conds
                else:
                    _rc_guider = copy.copy(guider)
                    _rc_base_conds = guider.original_conds
                _rc_aud_in = (aud_out if _margin_af == 0
                              else F.pad(aud_out, (0, _margin_af)))
                _rc_mask_a = torch.ones(1, 1, lanes_a, _rc_af,
                                        device=device)
                _rc_pe = (_blend_scene_cond(pos_conds[_plan_entry[0]],
                                            pos_conds[_plan_entry[1]],
                                            _plan_entry[2])
                          if _is_transition else pos_conds[scene_idx])
                _rc_pe = {**_rc_pe}
                _rc_pe.pop("minimax_keyframes", None)
                if _rc_kf:
                    _rc_pe["minimax_keyframes"] = _rc_kf
                _rc_aud_ref_blk = _aud_ref_blk
                _rc_src = _aud_src
                if (int(audio_recompose_ref_ms) > 0 and _audio_tail is not None
                        and not is_first and _scene_aud_n == 0):
                    _rc_aud_ref_blk, _rc_src = _build_audio_ref_block(
                        _audio_tail,
                        span_px=float(audio_recompose_ref_ms) / 1000.0 * fps,
                        end_px=float(px_ol),
                        audio_vae=audio_vae, device=device)
                _rc_pe["minimax_refs"] = (
                    list(_rc_pe.get("minimax_refs") or [])
                    + [_rc_vid]
                    + ([_rc_aud_ref_blk] if _rc_aud_ref_blk is not None else []))
                if _aud_ref_blk is not None:
                    _rc_pe = _apply_scene_cont(_rc_pe)
                _rc_guider.original_conds = {
                    **_rc_base_conds, "positive": [_rc_pe]}
                _sigma_rc = float(audio_recompose_sigma)
                _rcs = torch.linspace(_sigma_rc, 0.0,
                                      audio_recompose_steps + 1)
                _full_tok = total_lf * (H // 2) * (W // 2)
                _rc_tok = (int(_vr.shape[2]) * (_vr.shape[3] // 2)
                           * (_vr.shape[4] // 2)
                           + _rc_lf * (_rc_w // 2))
                print(f"[CLSS] chunk {chunk_idx + 1}: recompose "
                      f"{audio_recompose_steps} steps from sigma "
                      f"{_sigma_rc:.2f} ("
                      f"{'refine of the joint take' if _sigma_rc < 0.999 else 'fresh take'}"
                      f", seed {_rc_seed}) | pack {_rc_tok} video-side "
                      f"tokens vs {_full_tok} full "
                      f"(~{_full_tok / max(_rc_tok, 1):.0f}x fewer) | "
                      f"ref {_vr.shape[2]}f@{_vr.shape[3]}x{_vr.shape[4]} "
                      f"pool x{_rc_pool} stride {_rc_stride} | arc margin "
                      f"{audio_arc_margin_ms}ms ({_margin_af}af) | acfg="
                      f"{float(getattr(_rc_guider, 'audio_cfg', 0.0)):.1f}"
                      + (f" | audref {_rc_aud_ref_blk['ref_audio_t']}af({_rc_src or 'lat'}"
                         + (f", span {int(audio_recompose_ref_ms)}ms"
                            if int(audio_recompose_ref_ms) > 0 else "")
                         + ")"
                         if _rc_aud_ref_blk is not None else ""))
                _aud_pre_rc = aud_out
                _lg_max = max(0, int(loop_guard_rerolls))
                _lg_thr_wc = float(loop_guard_wc)
                _lg_thr_loop = float(loop_guard_loop)
                _lg_lo = int(_d_af)
                _lg_hi = max(_lg_lo, _win_af - _tm_af)
                _lg_on = (_lg_max > 0 and _scene_aud_n == 0
                          and _lg_hi - _lg_lo >= 8)
                _lg_hist = None
                if _lg_on and acc_audio and _hist_scene_start < len(acc_audio):
                    _lg_hist = torch.cat(acc_audio[_hist_scene_start:],
                                         dim=-1)
                _lg_refs_retry = None
                _lg_ref_used = False
                if (_lg_on and _rc_aud_ref_blk is not None
                        and int(loop_guard_retry_ref_ms) > 0):
                    _lg_ref_blk, _lg_ref_src = _build_audio_ref_block(
                        _audio_tail,
                        span_px=float(loop_guard_retry_ref_ms) / 1000.0 * fps,
                        end_px=float(px_ol),
                        audio_vae=audio_vae, device=device)
                    if _lg_ref_blk is not None:
                        _lg_refs_retry = (
                            [b for b in _rc_pe["minimax_refs"]
                             if b is not _rc_aud_ref_blk] + [_lg_ref_blk])

                def _rc_take(_seed):
                    _g = torch.Generator(device="cpu").manual_seed(_seed)
                    _nz = _FreshAVNoise(
                        _dummy_vid.clone(),
                        torch.randn(*aud_out.shape[:-1], _rc_af,
                                    generator=_g,
                                    dtype=aud_out.dtype).to(device),
                        seed=_seed)
                    _lat = {
                        "samples": comfy.nested_tensor.NestedTensor(
                            (_dummy_vid.clone(), _rc_aud_in.clone())),
                        "noise_mask": comfy.nested_tensor.NestedTensor((
                            torch.zeros(1, 1, _rc_lf, 1, 1, device=device),
                            _rc_mask_a)),
                    }
                    _, _ro = SamplerCustomAdvanced().sample(
                        noise=_nz, guider=_rc_guider, sampler=sampler,
                        sigmas=_rcs, latent_image=_lat)
                    return _ro["samples"].unbind()[1]

                _t_rc = time.time()
                _lg_tries = []
                _lg_shot = 0
                while True:
                    if (_lg_refs_retry is not None and _lg_shot >= 1
                            and not _lg_ref_used):
                        _rc_guider.original_conds = {
                            **_rc_base_conds,
                            "positive": [{**_rc_pe,
                                          "minimax_refs": _lg_refs_retry}]}
                        _lg_ref_used = True
                    _take = _rc_take(_rc_seed_for(audio_recompose_seed,
                                                  _noise_seed, chunk_idx,
                                                  _lg_shot))
                    _wc = _lp = float("nan")
                    if _lg_on:
                        _wc, _lp = _take_loop_metrics(
                            _lg_hist, _take[..., _lg_lo:_lg_hi].cpu())
                    _lg_tries.append((_lg_shot, _wc, _lp, _take))
                    if (not _lg_on
                            or not _loop_guard_bad(_wc, _lp, _lg_thr_wc,
                                                   _lg_thr_loop)
                            or _lg_shot >= _lg_max):
                        break
                    _lg_shot += 1
                _dt_rc = time.time() - _t_rc
                _lg_i = (_loop_guard_pick(_lg_tries)
                         if len(_lg_tries) > 1 else 0)
                if len(_lg_tries) > 1:
                    _lg_fired_total += 1
                    _lg_rerolls_total += len(_lg_tries) - 1
                if _lg_on:
                    _fmtl = lambda _v: ("nan" if _v != _v else f"{_v:.3f}")
                    _lg_msg = (f"[CLSS] chunk {chunk_idx + 1}: loop guard: "
                               + " | ".join(
                                   f"try{a}: wc={_fmtl(_w)} loop={_fmtl(_l)}"
                                   for a, _w, _l, _t in _lg_tries))
                    if len(_lg_tries) > 1:
                        _lg_msg += (
                            f" -> kept try{_lg_tries[_lg_i][0]} (seed "
                            f"{_rc_seed_for(audio_recompose_seed, _noise_seed, chunk_idx, _lg_tries[_lg_i][0])}"
                            f", {len(_lg_tries) - 1}/{_lg_max} re-rolls)")
                    if _lg_ref_used:
                        _lg_msg += (f" [retry ref "
                                    f"{int(loop_guard_retry_ref_ms)}ms]")
                    print(_lg_msg)
                aud_out = _lg_tries[_lg_i][3][..., :_win_af]
                if audio_refine_guider is not None:
                    comfy.model_management.unload_model_and_clones(
                        audio_refine_guider.model_patcher)
                    comfy.model_management.soft_empty_cache()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                with torch.no_grad():
                    _rcm_cos = _aud_cos(_aud_pre_rc, aud_out)
                    _rcm_rms = float(
                        aud_out.float().pow(2).mean().sqrt()
                        / _aud_pre_rc.float().pow(2).mean().sqrt()
                        .clamp(min=1e-8))
                _trend["aud_rcm"].append(_rcm_cos)
                _trend["aud_rcm_rms"].append(_rcm_rms)
                _trend["aud_rcm_s"].append(_dt_rc)
                print(f"[CLSS] chunk {chunk_idx + 1}: recompose took "
                      f"{_dt_rc:.1f}s ({_dt_rc / audio_recompose_steps:.2f}s/"
                      f"step) | audio vs turbo take cos={_rcm_cos:.3f} rms "
                      f"x{_rcm_rms:.2f}")

            new_vid = vid_out[:, :, chunk_overlap - _head_lf:
                              chunk_overlap - _head_lf + _cur_new_lf]
            corrected = clss_state.post_process(new_vid)

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
                    _l_band = (_g_lo, _g_hi)
                    if abs(_g_lo - 1.0) > 0.005 or abs(_g_hi - 1.0) > 0.005:
                        corrected = (_da_low * _g_lo + _da_high * _g_hi).reshape(
                            _da_b, _da_t, _da_c, _da_h, _da_w
                        ).permute(0, 2, 1, 3, 4).contiguous().to(corrected.dtype)
                        _e_low_p, _e_high_p = _e_low * _g_lo ** 2, _e_high * _g_hi ** 2
                        _hf_share = _e_high_p / max(_e_low_p + _e_high_p, 1e-12)
            _trend["vid_hf"].append(_hf_share)

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

            if _s1_vid_std_ref is None:
                _s1_vid_std_ref = corrected.float().std().item()
            else:
                _cur_vstd = corrected.float().std().item()
                _ratio = _s1_vid_std_ref / max(_cur_vstd, 1e-6)
                if _ratio < 0.96 or _ratio > 1.04:
                    _g_v = 1.0 + 0.5 * (_ratio - 1.0)
                    _l_std = _g_v
                    _m = corrected.float().mean()
                    corrected = ((corrected.float() - _m) * _g_v + _m).to(corrected.dtype)
            _trend["vid_std"].append(corrected.float().std().item())

            clss_state.update_buffer(corrected)
            acc_video.append(corrected.cpu())
            if _up_active:
                _t_up0 = time.time()
                _ov_hr = chunk_overlap - _head_lf
                _up_in = torch.cat([vid_out[:, :, :_ov_hr], corrected],
                                   dim=2)
                _up_hr = _ups.upscale(_up_in, _up_scale, device=_up_dev)
                _dt_up = time.time() - _t_up0
                _up_secs += _dt_up
                print(f"[CLSS]   upscale {int(_up_in.shape[2])} tok -> "
                      f"{_up_hr.shape[-1] * 16}x{_up_hr.shape[-2] * 16} px in "
                      f"{_dt_up:.1f}s")
                _blend_upscaled_overlap(acc_video_hr, _up_hr, _ov_hr)
                acc_video_hr.append(_up_hr[:, :, _ov_hr:].cpu())
            _trend["vid_intra"].append(_frame_cos(corrected[:, :, 0], corrected[:, :, -1]))
            if _s1_prev_last is not None:
                _trend["vid_bnd"].append(_frame_cos(_s1_prev_last.to(device), corrected[:, :, 0]))
            _cur_vfeat = F.normalize(
                corrected.float().mean(dim=(3, 4)).mean(dim=2), dim=1)
            if _s1_prev_vfeat is not None:
                _trend["vid_prev"].append(float(F.cosine_similarity(
                    _cur_vfeat, _s1_prev_vfeat.to(device), dim=1).mean()))
            _s1_prev_vfeat = _cur_vfeat.detach().cpu()
            _s1_prev_last = corrected[:, :, -1].cpu()

            aud_drop = _d_af
            _splice_bad = False
            if aud_drop > 0 and aud_out.shape[-1] < aud_drop:
                aud_drop = 0
                _splice_bad = True
            if not is_first and audio_head_discard_ms > 0:
                _extra = min(int(round(audio_head_discard_ms
                                       * AUDIO_LATENT_FPS / 1000.0)),
                             max(0, aud_out.shape[-1] - aud_drop - 1))
                if _extra > 0:
                    aud_drop += _extra
                    _splice_bad = True
            _keep_end = max(int(aud_drop), int(aud_out.shape[-1]) - _tm_af)
            new_aud = aud_out[..., aud_drop:_keep_end]
            _splice["delivered_af"].append(_keep_end - int(aud_drop))
            _splice["head_af"].append(_head_af)
            _splice["fill_af"].append(_fill_af)
            _splice["bad"].append(bool(_splice_bad))
            _env = new_aud.detach().float().pow(2).mean(dim=(0, 1, 2)).cpu()
            if _prev_aud_env is not None and len(_prev_aud_env) > 8:
                _L = min(len(_env), len(_prev_aud_env))
                _ea = _env[:_L] - _env[:_L].mean()
                _eb = _prev_aud_env[:_L] - _prev_aud_env[:_L].mean()
                _trend["aud_env"].append(float((_ea * _eb).sum() /
                                               (_ea.norm() * _eb.norm() + 1e-8)))
            _prev_aud_env = _env
            if is_first:
                _n_fade = min(8, new_aud.shape[-1])
                if _n_fade >= 2:
                    _ramp = torch.linspace(0.125, 1.0, _n_fade, device=device)
                    new_aud = new_aud.clone()
                    new_aud[..., :_n_fade] = new_aud[..., :_n_fade] * _ramp
            _fa = new_aud.float()
            _sig = _fa.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
            _over = (_fa.abs() - _sig * 3.5).clamp(min=0)
            new_aud = (_fa - torch.sign(_fa)
                       * (_over - 1.5 * _sig
                          * torch.tanh(_over / (1.5 * _sig)))).to(aud_out.dtype)
            _fa2 = new_aud.float()
            _trend["aud_peak"].append(float(
                _fa2.abs().max() / _fa2.std().clamp(min=1e-8)))
            _aud_sims = _aud_within_chunk_sims(new_aud)
            if _aud_sims:
                _trend["aud_wc"].append(_aud_sims[-1])
            if _s1_aud_prev_last is not None:
                _trend["aud_bnd"].append(_aud_cos(_s1_aud_prev_last.to(device), new_aud[..., :1]))
            _cur_arms = new_aud.float().pow(2).mean().sqrt().item()
            if _s1_aud_rms_ref is None:
                _s1_aud_rms_ref = _cur_arms
                if _s1_aud_level_ref is None:
                    _s1_aud_level_ref = _cur_arms
            else:
                _ratio_a = _s1_aud_rms_ref / max(_cur_arms, 1e-6)
                if _ratio_a < 0.94 or _ratio_a > 1.06:
                    _g_a = 1.0 + 0.5 * (_ratio_a - 1.0)
                    _l_arms = _g_a
                    new_aud = (new_aud.float() * _g_a).to(new_aud.dtype)
                    _cur_arms *= _g_a
            _trend["aud_rms"].append(_cur_arms)
            _tail_n = min(int(round(_TAIL_ANCHOR_S * AUDIO_LATENT_FPS)),
                          new_aud.shape[-1])
            if _tail_n > 0 and _s1_aud_level_ref is not None:
                _tail = new_aud[..., -_tail_n:]
                _t_rms = float(_tail.float().pow(2).mean().sqrt().item())
                if _t_rms > 1e-9:
                    _g_t = float(_s1_aud_level_ref) / _t_rms
                    _g_t = max(0.5, min(3.0, _g_t))
                    if abs(_g_t - 1.0) > 0.02:
                        _l_tg = _g_t
                        _r = torch.linspace(1.0, _g_t, _tail_n,
                                            device=new_aud.device,
                                            dtype=torch.float32)
                        new_aud = new_aud.float().clone()
                        new_aud[..., -_tail_n:] = (
                            _tail.float() * _r.view(1, 1, 1, -1))
                        new_aud = new_aud.to(aud_out.dtype)
            with torch.no_grad():
                _freq_e = new_aud.float().abs().mean(dim=(0, 2, 3)).tolist()
            if _s1_audio_freq_ref is None:
                _s1_audio_freq_ref = _freq_e
            else:
                _freq_raw = [e / r if r > 1e-6 else 0.0
                             for e, r in zip(_freq_e, _s1_audio_freq_ref)]
                if len(_freq_raw) >= 4:
                    _trend["aud_hf_raw"].append(sum(_freq_raw[-4:]) / 4.0)
                _n_ch = len(_freq_e)
                _lo_e = sum(_freq_e[:-4]) / max(1, _n_ch - 4)
                _hi_e = sum(_freq_e[-4:]) / 4.0
                _lo_r = sum(_s1_audio_freq_ref[:-4]) / max(1, _n_ch - 4)
                _hi_r = sum(_s1_audio_freq_ref[-4:]) / 4.0
                _g_lo = min(1.12, max(0.90, _lo_r / max(_lo_e, 1e-6)))
                _g_hi = min(1.20, max(0.90, _hi_r / max(_hi_e, 1e-6)))
                _l_sp = (_g_lo, _g_hi)
                if abs(_g_lo - 1.0) > 0.005 or abs(_g_hi - 1.0) > 0.005:
                    _gt = torch.ones(1, _n_ch, 1, 1, dtype=new_aud.dtype,
                                     device=new_aud.device)
                    _gt[0, :-4] = _g_lo
                    _gt[0, -4:] = _g_hi
                    new_aud = (new_aud * _gt).to(new_aud.dtype)
                    _freq_e = [_e * (_g_hi if _i >= _n_ch - 4 else _g_lo)
                               for _i, _e in enumerate(_freq_e)]
                _freq_ratio = [e / r if r > 1e-6 else 0.0
                               for e, r in zip(_freq_e, _s1_audio_freq_ref)]
                if len(_freq_ratio) >= 4:
                    _trend["aud_hf"].append(sum(_freq_ratio[-4:]) / 4.0)
            _audio_tail = (new_aud.cpu() if _audio_tail is None
                           else torch.cat([_audio_tail, new_aud.cpu()], dim=-1))
            _tail_keep = max(1, Ta_ol, int(round(
                max(int(audio_recompose_ref_ms),
                    int(loop_guard_retry_ref_ms))
                / 1000.0 * AUDIO_LATENT_FPS)))
            _tail_keep += _REF_DECODE_MARGIN_AF
            if _audio_tail.shape[-1] > _tail_keep:
                _audio_tail = _audio_tail[..., -_tail_keep:]
            if not is_first and acc_audio and new_aud.shape[-1] > 0:
                _trend["aud_dlv"].append(_aud_cos(
                    acc_audio[-1][..., -1:].to(device), new_aud[..., :1]))
                _Lw = min(round(AUDIO_LATENT_FPS), acc_audio[-1].shape[-1],
                          new_aud.shape[-1])
                if _Lw > 0:
                    _lv_prev = acc_audio[-1][..., -_Lw:].float().pow(2).mean().sqrt()
                    _lv_new = new_aud[..., :_Lw].float().pow(2).mean().sqrt()
                    _trend["aud_lvl"].append(
                        20.0 * math.log10(max(float(_lv_new), 1e-12)
                                          / max(float(_lv_prev), 1e-8)))
                _trend["aud_step"].append(_aud_seam_step(
                    acc_audio[-1], new_aud.cpu()))
                _lag_c, _lag_f = _aud_best_lag(acc_audio[-1], new_aud.cpu())
                _trend["aud_lag"].append(_lag_c)
                _trend["aud_lagf"].append(float(_lag_f))
                _hist = (torch.cat(acc_audio[_hist_scene_start:], dim=-1)
                         if _hist_scene_start < len(acc_audio) else None)
                if _hist is not None:
                    _loop_c, _loop_t = _aud_loop_ncc(_hist, new_aud.cpu())
                    _trend["aud_loop"].append(_loop_c)
                    _trend["aud_loopt"].append(
                        _loop_t + sum(a.shape[-1]
                                      for a in acc_audio[:_hist_scene_start])
                        / AUDIO_LATENT_FPS)
            acc_audio.append(new_aud.cpu())
            audio_chunk_ends.append(sum(a.shape[-1] for a in acc_audio))
            _s1_aud_prev_last = new_aud[..., -1:].cpu()

            _fg = (lambda x: "-" if x is None else
                   (f"{x:.3f}" if isinstance(x, float) else f"{x[0]:.3f}/{x[1]:.3f}"))
            print(f"[CLSS]   corr@{chunk_idx + 1}: band={_fg(_l_band)} "
                  f"std={_fg(_l_std)} aud_rms={_fg(_l_arms)} "
                  f"aud_tail={_fg(_l_tg)} aud_spec={_fg(_l_sp)}")

            _vid_pos += _cur_new_lf
            _aud_pos += cur_new_af

        if _up_active:
            _n_up = len(acc_video_hr)
            full_vid = torch.cat(acc_video_hr, dim=2)
            acc_video.clear()
        else:
            full_vid = torch.cat(acc_video, dim=2)
        try:
            _fv = full_vid.float().permute(2, 0, 1, 3, 4).reshape(full_vid.shape[2], -1)
            if _fv.shape[0] >= 8:
                _cos = torch.cosine_similarity(_fv[1:], _fv[:-1], dim=1).cpu()
                _diss = (1.0 - _cos)
                _k = 15
                _med = torch.stack([
                    _diss[max(0, i - _k):i + _k + 1].median()
                    for i in range(_diss.numel())
                ]).clamp(min=1e-4)
                _ratio = _diss / _med
                _flag = (_ratio > 4.0) & (_diss > 0.05)
                _idx = _flag.nonzero().flatten().tolist()
                if _idx:
                    _worst = sorted(_idx, key=lambda i: -float(_ratio[i]))[:3]
                    _desc = ", ".join(
                        f"#{i + 1} ({(i + 1) / fps:.2f}s, {float(_ratio[i]):.1f}x, "
                        f"diss {float(_diss[i]):.3f})" for i in _worst)
                    print(f"[CLSS] WARNING: {len(_idx)} hard cut(s) in the "
                          f"delivered video — worst: {_desc}; try another "
                          f"seed or more steps.")
                else:
                    print(f"[CLSS] video continuity check: no hard cuts "
                          f"(max step {float(_ratio.max()):.2f}x local median)")
        except Exception as _exc:
            print(f"[CLSS] WARNING: cut check skipped ({_exc!r})")
        try:
            _vr = full_vid.float().mean(dim=1, keepdim=True)
            _vr = _vr.permute(0, 2, 1, 3, 4).reshape(
                -1, 1, full_vid.shape[-2], full_vid.shape[-1])
            _fp = F.avg_pool2d(_vr, 4)[:, 0].reshape(_vr.shape[0], -1)
            _fp = _fp - _fp.mean(dim=1, keepdim=True)
            _fp = _fp / _fp.norm(dim=1, keepdim=True).clamp(min=1e-6)
            _lag = max(1, int(round(12.4 * 24 / (17.0 / 5))))
            if _fp.shape[0] > 2 * _lag + 2:
                _c = (_fp[_lag:] * _fp[:-_lag]).sum(dim=1)
                _cm, _cx = float(_c.mean()), float(_c.max())
                print(f"[CLSS] vid_copy (frame NCC one chunk back): mean "
                      f"{_cm:.3f} max {_cx:.3f}"
                      + (" (re-presenting earlier content)"
                         if _cm > 0.70 else ""))
        except Exception as _exc:
            print(f"[CLSS] WARNING: vid_copy check skipped ({_exc!r})")
        for _k, _v in _trend.items():
            if _v:
                print(f"[CLSS] trend {_k}: " + " ".join(f"{_x:.3f}" for _x in _v))
        _vo = _trend.get("vid_origin") or []
        if len(_vo) >= 8:
            _h = len(_vo) // 2
            _a = sum(_vo[:_h]) / max(1, _h)
            _b = sum(_vo[_h:]) / max(1, len(_vo) - _h)
            if _b < _a - 0.06:
                print(f"[CLSS] WARNING: video content drift — vid_origin "
                      f"{_a:.3f} (first half) -> {_b:.3f} (second half); "
                      f"raise overlap or check the scene prompt.")
        _al = _trend.get("aud_loop") or []
        _aw = _trend.get("aud_wc") or []
        if (_al and max(_al) > 0.70) or (_aw and max(_aw[-6:]) > 0.99):
            print(f"[CLSS] WARNING: audio repetition suspected — aud_loop max "
                  f"{max(_al) if _al else 0.0:.2f}, aud_wc tail max "
                  f"{max(_aw[-6:]) if _aw else 0.0:.2f}")
        if _lg_fired_total:
            print(f"[CLSS] loop guard: re-rolled {_lg_fired_total} chunk(s), "
                  f"{_lg_rerolls_total} extra take(s); kept the lowest "
                  f"aud_wc per chunk")
        if _up_active:
            print(f"[CLSS] upscaler: {_n_up} chunk(s) upscaled in {_up_secs:.1f}s | "
                  f"output video {tuple(full_vid.shape)} "
                  f"({full_vid.shape[-1] * 16}x{full_vid.shape[-2] * 16} px)")
        full_aud = torch.cat(acc_audio, dim=-1)
        full_aud = _post_process_audio_latent(full_aud, audio_chunk_ends,
                                              energy_beta=0.0, label=" S1")
        output_samples = comfy.nested_tensor.NestedTensor((full_vid, full_aud))
        if _ups is not None:
            _ups.offload()
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _cum_px = 0
        for _cf in chunk_plan[:-1]:
            _cum_px += int(_cf[2])
            _splice["join_af_exact"].append(_cum_px * float(FRAME_RESCALE))
        _splice["total_af_exact"] = (
            sum(int(_cf[2]) for _cf in chunk_plan) * float(FRAME_RESCALE))
        return ({"samples": output_samples,
                 "clss_audio_chunk_ends": list(audio_chunk_ends),
                 "clss_audio_splice": _splice,
                 "clss_ctx_trim": _ctx_trim_rec},)


def _splice_delivered_audio(audio: torch.Tensor, splice: dict,
                            spf: float, trim_af: float = 0.0
                            ) -> torch.Tensor:
    def _trim_front(a: torch.Tensor) -> torch.Tensor:
        if trim_af <= 0.0:
            return a
        d = int(round(float(trim_af) * spf))
        if d <= 0 or d >= int(a.shape[-1]):
            return a
        a = a[..., d:].contiguous()
        want_t = int(round(max(
            0.0, float(splice.get("total_af_exact") or 0.0)
            - float(trim_af)) * spf))
        if want_t > 0:
            if a.shape[-1] < want_t:
                miss = want_t - int(a.shape[-1])
                if miss <= max(4, int(round(spf))):
                    a = F.pad(a, (0, miss))
            elif a.shape[-1] > want_t:
                a = a[..., :want_t]
        print(f"[CLSS] decode_save: context head trim: {d} smp "
              f"({trim_af:.2f} af) dropped - the take starts at the new span")
        return a

    lens = [max(0, int(round(float(a) * spf)))
            for a in splice.get("delivered_af") or []]
    if len(lens) < 2:
        return _trim_front(audio)
    heads = [max(0, int(round(float(a) * spf)))
             for a in splice.get("head_af") or []]
    fills = [int(round(float(a) * spf)) for a in splice.get("fill_af") or []]
    bad = list(splice.get("bad") or [])
    total = int(audio.shape[-1])
    starts: list[int] = []
    _acc = 0
    for _L in lens:
        starts.append(_acc)
        _acc += _L
    if _acc > total + 8:
        print(f"[CLSS] decode_save: seam splice skipped \u2014 decoded audio "
              f"{total} smp < expected {_acc}")
        return audio
    parts = [audio[..., :starts[1]]]
    repaired = 0
    blended = 0
    for i in range(1, len(lens)):
        s = starts[i]
        e = starts[i + 1] if i + 1 < len(lens) else total
        seg = audio[..., s:e]
        h = heads[i] if i < len(heads) else 0
        f = fills[i - 1] if i - 1 < len(fills) else 0
        ok = (h > 0 and 0 <= f <= h and int(seg.shape[-1]) > h
              and i < len(bad) and not bad[i] and not bad[i - 1])
        if ok:
            x = h - f
            if x > 0 and parts[-1].shape[-1] >= x and int(seg.shape[-1]) > x:
                _w = torch.linspace(0.0, 1.0, x, device=audio.device,
                                    dtype=torch.float32)
                _a = parts[-1][..., -x:].float()
                _b = seg[..., :x].float()
                parts[-1] = parts[-1][..., :-x]
                parts.append((_a * (1.0 - _w) + _b * _w).to(audio.dtype))
                blended += x
            parts.append(seg[..., x:])
            repaired += 1
        else:
            parts.append(seg)
    if repaired == 0:
        return _trim_front(audio)
    audio = torch.cat(parts, dim=-1)
    want = int(round(float(splice.get("total_af_exact") or 0.0) * spf))
    if want > 0:
        if audio.shape[-1] < want:
            miss = want - audio.shape[-1]
            if miss <= max(4, int(round(spf))):
                audio = F.pad(audio, (0, miss))
            else:
                print(f"[CLSS] decode_save: seam splice tail short by "
                      f"{miss} smp (> 1 audio step) \u2014 left as is")
        elif audio.shape[-1] > want:
            audio = audio[..., :want]
    print(f"[CLSS] decode_save: seam splice: {repaired}/{len(lens) - 1} "
          f"junction(s) spliced, {blended} smp "
          f"({blended / (spf * 40.0) * 1000.0:.1f} ms) crossfaded")
    return _trim_front(audio)


class CLSSH3VideoDecodeSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae":       ("VAE",    {"tooltip": 'Video VAE. Decoding is temporally sliced: each slice is decoded standalone and its PNG frames are written to disk before the next slice decodes, so the whole decoded video never sits in RAM.'}),
                "audio_vae": ("VAE",    {"tooltip": 'Audio VAE (MiniMaxH3AudioVAE) for the audio stream; decoded in one shot via the same logic as the stock VAEDecodeAudio node.'}),
                "latent":    ("LATENT", {"tooltip": 'Full AV latent from CLSSH3StreamingSampler (video + audio).'}),
                "filename_prefix": ("STRING", {"default": "clss_h3/CLSSH3_frame_",
                                               "tooltip": 'Output filename prefix; frames are written as <prefix>_NNNNN.png under the ComfyUI output directory.',
                                               }),
                "frames_per_slice": ("INT", {"default": 27, "min": 5, "max": 502,
                                             "tooltip": "Video latent tokens decoded per VAE call, snapped to a multiple of 5 (the grid's group size; 27 -> 25 tokens = 85 px frames = 3.5 s). Slice boundaries stay on the absolute 5-token grid so the slices tile the timeline exactly.",
                                             }),
                "context_frames": ("INT", {"default": 0, "min": 0, "max": 16,
                                           "tooltip": "Extra latent tokens of temporal context prepended to each non-first slice and dropped after decode. Keep 0 on H3: the causal VAE's first tokens cover fewer frames than the rest, so prepended context misaligns the slice phase.",}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0, "step": 1.0,
                                  "tooltip": 'Informational only (duration logging). H3 is 24 fps native; frames are written in timeline order regardless.'}),
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
        _trim = (latent.get("clss_ctx_trim")
                 if isinstance(latent, dict) else None)
        _trim_px = int(_trim.get("px", 0)) if isinstance(_trim, dict) else 0
        _trim_af = (float(_trim.get("af", 0.0))
                    if isinstance(_trim, dict) else 0.0)

        fsm = getattr(vae, "first_stage_model", None)

        def _px_for_tokens(n: int) -> int:
            if fsm is not None and hasattr(fsm, "decode_output_shape"):
                return fsm.decode_output_shape((1, vid.shape[1], n, vid.shape[3], vid.shape[4]))[2]
            return _px_of_tokens(n, 0)

        output_dir = folder_paths.get_output_directory()
        full_folder, filename, _, _, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir)
        os.makedirs(full_folder, exist_ok=True)

        step = 5 * max(1, round(frames_per_slice / 5))
        ctx = max(0, int(context_frames))
        frame_idx = 0
        pos = 0
        while pos < T:
            end = min(pos + step, T)
            n_tok = end - pos
            c = 0 if pos == 0 else min(ctx, pos)
            px = vae.decode(vid[:, :, pos - c:end])
            drop = _px_for_tokens(c) if c else 0
            if pos == 0 and _trim_px > 0:
                drop = min(drop + _trim_px, int(px.shape[1]))
                print(f"[CLSS] decode_save: context head trim: first "
                      f"{_trim_px} frame(s) dropped - the take starts at "
                      f"the new span")
            arr = (px[0, drop:].float().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            for f in range(arr.shape[0]):
                Image.fromarray(arr[f]).save(
                    os.path.join(full_folder, f"{filename}_{frame_idx:05d}.png"),
                    compress_level=4)
                frame_idx += 1
            del px, arr
            pos = end

        audio = audio_vae.decode(aud).movedim(-1, 1)
        std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        audio = audio / std
        vae_sr = getattr(audio_vae, "audio_sample_rate_output",
                         getattr(audio_vae, "audio_sample_rate", 32000))
        _spf = float(vae_sr) / float(AUDIO_LATENT_FPS)
        _splice = (latent.get("clss_audio_splice")
                   if isinstance(latent, dict) else None)
        _ends = (latent.get("clss_audio_chunk_ends")
                 if isinstance(latent, dict) else None)
        if _trim_af > 0.0 and _ends:
            _ends = [float(_e) - _trim_af for _e in _ends]
        if _splice or _trim_af > 0.0:
            audio = _splice_delivered_audio(audio, _splice or {}, _spf,
                                            _trim_af)
        if _ends and len(_ends) >= 2:
            _b0 = [0] + [int(round(af / 40.0 * vae_sr)) for af in _ends[:-1]]
            _b0.append(int(audio.shape[-1]))
            _lvs = []
            for _i in range(len(_b0) - 1):
                _a, _b = _b0[_i], _b0[_i + 1]
                if _b > _a:
                    _r = float(audio[..., _a:_b].float().pow(2).mean().sqrt())
                    _lvs.append(20.0 * math.log10(max(_r, 1e-12)))
            if _lvs:
                print("[CLSS] decode_save: chunk RMS "
                      + " / ".join(f"{v:.1f}" for v in _lvs) + " dBFS")
        _pk = float(audio.abs().max())
        if _pk > 0.99:
            _gain = 0.99 / _pk
            audio = audio * _gain
            print(f"[CLSS] decode_save: headroom guard {20.0 * math.log10(_gain):+.1f} dB "
                  f"(pre-save peak {_pk:.3f} would clip)")
        vae_sr = getattr(audio_vae, "audio_sample_rate_output",
                         getattr(audio_vae, "audio_sample_rate", 32000))
        print(f"[CLSS] decode_save: {frame_idx} frames ({frame_idx / fps:.1f} s @ "
              f"{fps:g} fps) -> {full_folder}/{filename}_*.png; "
              f"audio {audio.shape[-1] / vae_sr:.1f} s @ {vae_sr} Hz")
        return ({"waveform": audio, "sample_rate": vae_sr},)


_FRAME_FILE_EXTS = (".png",)
_AUDIO_FILE_EXTS = (".flac", ".wav", ".mp3", ".ogg", ".opus", ".m4a")
_AUDIO_MIN_SAMPLES = 800


def _output_folder(prefix: str) -> tuple[str, str]:
    import folder_paths
    norm = os.path.normpath(prefix)
    return (os.path.join(folder_paths.get_output_directory(),
                         os.path.dirname(norm)),
            os.path.basename(norm))


def _indexed_files(folder: str, name: str,
                   exts: tuple) -> list[tuple[int, str]]:
    try:
        entries = os.listdir(folder)
    except OSError:
        return []
    out: list[tuple[int, str]] = []
    for fn in entries:
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in exts or not stem.startswith(name):
            continue
        tail = stem[len(name):]
        if not tail.startswith("_"):
            continue
        tail = tail[1:]
        if tail.isdigit():
            out.append((int(tail), os.path.join(folder, fn)))
    out.sort(key=lambda pair: pair[0])
    return out


def _load_frame_range(images_prefix: str, start: int, count: int,
                      h_lat: int, w_lat: int, tail_extra: int = 0
                      ) -> tuple[torch.Tensor, int]:
    from PIL import Image
    import numpy as np
    folder, name = _output_folder(images_prefix)
    found = dict(_indexed_files(folder, name, _FRAME_FILE_EXTS))
    if not found:
        raise ValueError(f"no frames found for prefix {images_prefix!r} in "
                         f"{folder} — point filename_prefix at the saved "
                         f"run's decode prefix")
    if start < 0:
        start = max(0, max(found) + 1 + start)
    frames = []
    idx = start
    while idx < start + count and idx in found:
        with Image.open(found[idx]) as im:
            frames.append(torch.from_numpy(
                np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0))
        idx += 1
    if len(frames) < count:
        if len(frames) < count - tail_extra:
            raise ValueError(
                f"{images_prefix!r} holds frames up to {max(found)} — need "
                f"{start}..{start + count - 1} ({count} frame(s)); a re-edit "
                f"needs the chunk's span plus the overlap frames before it")
        print(f"[CLSS] context frames: only {len(frames)}/{count} frame(s) "
              f"available from {start} — the tail is padded with the last "
              f"frame")
        frames = frames + [frames[-1]] * (count - len(frames))
    out = torch.stack(frames, dim=0)
    if out.shape[1] != h_lat * 16 or out.shape[2] != w_lat * 16:
        print(f"[CLSS] context frames: {out.shape[2]}x{out.shape[1]} resized "
              f"to the template canvas {w_lat * 16}x{h_lat * 16}")
        out = comfy.utils.common_upscale(
            out.movedim(-1, 1), w_lat * 16, h_lat * 16, "lanczos",
            "disabled").movedim(1, -1)
    return out, int(start)


def _load_audio_prefix(audio_prefix: str):
    folder, name = _output_folder(audio_prefix)
    for ext in _AUDIO_FILE_EXTS:
        found = _indexed_files(folder, name, (ext,))
        if found:
            path = found[-1][1]
            from comfy_extras.nodes_audio import load as _load_audio
            waveform, rate = _load_audio(path)
            print(f"[CLSS] context audio: {os.path.basename(path)} "
                  f"({waveform.shape[-1] / rate:.1f}s @ {rate} Hz)")
            return {"waveform": waveform.unsqueeze(0), "sample_rate": rate}
    print(f"[CLSS] WARNING: no audio file for prefix {audio_prefix!r} in "
          f"{folder} — no audio context is attached")
    return None


def _template_geometry(latent) -> tuple[int, int, int]:
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if not (getattr(samples, "is_nested", False)
            and len(samples.unbind()) == 2):
        raise ValueError("this node needs the MiniMax H3 AV latent template "
                         "from EmptyMiniMaxH3LatentAV")
    vid = samples.unbind()[0]
    if (vid.ndim != 5 or int(vid.shape[1]) != 24 or int(vid.shape[2]) < 7
            or int(vid.shape[2]) % 5 != 2):
        raise ValueError(f"template video latent {tuple(vid.shape)} is not on "
                         f"the H3 5k+2 grid")
    return int(vid.shape[2]), int(vid.shape[3]), int(vid.shape[4])


def _encode_context_video(vae, frames: torch.Tensor, h_lat: int, w_lat: int,
                          want_tokens: int, label: str) -> torch.Tensor:
    z = vae.encode(frames)
    if (z.ndim != 5 or int(z.shape[2]) != int(want_tokens)
            or int(z.shape[3]) != int(h_lat) or int(z.shape[4]) != int(w_lat)):
        raise ValueError(
            f"{label}: {frames.shape[0]} frame(s) encoded to {tuple(z.shape)} "
            f"— expected [1, 24, {want_tokens}, {h_lat}, {w_lat}]. The saved "
            f"frames must come from a run with the same template resolution "
            f"and the same overlap in its clss_config.")
    return z.cpu()


def _encode_pin_latent(vae, frame: torch.Tensor) -> torch.Tensor:
    z = vae.encode(frame.unsqueeze(0))
    if z.ndim != 5 or int(z.shape[2]) != 1:
        raise ValueError(f"a pinned frame encoded to {tuple(z.shape)} — "
                         f"expected [1, 24, 1, h, w]")
    return z.cpu()


def _edit_pins(vae, frames: torch.Tensor, need_ctx: int, new_px: int,
               mode: str, head_px: int = 0) -> list[dict]:
    out: list[dict] = []
    if mode in ("first+last", "first"):
        out.append({"index": need_ctx - head_px,
                    "latent": _encode_pin_latent(vae,
                                                 frames[need_ctx - head_px])})
    if mode in ("first+last", "last"):
        out.append({"index": need_ctx + new_px - 1,
                    "latent": _encode_pin_latent(
                        vae, frames[need_ctx + new_px - 1])})
    return out


def _context_audio_block(audio_vae, audio, start_px: int, span_px: int):
    if audio_vae is None:
        raise ValueError("an audio context needs the audio_vae input")
    waveform, vae_sr = _ref_audio_waveform(audio_vae, audio)
    spf = float(vae_sr) / float(_NATIVE_FPS)
    total = int(waveform.shape[-1])
    a = max(0, min(total, int(round(start_px * spf))))
    b = max(a, min(total, int(round((start_px + span_px) * spf))))
    if b - a < _AUDIO_MIN_SAMPLES:
        print(f"[CLSS] WARNING: audio context span {b - a} sample(s) at "
              f"{start_px} px is shorter than one VAE step "
              f"({_AUDIO_MIN_SAMPLES}) — no audio context is attached")
        return None
    _item, blk = _encode_ref_audio_slice(audio_vae,
                                         waveform[..., a:b].contiguous())
    blk[_MC_AUDIO_KEY] = float(span_px)
    print(f"[CLSS] context audio ref: samples [{a}, {b}) -> "
          f"{int(blk['ref_audio_t'])} af "
          f"({int(blk['ref_audio_t']) / AUDIO_LATENT_FPS:.2f} s) ending at "
          f"the join")
    return blk


def _video_ref_block(vae, clip_frames: torch.Tensor, pool: int,
                     stride: int) -> dict:
    z = vae.encode(clip_frames).float()
    if z.ndim != 5:
        raise ValueError(f"the video-ref clip encoded to {tuple(z.shape)}")
    pool = max(1, int(pool))
    stride = max(1, int(stride))
    if pool > 1:
        m = 2 * pool
        z = z[..., :(z.shape[-2] // m) * m, :(z.shape[-1] // m) * m]
        z = F.avg_pool3d(z, (1, pool, pool))
    if stride > 1:
        z = z[:, :, ::stride]
    blk = {"kind": "video", "latent_t": int(z.shape[2]),
           "latent_h": int(z.shape[3]), "latent_w": int(z.shape[4]),
           "ref_audio_t": 0, "latent": z.contiguous(),
           "audio_latent": None}
    if stride > 1:
        blk[_MC_VIDEO_KEY] = stride
    return blk


class CLSSH3ContinueFromVideo(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLSSH3ContinueFromVideo",
            display_name="CLSS H3 Continue From Video",
            category="MiniMaxH3-CLSS",
            description="Continue a finished CLSS run where it ended: the last overlap span of its saved PNG frames is re-encoded into the opening context (keyframe replay + SLB seed) and the tail of its saved audio becomes the tail ref, so the new run's first window generates the next span as a continuation of the real previous take. The first delivered window re-covers the saved run's last 5 px (0.21 s) as a 2-token head - the 17k+5 head a fresh piece starts with, which fixes the token grid phase. The head is decoded with the run but excluded from the saved output (decode-save drops those frames and the matching 0.21 s of audio), so the take carries exactly the new span and appends after the saved run with nothing to trim.",
            inputs=[
                io.Latent.Input("latent", tooltip="Per-chunk AV template (EmptyMiniMaxH3LatentAV) the new run will use: its resolution must match the saved frames, its length sets the chunk geometry."),
                io.Vae.Input("vae", tooltip="Video VAE, used to re-encode the saved tail frames into context rows."),
                io.Custom("CLSS_CONFIG").Input("clss_config", tooltip="CLSSH3Config of the new run: the overlap token count defines the context span, so it must match the config the sampler runs with."),
                io.String.Input("filename_prefix", default="clss_h3/t2v/CLSSH3_frame_", tooltip="Frame prefix of the finished run - the same string its decode-save node wrote. The last overlap-span frames under that prefix become the context."),
                io.Int.Input("scene_index", default=1, min=1, max=64, tooltip="Which scene block of the prompt node the saved run ENDS in (1-based): multi-scene prompts use one conditioning entry per '---' block, and the continuation window must carry that scene's text. Single-scene prompts: leave 1."),
                io.String.Input("audio_prefix", default="audio/t2v/CLSSH3", tooltip="Audio prefix of the finished run - the same string its save-audio node wrote. The newest matching file's tail (overlap span, ending at the join) becomes the run's audio tail ref. Empty = no audio context."),
                io.Vae.Input("audio_vae", optional=True, tooltip="Audio VAE (MiniMaxH3AudioVAE), required when audio_prefix resolves to a file."),
            ],
            outputs=[
                io.Custom("CLSS_CONTEXT").Output(display_name="context"),
                io.Int.Output(display_name="context_frames", tooltip="Pixel frames the context carries (the run's overlap span)."),
            ],
        )

    @classmethod
    @torch.inference_mode()
    def execute(cls, latent, vae, clss_config, filename_prefix,
                scene_index=1, audio_prefix="audio/t2v/CLSSH3", audio_vae=None):
        _lf0, h_lat, w_lat = _template_geometry(latent)
        overlap = _snap_overlap(clss_config.overlap_latent_frames)
        px_ol = _px_of_tokens(overlap, 0)
        frames, abs_start = _load_frame_range(filename_prefix, -px_ol, px_ol,
                                              h_lat, w_lat)
        ctx_video = _encode_context_video(vae, frames, h_lat, w_lat, overlap,
                                          "continuation context")
        aud_blk = None
        if audio_prefix:
            audio = _load_audio_prefix(audio_prefix)
            if audio is not None:
                aud_blk = _context_audio_block(audio_vae, audio, abs_start,
                                               px_ol)
        print(f"[CLSS] continuation context: frames {abs_start}.."
              f"{abs_start + px_ol - 1} of {filename_prefix} -> "
              f"{int(ctx_video.shape[2])} row(s) ({px_ol} px)"
              + (f" + audio ref {int(aud_blk['ref_audio_t'])}af"
                 if aud_blk is not None else " + no audio ref"))
        return io.NodeOutput(
            {"video": ctx_video, "aud_ref": aud_blk, "pins": [],
             "vref": None, "px_ol": int(px_ol), "scene": int(scene_index),
             "trim_head": True},
            int(px_ol))


class CLSSH3ReeditChunk(io.ComfyNode):

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CLSSH3ReeditChunk",
            display_name="CLSS H3 Re-edit Chunk",
            category="MiniMaxH3-CLSS",
            description="Re-render one chunk of a finished CLSS run without regenerating the run. The chunk is rebuilt as a continuation of the saved frames right before it (context rows + audio tail ref) and re-covers the saved run's last 5 px (0.21 s) as a 2-token head - the same 17k+5 head a fresh piece starts with, which keeps the token grid phase identical to the decoded stream. The head is part of the re-rendered span (kept in the output): it replaces the saved frames at start_frame. Optional first/last frame pins copied from the saved video anchor the seam frames, so the re-rendered chunk drops into place. Chunk 1 has no predecessor and is rebuilt from its pins alone.",
            inputs=[
                io.Latent.Input("latent", tooltip="Per-chunk AV template (EmptyMiniMaxH3LatentAV) the saved run used: its resolution must match the saved frames, its length sets the chunk geometry the re-rendered span is derived from."),
                io.Vae.Input("vae", tooltip="Video VAE, used to re-encode the saved frames (context rows, pinned frames, optional video reference)."),
                io.Custom("CLSS_CONFIG").Input("clss_config", tooltip="CLSSH3Config of the saved run: the overlap token count defines the context span and must match the config the sampler runs with."),
                io.String.Input("filename_prefix", default="clss_h3/t2v/CLSSH3_frame_", tooltip="Frame prefix of the finished run - the same string its decode-save node wrote. Chunk k starts at the cumulative delivered span of the chunks before it (chunk 1 = 17k+5 px at the template length, every later chunk = 17k px)."),
                io.Int.Input("chunk_index", default=2, min=1, max=500, tooltip="1-based chunk of the saved run to re-render. Chunks 2+ are rebuilt as continuations of the saved frames before them; chunk 1 has no predecessor and is rebuilt from its pins alone."),
                io.Int.Input("scene_index", default=1, min=1, max=64, tooltip="Which scene block of the prompt node the saved chunk belongs to (1-based). Multi-scene prompts use one conditioning entry per '---' block - set the prompt text to that scene's text so the re-render carries it. Single-scene prompts: leave 1."),
                io.Combo.Input("pin_frames", options=["first+last", "last", "first", "off"], default="first+last", tooltip="Frames copied from the saved video and pinned as clean keyframes: first = the re-rendered span's first delivered frame, last = its last delivered frame (kept at the saved pixels so the saved neighbours still fit). off = a free re-render that may drift from the saved take."),
                io.Combo.Input("video_ref", options=["off", "on"], default="off", tooltip="Present the saved window's own motion (starting at its context rows, so it lands on the target time grid) to the model as a video reference - the same mechanism the audio recompose uses for its video ref. on = stronger identity to the saved take, more ref tokens; off = the re-render is driven by the context rows and pins only."),
                io.Int.Input("ref_pool", default=2, min=1, max=4, step=1, optional=True, tooltip="Spatial downscale of the video reference (2 = quarter of the reference tokens). Used when video_ref is on."),
                io.Int.Input("ref_stride", default=2, min=1, max=5, step=1, optional=True, tooltip="Temporal stride of the video reference; the kept frames are re-spaced onto the target time grid, so the reference keeps true time at fewer tokens. Used when video_ref is on."),
                io.String.Input("audio_prefix", default="audio/t2v/CLSSH3", tooltip="Audio prefix of the finished run - the same string its save-audio node wrote. The overlap span right before the chunk becomes the audio tail ref. Empty = no audio context."),
                io.Vae.Input("audio_vae", optional=True, tooltip="Audio VAE (MiniMaxH3AudioVAE), required when audio_prefix resolves to a file."),
            ],
            outputs=[
                io.Custom("CLSS_CONTEXT").Output(display_name="context"),
                io.Int.Output(display_name="start_frame", tooltip="Saved-video frame index where the re-rendered span starts (its first delivered frame; 5 px before the chunk boundary on chunks 2+, where the span carries the 2-token head)."),
                io.Int.Output(display_name="frames", tooltip="Pixel frames the re-rendered span covers - replace exactly this many saved frames at start_frame (the span length is unchanged by the head: 243 px for the default 243 px template)."),
            ],
        )

    @classmethod
    @torch.inference_mode()
    def execute(cls, latent, vae, clss_config, filename_prefix, chunk_index=2,
                scene_index=1, pin_frames="first+last", video_ref="off",
                ref_pool=2, ref_stride=2, audio_prefix="audio/t2v/CLSSH3",
                audio_vae=None):
        _lf0, h_lat, w_lat = _template_geometry(latent)
        new_cont = _lf0 - 2
        px0 = _px_of_tokens(_lf0, 0)
        pxc = _px_of_tokens(new_cont, 2)
        overlap = _snap_overlap(clss_config.overlap_latent_frames)
        px_ol = _px_of_tokens(overlap, 0)
        k = int(chunk_index)
        if k < 1:
            raise ValueError("chunk_index is 1-based")
        start = 0 if k == 1 else px0 + (k - 2) * pxc
        need_ctx = 0 if k == 1 else px_ol
        new_px = px0 if k == 1 else pxc
        count = need_ctx + new_px
        _extra = 0
        if video_ref == "on" and px0 > new_px:
            _extra = px0 - new_px
            count = need_ctx + px0
        frames, abs_start = _load_frame_range(filename_prefix,
                                              start - need_ctx, count,
                                              h_lat, w_lat,
                                              tail_extra=_extra)
        ctx_video = None
        if need_ctx:
            ctx_video = _encode_context_video(
                vae, frames[:need_ctx], h_lat, w_lat, overlap,
                f"re-edit context of chunk {k}")
        _head_px = _px_of_tokens(2, (overlap - 2) % 5) if need_ctx else 0
        pins = _edit_pins(vae, frames, need_ctx, new_px, pin_frames,
                          head_px=_head_px)
        aud_blk = None
        if need_ctx and audio_prefix:
            audio = _load_audio_prefix(audio_prefix)
            if audio is not None:
                aud_blk = _context_audio_block(audio_vae, audio, abs_start,
                                               px_ol)
        vref = None
        if video_ref == "on":
            vref = _video_ref_block(vae, frames[:px0], ref_pool, ref_stride)
        print(f"[CLSS] re-edit chunk {k}: saved frames {abs_start}.."
              f"{abs_start + count - 1} of {filename_prefix} | re-render span "
              f"{start - _head_px}..{start + new_px - 1} "
              f"({new_px + _head_px} px = head {_head_px} px + "
              f"{new_px} px) | context {need_ctx} px | pins {len(pins)}"
              + (f" | audio ref {int(aud_blk['ref_audio_t'])}af"
                 if aud_blk is not None else " | no audio ref")
              + (f" | video ref {int(vref['latent_t'])}tok"
                 if vref is not None else ""))
        return io.NodeOutput(
            {"video": ctx_video, "aud_ref": aud_blk, "pins": pins,
             "vref": vref, "px_ol": int(px_ol), "scene": int(scene_index),
             "trim_head": False},
            int(start - _head_px), int(new_px + _head_px))


NODE_CLASS_MAPPINGS = {
    "CLSSH3Config":           CLSSH3Config,
    "CLSSH3AudioConfig":      CLSSH3AudioConfig,
    "CLSSH3ScenePrompts":     CLSSH3ScenePrompts,
    "CLSSH3SceneReference":   CLSSH3SceneReference,
    "CLSSH3SceneReferences":  CLSSH3SceneReferences,
    "CLSSH3SceneReferencesAll": CLSSH3SceneReferencesAll,
    "CLSSH3ContinueFromVideo": CLSSH3ContinueFromVideo,
    "CLSSH3ReeditChunk":     CLSSH3ReeditChunk,
    "CLSSH3LoadLatentUpscaleModel": CLSSH3LoadLatentUpscaleModel,
    "CLSSH3StreamingSampler": CLSSH3StreamingSampler,
    "CLSSH3Guider":           CLSSH3Guider,
    "CLSSH3VideoDecodeSave":  CLSSH3VideoDecodeSave,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CLSSH3Config":           "CLSS H3 Config",
    "CLSSH3AudioConfig":      "CLSS H3 Audio Config",
    "CLSSH3ScenePrompts":     "CLSS H3 Scene Prompts",
    "CLSSH3SceneReference":   "CLSS H3 Scene Reference (R2V)",
    "CLSSH3SceneReferences":  "CLSS H3 Scene References (R2V multi)",
    "CLSSH3SceneReferencesAll": "CLSS H3 Scene References (R2V all scenes)",
    "CLSSH3ContinueFromVideo": "CLSS H3 Continue From Video",
    "CLSSH3ReeditChunk":      "CLSS H3 Re-edit Chunk",
    "CLSSH3LoadLatentUpscaleModel": "CLSS H3 Load Latent Upscale Model",
    "CLSSH3StreamingSampler": "CLSS H3 Streaming Sampler",
    "CLSSH3Guider":           "CLSS H3 Guider (Split AV CFG)",
    "CLSSH3VideoDecodeSave":  "CLSS H3 Video Decode+Save (streaming)",
}
