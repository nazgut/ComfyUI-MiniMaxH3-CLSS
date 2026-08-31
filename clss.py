"""
Closed-Loop Streaming Synthesis (CLSS) algorithm components.

CLSS extends Latent Streaming Synthesis (LSS) with closed-loop corrections that
eliminate the exposure-bias drift accumulation inherent in open-loop streaming:

  §2.1  Calibrated context re-noising
        The overlap latents fed as context to chunk N are re-noised to level τc:
            L̃_overlap = α(τc)·L̂_overlap + σ(τc)·ε   (α = 1−τc, σ = τc)
        How the re-noised context is presented to the transformer is the host
        sampler's job (model-specific); this module tracks the SLB itself.
        The denoising transformer must see those tokens tagged with a non-zero
        timestep so it can actively re-project the re-noised context back onto
        the data manifold.

  §2.3  EMA-tracked per-channel distribution reference
        A slow EMA (rate λ ≈ 0.05) tracks per-channel mean and std across chunks.
        Fast drift (single-chunk statistical error) is suppressed; slow intended
        evolution (lighting changes, scene content) passes through.  Applied as a
        per-channel AdaIN-style renormalisation blended with factor β.

Algorithm 1 (per-chunk step):
  1.  L_overlap  ← SLB.read()
  2.  L̃_overlap ← α(τc)·L_overlap + σ(τc)·ε          [§2.1, via conditioning]
  3.  L_N        ← Generate(L̃_overlap, prompt)
  4.  L_N        ← AdaIN lerp toward per-channel EMA ref, factor β  [§2.3]
  5.  EMA refs   ← update(L_N)
  6.  SLB.push(trailing frames)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _chan_stats(x: torch.Tensor) -> tuple[list[float], list[float]]:
    """Return (per-channel means, per-channel stds) for a [B, C, F, H, W] latent."""
    with torch.no_grad():
        flat = x.float().permute(1, 0, 2, 3, 4).flatten(1)
        return flat.mean(1).tolist(), flat.std(1).tolist()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CLSSConfig:
    """Hyperparameters for Closed-Loop Streaming Synthesis.

    Parameters
    ----------
    tau_c:
        Re-noising noise level applied to overlap context (§2.1).
        0 recovers LSS behaviour (maximal continuity, maximal drift accumulation);
        larger values give more distributional repair at the cost of softer
        motion lock.  Recommended range: 0.10–0.20.
    beta:
        AdaIN correction blend factor (§2.3).  0 = no correction, 1 = full
        replacement with reference statistics.  Recommended: 0.2–0.4.
    ema_lambda:
        EMA update rate per chunk (§2.3).  Slow enough to track intended
        scene evolution while suppressing per-chunk drift.  Recommended: 0.05–0.10.
    overlap_latent_frames:
        Number of latent frames shared between consecutive chunks (the SLB size).
        Corresponds to the temporal context the model sees from the previous chunk.
    new_latent_frames:
        Number of genuinely new latent frames generated per chunk.  The pixel
        frame count is (new_latent_frames − 1) × 8 + 1 with the default VAE
        scale factor of 8.
    """

    # §2.1  Calibrated context re-noising
    # 0.0 = full overlap strength (maximal continuity, some drift risk)
    # 0.05 = very light re-noising, very strong temporal constraint
    # 0.15–0.20 = paper recommended range (too loose for 22B GGUF streaming)
    tau_c: float = 0.05

    # §2.3  EMA-tracked per-channel distribution reference
    beta: float = 0.4
    ema_lambda: float = 0.05
    # Cap on how far the per-channel EMA std may grow relative to its initial
    # value (chunk-0 statistics).  0.0 = uncapped (old behaviour, allows σ drift).
    # With the default 0.05 the reference std is allowed to increase at most 5 %
    # from chunk 0's value, which prevents AdaIN from quietly amplifying late chunks
    # while still permitting slow intentional brightening / saturation changes.
    ema_sigma_max_drift: float = 0.05
    # Maximum per-channel AdaIN upward amplification factor.
    # When the EMA reference std for a channel exceeds the current chunk's std,
    # AdaIN would amplify that channel's variance — boosting any residual
    # denoising noise and causing visible grain in the decoded video.
    # This cap limits how aggressively AdaIN can amplify: 1.2 = allow at most
    # 20 % upward scaling.  0.0 = no cap (original behaviour).
    # Recommended: 1.2 when noise/grain is visible, especially with < 30 steps.
    adain_max_amplification: float = 0.0

    # Streaming buffer dimensions
    # 8 latent frames ≈ 57 pixel frames ≈ 2.4 s of hard context from previous chunk.
    # Combined with tau_c=0.05 this makes the overlap frames nearly frozen, giving
    # strong visual continuity at the cost of ~17% more denoising work per chunk.
    overlap_latent_frames: int = 8
    new_latent_frames: int = 13

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau_c <= 1.0:
            raise ValueError(f"tau_c must be in [0, 1], got {self.tau_c}")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")


# ---------------------------------------------------------------------------
# §2.3  Per-channel EMA reference
# ---------------------------------------------------------------------------


class _PerChannelEMA:
    """Exponential moving average of per-channel mean and standard deviation.

    Operates on unpatchified latents of shape [B, C, F, H, W].
    Statistics are computed over all non-channel dimensions (B, F, H, W).
    """

    def __init__(self) -> None:
        self.mean: Optional[torch.Tensor] = None  # [C]
        self.std: Optional[torch.Tensor] = None   # [C]
        self._init_std: Optional[torch.Tensor] = None  # [C] anchored to chunk-0

    def update(self, latent: torch.Tensor, lam: float, sigma_max_drift: float = 0.0) -> None:
        """Update EMA statistics.  latent: [B, C, F, H, W]

        sigma_max_drift: if > 0, caps the per-channel EMA std to at most
        (1 + sigma_max_drift) × the chunk-0 std, preventing gradual amplification.
        """
        # [C, B*F*H*W]
        x = latent.float().permute(1, 0, 2, 3, 4).flatten(1)
        mu = x.mean(1)
        sig = x.std(1).clamp(min=1e-5)
        if self.mean is None:
            self.mean = mu.clone()
            self.std = sig.clone()
            self._init_std = sig.clone()
        else:
            self.mean = (1.0 - lam) * self.mean + lam * mu
            self.std  = (1.0 - lam) * self.std  + lam * sig
            if sigma_max_drift > 0.0 and self._init_std is not None:
                self.std = self.std.clamp(max=self._init_std * (1.0 + sigma_max_drift))

    def apply_adain(
        self, latent: torch.Tensor, beta: float, max_amplification: float = 0.0
    ) -> torch.Tensor:
        """Blend *latent* toward the EMA statistics via per-channel AdaIN.

        Result = (1−β)·latent + β·AdaIN(latent → EMA).
        Returns the original latent unchanged when the EMA is uninitialised
        (first chunk).

        max_amplification: if > 0, caps per-channel upward std scaling to this
        factor.  E.g. 1.2 = allow the EMA reference to push a channel's std up
        by at most 20 %.  Attenuation (EMA std < current std) is never capped.
        Set to 0.0 to disable (original behaviour).
        """
        if self.mean is None:
            return latent
        B, C, F, H, W = latent.shape
        # [C, N] where N = B*F*H*W
        x = latent.float().permute(1, 0, 2, 3, 4).flatten(1)
        mu_cur = x.mean(1, keepdim=True)
        sig_cur = x.std(1, keepdim=True).clamp(min=1e-5)
        # Per-channel target std: optionally cap upward amplification so that
        # channels with EMA std >> current std don't boost residual denoising noise.
        target_std = self.std.unsqueeze(1)  # [C, 1]
        if max_amplification > 0.0:
            cap = sig_cur * max_amplification  # [C, 1]
            target_std = torch.minimum(target_std, cap)
        # Normalise to zero-mean/unit-std, then scale to (capped) EMA reference
        corrected = (x - mu_cur) / sig_cur * target_std + self.mean.unsqueeze(1)
        blended = (1.0 - beta) * x + beta * corrected
        return blended.view(C, B, F, H, W).permute(1, 0, 2, 3, 4).to(latent.dtype)


class CLSSState:
    """Persistent CLSS state maintained across all chunk steps.

    Manages the two CLSS components:
    - Streaming Latent Buffer (SLB): the overlap latent passed to the next chunk
    - Per-channel EMA reference for AdaIN correction (§2.3)

    Usage::

        clss = CLSSState(config)
        for chunk_latent_frames in chunk_schedule:
            # 1. Read the SLB overlap and feed it as context (host sampler's job)
            overlap = clss.overlap_latent

            # 2. Generate the chunk; extract the new frames (drop the overlap)
            new_frames = generated[:, :, overlap_lf:] if chunk_idx > 0 else generated

            # 3. Apply CLSS corrections (§2.3), update the EMA reference
            corrected = clss.post_process(new_frames)

            # 4. Update the SLB for the next iteration
            clss.update_buffer(corrected)

            # 5. Accumulate corrected for final decoding
            all_latents.append(corrected)
    """

    def __init__(self, config: CLSSConfig) -> None:
        self.config = config
        self._ema = _PerChannelEMA()
        # Streaming Latent Buffer: last overlap_latent_frames from the previous chunk
        self._overlap_latent: Optional[torch.Tensor] = None   # [1, C, overlap_F, H, W]
        self._chunk_index: int = 0

    def reset_drift_refs(self) -> None:
        """Drop the §2.3 EMA reference (call on a scene change).

        The next chunk is then treated like chunk 0: ``apply_adain`` is a no-op
        for it and ``update`` re-seeds the EMA from its statistics — including
        ``_init_std``, the anchor for the ``ema_sigma_max_drift`` cap.  Without
        this, a scene change keeps pulling the new scene's per-channel stats
        toward the old scene's EMA for several chunks (β-weighted color drag).
        """
        self._ema = _PerChannelEMA()

    @property
    def overlap_latent(self) -> Optional[torch.Tensor]:
        """The current SLB overlap latent ([1, C, F, H, W]) or None on chunk 0.

        How the overlap is fed to the model (frozen pinning, re-noising, keyframe
        conditioning rows) is the host sampler's business — it is model-specific.
        """
        return self._overlap_latent

    # ------------------------------------------------------------------
    # Steps 5–6 — Post-process generated new frames
    # ------------------------------------------------------------------

    def post_process(self, new_frames: torch.Tensor) -> torch.Tensor:
        """Apply AdaIN (§2.3) to *new_frames*.

        The correction is applied to the *new* frames only (the overlap region
        is not post-processed — it was already corrected when it was generated).
        EMA references are updated with the fully corrected output (Algorithm 1,
        step 6).

        new_frames: [1, C, F, H, W]
        Returns corrected latent with the same shape and dtype.
        """
        cfg = self.config
        cidx = self._chunk_index

        if logger.isEnabledFor(logging.DEBUG):
            mu_raw, sig_raw = _chan_stats(new_frames)
            logger.debug(
                "[CLSS] chunk=%d  raw_latent  μ̄=%.4f  σ̄=%.4f  "
                "μ_range=[%.4f, %.4f]  σ_range=[%.4f, %.4f]",
                cidx,
                sum(mu_raw) / len(mu_raw), sum(sig_raw) / len(sig_raw),
                min(mu_raw), max(mu_raw), min(sig_raw), max(sig_raw),
            )

        _pre_mean = new_frames.float().mean().item()
        _pre_std  = new_frames.float().std().item()
        out = self._ema.apply_adain(new_frames, cfg.beta, cfg.adain_max_amplification)
        # §2.3 always-on telemetry: EMA reference + AdaIN correction direction
        if logger.isEnabledFor(logging.DEBUG):
            print(
                f"[CLSS] chunk={cidx}"
                f"  adain_delta_mean={out.float().mean().item() - _pre_mean:+.5f}"
                f"  delta_std={out.float().std().item() - _pre_std:+.5f}"
            )

        if logger.isEnabledFor(logging.DEBUG) and self._ema.mean is not None:
            mu_adain, sig_adain = _chan_stats(out)
            ema_mu_mean = sum(self._ema.mean.tolist()) / len(self._ema.mean)
            ema_sig_mean = sum(self._ema.std.tolist()) / len(self._ema.std)
            logger.debug(
                "[CLSS] chunk=%d  after_adain  μ̄=%.4f  σ̄=%.4f  β=%.4f  "
                "ema_ref: μ̄_ema=%.4f  σ̄_ema=%.4f",
                cidx,
                sum(mu_adain) / len(mu_adain), sum(sig_adain) / len(sig_adain),
                cfg.beta, ema_mu_mean, ema_sig_mean,
            )

        # §2.3 Update per-channel EMA with the corrected output (step 6)
        self._ema.update(out, cfg.ema_lambda, sigma_max_drift=cfg.ema_sigma_max_drift)

        if logger.isEnabledFor(logging.DEBUG):
            mu_final, sig_final = _chan_stats(out)
            logger.debug(
                "[CLSS] chunk=%d  post_process_done  μ̄=%.4f  σ̄=%.4f",
                cidx,
                sum(mu_final) / len(mu_final), sum(sig_final) / len(sig_final),
            )

        return out

    # ------------------------------------------------------------------
    # Update SLB
    # ------------------------------------------------------------------

    def update_buffer(self, output_latent: torch.Tensor) -> None:
        """Push the trailing overlap frames to the SLB.

        output_latent: [1, C, F, H, W] — corrected new frames from this chunk.
        The last min(overlap_latent_frames, F) frames become the SLB for the
        next chunk.
        """
        cfg = self.config
        F_total = output_latent.shape[2]

        # SLB: last overlap_latent_frames frames (or all if chunk is smaller)
        n_overlap = min(cfg.overlap_latent_frames, F_total)
        self._overlap_latent = output_latent[:, :, -n_overlap:].clone()

        self._chunk_index += 1
