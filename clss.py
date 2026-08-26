"""
Closed-Loop Streaming Synthesis (CLSS) algorithm components.

CLSS extends Latent Streaming Synthesis (LSS) with three closed-loop corrections that
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

  §2.5  Dynamic anchor bank
        A small bank of anchor keyframes is maintained.  The first frame of the
        first chunk seeds the bank.  Persistent cosine-similarity drops (below
        threshold ρ for two consecutive chunks) trigger a new anchor.  Chunk N
        cross-attends to the top-m most similar anchors, providing soft long-range
        identity without dragging back to a single frozen frame-0 reference.
        Anchor similarity is computed from mean-pooled, L2-normalised latent
        features (no separate image encoder is assumed to be at hand).

Algorithm 1 (per-chunk step):
  1.  L_overlap  ← SLB.read()
  2.  L̃_overlap ← α(τc)·L_overlap + σ(τc)·ε          [§2.1, via conditioning]
  3.  A_m        ← top-m anchors by feature similarity  [§2.5]
  4.  L_N        ← Generate(L̃_overlap, A_m, prompt)
  5.  L_N        ← AdaIN lerp toward per-channel EMA ref, factor β  [§2.3]
  6.  EMA refs   ← update(L_N); anchor bank ← update on persistent scene change
  7.  SLB.push(trailing frames)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

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
    anchor_threshold:
        Cosine-similarity threshold below which a chunk is considered a candidate
        scene change (§2.5).  Default 0.78 (from the paper).
    anchor_persistence:
        Number of consecutive below-threshold chunks required before a new anchor
        is committed (persistence test, §2.5).  Filters transient occlusions.
    anchor_max_size:
        Maximum number of anchors retained in the bank (§2.5).  Oldest
        least-recently-retrieved anchor is evicted when the cap is exceeded.
    anchor_top_m:
        Number of anchors retrieved per chunk for conditioning (§2.5).
    anchor_strength:
        Denoising strength for anchor keyframe tokens (1.0 = fully clean / no
        denoising; use 1.0 to prevent anchor frames from being corrupted).
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

    # §2.5  Dynamic anchor bank
    anchor_threshold: float = 0.78
    anchor_persistence: int = 2
    anchor_max_size: int = 8
    anchor_top_m: int = 2
    anchor_strength: float = 1.0
    # Force a new anchor every N chunks regardless of similarity.  0 = disabled.
    # Without this, on visually-stable scenes the bank stagnates (similarity stays
    # above threshold, persistence never fires) and the model loses recent reference
    # frames — the primary cause of content collapse after ~10-15 chunks.
    anchor_force_every: int = 5

    # Streaming buffer dimensions
    # 8 latent frames ≈ 57 pixel frames ≈ 2.4 s of hard context from previous chunk.
    # Combined with tau_c=0.05 this makes the overlap frames nearly frozen, giving
    # strong visual continuity at the cost of ~17% more denoising work per chunk.
    overlap_latent_frames: int = 8
    new_latent_frames: int = 13

    # §3.7  Open-loop model gain measurement.
    # measure_g: run a second denoising pass with a perturbed overlap on every chunk
    #   that has overlap, to track g = ||Δ_output||_F / ||δ_input||_F per-chunk.
    #   Doubles denoising time for non-first chunks.  Requires a non-None generator.
    # measure_g_epsilon: perturbation magnitude as a fraction of the overlap norm.
    measure_g: bool = True
    measure_g_epsilon: float = 0.01

    # Temporally-correlated initial video noise (EXPERIMENTAL, off by default —
    # unvalidated until a live run).  Targets the measured ~4 s layout
    # oscillation, which is invariant to chunk length / overlap length / SLB
    # strength: with i.i.d. noise every ~4 s span of frames carries an
    # independent low-frequency "content suggestion" that the model resolves as
    # a fresh motion arc at its trained temporal horizon.  Mixing a run-constant
    # shared frame into every video noise frame,
    #     n_t = sqrt(1-a)·eps_t + sqrt(a)·eps_shared,
    # keeps each frame's marginal exactly N(0,1) but raises frame-to-frame noise
    # correlation to a at ALL lags (inference-time correlated noise prior, same
    # family as FreeNoise noise rescheduling / PYoCo mixed noise).  Unlike
    # block-repeat schemes this cannot introduce a new periodicity.  0.0 = off
    # (bit-exact baseline).  Worst case behaves like a seed change with more
    # static content.
    noise_temporal_corr: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau_c <= 1.0:
            raise ValueError(f"tau_c must be in [0, 1], got {self.tau_c}")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")
        if not 0.0 <= self.noise_temporal_corr < 1.0:
            raise ValueError(
                f"noise_temporal_corr must be in [0, 1), got {self.noise_temporal_corr}"
            )


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


# ---------------------------------------------------------------------------
# §2.5  Dynamic anchor bank
# ---------------------------------------------------------------------------


@dataclass
class _AnchorEntry:
    """Single keyframe anchor."""
    feature: torch.Tensor    # [C] L2-normalised mean-pooled latent feature
    latent: torch.Tensor     # [1, C, 1, H, W] single-frame latent
    frame_idx: int           # absolute latent frame index in the full generated video
    last_retrieved: int = 0  # chunk index of most recent retrieval (for LRU eviction)


class _AnchorBank:
    """Dynamic keyframe anchor bank with LRU eviction (§2.5)."""

    def __init__(self, threshold: float, persistence: int, max_size: int) -> None:
        self.threshold = threshold
        self.persistence = persistence
        self.max_size = max_size
        self.anchors: list[_AnchorEntry] = []
        self._below_count = 0

    @staticmethod
    def _feature(frame_latent: torch.Tensor) -> torch.Tensor:
        """Compute a [C] feature from a [1, C, 1, H, W] single-frame latent."""
        v = frame_latent.float().mean(dim=(0, 2, 3, 4))  # [C]
        return F.normalize(v.unsqueeze(0), dim=1).squeeze(0)

    def _max_cosine_sim(self, feat: torch.Tensor) -> float:
        if not self.anchors:
            return 0.0
        return max(
            F.cosine_similarity(feat.unsqueeze(0), a.feature.unsqueeze(0)).item()
            for a in self.anchors
        )

    def initialize(self, frame_latent: torch.Tensor, frame_idx: int) -> None:
        """Seed the bank with the first anchor (called once for chunk 0)."""
        feat = self._feature(frame_latent)
        self.anchors = [_AnchorEntry(feature=feat, latent=frame_latent.clone(), frame_idx=frame_idx)]

    def update(
        self,
        frame_latent: torch.Tensor,
        frame_idx: int,
        chunk_idx: int,
        force: bool = False,
    ) -> bool:
        """Evaluate scene-change condition; commit a new anchor if triggered.

        force=True bypasses the threshold/persistence check and always adds an
        anchor.  Used for periodic forced insertions so the bank always contains
        recent frames (prevents content drift on long, visually-stable videos).
        Returns True if a new anchor was committed this call.
        """
        feat = self._feature(frame_latent)
        sim = self._max_cosine_sim(feat)
        if sim < self.threshold:
            self._below_count += 1
        else:
            self._below_count = 0

        committed = False
        if force or self._below_count >= self.persistence:
            if force:
                self._below_count = 0  # forced insert resets streak
            else:
                self._below_count = 0
            self.anchors.append(
                _AnchorEntry(
                    feature=feat,
                    latent=frame_latent.clone(),
                    frame_idx=frame_idx,
                    last_retrieved=chunk_idx,
                )
            )
            committed = True
            # LRU eviction: remove the anchor retrieved least recently
            while len(self.anchors) > self.max_size:
                self.anchors.sort(key=lambda a: a.last_retrieved)
                self.anchors.pop(0)
        return committed

    # Anchors whose similarity to the current overlap exceeds this are redundant:
    # the overlap conditioning already captures them.  Injecting redundant anchors
    # alongside the overlap can confuse the model (two near-identical conditions
    # at the same temporal position).
    _REDUNDANCY_SIM_THRESHOLD: float = 0.85

    def retrieve(
        self, context_latent: torch.Tensor, top_m: int, chunk_idx: int
    ) -> list[_AnchorEntry]:
        """Return top-m non-redundant anchors by cosine similarity to *context_latent*.

        context_latent: [1, C, F, H, W] — uses the last frame as the query.
        Anchors with similarity > _REDUNDANCY_SIM_THRESHOLD are skipped: they are
        nearly identical to the current overlap and the overlap conditioning
        already handles them.  This prevents injecting a "chunk N-1 last frame"
        anchor that duplicates the overlap and conflicts with it at the start of
        the new chunk.
        Updates last_retrieved for returned anchors (prevents premature eviction).
        """
        if not self.anchors or top_m == 0:
            return []
        query_feat = self._feature(context_latent[:, :, -1:])
        sims = [
            F.cosine_similarity(query_feat.unsqueeze(0), a.feature.unsqueeze(0)).item()
            for a in self.anchors
        ]
        # Keep only anchors that add information the overlap doesn't already provide
        candidates = [
            i for i, s in enumerate(sims) if s < self._REDUNDANCY_SIM_THRESHOLD
        ]
        if not candidates:
            return []
        top_indices = sorted(candidates, key=lambda i: sims[i], reverse=True)[:top_m]
        result = []
        for idx in top_indices:
            self.anchors[idx].last_retrieved = chunk_idx
            result.append(self.anchors[idx])
        return result


# ---------------------------------------------------------------------------
# CLSSState — orchestrates all per-chunk state
# ---------------------------------------------------------------------------


class CLSSState:
    """Persistent CLSS state maintained across all chunk steps.

    Manages the three CLSS components:
    - Streaming Latent Buffer (SLB): the overlap latent passed to the next chunk
    - Per-channel EMA reference for AdaIN correction (§2.3)
    - Dynamic anchor bank for long-range identity (§2.5)

    Usage::

        clss = CLSSState(config)
        for chunk_latent_frames in chunk_schedule:
            # 1. Build conditionings: overlap (§2.1) + anchors (§2.5)
            conds = clss.get_overlap_conditioning() + clss.get_anchor_conditioning()

            # 2. Run DiffusionStage with *conds*; get chunk output [1,C,F,H,W]
            generated = ...

            # 3. Extract new frames (drop the overlap region)
            new_frames = generated[:, :, overlap_lf:] if chunk_idx > 0 else generated

            # 4. Apply CLSS corrections (§2.3), update EMA + anchor bank
            corrected = clss.post_process(new_frames)

            # 5. Update the SLB for the next iteration
            clss.update_buffer(corrected)

            # 6. Accumulate corrected for final decoding
            all_latents.append(corrected)
    """

    def __init__(self, config: CLSSConfig) -> None:
        self.config = config
        self._ema = _PerChannelEMA()
        self._anchor_bank = _AnchorBank(
            threshold=config.anchor_threshold,
            persistence=config.anchor_persistence,
            max_size=config.anchor_max_size,
        )
        # Streaming Latent Buffer: last overlap_latent_frames from the previous chunk
        self._overlap_latent: Optional[torch.Tensor] = None   # [1, C, overlap_F, H, W]
        self._chunk_index: int = 0
        self._abs_frame_idx: int = 0  # next absolute latent frame index to be written
        # §3.7 Open-loop model gain: set by the pipeline after a perturbation experiment
        self.g_measured: Optional[float] = None
        self.rho_closed: Optional[float] = None

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

    def top_anchors(self) -> list:
        """Return the top-m anchor entries most similar to the current SLB overlap.

        Same contract note as ``overlap_latent``: turning anchors into model
        conditioning is the host sampler's job.
        """
        if self._overlap_latent is None:
            return []
        return self._anchor_bank.retrieve(
            self._overlap_latent, self.config.anchor_top_m, self._chunk_index
        )

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
    # Step 8  — Update SLB and anchor bank
    # ------------------------------------------------------------------

    def update_buffer(self, output_latent: torch.Tensor) -> None:
        """Push the trailing overlap frames to the SLB; update the anchor bank.

        output_latent: [1, C, F, H, W] — corrected new frames from this chunk.
        The last min(overlap_latent_frames, F) frames become the SLB for the
        next chunk.  The anchor bank is updated with the final frame of this
        chunk (scene-change detection, §2.5).
        """
        cfg = self.config
        F_total = output_latent.shape[2]

        # §2.5 Anchor bank: evaluate scene change with the last frame
        last_frame = output_latent[:, :, -1:]  # [1, C, 1, H, W]
        abs_last_frame_idx = self._abs_frame_idx + F_total - 1
        if self._chunk_index == 0:
            self._anchor_bank.initialize(last_frame, abs_last_frame_idx)
            _new_anchor = True
        else:
            forced = (
                cfg.anchor_force_every > 0
                and self._chunk_index % cfg.anchor_force_every == 0
            )
            _new_anchor = self._anchor_bank.update(
                last_frame, abs_last_frame_idx, self._chunk_index, force=forced,
            )

        # §2.5 always-on telemetry: anchor bank events
        _feat_last  = _AnchorBank._feature(last_frame)
        _max_sim    = self._anchor_bank._max_cosine_sim(_feat_last)
        _bank_fids  = [a.frame_idx for a in self._anchor_bank.anchors]
        print(
            f"[CLSS] chunk={self._chunk_index}"
            f"  anchors: bank_size={len(self._anchor_bank.anchors)}"
            f"  last_frame_max_sim={_max_sim:.4f}"
            f"  new_anchor={_new_anchor}"
            f"  scene_change_streak={self._anchor_bank._below_count}"
            f"  bank_frame_ids={_bank_fids}"
        )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[CLSS] chunk=%d  anchor_bank  n_anchors=%d  "
                "last_frame_max_sim=%.4f  below_streak=%d  threshold=%.2f",
                self._chunk_index,
                len(self._anchor_bank.anchors),
                _max_sim,
                self._anchor_bank._below_count,
                cfg.anchor_threshold,
            )

        # SLB: last overlap_latent_frames frames (or all if chunk is smaller)
        n_overlap = min(cfg.overlap_latent_frames, F_total)
        self._overlap_latent = output_latent[:, :, -n_overlap:].clone()

        self._abs_frame_idx += F_total
        self._chunk_index += 1

    # ------------------------------------------------------------------
    # Stability diagnostics (§2.6)
    # ------------------------------------------------------------------

    @property
    def chunk_index(self) -> int:
        """Number of chunks processed so far."""
        return self._chunk_index

    def loop_gain_estimate(self, beta: Optional[float] = None, tau_c: Optional[float] = None) -> float:
        """Estimate the closed-loop gain ρ_loop = (1 − β)·α(τc) (§2.6, Eq. 6).

        For bounded drift we need ρ_loop < 1.  The formula omits the open-loop
        model gain g (which must be measured empirically, §2.6), so the returned
        value is a lower bound on ρ_loop.

        Returns
        -------
        float
            (1 − β) · (1 − τc).  Values well below 1 give large stability margins.
        """
        b = beta if beta is not None else self.config.beta
        t = tau_c if tau_c is not None else self.config.tau_c
        return (1.0 - b) * (1.0 - t)
