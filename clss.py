from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def _chan_stats(x: torch.Tensor) -> tuple[list[float], list[float]]:
    with torch.no_grad():
        flat = x.float().permute(1, 0, 2, 3, 4).flatten(1)
        return flat.mean(1).tolist(), flat.std(1).tolist()


_MEAN_PULL = 0.9


@dataclass
class CLSSConfig:

    tau_c: float = 0.05

    beta: float = 0.4
    ema_lambda: float = 0.05
    ema_sigma_max_drift: float = 0.05
    ema_mean_max_drift: float = 0.25
    adain_max_amplification: float = 0.0

    overlap_latent_frames: int = 8
    new_latent_frames: int = 13

    def __post_init__(self) -> None:
        if not 0.0 <= self.tau_c <= 1.0:
            raise ValueError(f"tau_c must be in [0, 1], got {self.tau_c}")
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {self.beta}")


class _PerChannelEMA:

    def __init__(self) -> None:
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self._init_std: Optional[torch.Tensor] = None
        self._init_mean: Optional[torch.Tensor] = None

    def update(self, latent: torch.Tensor, lam: float, sigma_max_drift: float = 0.0,
               mean_max_drift: float = 0.0) -> None:
        x = latent.float().permute(1, 0, 2, 3, 4).flatten(1)
        mu = x.mean(1)
        sig = x.std(1).clamp(min=1e-5)
        if self.mean is None:
            self.mean = mu.clone()
            self.std = sig.clone()
            self._init_std = sig.clone()
            self._init_mean = mu.clone()
        else:
            self.mean = (1.0 - lam) * self.mean + lam * mu
            self.std  = (1.0 - lam) * self.std  + lam * sig
            if sigma_max_drift > 0.0 and self._init_std is not None:
                self.std = self.std.clamp(max=self._init_std * (1.0 + sigma_max_drift))
            if (mean_max_drift > 0.0 and self._init_mean is not None
                    and self._init_std is not None):
                band = self._init_std * mean_max_drift
                self.mean = self.mean.clamp(min=self._init_mean - band,
                                            max=self._init_mean + band)

    def apply_adain(
        self, latent: torch.Tensor, beta: float, max_amplification: float = 0.0
    ) -> torch.Tensor:
        if beta <= 0.0:
            return latent
        if self.mean is None:
            return latent
        B, C, F, H, W = latent.shape
        x = latent.float().permute(1, 0, 2, 3, 4).flatten(1)
        mu_cur = x.mean(1, keepdim=True)
        sig_cur = x.std(1, keepdim=True).clamp(min=1e-5)
        target_std = self.std.unsqueeze(1)
        if max_amplification > 0.0:
            cap = sig_cur * max_amplification
            target_std = torch.minimum(target_std, cap)
        std_mix = (1.0 - beta) * sig_cur + beta * target_std
        mean_mix = ((1.0 - _MEAN_PULL) * mu_cur
                    + _MEAN_PULL * self.mean.unsqueeze(1))
        corrected = (x - mu_cur) / sig_cur * std_mix + mean_mix
        return corrected.view(C, B, F, H, W).permute(1, 0, 2, 3, 4).to(latent.dtype)


class CLSSState:

    def __init__(self, config: CLSSConfig) -> None:
        self.config = config
        self._ema = _PerChannelEMA()
        self._overlap_latent: Optional[torch.Tensor] = None
        self._chunk_index: int = 0

    def reset_drift_refs(self) -> None:
        self._ema = _PerChannelEMA()

    @property
    def overlap_latent(self) -> Optional[torch.Tensor]:
        return self._overlap_latent


    def post_process(self, new_frames: torch.Tensor) -> torch.Tensor:
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

        self._ema.update(out, cfg.ema_lambda, sigma_max_drift=cfg.ema_sigma_max_drift,
                         mean_max_drift=getattr(cfg, "ema_mean_max_drift", 0.0))

        if logger.isEnabledFor(logging.DEBUG):
            mu_final, sig_final = _chan_stats(out)
            logger.debug(
                "[CLSS] chunk=%d  post_process_done  μ̄=%.4f  σ̄=%.4f",
                cidx,
                sum(mu_final) / len(mu_final), sum(sig_final) / len(sig_final),
            )

        return out


    def update_buffer(self, output_latent: torch.Tensor) -> None:
        cfg = self.config
        F_total = output_latent.shape[2]

        n_overlap = min(cfg.overlap_latent_frames, F_total)
        self._overlap_latent = output_latent[:, :, -n_overlap:].clone()

        self._chunk_index += 1
