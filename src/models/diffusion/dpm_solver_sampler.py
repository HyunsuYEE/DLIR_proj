from dataclasses import dataclass
from typing import Any, List, Tuple

import torch
from torch import Tensor

from .teacache import TeaCacheState


@dataclass
class DpmSolverSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5.0
    rho: int = 7
    order: int = 2
    method: str = "multistep"
    teacache_enabled: bool = False
    teacache_rel_l1_thresh: float = 0.2
    teacache_force_last: bool = True


class DenoiserDpmAdapter:
    def __init__(
        self,
        denoiser: Any,
        obs: Tensor,
        act: Tensor,
        teacache_state: TeaCacheState | None = None,
        num_steps: int | None = None,
    ) -> None:
        self.denoiser = denoiser
        self.obs = obs
        self.act = act
        self.teacache_state = teacache_state
        self.num_steps = num_steps

    def predict_x0(self, x: Tensor, sigma: Tensor, step_index: int | None = None) -> Tensor:
        if self.teacache_state is not None:
            if step_index is None or self.num_steps is None:
                raise ValueError("TeaCache requires step_index and num_steps.")
            return self.denoiser.denoise_teacache(
                x,
                sigma,
                self.obs,
                self.act,
                teacache_state=self.teacache_state,
                step_index=step_index,
                num_steps=self.num_steps,
            )
        return self.denoiser.denoise(x, sigma, self.obs, self.act)

    def derivative(self, x: Tensor, sigma: Tensor, step_index: int | None = None) -> Tensor:
        return edm_derivative(x, self.predict_x0(x, sigma, step_index=step_index), sigma)


class DpmSolverSampler:
    def __init__(self, denoiser: Any, cfg: DpmSolverSamplerConfig) -> None:
        if cfg.method != "multistep":
            raise ValueError(f"Unsupported DPM-Solver method: {cfg.method}")
        if cfg.order not in (1, 2):
            raise ValueError(f"Unsupported DPM-Solver order: {cfg.order}")
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser.device)
        self.num_denoiser_evals_last_sample = 0
        self.num_teacache_full_evals_last_sample = 0
        self.num_teacache_cache_hits_last_sample = 0

    @torch.no_grad()
    def sample(self, prev_obs: Tensor, prev_act: Tensor) -> Tuple[Tensor, List[Tensor]]:
        device = prev_obs.device
        b, t, c, h, w = prev_obs.size()
        obs = prev_obs.reshape(b, t * c, h, w)
        num_steps = len(self.sigmas) - 1
        teacache_state = self._make_teacache_state()
        adapter = DenoiserDpmAdapter(self.denoiser, obs, prev_act, teacache_state=teacache_state, num_steps=num_steps)
        self.num_denoiser_evals_last_sample = 0
        self.num_teacache_full_evals_last_sample = 0
        self.num_teacache_cache_hits_last_sample = 0

        x = torch.randn(b, c, h, w, device=device)
        trajectory = [x]

        previous_derivative = None
        previous_sigma = None
        for step_index, (sigma, next_sigma) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
            derivative = adapter.derivative(x, sigma, step_index=step_index)
            self.num_denoiser_evals_last_sample += 1
            if self.cfg.order == 1 or previous_derivative is None:
                x = x + derivative * (next_sigma - sigma)
            else:
                x = edm_ab2_step(
                    x=x,
                    d_cur=derivative,
                    d_prev=previous_derivative,
                    sigma_cur=sigma,
                    sigma_prev=previous_sigma,
                    sigma_next=next_sigma,
                )

            previous_derivative = derivative
            previous_sigma = sigma
            trajectory.append(x)

        if teacache_state is not None:
            self.num_teacache_full_evals_last_sample = teacache_state.stats.full_evals
            self.num_teacache_cache_hits_last_sample = teacache_state.stats.cache_hits
        return x, trajectory

    def _make_teacache_state(self) -> TeaCacheState | None:
        if not self.cfg.teacache_enabled:
            return None
        return TeaCacheState(
            rel_l1_thresh=self.cfg.teacache_rel_l1_thresh,
            force_last=self.cfg.teacache_force_last,
            enabled=True,
        )


def edm_derivative(x: Tensor, x0: Tensor, sigma: Tensor) -> Tensor:
    return (x - x0) / expand_to(sigma, x)


def edm_euler_step(x: Tensor, x0: Tensor, sigma: Tensor, next_sigma: Tensor) -> Tensor:
    return x + edm_derivative(x, x0, sigma) * expand_to(next_sigma - sigma, x)


def edm_ab2_step(
    x: Tensor,
    d_cur: Tensor,
    d_prev: Tensor,
    sigma_cur: Tensor,
    sigma_prev: Tensor,
    sigma_next: Tensor,
) -> Tensor:
    h = sigma_next - sigma_cur
    slope = (d_cur - d_prev) / expand_to(sigma_cur - sigma_prev, d_cur)
    return x + d_cur * expand_to(h, x) + 0.5 * slope * expand_to(h, x) ** 2


def expand_to(value: Tensor, target: Tensor) -> Tensor:
    value = torch.as_tensor(value, device=target.device, dtype=target.dtype)
    if value.ndim == 0:
        return value
    return value.reshape(value.shape + (1,) * (target.ndim - value.ndim))


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))
