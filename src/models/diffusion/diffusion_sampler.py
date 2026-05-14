from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor

from .denoiser import Denoiser
from .dpm_solver_sampler import DpmSolverSampler, DpmSolverSamplerConfig
from .teacache import TeaCacheState
from ..prg import PRGGateConfig, PRGRiskWeights, RolloutRiskInputs, make_prg_decision, proxy_margin


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1
    solver_type: str = "euler_heun"
    dpm_solver_order: int = 2
    dpm_solver_method: str = "multistep"
    teacache_enabled: bool = False
    teacache_rel_l1_thresh: float = 0.2
    teacache_force_last: bool = True
    prg_risk_threshold: float = 0.5
    prg_depth_weight: float = 1.0
    prg_policy_weight: float = 1.0
    prg_proxy_weight: float = 1.0


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig) -> None:
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser.device)
        self.num_denoiser_evals_last_sample = 0
        self.num_teacache_full_evals_last_sample = 0
        self.num_teacache_cache_hits_last_sample = 0
        self.num_prg_aggressive_calls_last_sample = 0
        self.num_prg_conservative_calls_last_sample = 0
        self.prg_risk_score_last_sample = 0.0
        self.prg_proxy_margin_last_sample = 0.0
        self.dpm_solver_sampler = None
        self.prg_conservative_sampler = None
        self.prg_aggressive_sampler = None
        self._prg_previous_proxy = None
        self.uses_prg = cfg.solver_type == "prg"
        if cfg.solver_type == "dpm_solver":
            dpm_cfg = DpmSolverSamplerConfig(
                num_steps_denoising=cfg.num_steps_denoising,
                sigma_min=cfg.sigma_min,
                sigma_max=cfg.sigma_max,
                rho=cfg.rho,
                order=cfg.dpm_solver_order,
                method=cfg.dpm_solver_method,
                teacache_enabled=cfg.teacache_enabled,
                teacache_rel_l1_thresh=cfg.teacache_rel_l1_thresh,
                teacache_force_last=cfg.teacache_force_last,
            )
            self.dpm_solver_sampler = DpmSolverSampler(denoiser, dpm_cfg)
        elif cfg.solver_type == "prg":
            conservative_cfg = DpmSolverSamplerConfig(
                num_steps_denoising=cfg.num_steps_denoising,
                sigma_min=cfg.sigma_min,
                sigma_max=cfg.sigma_max,
                rho=cfg.rho,
                order=cfg.dpm_solver_order,
                method=cfg.dpm_solver_method,
                teacache_enabled=False,
            )
            aggressive_cfg = DpmSolverSamplerConfig(
                num_steps_denoising=cfg.num_steps_denoising,
                sigma_min=cfg.sigma_min,
                sigma_max=cfg.sigma_max,
                rho=cfg.rho,
                order=cfg.dpm_solver_order,
                method=cfg.dpm_solver_method,
                teacache_enabled=True,
                teacache_rel_l1_thresh=cfg.teacache_rel_l1_thresh,
                teacache_force_last=cfg.teacache_force_last,
            )
            self.prg_conservative_sampler = DpmSolverSampler(denoiser, conservative_cfg)
            self.prg_aggressive_sampler = DpmSolverSampler(denoiser, aggressive_cfg)
        elif cfg.solver_type != "euler_heun":
            raise ValueError(f"Unsupported diffusion solver_type: {cfg.solver_type}")

    @torch.no_grad()
    def sample(
        self,
        prev_obs: Tensor,
        prev_act: Tensor,
        rollout_depth_fraction: float = 0.0,
        policy_entropy_norm: float | Tensor | None = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        if self.dpm_solver_sampler is not None:
            out, trajectory = self.dpm_solver_sampler.sample(prev_obs, prev_act)
            self.num_denoiser_evals_last_sample = self.dpm_solver_sampler.num_denoiser_evals_last_sample
            self.num_teacache_full_evals_last_sample = self.dpm_solver_sampler.num_teacache_full_evals_last_sample
            self.num_teacache_cache_hits_last_sample = self.dpm_solver_sampler.num_teacache_cache_hits_last_sample
            self.num_prg_aggressive_calls_last_sample = 0
            self.num_prg_conservative_calls_last_sample = 0
            self.prg_risk_score_last_sample = 0.0
            self.prg_proxy_margin_last_sample = 0.0
            return out, trajectory

        if self.prg_conservative_sampler is not None and self.prg_aggressive_sampler is not None:
            decision = self._make_prg_decision(prev_obs, prev_act, rollout_depth_fraction, policy_entropy_norm)
            sampler = self.prg_aggressive_sampler if decision.mode == "aggressive" else self.prg_conservative_sampler
            out, trajectory = sampler.sample(prev_obs, prev_act)
            self.num_denoiser_evals_last_sample = sampler.num_denoiser_evals_last_sample
            self.num_teacache_full_evals_last_sample = sampler.num_teacache_full_evals_last_sample
            self.num_teacache_cache_hits_last_sample = sampler.num_teacache_cache_hits_last_sample
            self.num_prg_aggressive_calls_last_sample = int(decision.mode == "aggressive")
            self.num_prg_conservative_calls_last_sample = int(decision.mode == "conservative")
            self.prg_risk_score_last_sample = float(decision.risk_score)
            self.prg_proxy_margin_last_sample = float(decision.inputs.proxy_margin)
            return out, trajectory

        device = prev_obs.device
        self.num_denoiser_evals_last_sample = 0
        self.num_teacache_full_evals_last_sample = 0
        self.num_teacache_cache_hits_last_sample = 0
        self.num_prg_aggressive_calls_last_sample = 0
        self.num_prg_conservative_calls_last_sample = 0
        self.prg_risk_score_last_sample = 0.0
        self.prg_proxy_margin_last_sample = 0.0
        b, t, c, h, w = prev_obs.size()
        prev_obs = prev_obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        x = torch.randn(b, c, h, w, device=device)
        trajectory = [x]
        teacache_state = self._make_teacache_state()
        num_steps = len(self.sigmas) - 1
        for step_index, (sigma, next_sigma) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x) * self.cfg.s_noise
                x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
            denoised = self._denoise(x, sigma, prev_obs, prev_act, teacache_state, step_index, num_steps)
            self.num_denoiser_evals_last_sample += 1
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.cfg.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                denoised_2 = self._denoise(
                    x_2,
                    next_sigma * s_in,
                    prev_obs,
                    prev_act,
                    teacache_state,
                    step_index,
                    num_steps,
                )
                self.num_denoiser_evals_last_sample += 1
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
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

    def _make_prg_decision(
        self,
        prev_obs: Tensor,
        prev_act: Tensor,
        rollout_depth_fraction: float,
        policy_entropy_norm: float | Tensor | None,
    ):
        current_proxy = self._build_prg_proxy(prev_obs, prev_act)
        if self._prg_previous_proxy is None or self._prg_previous_proxy.shape != current_proxy.shape:
            proxy_margin_value = 0.0
        else:
            proxy_margin_value = float(proxy_margin(current_proxy, self._prg_previous_proxy).detach().cpu().item())
        self._prg_previous_proxy = current_proxy.detach()

        entropy_value = 1.0
        if policy_entropy_norm is not None:
            entropy_tensor = torch.as_tensor(policy_entropy_norm)
            entropy_value = float(entropy_tensor.detach().float().mean().cpu().item())

        inputs = RolloutRiskInputs(
            depth_fraction=float(rollout_depth_fraction),
            normalized_policy_entropy=entropy_value,
            proxy_margin=proxy_margin_value,
        )
        weights = PRGRiskWeights(
            depth=self.cfg.prg_depth_weight,
            policy=self.cfg.prg_policy_weight,
            proxy=self.cfg.prg_proxy_weight,
        )
        gate_cfg = PRGGateConfig(risk_threshold=self.cfg.prg_risk_threshold)
        return make_prg_decision(inputs, weights, gate_cfg)

    @staticmethod
    def _build_prg_proxy(prev_obs: Tensor, prev_act: Tensor) -> Tensor:
        obs = prev_obs.float()
        act = prev_act.float()
        obs_mean = obs.mean(dim=(1, 2, 3, 4), keepdim=False).unsqueeze(1)
        obs_std = obs.std(dim=(1, 2, 3, 4), unbiased=False, keepdim=False).unsqueeze(1)
        act_mean = act.mean(dim=1, keepdim=True)
        act_std = act.std(dim=1, unbiased=False, keepdim=True)
        return torch.cat((obs_mean, obs_std, act_mean, act_std), dim=1)

    def _denoise(
        self,
        x: Tensor,
        sigma: Tensor,
        prev_obs: Tensor,
        prev_act: Tensor,
        teacache_state: TeaCacheState | None,
        step_index: int,
        num_steps: int,
    ) -> Tensor:
        if teacache_state is None:
            return self.denoiser.denoise(x, sigma, prev_obs, prev_act)
        return self.denoiser.denoise_teacache(
            x,
            sigma,
            prev_obs,
            prev_act,
            teacache_state=teacache_state,
            step_index=step_index,
            num_steps=num_steps,
        )


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))
