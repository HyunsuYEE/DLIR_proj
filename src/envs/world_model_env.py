import time
from dataclasses import dataclass
from typing import Any, Dict, Generator, List, Tuple

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical
from torch.utils.data import DataLoader

from coroutines import coroutine
from models.diffusion import Denoiser, DiffusionSampler, DiffusionSamplerConfig
from models.prg import normalized_rollout_depth
from models.rew_end_model import RewEndModel

ResetOutput = Tuple[torch.FloatTensor, Dict[str, Any]]
StepOutput = Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]
InitialCondition = Tuple[Tensor, Tensor, Tuple[Tensor, Tensor]]


@dataclass
class WorldModelEnvConfig:
    horizon: int
    num_batches_to_preload: int
    diffusion_sampler: DiffusionSamplerConfig


class WorldModelEnv:
    def __init__(
        self,
        denoiser: Denoiser,
        rew_end_model: RewEndModel,
        data_loader: DataLoader,
        cfg: WorldModelEnvConfig,
        return_denoising_trajectory: bool = False,
    ) -> None:
        self.sampler = DiffusionSampler(denoiser, cfg.diffusion_sampler)
        self.uses_prg = self.sampler.uses_prg
        self.rew_end_model = rew_end_model
        self.horizon = cfg.horizon
        self.return_denoising_trajectory = return_denoising_trajectory
        self.num_envs = data_loader.batch_sampler.batch_size
        self.generator_init = self.make_generator_init(data_loader, cfg.num_batches_to_preload)
        self._reset_timing_stats()

    def _reset_timing_stats(self) -> None:
        self._diffusion_time_s = 0.0
        self._diffusion_calls = 0
        self._denoiser_evals = 0
        self._teacache_full_evals = 0
        self._teacache_cache_hits = 0
        self._prg_aggressive_calls = 0
        self._prg_conservative_calls = 0
        self._prg_risk_score_sum = 0.0
        self._prg_proxy_margin_sum = 0.0
        self._rew_end_time_s = 0.0
        self._rew_end_calls = 0

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def pop_timing_stats(self) -> Dict[str, float]:
        # Synchronize once so any pending GPU work from the last timed call is reflected.
        self._sync()
        d_calls = self._diffusion_calls
        r_calls = self._rew_end_calls
        stats = {
            "diffusion_inference_total_s": self._diffusion_time_s,
            "diffusion_inference_ms_per_step": (1000.0 * self._diffusion_time_s / d_calls) if d_calls > 0 else 0.0,
            "diffusion_inference_calls": float(d_calls),
            "denoiser_evals": float(self._denoiser_evals),
            "denoiser_evals_per_diffusion_call": (self._denoiser_evals / d_calls) if d_calls > 0 else 0.0,
            "teacache_full_evals": float(self._teacache_full_evals),
            "teacache_cache_hits": float(self._teacache_cache_hits),
            "prg_aggressive_calls": float(self._prg_aggressive_calls),
            "prg_conservative_calls": float(self._prg_conservative_calls),
            "prg_risk_score_sum": float(self._prg_risk_score_sum),
            "prg_proxy_margin_sum": float(self._prg_proxy_margin_sum),
            "rew_end_inference_total_s": self._rew_end_time_s,
            "rew_end_inference_ms_per_step": (1000.0 * self._rew_end_time_s / r_calls) if r_calls > 0 else 0.0,
            "rew_end_inference_calls": float(r_calls),
        }
        self._reset_timing_stats()
        return stats

    @property
    def device(self) -> torch.device:
        return self.sampler.denoiser.device

    @torch.no_grad()
    def reset(self, **kwargs) -> ResetOutput:
        obs, act, (hx, cx) = self.generator_init.send(self.num_envs)
        self.obs_buffer = obs
        self.act_buffer = act
        self.hx_rew_end = hx
        self.cx_rew_end = cx
        self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=obs.device)
        return self.obs_buffer[:, -1], {}

    @torch.no_grad()
    def reset_dead(self, dead: torch.BoolTensor) -> None:
        obs, act, (hx, cx) = self.generator_init.send(dead.sum().item())
        self.obs_buffer[dead] = obs
        self.act_buffer[dead] = act
        self.hx_rew_end[:, dead] = hx
        self.cx_rew_end[:, dead] = cx
        self.ep_len[dead] = 0

    @torch.no_grad()
    def step(self, act: torch.LongTensor, policy_entropy_norm: Tensor | float | None = None) -> StepOutput:
        self.act_buffer[:, -1] = act

        self._sync()
        t0 = time.perf_counter()
        rollout_depth_fraction = normalized_rollout_depth(float(self.ep_len.float().mean().item()), self.horizon)
        next_obs, denoising_trajectory = self.predict_next_obs(rollout_depth_fraction, policy_entropy_norm)
        self._sync()
        self._diffusion_time_s += time.perf_counter() - t0
        self._diffusion_calls += 1
        self._denoiser_evals += self.sampler.num_denoiser_evals_last_sample
        self._teacache_full_evals += self.sampler.num_teacache_full_evals_last_sample
        self._teacache_cache_hits += self.sampler.num_teacache_cache_hits_last_sample
        self._prg_aggressive_calls += self.sampler.num_prg_aggressive_calls_last_sample
        self._prg_conservative_calls += self.sampler.num_prg_conservative_calls_last_sample
        self._prg_risk_score_sum += self.sampler.prg_risk_score_last_sample
        self._prg_proxy_margin_sum += self.sampler.prg_proxy_margin_last_sample

        t0 = time.perf_counter()
        rew, end = self.predict_rew_end(next_obs.unsqueeze(1))
        self._sync()
        self._rew_end_time_s += time.perf_counter() - t0
        self._rew_end_calls += 1

        self.ep_len += 1
        trunc = (self.ep_len >= self.horizon).long()

        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
        self.act_buffer = self.act_buffer.roll(-1, dims=1)
        self.obs_buffer[:, -1] = next_obs

        dead = torch.logical_or(end, trunc)

        info = {}
        if self.return_denoising_trajectory:
            info["denoising_trajectory"] = torch.stack(denoising_trajectory, dim=1)

        if dead.any():
            self.reset_dead(dead)
            info["final_observation"] = next_obs[dead]
            info["burnin_obs"] = self.obs_buffer[dead, :-1]

        return self.obs_buffer[:, -1], rew, end, trunc, info

    @torch.no_grad()
    def predict_next_obs(
        self,
        rollout_depth_fraction: float = 0.0,
        policy_entropy_norm: Tensor | float | None = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        return self.sampler.sample(self.obs_buffer, self.act_buffer, rollout_depth_fraction, policy_entropy_norm)

    @torch.no_grad()
    def predict_rew_end(self, next_obs: Tensor) -> Tuple[Tensor, Tensor]:
        logits_rew, logits_end, (self.hx_rew_end, self.cx_rew_end) = self.rew_end_model.predict_rew_end(
            self.obs_buffer[:, -1:],
            self.act_buffer[:, -1:],
            next_obs,
            (self.hx_rew_end, self.cx_rew_end),
        )
        rew = Categorical(logits=logits_rew).sample().squeeze(1) - 1.0  # in {-1, 0, 1}
        end = Categorical(logits=logits_end).sample().squeeze(1)
        return rew, end

    @coroutine
    def make_generator_init(
        self,
        data_loader: DataLoader,
        num_batches_to_preload: int,
    ) -> Generator[InitialCondition, None, None]:
        num_dead = yield
        data_iterator = iter(data_loader)

        while True:
            # Preload on device and burnin rew/end model
            obs_, act_, hx_, cx_ = [], [], [], []
            for _ in range(num_batches_to_preload):
                batch = next(data_iterator)
                obs = batch.obs.to(self.device)
                act = batch.act.to(self.device)
                with torch.no_grad():
                    *_, (hx, cx) = self.rew_end_model.predict_rew_end(obs[:, :-1], act[:, :-1], obs[:, 1:])  # Burn-in of rew/end model
                assert hx.size(0) == cx.size(0) == 1
                obs_.extend(list(obs))
                act_.extend(list(act))
                hx_.extend(list(hx[0]))
                cx_.extend(list(cx[0]))

            # Yield new initial conditions for dead envs
            c = 0
            while c + num_dead <= len(obs_):
                obs = torch.stack(obs_[c : c + num_dead])
                act = torch.stack(act_[c : c + num_dead])
                hx = torch.stack(hx_[c : c + num_dead]).unsqueeze(0)
                cx = torch.stack(cx_[c : c + num_dead]).unsqueeze(0)
                c += num_dead
                num_dead = yield obs, act, (hx, cx)
