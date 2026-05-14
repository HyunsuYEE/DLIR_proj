from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
from torch import Tensor


@dataclass
class TeaCacheDecision:
    should_compute: bool
    reason: str


@dataclass
class TeaCacheStats:
    full_evals: int = 0
    cache_hits: int = 0


@dataclass
class TeaCacheState:
    rel_l1_thresh: float
    force_last: bool = True
    enabled: bool = True
    eps: float = 1e-6
    previous_proxy: Optional[Tensor] = None
    previous_residual: Optional[Tensor] = None
    accumulated_rel_l1: float = 0.0
    stats: TeaCacheStats = field(default_factory=TeaCacheStats)

    def reset(self) -> None:
        self.previous_proxy = None
        self.previous_residual = None
        self.accumulated_rel_l1 = 0.0
        self.stats = TeaCacheStats()

    def update_and_decide(self, proxy: Tensor, step_index: int, num_steps: int) -> TeaCacheDecision:
        proxy = proxy.detach()
        if self.previous_proxy is None:
            self.previous_proxy = proxy
            self.accumulated_rel_l1 = 0.0
            return TeaCacheDecision(should_compute=True, reason="first")

        rel_l1 = relative_l1(proxy, self.previous_proxy, eps=self.eps).detach().item()
        self.accumulated_rel_l1 += rel_l1
        self.previous_proxy = proxy

        if self.force_last and step_index == num_steps - 1:
            self.accumulated_rel_l1 = 0.0
            return TeaCacheDecision(should_compute=True, reason="last")

        if self.accumulated_rel_l1 >= self.rel_l1_thresh:
            self.accumulated_rel_l1 = 0.0
            return TeaCacheDecision(should_compute=True, reason="threshold")

        return TeaCacheDecision(should_compute=False, reason="cache")


def relative_l1(current: Tensor, previous: Tensor, eps: float = 1e-6) -> Tensor:
    denom = previous.abs().mean().clamp_min(eps)
    return (current - previous).abs().mean() / denom


def apply_teacache_residual(
    x: Tensor,
    cond: Tensor,
    proxy: Tensor,
    expensive_block: Callable[[Tensor, Tensor], Tensor],
    state: TeaCacheState,
    step_index: int,
    num_steps: int,
) -> Tensor:
    if not state.enabled:
        return expensive_block(x, cond)

    decision = state.update_and_decide(proxy, step_index=step_index, num_steps=num_steps)
    if not decision.should_compute and state.previous_residual is not None:
        state.stats.cache_hits += 1
        return x + state.previous_residual.to(device=x.device, dtype=x.dtype)

    out = expensive_block(x, cond)
    state.previous_residual = (out - x).detach()
    state.stats.full_evals += 1
    return out
