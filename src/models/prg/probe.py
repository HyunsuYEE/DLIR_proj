from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ProbeThresholds:
    target_drift: float
    reward_disagreement: float
    action_disagreement: float


@dataclass(frozen=True)
class ProbeResult:
    unsafe: bool
    target_drift: float
    reward_disagreement: float
    action_disagreement: float
    reasons: list[str]


def compute_target_drift(conservative_targets: Tensor, aggressive_targets: Tensor, eps: float = 1e-6) -> Tensor:
    denom = conservative_targets.abs().mean().clamp_min(eps)
    return (aggressive_targets - conservative_targets).abs().mean() / denom


def compute_disagreement_rate(conservative: Tensor, aggressive: Tensor) -> Tensor:
    if conservative.shape != aggressive.shape:
        raise ValueError("disagreement tensors must have the same shape.")
    return (conservative != aggressive).float().mean()


def classify_probe(
    conservative_targets: Tensor,
    aggressive_targets: Tensor,
    conservative_rewards: Tensor,
    aggressive_rewards: Tensor,
    conservative_actions: Tensor,
    aggressive_actions: Tensor,
    thresholds: ProbeThresholds,
) -> ProbeResult:
    target_drift = float(compute_target_drift(conservative_targets, aggressive_targets).detach().cpu().item())
    reward_disagreement = float(
        compute_disagreement_rate(conservative_rewards, aggressive_rewards).detach().cpu().item()
    )
    action_disagreement = float(
        compute_disagreement_rate(conservative_actions, aggressive_actions).detach().cpu().item()
    )

    reasons = []
    if target_drift > thresholds.target_drift:
        reasons.append("target_drift")
    if reward_disagreement > thresholds.reward_disagreement:
        reasons.append("reward_disagreement")
    if action_disagreement > thresholds.action_disagreement:
        reasons.append("action_disagreement")

    return ProbeResult(
        unsafe=bool(reasons),
        target_drift=target_drift,
        reward_disagreement=reward_disagreement,
        action_disagreement=action_disagreement,
        reasons=reasons,
    )
