from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.distributions.categorical import Categorical


@dataclass(frozen=True)
class RolloutRiskInputs:
    depth_fraction: float
    normalized_policy_entropy: float
    proxy_margin: float


@dataclass(frozen=True)
class PRGRiskWeights:
    depth: float = 1.0
    policy: float = 1.0
    proxy: float = 1.0


def normalized_rollout_depth(step_index: int | float, horizon: int) -> float:
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if horizon == 1:
        return 1.0
    return _clamp01(float(step_index) / float(horizon - 1))


def normalized_policy_entropy(logits: Tensor) -> Tensor:
    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension.")
    num_actions = logits.size(-1)
    if num_actions <= 1:
        return torch.zeros((), device=logits.device, dtype=logits.dtype)
    entropy = Categorical(logits=logits).entropy()
    return (entropy / math.log(num_actions)).mean().clamp(0.0, 1.0)


def policy_entropy_risk(logits: Tensor) -> Tensor:
    return 1.0 - normalized_policy_entropy(logits)


def proxy_margin(current_proxy: Tensor, previous_proxy: Tensor, eps: float = 1e-6) -> Tensor:
    denom = previous_proxy.abs().mean().clamp_min(eps)
    return (current_proxy - previous_proxy).abs().mean() / denom


def compute_risk_score(inputs: RolloutRiskInputs, weights: PRGRiskWeights) -> float:
    depth_risk = _clamp01(inputs.depth_fraction)
    policy_risk = _clamp01(1.0 - inputs.normalized_policy_entropy)
    proxy_risk = _clamp01(inputs.proxy_margin)

    total_weight = weights.depth + weights.policy + weights.proxy
    if total_weight <= 0:
        raise ValueError("risk weights must sum to a positive value.")

    score = (
        weights.depth * depth_risk
        + weights.policy * policy_risk
        + weights.proxy * proxy_risk
    ) / total_weight
    return _clamp01(score)


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))
