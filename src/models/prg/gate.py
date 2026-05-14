from dataclasses import dataclass

from .risk import PRGRiskWeights, RolloutRiskInputs, compute_risk_score


@dataclass(frozen=True)
class PRGGateConfig:
    risk_threshold: float = 0.5


@dataclass(frozen=True)
class PRGDecision:
    mode: str
    risk_score: float
    inputs: RolloutRiskInputs


def select_mode(risk_score: float, cfg: PRGGateConfig) -> str:
    return "aggressive" if risk_score <= cfg.risk_threshold else "conservative"


def make_prg_decision(inputs: RolloutRiskInputs, weights: PRGRiskWeights, cfg: PRGGateConfig) -> PRGDecision:
    score = compute_risk_score(inputs, weights)
    return PRGDecision(mode=select_mode(score, cfg), risk_score=score, inputs=inputs)
