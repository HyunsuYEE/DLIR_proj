import torch
from torch import Tensor


def calibrate_risk_threshold(
    risk_scores: Tensor,
    unsafe_labels: Tensor,
    max_false_safe_rate: float = 0.0,
) -> float:
    if risk_scores.numel() == 0:
        raise ValueError("risk_scores must not be empty.")
    if risk_scores.shape != unsafe_labels.shape:
        raise ValueError("risk_scores and unsafe_labels must have the same shape.")
    if not 0.0 <= max_false_safe_rate <= 1.0:
        raise ValueError("max_false_safe_rate must be in [0, 1].")

    scores = risk_scores.detach().flatten().float().cpu()
    unsafe = unsafe_labels.detach().flatten().bool().cpu()
    candidates = torch.unique(scores).sort().values

    best = float(candidates[0].item())
    for threshold in candidates:
        selected_aggressive = scores <= threshold
        unsafe_total = int(unsafe.sum().item())
        false_safe = int((selected_aggressive & unsafe).sum().item())
        false_safe_rate = 0.0 if unsafe_total == 0 else false_safe / unsafe_total
        if false_safe_rate <= max_false_safe_rate:
            best = float(threshold.item())
        else:
            break
    return best
