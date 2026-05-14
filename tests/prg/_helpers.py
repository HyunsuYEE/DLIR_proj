from pathlib import Path
import sys

import torch


def add_src_to_path() -> None:
    src_dir = Path(__file__).resolve().parents[2] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def uniform_logits(batch_size: int = 2, num_actions: int = 4) -> torch.Tensor:
    return torch.zeros(batch_size, num_actions)


def peaked_logits(batch_size: int = 2, num_actions: int = 4, peak: float = 20.0) -> torch.Tensor:
    logits = torch.zeros(batch_size, num_actions)
    logits[:, 0] = peak
    return logits
