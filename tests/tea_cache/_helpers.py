from pathlib import Path
import sys

import torch
from torch import nn


def add_src_to_path() -> None:
    src_dir = Path(__file__).resolve().parents[2] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


class CountingBlock(nn.Module):
    def __init__(self, residual_scale: float = 2.0) -> None:
        super().__init__()
        self.residual_scale = residual_scale
        self.calls = 0

    def forward(self, x, cond):
        self.calls += 1
        return x * self.residual_scale


def make_proxy(values):
    return torch.tensor(values, dtype=torch.float32).reshape(1, -1)

