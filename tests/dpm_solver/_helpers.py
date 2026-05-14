from pathlib import Path
import sys

import torch


def add_src_to_path() -> None:
    src_dir = Path(__file__).resolve().parents[2] / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


class FakeDenoiser:
    def __init__(self, mode: str = "zero", scale: float = 0.5, device: str = "cpu") -> None:
        self.mode = mode
        self.scale = scale
        self.device = torch.device(device)
        self.calls = []

    def denoise(self, noisy_next_obs, sigma, obs, act):
        self.calls.append(
            {
                "x_shape": tuple(noisy_next_obs.shape),
                "sigma_shape": tuple(sigma.shape) if isinstance(sigma, torch.Tensor) else (),
                "obs_shape": tuple(obs.shape),
                "act_shape": tuple(act.shape),
            }
        )
        if self.mode == "zero":
            return torch.zeros_like(noisy_next_obs)
        if self.mode == "identity":
            return noisy_next_obs
        if self.mode == "scale":
            return noisy_next_obs * self.scale
        raise ValueError(f"unknown FakeDenoiser mode: {self.mode}")

