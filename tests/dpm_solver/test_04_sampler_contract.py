import importlib
import unittest

import torch

from tests.dpm_solver._helpers import FakeDenoiser, add_src_to_path


add_src_to_path()


class DpmSolverSamplerContractTest(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module("models.diffusion.dpm_solver_sampler")

    def test_sampler_returns_diamond_compatible_output_and_trajectory(self):
        cfg = self.mod.DpmSolverSamplerConfig(
            num_steps_denoising=3,
            sigma_min=0.1,
            sigma_max=1.0,
            rho=1,
            order=2,
            method="multistep",
        )
        denoiser = FakeDenoiser(mode="zero")
        sampler = self.mod.DpmSolverSampler(denoiser, cfg)
        prev_obs = torch.zeros(2, 4, 3, 8, 8)
        prev_act = torch.zeros(2, 4, dtype=torch.long)

        torch.manual_seed(0)
        out, trajectory = sampler.sample(prev_obs, prev_act)

        self.assertEqual(out.shape, (2, 3, 8, 8))
        self.assertEqual(len(trajectory), cfg.num_steps_denoising + 1)
        self.assertEqual(len(denoiser.calls), cfg.num_steps_denoising)
        self.assertTrue(torch.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
