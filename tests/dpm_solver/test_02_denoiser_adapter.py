import importlib
import unittest

import torch

from tests.dpm_solver._helpers import FakeDenoiser, add_src_to_path


add_src_to_path()


class DenoiserAdapterTest(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module("models.diffusion.dpm_solver_sampler")

    def test_predict_x0_delegates_to_diamond_denoiser(self):
        denoiser = FakeDenoiser(mode="scale", scale=0.25)
        obs = torch.zeros(2, 12, 8, 8)
        act = torch.zeros(2, 4, dtype=torch.long)
        adapter = self.mod.DenoiserDpmAdapter(denoiser, obs, act)
        x = torch.ones(2, 3, 8, 8)
        sigma = torch.tensor([1.0, 0.5])

        x0 = adapter.predict_x0(x, sigma)

        self.assertTrue(torch.allclose(x0, x * 0.25))
        self.assertEqual(len(denoiser.calls), 1)
        self.assertEqual(denoiser.calls[0]["x_shape"], tuple(x.shape))
        self.assertEqual(denoiser.calls[0]["obs_shape"], tuple(obs.shape))
        self.assertEqual(denoiser.calls[0]["act_shape"], tuple(act.shape))

    def test_derivative_uses_adapter_prediction(self):
        denoiser = FakeDenoiser(mode="zero")
        adapter = self.mod.DenoiserDpmAdapter(
            denoiser,
            obs=torch.zeros(1, 12, 4, 4),
            act=torch.zeros(1, 4, dtype=torch.long),
        )
        x = torch.full((1, 3, 4, 4), 2.0)
        sigma = torch.tensor([4.0])

        derivative = adapter.derivative(x, sigma)

        self.assertTrue(torch.allclose(derivative, torch.full_like(x, 0.5)))


if __name__ == "__main__":
    unittest.main()
