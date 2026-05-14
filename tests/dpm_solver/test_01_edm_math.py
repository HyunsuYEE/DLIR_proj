import importlib
import unittest

import torch

from tests.dpm_solver._helpers import add_src_to_path


add_src_to_path()


class EdmMathTest(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module("models.diffusion.dpm_solver_sampler")

    def test_edm_derivative_matches_current_diamond_ode(self):
        x = torch.tensor([[[[4.0]]], [[[9.0]]]])
        x0 = torch.tensor([[[[1.0]]], [[[1.0]]]])
        sigma = torch.tensor([3.0, 4.0])

        derivative = self.mod.edm_derivative(x, x0, sigma)

        expected = torch.tensor([[[[1.0]]], [[[2.0]]]])
        self.assertTrue(torch.allclose(derivative, expected))

    def test_edm_euler_step_matches_existing_sampler_update(self):
        x = torch.tensor([[[[4.0]]]])
        x0 = torch.tensor([[[[1.0]]]])
        sigma = torch.tensor(3.0)
        next_sigma = torch.tensor(1.0)

        out = self.mod.edm_euler_step(x, x0, sigma, next_sigma)

        expected = x + (x - x0) / sigma * (next_sigma - sigma)
        self.assertTrue(torch.allclose(out, expected))


if __name__ == "__main__":
    unittest.main()
