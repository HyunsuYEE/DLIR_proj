import importlib
import unittest

import torch

from tests.dpm_solver._helpers import add_src_to_path


add_src_to_path()


class MultistepUpdateTest(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module("models.diffusion.dpm_solver_sampler")

    def test_ab2_reduces_to_euler_for_constant_derivative(self):
        x = torch.tensor([[[[10.0]]]])
        d_prev = torch.tensor([[[[3.0]]]])
        d_cur = torch.tensor([[[[3.0]]]])

        out = self.mod.edm_ab2_step(
            x=x,
            d_cur=d_cur,
            d_prev=d_prev,
            sigma_cur=torch.tensor(2.0),
            sigma_prev=torch.tensor(4.0),
            sigma_next=torch.tensor(1.0),
        )

        self.assertTrue(torch.allclose(out, torch.tensor([[[[7.0]]]])))

    def test_ab2_integrates_linear_derivative_exactly(self):
        # f(sigma) = 2 * sigma + 1. Integral from 2 to 1 is -4.
        x = torch.tensor([[[[10.0]]]])
        d_prev = torch.tensor([[[[9.0]]]])  # f(4)
        d_cur = torch.tensor([[[[5.0]]]])  # f(2)

        out = self.mod.edm_ab2_step(
            x=x,
            d_cur=d_cur,
            d_prev=d_prev,
            sigma_cur=torch.tensor(2.0),
            sigma_prev=torch.tensor(4.0),
            sigma_next=torch.tensor(1.0),
        )

        self.assertTrue(torch.allclose(out, torch.tensor([[[[6.0]]]])))


if __name__ == "__main__":
    unittest.main()
