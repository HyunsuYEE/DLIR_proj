import unittest

import torch

from tests.dpm_solver._helpers import add_src_to_path


add_src_to_path()

from models.diffusion.diffusion_sampler import build_sigmas


class KarrasSigmasTest(unittest.TestCase):
    def test_build_sigmas_has_expected_endpoints_and_order(self):
        sigmas = build_sigmas(
            num_steps=5,
            sigma_min=0.002,
            sigma_max=5.0,
            rho=7,
            device=torch.device("cpu"),
        )

        self.assertEqual(sigmas.shape, (6,))
        self.assertTrue(torch.isclose(sigmas[0], torch.tensor(5.0), rtol=1e-6, atol=1e-6))
        self.assertTrue(torch.isclose(sigmas[-2], torch.tensor(0.002), rtol=1e-5, atol=1e-7))
        self.assertEqual(sigmas[-1].item(), 0.0)
        self.assertTrue(torch.all(sigmas[:-1].diff() < 0))

    def test_build_sigmas_preserves_device(self):
        sigmas = build_sigmas(3, 0.01, 1.0, 7, torch.device("cpu"))

        self.assertEqual(sigmas.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
