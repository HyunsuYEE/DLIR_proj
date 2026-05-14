import unittest

from tests.dpm_solver._helpers import add_src_to_path


add_src_to_path()

from models.diffusion.diffusion_sampler import DiffusionSamplerConfig


class DiffusionSamplerModeDispatchTest(unittest.TestCase):
    def test_config_accepts_dpm_solver_backend_selector(self):
        cfg = DiffusionSamplerConfig(
            num_steps_denoising=3,
            solver_type="dpm_solver",
            dpm_solver_order=2,
            dpm_solver_method="multistep",
        )

        self.assertEqual(cfg.solver_type, "dpm_solver")
        self.assertEqual(cfg.dpm_solver_order, 2)
        self.assertEqual(cfg.dpm_solver_method, "multistep")


if __name__ == "__main__":
    unittest.main()
