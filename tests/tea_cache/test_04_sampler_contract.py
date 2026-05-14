import importlib
import unittest

import torch

from tests.tea_cache._helpers import add_src_to_path


add_src_to_path()


class TeaCacheSamplerContractTest(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module("models.diffusion.diffusion_sampler")

    def test_diffusion_config_accepts_teacache_options(self):
        cfg = self.mod.DiffusionSamplerConfig(
            num_steps_denoising=3,
            solver_type="dpm_solver",
            dpm_solver_order=2,
            teacache_enabled=True,
            teacache_rel_l1_thresh=0.2,
            teacache_force_last=False,
        )

        self.assertTrue(cfg.teacache_enabled)
        self.assertEqual(cfg.teacache_rel_l1_thresh, 0.2)
        self.assertFalse(cfg.teacache_force_last)


if __name__ == "__main__":
    unittest.main()

