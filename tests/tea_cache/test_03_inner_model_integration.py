import importlib
import unittest

import torch

from tests.tea_cache._helpers import add_src_to_path


add_src_to_path()

from models.diffusion.inner_model import InnerModel, InnerModelConfig


class InnerModelTeaCacheIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.teacache = importlib.import_module("models.diffusion.teacache")
        torch.manual_seed(0)
        cfg = InnerModelConfig(
            img_channels=3,
            num_steps_conditioning=2,
            cond_channels=8,
            depths=[1, 1],
            channels=[8, 8],
            attn_depths=[0, 0],
            num_actions=4,
        )
        self.model = InnerModel(cfg).eval()

    def _inputs(self):
        noisy_next_obs = torch.randn(2, 3, 8, 8)
        c_noise = torch.tensor([0.1, 0.2])
        obs = torch.randn(2, 6, 8, 8)
        act = torch.zeros(2, 2, dtype=torch.long)
        return noisy_next_obs, c_noise, obs, act

    def test_disabled_teacache_matches_regular_forward(self):
        args = self._inputs()
        state = self.teacache.TeaCacheState(rel_l1_thresh=1.0, enabled=False)

        regular = self.model(*args)
        cached = self.model.forward_teacache(*args, teacache_state=state, step_index=0, num_steps=3)

        self.assertTrue(torch.allclose(cached, regular))
        self.assertEqual(state.stats.full_evals, 0)
        self.assertEqual(state.stats.cache_hits, 0)

    def test_enabled_teacache_tracks_full_eval_and_cache_hit(self):
        args = self._inputs()
        state = self.teacache.TeaCacheState(rel_l1_thresh=10.0, force_last=False)

        _ = self.model.forward_teacache(*args, teacache_state=state, step_index=0, num_steps=3)
        _ = self.model.forward_teacache(*args, teacache_state=state, step_index=1, num_steps=3)

        self.assertEqual(state.stats.full_evals, 1)
        self.assertEqual(state.stats.cache_hits, 1)


if __name__ == "__main__":
    unittest.main()

