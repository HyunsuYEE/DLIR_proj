import importlib
import unittest

import torch

from tests.tea_cache._helpers import CountingBlock, add_src_to_path


add_src_to_path()


class ResidualCacheTest(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module("models.diffusion.teacache")

    def test_full_compute_stores_residual_and_updates_stats(self):
        state = self.mod.TeaCacheState(rel_l1_thresh=0.5)
        block = CountingBlock(residual_scale=3.0)
        x = torch.ones(1, 2, 3, 3)
        cond = torch.zeros(1, 4)
        proxy = torch.ones(1, 4)

        out = self.mod.apply_teacache_residual(
            x=x,
            cond=cond,
            proxy=proxy,
            expensive_block=block,
            state=state,
            step_index=0,
            num_steps=3,
        )

        self.assertTrue(torch.allclose(out, x * 3.0))
        self.assertTrue(torch.allclose(state.previous_residual, x * 2.0))
        self.assertEqual(block.calls, 1)
        self.assertEqual(state.stats.full_evals, 1)
        self.assertEqual(state.stats.cache_hits, 0)

    def test_cache_hit_skips_expensive_block_and_reuses_residual(self):
        state = self.mod.TeaCacheState(rel_l1_thresh=10.0, force_last=False)
        block = CountingBlock(residual_scale=3.0)
        x = torch.ones(1, 2, 3, 3)
        cond = torch.zeros(1, 4)

        self.mod.apply_teacache_residual(x, cond, torch.ones(1, 4), block, state, step_index=0, num_steps=3)
        out = self.mod.apply_teacache_residual(x * 2.0, cond, torch.ones(1, 4) * 1.01, block, state, step_index=1, num_steps=3)

        self.assertTrue(torch.allclose(out, x * 2.0 + x * 2.0))
        self.assertEqual(block.calls, 1)
        self.assertEqual(state.stats.full_evals, 1)
        self.assertEqual(state.stats.cache_hits, 1)


if __name__ == "__main__":
    unittest.main()

