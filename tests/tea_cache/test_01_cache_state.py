import importlib
import unittest

import torch

from tests.tea_cache._helpers import make_proxy, add_src_to_path


add_src_to_path()


class TeaCacheStateTest(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module("models.diffusion.teacache")

    def test_first_step_forces_full_compute(self):
        state = self.mod.TeaCacheState(rel_l1_thresh=0.5)

        decision = state.update_and_decide(make_proxy([1.0, 2.0]), step_index=0, num_steps=4)

        self.assertTrue(decision.should_compute)
        self.assertEqual(decision.reason, "first")
        self.assertEqual(state.accumulated_rel_l1, 0.0)

    def test_small_accumulated_change_hits_cache(self):
        state = self.mod.TeaCacheState(rel_l1_thresh=0.5)
        state.update_and_decide(make_proxy([1.0, 1.0]), step_index=0, num_steps=4)

        decision = state.update_and_decide(make_proxy([1.1, 1.1]), step_index=1, num_steps=4)

        self.assertFalse(decision.should_compute)
        self.assertEqual(decision.reason, "cache")
        self.assertGreater(state.accumulated_rel_l1, 0.0)

    def test_large_accumulated_change_refreshes_cache(self):
        state = self.mod.TeaCacheState(rel_l1_thresh=0.05)
        state.update_and_decide(make_proxy([1.0, 1.0]), step_index=0, num_steps=4)

        decision = state.update_and_decide(make_proxy([1.2, 1.2]), step_index=1, num_steps=4)

        self.assertTrue(decision.should_compute)
        self.assertEqual(decision.reason, "threshold")
        self.assertEqual(state.accumulated_rel_l1, 0.0)

    def test_last_step_can_force_full_compute(self):
        state = self.mod.TeaCacheState(rel_l1_thresh=10.0, force_last=True)
        state.update_and_decide(make_proxy([1.0, 1.0]), step_index=0, num_steps=3)

        decision = state.update_and_decide(make_proxy([1.01, 1.01]), step_index=2, num_steps=3)

        self.assertTrue(decision.should_compute)
        self.assertEqual(decision.reason, "last")

    def test_reset_clears_cached_proxy_and_residual(self):
        state = self.mod.TeaCacheState(rel_l1_thresh=0.5)
        state.previous_proxy = torch.ones(1, 2)
        state.previous_residual = torch.ones(1, 2, 1, 1)
        state.accumulated_rel_l1 = 3.0

        state.reset()

        self.assertIsNone(state.previous_proxy)
        self.assertIsNone(state.previous_residual)
        self.assertEqual(state.accumulated_rel_l1, 0.0)


if __name__ == "__main__":
    unittest.main()

