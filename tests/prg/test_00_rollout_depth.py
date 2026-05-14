import importlib
import unittest

from tests.prg._helpers import add_src_to_path


add_src_to_path()


class RolloutDepthRiskTest(unittest.TestCase):
    def setUp(self):
        self.risk = importlib.import_module("models.prg.risk")

    def test_depth_fraction_is_zero_at_first_step_and_one_at_last_step(self):
        self.assertEqual(self.risk.normalized_rollout_depth(step_index=0, horizon=5), 0.0)
        self.assertEqual(self.risk.normalized_rollout_depth(step_index=4, horizon=5), 1.0)

    def test_depth_fraction_is_monotonic_and_clamped(self):
        early = self.risk.normalized_rollout_depth(step_index=1, horizon=5)
        late = self.risk.normalized_rollout_depth(step_index=3, horizon=5)

        self.assertLess(early, late)
        self.assertEqual(self.risk.normalized_rollout_depth(step_index=-3, horizon=5), 0.0)
        self.assertEqual(self.risk.normalized_rollout_depth(step_index=99, horizon=5), 1.0)

    def test_single_step_horizon_is_max_depth(self):
        self.assertEqual(self.risk.normalized_rollout_depth(step_index=0, horizon=1), 1.0)

    def test_invalid_horizon_raises(self):
        with self.assertRaises(ValueError):
            self.risk.normalized_rollout_depth(step_index=0, horizon=0)


if __name__ == "__main__":
    unittest.main()
