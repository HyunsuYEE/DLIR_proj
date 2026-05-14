import importlib
import unittest

import torch

from tests.prg._helpers import add_src_to_path


add_src_to_path()


class PRGProbeLabelTest(unittest.TestCase):
    def setUp(self):
        self.probe = importlib.import_module("models.prg.probe")

    def test_target_drift_is_relative_mean_absolute_difference(self):
        conservative = torch.tensor([1.0, 2.0, 3.0])
        aggressive = torch.tensor([1.5, 2.5, 3.5])

        drift = self.probe.compute_target_drift(conservative, aggressive)

        expected = (aggressive - conservative).abs().mean() / conservative.abs().mean()
        self.assertTrue(torch.isclose(torch.as_tensor(drift), expected))

    def test_disagreement_rate_counts_reward_or_action_mismatch(self):
        conservative = torch.tensor([0, 1, 1, 2])
        aggressive = torch.tensor([0, 0, 1, 1])

        rate = self.probe.compute_disagreement_rate(conservative, aggressive)

        self.assertAlmostEqual(float(rate), 0.50, places=6)

    def test_probe_classifies_safe_when_control_drift_is_below_thresholds(self):
        thresholds = self.probe.ProbeThresholds(
            target_drift=0.20,
            reward_disagreement=0.25,
            action_disagreement=0.25,
        )

        result = self.probe.classify_probe(
            conservative_targets=torch.tensor([1.0, 2.0, 3.0]),
            aggressive_targets=torch.tensor([1.01, 2.01, 3.01]),
            conservative_rewards=torch.tensor([0, 1, 0, 1]),
            aggressive_rewards=torch.tensor([0, 1, 0, 1]),
            conservative_actions=torch.tensor([0, 2, 1, 3]),
            aggressive_actions=torch.tensor([0, 2, 1, 3]),
            thresholds=thresholds,
        )

        self.assertFalse(result.unsafe)
        self.assertEqual(result.reasons, [])

    def test_probe_classifies_unsafe_when_any_control_drift_exceeds_threshold(self):
        thresholds = self.probe.ProbeThresholds(
            target_drift=0.10,
            reward_disagreement=0.25,
            action_disagreement=0.25,
        )

        result = self.probe.classify_probe(
            conservative_targets=torch.tensor([1.0, 2.0, 3.0]),
            aggressive_targets=torch.tensor([1.5, 2.5, 3.5]),
            conservative_rewards=torch.tensor([0, 1, 0, 1]),
            aggressive_rewards=torch.tensor([0, 0, 0, 1]),
            conservative_actions=torch.tensor([0, 2, 1, 3]),
            aggressive_actions=torch.tensor([0, 1, 1, 0]),
            thresholds=thresholds,
        )

        self.assertTrue(result.unsafe)
        self.assertIn("target_drift", result.reasons)


if __name__ == "__main__":
    unittest.main()
