import importlib
import unittest

import torch

from tests.prg._helpers import add_src_to_path, peaked_logits, uniform_logits


add_src_to_path()


class PolicyEntropyRiskTest(unittest.TestCase):
    def setUp(self):
        self.risk = importlib.import_module("models.prg.risk")

    def test_uniform_policy_has_normalized_entropy_near_one(self):
        entropy = self.risk.normalized_policy_entropy(uniform_logits(batch_size=3, num_actions=4))

        self.assertTrue(torch.isclose(torch.as_tensor(entropy), torch.tensor(1.0), atol=1e-6))

    def test_peaked_policy_has_low_normalized_entropy(self):
        entropy = self.risk.normalized_policy_entropy(peaked_logits(batch_size=3, num_actions=4, peak=20.0))

        self.assertLess(float(entropy), 0.01)

    def test_entropy_risk_is_high_for_sharp_policy_and_low_for_uniform_policy(self):
        uniform_risk = self.risk.policy_entropy_risk(uniform_logits(batch_size=3, num_actions=4))
        peaked_risk = self.risk.policy_entropy_risk(peaked_logits(batch_size=3, num_actions=4, peak=20.0))

        self.assertLess(float(uniform_risk), 0.01)
        self.assertGreater(float(peaked_risk), 0.99)


if __name__ == "__main__":
    unittest.main()
