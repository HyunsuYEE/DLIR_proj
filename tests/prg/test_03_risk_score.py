import importlib
import unittest

from tests.prg._helpers import add_src_to_path


add_src_to_path()


class PRGRiskScoreTest(unittest.TestCase):
    def setUp(self):
        self.risk = importlib.import_module("models.prg.risk")

    def test_score_is_weighted_average_of_depth_policy_certainty_and_proxy_margin(self):
        inputs = self.risk.RolloutRiskInputs(
            depth_fraction=0.50,
            normalized_policy_entropy=0.25,
            proxy_margin=0.20,
        )
        weights = self.risk.PRGRiskWeights(depth=1.0, policy=1.0, proxy=1.0)

        score = self.risk.compute_risk_score(inputs, weights)

        expected = (0.50 + 0.75 + 0.20) / 3.0
        self.assertAlmostEqual(float(score), expected, places=6)

    def test_score_is_clamped_to_unit_interval(self):
        inputs = self.risk.RolloutRiskInputs(
            depth_fraction=10.0,
            normalized_policy_entropy=-2.0,
            proxy_margin=99.0,
        )
        weights = self.risk.PRGRiskWeights(depth=1.0, policy=1.0, proxy=1.0)

        score = self.risk.compute_risk_score(inputs, weights)

        self.assertGreaterEqual(float(score), 0.0)
        self.assertLessEqual(float(score), 1.0)

    def test_score_increases_with_depth_policy_sharpness_and_proxy_margin(self):
        weights = self.risk.PRGRiskWeights(depth=1.0, policy=1.0, proxy=1.0)
        low_risk = self.risk.compute_risk_score(
            self.risk.RolloutRiskInputs(
                depth_fraction=0.10,
                normalized_policy_entropy=0.95,
                proxy_margin=0.05,
            ),
            weights,
        )
        high_risk = self.risk.compute_risk_score(
            self.risk.RolloutRiskInputs(
                depth_fraction=0.90,
                normalized_policy_entropy=0.05,
                proxy_margin=0.80,
            ),
            weights,
        )

        self.assertLess(float(low_risk), float(high_risk))


if __name__ == "__main__":
    unittest.main()
