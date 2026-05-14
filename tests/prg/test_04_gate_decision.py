import importlib
import unittest

from tests.prg._helpers import add_src_to_path


add_src_to_path()


class PRGGateDecisionTest(unittest.TestCase):
    def setUp(self):
        self.gate = importlib.import_module("models.prg.gate")
        self.risk = importlib.import_module("models.prg.risk")

    def test_low_risk_selects_aggressive_and_high_risk_selects_conservative(self):
        cfg = self.gate.PRGGateConfig(risk_threshold=0.50)

        self.assertEqual(self.gate.select_mode(0.20, cfg), "aggressive")
        self.assertEqual(self.gate.select_mode(0.80, cfg), "conservative")

    def test_decision_contains_score_and_input_breakdown(self):
        cfg = self.gate.PRGGateConfig(risk_threshold=0.50)
        inputs = self.risk.RolloutRiskInputs(
            depth_fraction=0.25,
            normalized_policy_entropy=1.00,
            proxy_margin=0.05,
        )
        weights = self.risk.PRGRiskWeights(depth=1.0, policy=1.0, proxy=1.0)

        decision = self.gate.make_prg_decision(inputs, weights, cfg)

        self.assertEqual(decision.mode, "aggressive")
        self.assertLess(float(decision.risk_score), cfg.risk_threshold)
        self.assertEqual(decision.inputs, inputs)


if __name__ == "__main__":
    unittest.main()
