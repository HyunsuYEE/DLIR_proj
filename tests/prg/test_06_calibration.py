import importlib
import unittest

import torch

from tests.prg._helpers import add_src_to_path


add_src_to_path()


class PRGCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.calibration = importlib.import_module("models.prg.calibration")
        self.gate = importlib.import_module("models.prg.gate")

    def test_calibration_chooses_largest_threshold_with_no_false_safe_probe(self):
        risk_scores = torch.tensor([0.10, 0.20, 0.55, 0.80])
        unsafe_labels = torch.tensor([False, False, True, True])

        threshold = self.calibration.calibrate_risk_threshold(
            risk_scores,
            unsafe_labels,
            max_false_safe_rate=0.0,
        )

        self.assertAlmostEqual(float(threshold), 0.20, places=6)
        cfg = self.gate.PRGGateConfig(risk_threshold=float(threshold))
        self.assertEqual(self.gate.select_mode(0.20, cfg), "aggressive")
        self.assertEqual(self.gate.select_mode(0.55, cfg), "conservative")

    def test_calibration_falls_back_to_max_observed_score_when_no_probe_is_unsafe(self):
        risk_scores = torch.tensor([0.10, 0.20, 0.55])
        unsafe_labels = torch.tensor([False, False, False])

        threshold = self.calibration.calibrate_risk_threshold(
            risk_scores,
            unsafe_labels,
            max_false_safe_rate=0.0,
        )

        self.assertAlmostEqual(float(threshold), 0.55, places=6)


if __name__ == "__main__":
    unittest.main()
