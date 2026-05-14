import importlib
import unittest

import torch

from tests.prg._helpers import add_src_to_path


add_src_to_path()


class ProxyMarginRiskTest(unittest.TestCase):
    def setUp(self):
        self.risk = importlib.import_module("models.prg.risk")

    def test_proxy_margin_matches_relative_l1_ratio(self):
        current = torch.tensor([[2.0, 4.0, 8.0]])
        previous = torch.tensor([[1.0, 2.0, 4.0]])

        margin = self.risk.proxy_margin(current, previous)

        expected = (current - previous).abs().mean() / previous.abs().mean()
        self.assertTrue(torch.isclose(torch.as_tensor(margin), expected))

    def test_proxy_margin_is_finite_for_zero_previous_proxy(self):
        margin = self.risk.proxy_margin(torch.ones(2, 3), torch.zeros(2, 3))

        self.assertTrue(torch.isfinite(torch.as_tensor(margin)))
        self.assertGreater(float(margin), 0.0)

    def test_proxy_margin_grows_with_proxy_change(self):
        previous = torch.ones(2, 3)
        small = self.risk.proxy_margin(previous * 1.01, previous)
        large = self.risk.proxy_margin(previous * 1.50, previous)

        self.assertLess(float(small), float(large))


if __name__ == "__main__":
    unittest.main()
