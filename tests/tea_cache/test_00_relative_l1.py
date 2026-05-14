import importlib
import unittest

import torch

from tests.tea_cache._helpers import add_src_to_path


add_src_to_path()


class RelativeL1Test(unittest.TestCase):
    def setUp(self):
        self.mod = importlib.import_module("models.diffusion.teacache")

    def test_relative_l1_matches_manual_mean_ratio(self):
        current = torch.tensor([[2.0, 4.0, 8.0]])
        previous = torch.tensor([[1.0, 2.0, 4.0]])

        out = self.mod.relative_l1(current, previous)

        expected = (current - previous).abs().mean() / previous.abs().mean()
        self.assertTrue(torch.isclose(out, expected))

    def test_relative_l1_is_finite_for_zero_previous(self):
        current = torch.ones(2, 3)
        previous = torch.zeros(2, 3)

        out = self.mod.relative_l1(current, previous)

        self.assertTrue(torch.isfinite(out))
        self.assertGreater(out.item(), 0.0)


if __name__ == "__main__":
    unittest.main()

