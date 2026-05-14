import unittest
from pathlib import Path


class RunModeContractTest(unittest.TestCase):
    def test_aggressive_mode_is_reserved_for_dpm_solver_plus_teacache(self):
        run_sh = Path("run.sh").read_text()

        self.assertIn("--aggressive", run_sh)
        self.assertIn("solver_type=dpm_solver", run_sh)
        self.assertIn("teacache_enabled=true", run_sh)


if __name__ == "__main__":
    unittest.main()
