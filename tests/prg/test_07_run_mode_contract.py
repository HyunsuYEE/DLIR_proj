import unittest
from pathlib import Path


class PRGRunModeContractTest(unittest.TestCase):
    def test_run_sh_exposes_prg_mode(self):
        run_sh = Path("run.sh").read_text()

        self.assertIn("--prg", run_sh)
        self.assertIn("solver_type=prg", run_sh)
        self.assertIn("prg_risk_threshold", run_sh)

    def test_benchmark_runs_and_summarizes_prg_mode(self):
        benchmark_sh = Path("run_benchmark.sh").read_text()

        self.assertIn("./run.sh --prg", benchmark_sh)
        self.assertIn('extract_metrics_summary "prg"', benchmark_sh)


if __name__ == "__main__":
    unittest.main()
