import contextlib
import csv
import io
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import run_uataf_alpha_sensitivity as experiment


class AlphaSensitivityTests(unittest.TestCase):
    def test_fixed_evidence_sweep_and_summary(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            rng = np.random.default_rng(42)
            for iteration in (0, 1):
                evidence = root / "training" / "RumourEval2019" / f"iter{iteration}" / "evidence"
                evidence.mkdir(parents=True)
                for split in ("dev", "test"):
                    for branch in ("graph", "root", "stance"):
                        logits = rng.normal(size=(9, 3)).astype(np.float32)
                        raw = rng.normal(size=(3, 9, 3)).astype(np.float32)
                        samples = np.exp(raw - raw.max(-1, keepdims=True))
                        samples /= samples.sum(-1, keepdims=True)
                        np.savez(evidence / f"{split}_{branch}.npz",
                                 event_ids=np.arange(9).astype(str), labels=np.arange(9) % 3,
                                 mc_probability_samples=samples, **{f"z_{branch}": logits})
            args = Namespace(datasets=["RumourEval2019"], iterations=[0, 1],
                             alphas=list(experiment.DEFAULT_ALPHAS), fit_split="dev",
                             evaluate_split="test", temperature_reg=0.01,
                             training_root=root / "training", output_dir=root / "out")
            with patch.object(experiment, "fit_dev_temperatures", return_value={
                b: 1.0 for b in ("graph", "root", "stance")
            }) as fit, contextlib.redirect_stdout(io.StringIO()):
                experiment.main(args)
            self.assertEqual(fit.call_count, 2)  # once per run, not once per alpha
            summary = json.loads((root / "out" / "alpha_sensitivity_summary.json").read_text())
            self.assertEqual([r["alpha"] for r in summary], list(experiment.DEFAULT_ALPHAS))
            self.assertTrue(all(r["runs"] == 2 for r in summary))
            with (root / "out" / "alpha_sensitivity_runs.csv").open(encoding="utf-8-sig") as handle:
                runs = list(csv.DictReader(handle))
            self.assertEqual(len(runs), 10)
            for row in summary:
                values = [float(r["macro_f1"]) for r in runs if float(r["alpha"]) == row["alpha"]]
                self.assertAlmostEqual(row["macro_f1_std"], np.std(values, ddof=1))
            directory = root / "out" / "RumourEval2019" / "iter0" / "test"
            with np.load(directory / "alpha_0.npz") as zero, np.load(directory / "alpha_1.npz") as one:
                np.testing.assert_allclose(zero["weights"], 1 / 3)
                np.testing.assert_array_equal(zero["event_ids"], one["event_ids"])
                self.assertFalse(np.allclose(zero["weights"], one["weights"]))

    def test_macro_f1_includes_absent_classes(self):
        result = experiment.probability_metrics(np.array([[0.9, 0.05, 0.05]]), np.array([0]))
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
