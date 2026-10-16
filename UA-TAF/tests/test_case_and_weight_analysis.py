import contextlib
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
import run_uataf_case_analysis as cases
import run_uataf_weight_analysis as weight_analysis
from run_uataf_weight_analysis import align_test, summarize_weights


class CaseAndWeightTests(unittest.TestCase):
    def test_stability_requires_all_five_runs(self):
        branches = np.tile(np.array([[0, 0, 0], [0, 1, 1], [1, 1, 1], [0, 0, 0]]), (5, 1, 1))
        fusion = np.tile([0, 0, 1, 0], (5, 1))
        fusion[4, 3] = 1  # four successes is not stable
        masks = cases.classify_cases(branches, fusion, np.zeros(4, dtype=int))
        np.testing.assert_array_equal(masks["evidence_agreement"].all(0), [True, False, False, False])
        np.testing.assert_array_equal(masks["conflict_corrected"].all(0), [False, True, False, False])
        np.testing.assert_array_equal(masks["fusion_failure"].all(0), [False, False, True, False])

    def test_empty_categories_and_deterministic_selection(self):
        probs = np.tile([.9, .05, .05], (5, 2, 3, 1))
        fused = probs[:, :, 0]
        weights = np.full((5, 2, 3), 1 / 3)
        selected, candidates, _ = cases.select_stable_cases(probs, fused, weights, np.zeros(2, dtype=int), ["b", "a"])
        self.assertEqual(selected["evidence_agreement"], 1)
        self.assertIsNone(selected["conflict_corrected"])
        self.assertIsNone(selected["fusion_failure"])
        self.assertEqual(candidates[0]["event_id"], "a")

    def test_uniform_ties_and_separate_variability(self):
        weights = np.full((5, 3, 3), 1 / 3)
        run, events, branches, overview = summarize_weights(weights, ["a", "b", "c"], [0, 1, 2], list(range(5)))
        self.assertEqual(overview["nonuniform_event_run_fraction"], 0)
        self.assertEqual(overview["tied_maximum_event_run_fraction"], 1)
        for row in branches:
            self.assertAlmostEqual(row["dominant_share_tie_adjusted"], 1 / 3)
            self.assertAlmostEqual(row["mean_within_run_event_std"], 0)

    def test_run_alignment_checks_labels(self):
        evidence = {"event_ids": np.array(["b", "a"]), "labels": np.array([1, 0]),
                    "logits": {b: np.array([[2], [1]]) for b in cases.BRANCHES},
                    "mc_samples": {b: np.array([[[2], [1]], [[2], [1]]]) for b in cases.BRANCHES}}
        aligned = align_test(evidence, ["a", "b"], np.array([0, 1]))
        np.testing.assert_array_equal(aligned["logits"]["root"], [[1], [2]])
        with self.assertRaises(ValueError):
            align_test(evidence, ["a", "b"], np.array([1, 0]))

    def test_table_generation_on_five_synthetic_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            # Agreement, conflict corrected (uniform vote would be wrong), failure.
            distributions = np.array([
                [[.9, .05, .05], [.9, .05, .05], [.9, .05, .05]],
                [[.99, .005, .005], [.3, .4, .3], [.3, .4, .3]],
                [[.05, .9, .05], [.05, .9, .05], [.05, .9, .05]],
            ], dtype=np.float32)
            for iteration in range(5):
                directory = root / "training" / "RumourEval2019" / f"iter{iteration}" / "evidence"
                directory.mkdir(parents=True)
                for split in ("dev", "test"):
                    order = [2, 0, 1] if iteration % 2 else [0, 1, 2]
                    for b_id, branch in enumerate(cases.BRANCHES):
                        p = distributions[order, b_id]
                        np.savez(directory / f"{split}_{branch}.npz",
                                 event_ids=np.array([f"{split}{i}" for i in order]), labels=np.zeros(3, dtype=int),
                                 mc_probability_samples=np.stack([p, p]), **{f"z_{branch}": np.log(p)})
            args = Namespace(training_root=root / "training", output_dir=root / "out",
                             iterations=list(range(5)), alpha=1., temperature_reg=.01, threads_csv=None)
            with patch.object(cases, "fit_dev_temperatures", return_value={b: 1. for b in cases.BRANCHES}) as fit, \
                 contextlib.redirect_stdout(io.StringIO()):
                cases.main(args)
            self.assertEqual(fit.call_count, 5)
            report = json.loads((root / "out" / "selection_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["candidate_counts"], dict.fromkeys(cases.CASE_TYPES, 1))
            self.assertEqual(report["selected"]["conflict_corrected"], "test1")
            self.assertIn("FR:5", (root / "out" / "table6.md").read_text(encoding="utf-8"))
            args.output_dir = root / "weights"
            with patch.object(weight_analysis, "fit_dev_temperatures", return_value={b: 1. for b in cases.BRANCHES}), \
                 contextlib.redirect_stdout(io.StringIO()):
                weight_analysis.main(args)
            with np.load(root / "weights" / "fusion_weights.npz") as data:
                self.assertEqual(data["weights"].shape, (5, 3, 3))
                np.testing.assert_array_equal(data["event_ids"], ["test0", "test1", "test2"])
            summary = json.loads((root / "weights" / "analysis_summary.json").read_text())
            self.assertEqual(summary["overview"]["event_run_observations"], 15)


if __name__ == "__main__":
    unittest.main()
