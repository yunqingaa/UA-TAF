import sys
import unittest
from pathlib import Path

import torch as th


FUSION_DIR = Path(__file__).resolve().parents[1] / "fusion"
sys.path.insert(0, str(FUSION_DIR))

from dual_uncertainty_fusion import (  # noqa: E402
    decompose_predictive_uncertainty,
    fuse_mc_branches,
    fusion_weights,
    reliability_contrast,
)


class DualUncertaintyFusionTests(unittest.TestCase):
    def test_removed_branch_cannot_affect_prediction(self):
        th.manual_seed(22)
        samples = {b: th.softmax(th.randn(3, 4, 3), -1)
                   for b in ("graph", "root", "stance")}
        temperatures = {b: 1.3 for b in samples}
        for ablation, removed in (("no_root", "root"),
                                  ("no_propagation", "graph"),
                                  ("no_stance", "stance")):
            result = fuse_mc_branches(samples, temperatures, ablation=ablation)
            subset = {b: x for b, x in samples.items() if b != removed}
            other = fuse_mc_branches(subset, temperatures, ablation=ablation)
            self.assertTrue(th.equal(result["fused_probability"], other["fused_probability"]))
            self.assertNotIn(removed, result["branches"])
            self.assertEqual(result["weights"].shape, (4, 2))
            self.assertTrue(th.allclose(result["weights"].sum(1), th.ones(4)))

    def test_single_branch_is_calibrated_mc_mean(self):
        from dual_uncertainty_fusion import calibrate_probability_samples
        samples = th.softmax(th.randn(3, 4, 3), -1)
        for ablation, branch in (("propagation_only", "graph"),
                                 ("root_only", "root"), ("stance_only", "stance")):
            result = fuse_mc_branches({branch: samples}, {branch: 1.7}, ablation=ablation)
            expected = calibrate_probability_samples(samples, 1.7).mean(0)
            self.assertTrue(th.allclose(result["fused_probability"], expected, atol=1e-6))
            self.assertTrue(th.equal(result["weights"], th.ones(4, 1)))

    def test_identical_mc_passes_have_zero_model_instability(self):
        one_pass = th.tensor([[0.8, 0.1, 0.1], [1 / 3, 1 / 3, 1 / 3]])
        samples = one_pass.unsqueeze(0).repeat(5, 1, 1)
        _, total, expected, disagreement = decompose_predictive_uncertainty(samples)
        self.assertTrue(th.allclose(disagreement, th.zeros(2), atol=1e-6))
        self.assertTrue(th.allclose(total, expected, atol=1e-6))

    def test_conflicting_certain_passes_have_high_model_instability(self):
        samples = th.tensor([
            [[0.999, 0.0005, 0.0005]],
            [[0.0005, 0.999, 0.0005]],
            [[0.0005, 0.0005, 0.999]],
        ])
        _, total, expected, disagreement = decompose_predictive_uncertainty(samples)
        self.assertGreater(float(total[0]), 0.99)
        self.assertLess(float(expected[0]), 0.02)
        self.assertGreater(float(disagreement[0]), 0.98)
        self.assertTrue(th.allclose(total, expected + disagreement, atol=1e-6))

    def test_contrast_uniform_and_one_hot_limits(self):
        uniform = th.full((1, 3), 1 / 3)
        one_hot = th.tensor([[1.0, 0.0, 0.0]])
        self.assertAlmostEqual(float(reliability_contrast(uniform)[0]), 0.0, places=6)
        self.assertAlmostEqual(float(reliability_contrast(one_hot)[0]), 1.0, places=6)

    def test_alpha_zero_is_uniform_fusion(self):
        q = th.tensor([[0.8, 0.1, 0.1]])
        weights, contrast, strength = fusion_weights(q, alpha=0.0)
        self.assertTrue(th.allclose(weights, th.full((1, 3), 1 / 3)))
        self.assertGreater(float(contrast[0]), 0.0)
        self.assertEqual(float(strength[0]), 0.0)

    def test_end_to_end_invariants(self):
        th.manual_seed(13)
        samples = {
            branch: th.softmax(th.randn(7, 11, 3), dim=-1)
            for branch in ("graph", "root", "stance")
        }
        result = fuse_mc_branches(samples, {b: 1.0 for b in samples}, alpha=0.7)
        self.assertTrue(th.allclose(result["weights"].sum(1), th.ones(11), atol=1e-6))
        self.assertTrue(th.allclose(result["fused_probability"].sum(1), th.ones(11), atol=1e-6))
        self.assertTrue(bool((result["weights"] >= 0).all()))
        self.assertTrue(bool((result["contrast"] >= 0).all()))
        self.assertTrue(bool((result["contrast"] <= 1).all()))

    def test_without_uncertainty_is_uniform_average(self):
        th.manual_seed(15)
        samples = {
            branch: th.softmax(th.randn(5, 4, 3), dim=-1)
            for branch in ("graph", "root", "stance")
        }
        result = fuse_mc_branches(
            samples, {b: 1.0 for b in samples},
            ablation="no_uncertainty",
        )
        self.assertTrue(th.allclose(result["weights"], th.full((4, 3), 1 / 3)))
        self.assertTrue(th.allclose(result["contrast"], th.zeros(4)))

    def test_without_contrast_retains_fixed_alpha(self):
        th.manual_seed(16)
        samples = {
            branch: th.softmax(th.randn(5, 4, 3), dim=-1)
            for branch in ("graph", "root", "stance")
        }
        result = fuse_mc_branches(
            samples, {b: 1.0 for b in samples}, alpha=0.2,
            ablation="no_reliability_contrast",
        )
        self.assertTrue(th.allclose(result["weights"], 0.8 / 3 + 0.2 * result["normalized_reliability"]))
        self.assertTrue(th.equal(result["contrast"], th.ones(4)))
        self.assertTrue(th.allclose(result["adaptive_strength"], th.full((4,), 0.2)))
        for alpha in (0.0, 1.0):
            endpoint = fuse_mc_branches(samples, {b: 1.0 for b in samples},
                                       alpha=alpha, ablation="no_reliability_contrast")
            expected = th.full((4, 3), 1 / 3) if alpha == 0 else endpoint["normalized_reliability"]
            self.assertTrue(th.allclose(endpoint["weights"], expected))

    def test_without_calibration_ignores_fitted_temperatures(self):
        th.manual_seed(17)
        samples = {
            branch: th.softmax(th.randn(5, 4, 3), dim=-1)
            for branch in ("graph", "root", "stance")
        }
        first = fuse_mc_branches(
            samples, {b: 0.2 for b in samples},
            ablation="no_calibration",
        )
        second = fuse_mc_branches(
            samples, {b: 9.0 for b in samples},
            ablation="no_calibration",
        )
        self.assertTrue(th.equal(first["fused_probability"], second["fused_probability"]))


if __name__ == "__main__":
    unittest.main()
