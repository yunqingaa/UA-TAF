"""Equation-driven UA-TAF fusion with predictive uncertainty decomposition.

The three frozen evidence encoders are unchanged.  Temperatures are fitted on
Dev logits, applied to every Monte-Carlo dropout pass, and the event-level
fusion follows equations (1)--(9) of the revised UA-TAF formulation.
"""

import math
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F


BRANCHES = ("graph", "root", "stance")
ABLATION_LABELS = {
    "full": "UA-TAF",
    "no_root": "w/o Root",
    "no_propagation": "w/o Propagation",
    "no_stance": "w/o Stance",
    "no_calibration": "w/o Calibration",
    "no_uncertainty": "w/o Uncertainty",
    "no_reliability_contrast": "w/o Reliability Contrast",
    "propagation_only": "Propagation-only",
    "root_only": "Root-only",
    "stance_only": "Stance-only",
}
ABLATIONS = tuple(ABLATION_LABELS)


def active_branches(ablation):
    if ablation not in ABLATIONS:
        raise ValueError("unknown ablation: {}".format(ablation))
    return {
        "no_root": ("graph", "stance"),
        "no_propagation": ("root", "stance"),
        "no_stance": ("graph", "root"),
        "propagation_only": ("graph",),
        "root_only": ("root",),
        "stance_only": ("stance",),
    }.get(ablation, BRANCHES)
EPS = 1e-8


def _validate_alpha(alpha):
    value = float(alpha)
    if not 0.0 <= value <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    return value


def fit_temperature(logits, labels, regularization=0.01):
    """Fit one positive scalar temperature using Dev only."""
    values = th.as_tensor(logits, dtype=th.float32)
    targets = th.as_tensor(labels, dtype=th.long)
    log_temperature = th.nn.Parameter(th.zeros(()))
    optimizer = th.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.cross_entropy(values / temperature, targets)
        loss = loss + float(regularization) * log_temperature.square()
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def calibrate_probability_samples(probability_samples, temperature, eps=EPS):
    """Apply logit temperature scaling exactly from softmax probabilities."""
    samples = th.as_tensor(probability_samples, dtype=th.float32)
    if samples.ndim != 3 or samples.shape[0] < 2:
        raise ValueError("MC probability samples must have shape [M>=2, N, C]")
    if not th.isfinite(samples).all() or bool((samples < 0).any()):
        raise ValueError("MC probability samples must be finite and non-negative")
    if float(temperature) <= 0:
        raise ValueError("temperature must be positive")
    log_prob = th.log(samples.clamp_min(eps)) / float(temperature)
    return F.softmax(log_prob, dim=-1)


def normalized_entropy(probabilities, eps=EPS):
    if probabilities.shape[-1] < 2:
        raise ValueError("at least two classes are required")
    scale = math.log(probabilities.shape[-1])
    return -(probabilities * th.log(probabilities.clamp_min(eps))).sum(-1) / scale


def decompose_predictive_uncertainty(calibrated_samples):
    """Return mean probability and the U_P = U_A + U_E decomposition."""
    if calibrated_samples.ndim != 3 or calibrated_samples.shape[0] < 2:
        raise ValueError("calibrated samples must have shape [M>=2, N, C]")
    mean_probability = calibrated_samples.mean(0)
    mean_probability = mean_probability / mean_probability.sum(-1, keepdim=True).clamp_min(EPS)
    total_predictive = normalized_entropy(mean_probability)
    expected_entropy = normalized_entropy(calibrated_samples).mean(0)
    # U_E is mutual-information-like model disagreement. It is non-negative
    # analytically, so the clamp only removes floating-point round-off.
    model_disagreement = (total_predictive - expected_entropy).clamp(0.0, 1.0)
    return (
        mean_probability,
        total_predictive.clamp(0.0, 1.0),
        expected_entropy.clamp(0.0, 1.0),
        model_disagreement,
    )


def reliability_from_uncertainty(total_predictive):
    """Event-level evidence reliability r=1-U_P."""
    return (1.0 - total_predictive).clamp(0.0, 1.0)


def normalize_reliability(reliability, eps=EPS):
    if reliability.ndim != 2:
        raise ValueError("reliability must have shape [N, K]")
    adjusted = reliability.clamp_min(0.0) + float(eps)
    return adjusted / adjusted.sum(-1, keepdim=True)


def reliability_contrast(normalized_reliability):
    """Normalized L1 distance from uniform, in [0, 1]."""
    if normalized_reliability.ndim != 2:
        raise ValueError("normalized reliability must have shape [N, K]")
    branch_count = normalized_reliability.shape[1]
    if branch_count < 2:
        raise ValueError("at least two branches are required")
    factor = branch_count / (2.0 * (branch_count - 1.0))
    contrast = factor * (normalized_reliability - 1.0 / branch_count).abs().sum(-1)
    return contrast.clamp(0.0, 1.0)


def fusion_weights(normalized_reliability, alpha):
    alpha = _validate_alpha(alpha)
    contrast = reliability_contrast(normalized_reliability)
    adaptive_strength = alpha * contrast
    branch_count = normalized_reliability.shape[1]
    weights = (
        (1.0 - adaptive_strength).unsqueeze(-1) / branch_count
        + adaptive_strength.unsqueeze(-1) * normalized_reliability
    )
    return weights, contrast, adaptive_strength


def fuse_mc_branches(
    mc_samples,
    temperatures,
    alpha=1.0,
    eps=EPS,
    ablation="full",
):
    """Fuse aligned branch MC samples and return transparent diagnostics."""
    if ablation not in ABLATIONS:
        raise ValueError("unknown ablation {!r}; choose {}".format(ablation, ABLATIONS))
    branches = active_branches(ablation)
    missing = set(branches).difference(mc_samples)
    if missing:
        raise ValueError("missing MC branches: {}".format(sorted(missing)))
    if set(branches).difference(temperatures):
        raise ValueError("one temperature is required for every branch")

    means, u_total, u_expected, u_disagreement = {}, {}, {}, {}
    sample_count = None
    for branch in branches:
        temperature = (
            1.0 if ablation == "no_calibration" else temperatures[branch]
        )
        calibrated = calibrate_probability_samples(
            mc_samples[branch], temperature, eps=eps
        )
        mean, uncertainty_p, uncertainty_a, uncertainty_e = (
            decompose_predictive_uncertainty(calibrated)
        )
        if sample_count is None:
            sample_count = mean.shape[0]
        elif mean.shape[0] != sample_count:
            raise ValueError("branch event counts are not aligned")
        means[branch] = mean
        u_total[branch] = uncertainty_p
        u_expected[branch] = uncertainty_a
        u_disagreement[branch] = uncertainty_e

    total_predictive = th.stack([u_total[b] for b in branches], dim=1)
    expected_predictive = th.stack([u_expected[b] for b in branches], dim=1)
    model_disagreement = th.stack([u_disagreement[b] for b in branches], dim=1)
    if ablation == "no_uncertainty":
        reliability = th.ones_like(total_predictive)
    else:
        reliability = reliability_from_uncertainty(total_predictive)
    normalized = normalize_reliability(reliability, eps=eps)
    if len(branches) == 1:
        weights = th.ones_like(normalized)
        contrast = th.zeros_like(normalized[:, 0])
        adaptive_strength = contrast
    elif ablation == "no_reliability_contrast":
        # Remove only event-dependent contrast gating; retain fixed alpha.
        alpha = _validate_alpha(alpha)
        weights = (1.0 - alpha) / len(branches) + alpha * normalized
        contrast = th.ones(normalized.shape[0], dtype=normalized.dtype, device=normalized.device)
        adaptive_strength = alpha * contrast
    else:
        weights, contrast, adaptive_strength = fusion_weights(normalized, alpha)
    stacked = th.stack([means[b] for b in branches], dim=1)
    fused = (weights.unsqueeze(-1) * stacked).sum(1)
    fused = fused / fused.sum(-1, keepdim=True).clamp_min(eps)
    return {
        "probabilities": means,
        "total_predictive_uncertainty": total_predictive,
        "expected_predictive_uncertainty": expected_predictive,
        "model_disagreement_uncertainty": model_disagreement,
        "reliability": reliability,
        "normalized_reliability": normalized,
        "contrast": contrast,
        "adaptive_strength": adaptive_strength,
        "weights": weights,
        "fused_probability": fused,
        "ablation": ablation,
        "branches": branches,
    }


def load_aligned_mc_evidence(evidence_dir, split):
    """Load strict event-aligned logits and full MC samples from three NPZs."""
    evidence_dir = Path(evidence_dir)
    loaded = {}
    for branch in BRANCHES:
        path = evidence_dir / "{}_{}.npz".format(split, branch)
        with np.load(path, allow_pickle=False) as data:
            required = {"event_ids", "labels", "z_{}".format(branch), "mc_probability_samples"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError("{} missing {}; re-export evidence with MC samples".format(path, sorted(missing)))
            loaded[branch] = {
                "event_ids": data["event_ids"].astype(str),
                "labels": data["labels"].astype(np.int64),
                "logits": data["z_{}".format(branch)].astype(np.float32),
                "mc_samples": data["mc_probability_samples"].astype(np.float32),
            }

        values = loaded[branch]
        if len(set(values["event_ids"].tolist())) != len(values["event_ids"]):
            raise ValueError("duplicate event IDs in {}".format(branch))
        if values["logits"].ndim != 2:
            raise ValueError("{} logits must have shape [N, C]".format(branch))
        if values["mc_samples"].ndim != 3 or values["mc_samples"].shape[0] < 2:
            raise ValueError("{} MC samples must have shape [M>=2, N, C]".format(branch))
        if values["logits"].shape != values["mc_samples"].shape[1:]:
            raise ValueError("{} logits and MC samples disagree in shape".format(branch))
        if not np.all(np.isfinite(values["logits"])) or not np.all(np.isfinite(values["mc_samples"])):
            raise ValueError("{} evidence contains NaN/Inf".format(branch))

    reference_ids = loaded[BRANCHES[0]]["event_ids"]
    reference_labels = loaded[BRANCHES[0]]["labels"]
    result = {"event_ids": reference_ids, "labels": reference_labels, "logits": {}, "mc_samples": {}}
    for branch in BRANCHES:
        values = loaded[branch]
        lookup = {event_id: i for i, event_id in enumerate(values["event_ids"])}
        if set(lookup) != set(reference_ids):
            raise ValueError("{} event ID set differs".format(branch))
        order = np.asarray([lookup[event_id] for event_id in reference_ids])
        if not np.array_equal(values["labels"][order], reference_labels):
            raise ValueError("{} labels differ".format(branch))
        result["logits"][branch] = values["logits"][order]
        result["mc_samples"][branch] = values["mc_samples"][:, order, :]
    return result


def fit_dev_temperatures(evidence, regularization=0.01):
    return {
        branch: fit_temperature(
            evidence["logits"][branch], evidence["labels"], regularization
        )
        for branch in BRANCHES
    }
