"""Monte-Carlo-dropout uncertainty utilities for frozen encoders."""

import numpy as np
import torch as th


def enable_mc_dropout(model):
    """Use eval behavior except for stochastic dropout operations."""
    model.eval()
    for module in model.modules():
        if isinstance(module, th.nn.modules.dropout._DropoutNd):
            module.train()
        # Graph DirectionalGCN uses functional dropout and self.training.
        if module.__class__.__name__ == "DirectionalGCN":
            module.training = True


def seed_mc(seed):
    th.manual_seed(int(seed))
    if th.cuda.is_available():
        th.cuda.manual_seed_all(int(seed))


def summarize_mc_probabilities(probability_samples, eps=1e-8, include_samples=True):
    """Convert [K,N,C] probability samples to uncertainty statistics."""
    probabilities = np.asarray(probability_samples, dtype=np.float64)
    if probabilities.ndim != 3 or probabilities.shape[0] < 2:
        raise ValueError("MC probabilities must have shape [K>=2, N, C]")
    mean_probability = probabilities.mean(axis=0)
    mean_probability /= np.clip(
        mean_probability.sum(axis=1, keepdims=True), eps, None
    )
    entropy_scale = np.log(probabilities.shape[-1])
    predictive_entropy = -np.sum(
        mean_probability * np.log(np.clip(mean_probability, eps, None)),
        axis=1,
    ) / entropy_scale
    expected_entropy = (-np.sum(
        probabilities * np.log(np.clip(probabilities, eps, None)), axis=2
    ) / entropy_scale).mean(axis=0)
    summary = {
        "mc_mean_prob": mean_probability.astype(np.float32),
        "mc_predictive_entropy": predictive_entropy.astype(np.float32),
        "mc_expected_entropy": expected_entropy.astype(np.float32),
        "mc_mutual_information": np.maximum(
            predictive_entropy - expected_entropy, 0.0
        ).astype(np.float32),
        "mc_probability_variance": probabilities.var(axis=0).mean(
            axis=1
        ).astype(np.float32),
    }
    # Keeping the individual passes makes it possible to fit temperature on
    # Dev and then apply that calibration to every MC sample.  Probabilities
    # are sufficient because softmax(z / T) is exactly proportional to
    # softmax(z) ** (1 / T); the unknown additive logit offset cancels.
    if include_samples:
        summary["mc_probability_samples"] = probabilities.astype(np.float32)
    return summary
