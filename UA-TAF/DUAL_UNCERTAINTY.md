# Revised UA-TAF: predictive uncertainty decomposition

This implementation keeps the Root, Propagation (Graph), and Stance encoders
frozen. It fits one temperature per branch on Dev and applies that temperature
to every stored MC-dropout pass before computing uncertainty.

The implementation follows these equations:

- `U_A = mean(H(p)) / log(C)` (expected predictive uncertainty)
- `U_E = H(mean(p)) / log(C) - mean(H(p)) / log(C)` (model disagreement)
- `U_P = H(mean(p)) / log(C) = U_A + U_E` (total predictive uncertainty)
- `r = 1 - U_P`
- `q_k = (r_k + eps) / sum_j(r_j + eps)`
- `gamma = K / (2 * (K - 1)) * sum_k(abs(q_k - 1 / K))`
- `w_k = (1 - alpha * gamma) / K + alpha * gamma * q_k`
- `p_f = sum_k(w_k * mean(p_k))`

Export evidence first. Encoder export now stores the full
`mc_probability_samples` array in addition to the existing summary fields:

```bash
python UA-TAF/train_encoders.py \
  --dataset RumourEval2019 \
  --iteration 0 \
  --branch all \
  --mc-samples 20
```

Fit calibration and evaluate on Dev:

```bash
python UA-TAF/train_dual_uncertainty_fusion.py \
  --evidence-dir results/UA_TAF_training/RumourEval2019/iter0/evidence \
  --output-dir results/UA_TAF_training/RumourEval2019/iter0/dual_uncertainty \
  --fit-split dev \
  --evaluate-split dev \
  --alpha 1.0
```

Only after model selection is complete, export Test evidence and use
`--evaluate-split test`; temperatures are still fitted exclusively on Dev.

Outputs are `fusion_artifact.pt`, `summary.json`, and `diagnostics.csv`. The
diagnostics contain both uncertainties, raw reliability, normalized
reliability, contrast, adaptive strength, and final weight for every branch.

## Ablations

Use `--ablation` with one of the following values:

- `full`: `r=1-U_P`, with `U_P=U_A+U_E`, and contrast-gated weights;
- `no_calibration`: use `T=1` for all MC samples;
- `no_uncertainty`: use equal reliability, which produces uniform averaging;
- `no_reliability_contrast`: set `gamma=1` and use `w=(1-alpha)/K+alpha*q` (equal to q at alpha=1). Included in the standard ten-variant suite.
