# RumourEval2019 alpha sensitivity

Evaluate alpha = {0, 0.25, 0.5, 0.75, 1.0} with the full UA-TAF model.
Encoders and MC-dropout samples stay fixed within each iteration. Fit the
three temperatures once on Dev per iteration and reuse them for every alpha.
The experiment changes only the reliability guidance strength:

`w_k = (1 - alpha * contrast) / K + alpha * contrast * q_k`

At alpha=0 this is uniform averaging. At alpha=1 it is the full
contrast-gated weighting, which is not necessarily equal to q.

## Prepare evidence

If five independent runs have already exported Dev and Test MC evidence,
skip training. Otherwise, after model choices are fixed, run:

```bash
for iteration in 0 1 2 3 4; do
  python UA-TAF/train_encoders.py --dataset RumourEval2019 \
    --iteration "$iteration" --branch all --mc-samples 20 --export-test || exit 1
done
```

Evidence must reside at
`results/UA_TAF_training/RumourEval2019/iterN/evidence/` with
`dev_graph.npz`, `dev_root.npz`, `dev_stance.npz` and corresponding Test
files. They must contain logits, aligned event IDs/labels and full
`mc_probability_samples`. This dataset uses its fixed Train/Dev/Test split;
iterations are independent training runs, not folds.

## Evaluate the predefined five alpha values

```bash
python run_uataf_alpha_sensitivity.py \
  --datasets RumourEval2019 --iterations 0,1,2,3,4 \
  --alphas 0,0.25,0.5,0.75,1.0 --evaluate-split test \
  --output-dir results/UA_TAF_alpha_sensitivity/RumourEval2019_test
```

If only iteration 0 exists, use `--iterations 0`. No retraining occurs
during the sweep. The default evaluation split is Dev; use Dev results for
parameter selection, and the predefined Test sweep to report sensitivity,
without selecting a new alpha from Test results.

Outputs:

- `alpha_sensitivity_runs.csv`: per-run Accuracy, Macro-F1, NLL and Brier score.
- `alpha_sensitivity_summary.csv` / `.json`: mean and sample standard deviation
  (`ddof=1`) across iterations for each alpha. Accuracy and Macro-F1 are in [0,1].
- `experiment_config.json`: parameter values, directories and weight formula.
- `RumourEval2019/iterN/test/calibration.json`: the shared Dev temperatures.
- `RumourEval2019/iterN/test/alpha_X.npz`: event IDs, labels, predicted
  probabilities/classes, weights and contrast for reproducibility.

One run has a displayed standard deviation of zero and does not estimate
run-to-run variability. Use separate output directories for Dev and Test.

## Plot measured results

```bash
python plot_alpha_sensitivity.py \
  --summary results/UA_TAF_alpha_sensitivity/RumourEval2019_test/alpha_sensitivity_summary.csv \
  --dataset RumourEval2019 \
  --output-prefix results/UA_TAF_alpha_sensitivity/RumourEval2019_test/alpha_curve
```

Requires matplotlib. Outputs PNG, PDF and SVG with Accuracy and Macro-F1
in percent. Error bars and shading denote sample standard deviation,
not confidence intervals. All plotted values are read from the experiment
CSV; the former hard-coded measurements are no longer used.
