# Standard ablation experiments

The standard suite has this exact order:

| Display name | CLI value | Definition |
| --- | --- | --- |
| UA-TAF | full | All three branches with calibration and uncertainty fusion |
| w/o Root | no_root | Propagation + Stance |
| w/o Propagation | no_propagation | Root + Stance |
| w/o Stance | no_stance | Propagation + Root |
| w/o Calibration | no_calibration | All temperatures fixed to 1 |
| w/o Uncertainty | no_uncertainty | Uniform average of calibrated branch probabilities |
| w/o Reliability Contrast | no_reliability_contrast | Contrast fixed to 1; weights = (1-alpha)/K + alpha*q |
| Propagation-only | propagation_only | Calibrated Propagation MC mean |
| Root-only | root_only | Calibrated Root MC mean |
| Stance-only | stance_only | Calibrated Stance MC mean |

Propagation is named `graph` internally. Removing a branch recomputes
reliability normalization and contrast over the remaining two branches.
Single-branch variants use weight 1 and bypass fusion. All variants reuse
the independently trained frozen encoders; no encoder retraining is needed
if their evidence already matches the desired data partition.

The CLI loads aligned evidence from all three branches for a controlled
comparison. Temperatures are fitted on Dev only and shared across variants;
the no-calibration variant uses 1 instead. `no_reliability_contrast` is
included in `all`: it removes event-dependent contrast gating but retains
the fixed alpha. At alpha=1 its weights equal q.

Run all ten on one fold (Linux shell):

```bash
python UA-TAF/train_dual_uncertainty_fusion.py \
  --evidence-dir results/UA_TAF_training/PHEME/iter0/fold0/evidence \
  --output-dir results/UA_TAF_training/PHEME/iter0/fold0/ablation \
  --evaluate-split test --ablation all
```

Test evidence must first be exported by training with `--export-test`,
after model choices are finalized. Use `--evaluate-split dev` for Dev
experiments. Each variant gets a separate directory, and the parent gets
`ablation_results.csv` with exactly these columns: `Variant`, `Acc`,
`Macro-F1`, `FR-F1`, `TR-F1`, `UR-F1`. Metrics are percentages with two
decimals and a percent sign (for example `66.67%`). Individual runs also
write `metrics.csv` in the variant directory. Raw full-precision metrics
remain in each `summary.json`. Three-class labels are FR/TR/UR; four-class
PHEME labels are NR/FR/TR/UR and Macro-F1 includes all four classes.
`ablation_metadata.json` stores the evaluation split for plotting.

Run the five PHEME folds:

```bash
for fold in 0 1 2 3 4; do
  python UA-TAF/train_dual_uncertainty_fusion.py \
    --evidence-dir results/UA_TAF_training/PHEME/iter0/fold${fold}/evidence \
    --output-dir results/UA_TAF_training/PHEME/iter0/fold${fold}/ablation \
    --evaluate-split test --ablation all || exit 1
done
```

Plot the actual results (requires matplotlib):

```bash
python plot_ablation.py \
  --results 'PHEME=results/UA_TAF_training/PHEME/iter0/fold*/ablation/ablation_results.csv' \
  --output-prefix ablation_pheme_nine
```

Repeat `--results 'Dataset=pattern'` for more datasets. The plot reports
the mean and sample standard deviation across matching CSVs, reads
Macro-F1 percentages (and supports the previous fractional CSV format), and rejects missing variants or mixed Dev/Test rows
within a dataset. For a single CSV it shows no error interval. The old
hard-coded six-variant plotting results have been removed; existing image
files are not regenerated until measured results are provided.
