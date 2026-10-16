# MC-dropout sampling sensitivity

Use RumourEval2019's five independently trained checkpoints. Freeze all
encoders and fix alpha=1.0. Evaluate M={5,10,20,30,50} using nested prefixes
of the same 50-pass sequence. Fit temperatures once per run on deterministic
Dev logits, and reuse them at every M. Recompute calibrated MC means,
predictive uncertainty, reliability and fusion weights for every M.
No training hyperparameters or checkpoints are changed.

Export a separate 50-pass evidence set from existing checkpoints:

```bash
for iteration in 0 1 2 3 4; do
  python UA-TAF/reexport_mc_evidence.py \
    --dataset RumourEval2019 --iteration "$iteration" --branch all \
    --mc-samples 50 --mc-evidence-name evidence_mc50 || exit 1
done
```

Existing Dev/Test evidence and best checkpoints are required under
`results/UA_TAF_training/RumourEval2019/iterN/`. The exporter preserves
event ordering and checks checkpoint logits against old evidence.
Supply the original dimension, dropout, batch-size and seed arguments
if training used non-default settings. The new samples are written to
`evidence_mc50`; original `evidence` files remain intact. The export
sequence depends on seed and batching; a new prefix need not equal an
older export produced with different settings.

Run the experiment:

```bash
python run_uataf_mc_sensitivity.py \
  --iterations 0 1 2 3 4 --samples 5 10 20 30 50
```

Default evaluation is Test. Use `--evaluate-split dev` and a separate
`--output-dir` if using the sweep for parameter selection. Fixed alpha=1
is enforced by the script and has no CLI override.

Outputs under `results/UA_TAF_mc_sensitivity/RumourEval2019/`:

- `mc_sensitivity_summary.csv` and `mc_sensitivity_table.md`: M, Accuracy
  (%), Macro-F1 (%), expressed as mean ± sample standard deviation,
  rounded to two decimal places.
- `mc_sensitivity_numeric.csv`: full-precision mean/std in percentage points.
- `mc_sensitivity_runs.csv`: per-run metrics in [0,1].
- `iterN/M*.npz`: aligned event IDs, labels, fused probabilities,
  predictive uncertainty and weights for each M.
- `experiment_config.json`: sampling protocol, calibration parameters
  and source directories.

Standard deviation uses ddof=1 across independent training runs, not
across individual dropout passes. A single run displays zero std and
cannot estimate between-run variation. Nested sampling controls sampling
variation for comparisons but does not guarantee monotonic performance.
The reported performance table alone does not establish that uncertainty
or weights are numerically stable; the saved arrays allow that analysis.

Synchronize the two modified scripts and the current UA-TAF encoder/fusion
modules, plus `run_uataf_alpha_sensitivity.py` and
`run_uataf_weight_analysis.py` (shared helper imports), to the server.
