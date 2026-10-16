# RumourEval2019: weights and stable cases

Both experiments use frozen Dev/Test MC evidence from five independently
trained runs (`iter0` through `iter4`) on the fixed RumourEval2019 split.
Calibrate each run's three branches on Dev only. Use one fixed alpha
(default 1.0) and the full UA-TAF fusion. Neither script trains encoders.
Test evidence must already have been exported with training `--export-test`.

Required files in each `results/UA_TAF_training/RumourEval2019/iterN/evidence/`:
`dev_graph.npz`, `dev_root.npz`, `dev_stance.npz`, and the three `test_*.npz`
files, containing logits, labels, event IDs and `mc_probability_samples`.
The scripts align events by ID across runs and reject differing Test sets
or labels. Propagation is called `graph` internally. Class IDs are
0=FR, 1=TR, 2=UR.

## Weight analysis

```bash
python run_uataf_weight_analysis.py --iterations 0 1 2 3 4 --alpha 1.0
python plot_uataf_weights.py
```

Outputs in `results/UA_TAF_weight_analysis/RumourEval2019/`:

- `event_run_weights.csv`: every event in every run, including true label,
  fused prediction, weight, uncertainty, reliability, q and contrast.
- `run_branch_summary.csv`: per-run mean, between-event standard deviation,
  quantiles, range, above-uniform fraction and dominant-branch share.
- `event_weight_summary.csv`: each event's branch weight mean and sample
  standard deviation across runs.
- `branch_summary.csv`: separate statistics for between-event variability
  and between-run variability. These are different sources of variation.
- `fusion_weights.npz`: aligned [run, event, branch] weights.
- `analysis_summary.json`: alpha, calibration temperatures and summary.

Tied maximum weights divide dominance credit evenly. Uniform weights do
not count as Propagation wins. Small differences use tolerance 1e-6.
Nonuniform weights alone are not evidence of changing behavior across
events: inspect `mean_within_run_event_std` and event-level plots as well.
These descriptive statistics do not treat 5 observations of the same
event as 5 independent events or claim statistical significance.

The plotting script requires matplotlib and produces PNG/PDF/SVG:
an event-mean weight distribution, a heatmap containing every Test event,
and the mean branch weights of individual runs. The heatmap order is saved
as CSV. It is sorted by dominant branch and weight, without label filtering.

## Stable three-case analysis (Table 6)

```bash
python run_uataf_case_analysis.py --iterations 0 1 2 3 4 --alpha 1.0
```

Optional `--threads-csv data/csv/RumourEval2019_threads.csv` adds source text
if the CSV contains a `source_text`, `root_text` or `text` column.
Set `--training-root` or `--output-dir` to use other directories.

Exactly five distinct runs are required. The three mutually exclusive
definitions are applied separately in every run:

| Case | Required condition in all five runs |
| --- | --- |
| Evidence agreement | All three calibrated branch predictions agree with the true label, and fusion is correct |
| Conflict corrected | Branch predictions disagree; at least one is correct and one is wrong; fusion is correct |
| Fusion failure | Fusion is wrong (branch agreement is unrestricted) |

Predictions use argmax of calibrated MC-mean probabilities. "Corrected"
means fusion remains correct despite incorrect branch evidence; it does
not require a wrong majority vote, nor does it claim all branches were
wrong. "Same case property" does not require each branch to predict the
same label in all runs; the table preserves label counts instead of hiding
changes behind a majority label. The failure case can predict different
wrong labels across runs.

The script first exports all stable candidates. For each category, it
selects the candidate closest to that category's median signature
(mean weights, branch confidence and fused confidence), standardized by
the candidate feature standard deviations. Lexicographic event ID breaks
ties. This is a reproducible descriptive choice, not evidence that a
selected event represents a whole dataset. No category requirements are
relaxed if a candidate is unavailable.

Outputs in `results/UA_TAF_case_analysis/RumourEval2019/`:

- `table6.md`, `table6.csv`: three cases, true labels, five-run prediction
  counts, and branch weight means ± sample standard deviations (ddof=1).
- `selected_case_runs.csv`: the selected events' 15 run records, including
  per-class branch/fusion probabilities, predictions and weights (fewer
  records if a category has no eligible event).
- `stable_candidates.csv`: every stable candidate, rank and selection score.
- `all_event_case_counts.csv`: number of runs satisfying each definition
  for every Test event, including those that failed the five-run rule.
- `case_evidence.npz`: aligned probabilities and weights of all events/runs.
- `selection_report.json`: definitions, counts, IDs, alpha and temperatures.

Missing categories appear as `NO_ELIGIBLE_EVENT` in the table and null in
the report. Do not describe the experiment as containing all three cases
unless all three actually have eligible events. Use qualitative case
analysis for interpretation, not to choose alpha or report aggregate
performance from selected examples.

Synchronize `run_uataf_case_analysis.py`, `run_uataf_weight_analysis.py`,
`plot_uataf_weights.py` and the current
`UA-TAF/fusion/dual_uncertainty_fusion.py` to the server.
