"""Export paper-ready auxiliary stance-classifier results.

The script reads one ``metrics.json`` per random seed produced by
``train_rumoureval_stance_classifier.py``. It never reads rumor-veracity
labels. Means and population standard deviations are computed across seeds.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("Support", "Deny", "Query", "Comment")
METRICS = (
    ("Accuracy", "accuracy", None),
    ("Macro-F1", "macro_f1", None),
    ("Support-F1", "f1", "Support"),
    ("Deny-F1", "f1", "Deny"),
    ("Query-F1", "f1", "Query"),
    ("Comment-F1", "f1", "Comment"),
)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def discover_seed_metrics(input_dir):
    paths = sorted(
        input_dir.glob("seed_*/metrics.json"),
        key=lambda path: path.parent.name,
    )
    if not paths:
        raise FileNotFoundError(
            "no seed_*/metrics.json files found under {}".format(input_dir)
        )
    records = []
    seen_seeds = set()
    for path in paths:
        record = load_json(path)
        seed = int(record.get("seed", path.parent.name.split("seed_", 1)[-1]))
        record["seed"] = seed
        if seed in seen_seeds:
            raise ValueError("duplicate stance-classifier seed {}".format(seed))
        seen_seeds.add(seed)
        for split in ("dev", "test"):
            if split not in record:
                raise ValueError("{} is missing '{}' metrics".format(path, split))
            bundle = record[split]
            for key in ("accuracy", "macro_f1", "per_class"):
                if key not in bundle:
                    raise ValueError(
                        "{} {} is missing '{}'".format(path, split, key)
                    )
            missing_classes = set(CLASS_NAMES) - set(bundle["per_class"])
            if missing_classes:
                raise ValueError(
                    "{} {} is missing classes {}".format(
                        path, split, sorted(missing_classes)
                    )
                )
        records.append(record)
    return sorted(records, key=lambda item: int(item["seed"]))


def metric_value(record, split, metric_key, class_name):
    if class_name is None:
        return float(record[split][metric_key])
    return float(record[split]["per_class"][class_name][metric_key])


def format_mean_std(mean, std, decimals):
    return ("{:.%df} ± {:.%df}" % (decimals, decimals)).format(mean, std)


def build_tables(records, decimals):
    summary_rows = []
    run_rows = []
    for metric_label, metric_key, class_name in METRICS:
        summary = {"Metric": metric_label}
        for split in ("dev", "test"):
            values = np.asarray([
                metric_value(record, split, metric_key, class_name)
                for record in records
            ], dtype=np.float64)
            mean = float(values.mean())
            std = float(values.std(ddof=0))
            prefix = split.capitalize()
            summary["{}_Mean".format(prefix)] = mean
            summary["{}_Std".format(prefix)] = std
            summary["{}_Mean_Std".format(prefix)] = format_mean_std(
                mean, std, decimals
            )
            for record, value in zip(records, values):
                run_rows.append({
                    "seed": int(record["seed"]),
                    "split": split,
                    "metric": metric_label,
                    "value": float(value),
                })
        summary_rows.append(summary)
    return summary_rows, run_rows


def write_csv(path, rows, fields):
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = [
        "| Metric | Dev Mean ± Std | Test Mean ± Std |",
        "|---|---:|---:|",
    ]
    lines.extend(
        "| {} | {} | {} |".format(
            row["Metric"], row["Dev_Mean_Std"], row["Test_Mean_Std"]
        )
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path, rows):
    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Metric & Dev Mean $\\pm$ Std & Test Mean $\\pm$ Std \\\\",
        "\\midrule",
    ]
    lines.extend(
        "{} & {} & {} \\\\".format(
            row["Metric"].replace("-", "--"),
            row["Dev_Mean_Std"].replace("±", "$\\pm$"),
            row["Test_Mean_Std"].replace("±", "$\\pm$"),
        )
        for row in rows
    )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default=str(PROJECT_ROOT / "checkpoint" / "rumoureval_stance"),
        help="Directory containing seed_*/metrics.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "results" / "stance_classifier"),
    )
    parser.add_argument("--decimals", type=int, default=3)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.decimals < 0:
        raise ValueError("decimals must be non-negative")
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = discover_seed_metrics(input_dir)
    summary_rows, run_rows = build_tables(records, args.decimals)
    summary_fields = [
        "Metric", "Dev_Mean", "Dev_Std", "Dev_Mean_Std",
        "Test_Mean", "Test_Std", "Test_Mean_Std",
    ]
    write_csv(
        output_dir / "stance_classifier_table.csv",
        summary_rows,
        summary_fields,
    )
    write_csv(
        output_dir / "stance_classifier_runs_long.csv",
        run_rows,
        ["seed", "split", "metric", "value"],
    )
    write_markdown(
        output_dir / "stance_classifier_table.md", summary_rows
    )
    write_latex(
        output_dir / "stance_classifier_table.tex", summary_rows
    )
    metadata = {
        "source_directory": str(input_dir),
        "num_runs": len(records),
        "seeds": [int(record["seed"]) for record in records],
        "class_order": list(CLASS_NAMES),
        "standard_deviation": "population standard deviation (ddof=0)",
        "selection_and_calibration": (
            "best epoch selected on Dev; temperature and decision thresholds "
            "estimated on Dev; official Test evaluated once per seed"
        ),
        "label_scope": (
            "stance labels only; rumor-veracity labels are not read or used "
            "by this exporter or auxiliary stance-classifier objective"
        ),
    }
    with open(
        output_dir / "stance_classifier_table_metadata.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)

    print("Metric\tDev Mean ± Std\tTest Mean ± Std")
    for row in summary_rows:
        print("{}\t{}\t{}".format(
            row["Metric"], row["Dev_Mean_Std"], row["Test_Mean_Std"]
        ))
    print("\nPaper-ready stance table:", output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
