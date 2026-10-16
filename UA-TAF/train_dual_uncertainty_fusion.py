"""Fit Dev-only calibration and run uncertainty-decomposition UA-TAF."""

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch as th
from sklearn.metrics import accuracy_score, f1_score


HERE = Path(__file__).resolve().parent
FUSION_DIR = HERE / "fusion"
sys.path.insert(0, str(FUSION_DIR))

from dual_uncertainty_fusion import (  # noqa: E402
    ABLATIONS,
    ABLATION_LABELS,
    BRANCHES,
    fit_dev_temperatures,
    fuse_mc_branches,
    load_aligned_mc_evidence,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fit-split", choices=["dev"], default="dev")
    parser.add_argument("--evaluate-split", default="dev")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--ablation", choices=(*ABLATIONS, "all"), default="full")
    parser.add_argument("--temperature-reg", type=float, default=0.01)
    return parser.parse_args()


def metrics(probabilities, labels):
    prediction = probabilities.argmax(1)
    class_count = probabilities.shape[1]
    if class_count not in (3, 4):
        raise ValueError("Expected FR/TR/UR or NR/FR/TR/UR class order")
    per_class = f1_score(labels, prediction, labels=list(range(class_count)),
                         average=None, zero_division=0)
    offset = 1 if class_count == 4 else 0
    return {
        "accuracy": float(accuracy_score(labels, prediction)),
        "macro_f1": float(per_class.mean()),
        "fr_f1": float(per_class[offset]),
        "tr_f1": float(per_class[offset + 1]),
        "ur_f1": float(per_class[offset + 2]),
    }


def formatted_metrics(variant, values):
    return {"Variant": variant, **{
        column: "{:.2f}%".format(100 * values[key])
        for column, key in (("Acc", "accuracy"), ("Macro-F1", "macro_f1"),
                            ("FR-F1", "fr_f1"), ("TR-F1", "tr_f1"), ("UR-F1", "ur_f1"))
    }}


def write_diagnostics(path, evidence, output):
    fields = ["event_id", "label", "prediction", "contrast", "adaptive_strength"]
    for branch in output["branches"]:
        fields.extend([
            "{}_u_total_predictive".format(branch),
            "{}_u_expected_predictive".format(branch),
            "{}_u_model_disagreement".format(branch),
            "{}_reliability".format(branch),
            "{}_q".format(branch),
            "{}_weight".format(branch),
        ])
    final = output["fused_probability"].detach().cpu().numpy()
    arrays = {key: value.detach().cpu().numpy() for key, value in output.items() if isinstance(value, th.Tensor)}
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, event_id in enumerate(evidence["event_ids"]):
            row = {
                "event_id": event_id,
                "label": int(evidence["labels"][index]),
                "prediction": int(final[index].argmax()),
                "contrast": float(arrays["contrast"][index]),
                "adaptive_strength": float(arrays["adaptive_strength"][index]),
            }
            for branch_id, branch in enumerate(output["branches"]):
                row.update({
                    "{}_u_total_predictive".format(branch): float(arrays["total_predictive_uncertainty"][index, branch_id]),
                    "{}_u_expected_predictive".format(branch): float(arrays["expected_predictive_uncertainty"][index, branch_id]),
                    "{}_u_model_disagreement".format(branch): float(arrays["model_disagreement_uncertainty"][index, branch_id]),
                    "{}_reliability".format(branch): float(arrays["reliability"][index, branch_id]),
                    "{}_q".format(branch): float(arrays["normalized_reliability"][index, branch_id]),
                    "{}_weight".format(branch): float(arrays["weights"][index, branch_id]),
                })
            writer.writerow(row)


def main(args=None, shared=None):
    args = parse_args() if args is None else args
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0,1]")
    evidence_dir = Path(args.evidence_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.fit_split != "dev":
        raise ValueError("temperature fitting requires Dev evidence")
    if shared is None:
        dev = load_aligned_mc_evidence(evidence_dir, args.fit_split)
        fitted = (None if args.ablation == "no_calibration"
                  else fit_dev_temperatures(dev, args.temperature_reg))
        evaluated = load_aligned_mc_evidence(evidence_dir, args.evaluate_split)
        shared = (fitted, evaluated)
    fitted, evaluated = shared
    if args.ablation == "all":
        rows = []
        for ablation, label in ABLATION_LABELS.items():
            variant_args = copy.copy(args)
            variant_args.ablation = ablation
            variant_args.output_dir = str(output_dir / ablation)
            main(variant_args, shared)
            with open(output_dir / ablation / "summary.json", encoding="utf-8") as handle:
                report = json.load(handle)
            rows.append(formatted_metrics(label, report["metrics"]))
        with open(output_dir / "ablation_results.csv", "w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with open(output_dir / "ablation_metadata.json", "w", encoding="utf-8") as handle:
            json.dump({"evaluate_split": args.evaluate_split, "fit_split": args.fit_split,
                       "metric_format": "percentage, two decimals", "alpha": args.alpha}, handle, indent=2)
        return 0
    temperatures = (
        {branch: 1.0 for branch in BRANCHES}
        if args.ablation == "no_calibration"
        else fitted
    )
    result = fuse_mc_branches(
        evaluated["mc_samples"], temperatures, args.alpha,
        ablation=args.ablation,
    )
    final = result["fused_probability"].detach().cpu().numpy()
    report = {
        "method": "UA-TAF-uncertainty-decomposition",
        "fit_split": args.fit_split,
        "evaluate_split": args.evaluate_split,
        "alpha": args.alpha,
        "ablation": args.ablation,
        "variant": ABLATION_LABELS.get(args.ablation, args.ablation),
        "active_branches": result["branches"],
        "temperatures": temperatures,
        "metrics": metrics(final, evaluated["labels"]),
        "equations": {
            "calibration": (
                "T=1" if args.ablation == "no_calibration"
                else "softmax(z/T), with T fitted on Dev"
            ),
            "decomposition": "U_P=U_A+U_E",
            "reliability": (
                "1" if args.ablation == "no_uncertainty" else "1-U_P"
            ),
            "contrast": (
                "0 (single branch)" if len(result["branches"]) == 1
                else "gamma=1" if args.ablation == "no_reliability_contrast"
                else "K/(2*(K-1))*sum(abs(q-1/K))"
            ),
            "weights": (
                "1 (single branch)" if len(result["branches"]) == 1
                else "(1-alpha)/K + alpha*q" if args.ablation == "no_reliability_contrast"
                else "(1-alpha*contrast)/K + alpha*contrast*q"
            ),
        },
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    formatted = formatted_metrics(report["variant"], report["metrics"])
    with open(output_dir / "metrics.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(formatted))
        writer.writeheader()
        writer.writerow(formatted)
    write_diagnostics(output_dir / "diagnostics.csv", evaluated, result)
    th.save({
        "temperatures": temperatures,
        "alpha": args.alpha,
        "ablation": args.ablation,
        "method": report["method"],
    }, output_dir / "fusion_artifact.pt")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
