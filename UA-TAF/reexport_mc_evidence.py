"""Add missing MC samples to existing Dev/Test evidence without retraining.

Accepts the same arguments as train_encoders.py. Model dimensions, dropout,
seed and MC count must match the original training/export settings.
"""

import os
import argparse
import sys
from pathlib import Path

import numpy as np
import train_encoders as training


def build_model(branch, args):
    if branch == "root":
        return training.RootEncoder(input_dim=args.in_feats, representation_dim=args.representation_dim,
                                    hidden_dim=args.root_hidden_dim, num_classes=args.num_classes, dropout=args.root_dropout)
    if branch == "graph":
        return training.GraphEncoder(in_feats=args.in_feats, hid_feats=args.graph_hidden_dim,
                                     out_feats=args.graph_out_dim, num_classes=args.num_classes, dropout=args.graph_dropout)
    return training.StanceEncoder(stance_dim=args.stance_dim, text_dim=args.reply_text_dim,
                                  text_hidden_dim=args.reply_text_hidden_dim, stance_hidden_dim=args.stance_hidden_dim,
                                  representation_dim=args.representation_dim, num_classes=args.num_classes,
                                  dropout=args.stance_dropout)


def main():
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--mc-evidence-name", default="evidence")
    extra, remaining = extra_parser.parse_known_args()
    if Path(extra.mc_evidence_name).name != extra.mc_evidence_name or extra.mc_evidence_name in ("", ".", ".."):
        raise ValueError("--mc-evidence-name must be a directory name")
    sys.argv = [sys.argv[0], *remaining]
    args = training.validate_and_resolve_args(training.parse_args())
    training.configure_dataset_args(args)
    folds = range(5) if args.dataset == "PHEME" and args.fold_id is None else [args.fold_id]
    branches = training.BRANCHES if args.branch == "all" else [args.branch]
    graph_dir = training.resolve_graph_dir(args.graph_dir, args.dataset)
    jobs = []
    for fold in folds:
        output = (Path(args.output_dir) if args.output_dir else
                  training.resolve_training_dir(args.dataset, args.iteration, fold))
        if args.output_dir and args.dataset == "PHEME" and args.fold_id is None:
            output = output / f"fold{fold}"
        for branch in branches:
            checkpoint = output / "checkpoints" / f"{branch}_best.m"
            for split in ("dev", "test"):
                evidence = output / "evidence" / f"{split}_{branch}.npz"
                if not checkpoint.is_file() or not evidence.is_file():
                    raise FileNotFoundError(f"Need existing checkpoint and evidence: {checkpoint}, {evidence}")
                destination = output / extra.mc_evidence_name / evidence.name
                jobs.append((branch, split, checkpoint, evidence, destination))
    device = training.th.device("cuda:0" if training.th.cuda.is_available() else "cpu")
    for branch, split, checkpoint, path, destination in jobs:
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        if destination == path and "mc_probability_samples" in arrays and arrays["mc_probability_samples"].shape[0] >= args.mc_samples:
            print("Already contains sufficient MC samples:", path)
            continue
        ids, labels = arrays["event_ids"].astype(str), arrays["labels"]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate event IDs: {path}")
        rows = [{"event_id": eid, "label": int(label)} for eid, label in zip(ids, labels)]
        model = build_model(branch, args).to(device)
        training.load_state(model, checkpoint)
        if branch == "root":
            dataset = training.RootNPZDataset(rows, graph_dir, input_dim=args.in_feats)
            extracted = training.extract_z_root(model, dataset, device, args.batch_size)
            extractor, offset = training.extract_mc_root, 0
        elif branch == "graph":
            dataset = training.GraphNPZDataset(rows, graph_dir)
            extracted = training.extract_z_graph(model, dataset, device, args.batch_size)
            extractor, offset = training.extract_mc_graph, 100
        else:
            dataset = training.StanceNPZDataset(rows, graph_dir, stance_dim=args.stance_dim, text_dim=args.reply_text_dim)
            extracted = training.extract_z_stance(model, dataset, device, args.batch_size)
            extractor, offset = training.extract_mc_stance, 200
        if not np.array_equal(np.asarray(extracted[0]).astype(str), ids) or not np.array_equal(extracted[-1], labels):
            raise ValueError(f"Event IDs/labels changed: {path}")
        if not np.allclose(extracted[1], arrays[f"z_{branch}"], rtol=1e-3, atol=1e-4):
            raise ValueError(f"Checkpoint logits disagree with {path}; check model arguments and checkpoint provenance")
        stats = extractor(model, dataset, device, args.batch_size, args.mc_samples,
                          args.seed + args.iteration + offset + {"dev": 20, "test": 30}[split])
        if "mc_probability_samples" not in stats:
            raise RuntimeError("Update UA-TAF/encoder/mc_uncertainty.py: exporter still omits MC samples")
        arrays.update(stats)
        arrays["mc_export_seed"] = args.seed + args.iteration + offset + {"dev": 20, "test": 30}[split]
        arrays["mc_export_batch_size"] = args.batch_size
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".mc.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        print("Updated:", destination, "MC shape:", stats["mc_probability_samples"].shape)
        del model
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
