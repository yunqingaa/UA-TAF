"""Shared automatic path construction for standalone evidence encoders."""

import random
import pickle
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DEV_RATIO = 0.1
DEFAULT_DEV_SPLIT_SEED = 2026


DATASET_SPECS = {
    "rumoureval2019": {
        "name": "RumourEval2019",
        "storage": "RumourEval2019",
        "csv": "RumourEval2019_threads.csv",
        "num_classes": 3,
        "class_names": ["FR", "TR", "UR"],
    },
    "pheme": {
        # PHEME4L is the four-label graph set containing generated stance_x.
        "name": "PHEME",
        "storage": "PHEME4L",
        "csv": "PHEME_threads.csv",
        "num_classes": 4,
        "class_names": ["NR", "FR", "TR", "UR"],
    },
    "pheme-scheme": {
        "name": "PHEME-SCHEME",
        "storage": "PHEMEScheme",
        "csv": "PHEME_RumourScheme_threads.csv",
        "num_classes": 3,
        "class_names": ["FR", "TR", "UR"],
    },
}


def get_dataset_spec(dataset):
    """Resolve paper-facing dataset names to processed-file locations."""
    key = str(dataset).strip().lower().replace("_", "-")
    aliases = {
        "rumoureval": "rumoureval2019",
        "pheme4": "pheme",
        "pheme4l": "pheme",
        "phemescheme": "pheme-scheme",
        "pheme-rumourscheme": "pheme-scheme",
    }
    key = aliases.get(key, key)
    if key not in DATASET_SPECS:
        raise ValueError(
            "unsupported UA-TAF dataset {!r}; choose RumourEval2019, "
            "PHEME, or PHEME-SCHEME".format(dataset)
        )
    return dict(DATASET_SPECS[key])


def configure_dataset_args(args):
    """Canonicalize ``args.dataset`` and infer/validate class count."""
    spec = get_dataset_spec(args.dataset)
    args.dataset = spec["name"]
    if getattr(args, "num_classes", None) is None:
        args.num_classes = spec["num_classes"]
    elif int(args.num_classes) != spec["num_classes"]:
        raise ValueError(
            "{} uses {} classes, but --num-classes {} was supplied".format(
                spec["name"], spec["num_classes"], args.num_classes
            )
        )
    return spec


def resolve_training_dir(dataset, iteration, fold_id=None):
    """Use a fold-specific directory so five-fold runs never overwrite."""
    spec = get_dataset_spec(dataset)
    path = (
        PROJECT_ROOT / "results" / "UA_TAF_training" / spec["name"]
        / "iter{}".format(iteration)
    )
    if fold_id is not None:
        path = path / "fold{}".format(fold_id)
    return path


def apply_npz_labels(rows, dataset=None, graph_dir=None):
    """Replace CSV labels with authoritative graph-NPZ ``y`` labels."""
    if graph_dir is None:
        if not dataset:
            raise ValueError(
                "dataset or graph_dir is required to load NPZ labels"
            )
        graph_dir = resolve_graph_dir(None, dataset)
    graph_dir = Path(graph_dir)

    labeled_rows = []
    mismatches = []
    missing = []
    for row in rows:
        event_id = str(row["event_id"])
        graph_path = graph_dir / "{}.npz".format(event_id)
        if not graph_path.is_file():
            missing.append(event_id)
            continue
        with np.load(graph_path, allow_pickle=True) as arrays:
            if "y" not in arrays.files:
                missing.append(event_id)
                continue
            npz_label = int(arrays["y"])
        csv_label = int(row.get("label", -1))
        if csv_label >= 0 and csv_label != npz_label:
            mismatches.append((event_id, csv_label, npz_label))
        labeled_row = dict(row)
        labeled_row["label"] = npz_label
        labeled_rows.append(labeled_row)

    if missing:
        raise ValueError(
            "{} selected events have no NPZ y label in {}; examples: {}".format(
                len(missing),
                graph_dir,
                missing[:5],
            )
        )
    if mismatches:
        print(
            "warning: replaced {} CSV labels with NPZ labels; examples: {}".format(
                len(mismatches),
                mismatches[:5],
            )
        )
    return labeled_rows


def build_bigcn_folds(rows, seed=DEFAULT_DEV_SPLIT_SEED):
    """Original Twitter BiGCN class-wise floor(20%) slicing, including tails."""
    rng = random.Random(seed)
    classes = [[] for _ in range(4)]
    seen = set()
    for row in rows:
        event_id = str(row["event_id"])
        label = int(row["label"])
        if event_id in seen or label not in range(4):
            raise ValueError("PHEME requires unique event IDs and labels 0..3")
        seen.add(event_id)
        classes[label].append(event_id)
    for ids in classes:
        rng.shuffle(ids)
    folds = []
    for fold in range(5):
        test_ids, train_ids = [], []
        for ids in classes:
            size = int(len(ids) * 0.2)
            start, end = fold * size, (fold + 1) * size
            test_ids.extend(ids[start:end])
            train_ids.extend(ids[:start] + ids[end:])
        rng.shuffle(test_ids)
        rng.shuffle(train_ids)
        folds.append((test_ids, train_ids))
    return folds


def select_rows_for_split(
    rows,
    split,
    dataset=None,
    fold_id=None,
    graph_dir=None,
    dev_ratio=DEFAULT_DEV_RATIO,
    seed=DEFAULT_DEV_SPLIT_SEED,
    iteration=0,
    split_seed=DEFAULT_DEV_SPLIT_SEED,
):
    """Select one split from a single authoritative partition source.

    PHEME regenerates original BiGCN folds using split_seed + iteration.
    For other datasets, when ``data/<dataset>/folds.pkl`` exists, its ``(test_ids, train_ids)``
    tuple defines both the official Test set and the pool from which Train and
    Dev are derived. CSV ``split`` values are then ignored; the CSV supplies
    only event content and labels. This prevents mixed partition authorities.
    """
    split = str(split).lower()
    if split == "all":
        return list(rows)
    if split not in {"train", "dev", "test"}:
        raise ValueError("unsupported split: {}".format(split))

    spec = get_dataset_spec(dataset) if dataset else None
    storage_dataset = spec["storage"] if spec else dataset
    row_by_id = {str(row["event_id"]): row for row in rows}
    folds_path = (
        PROJECT_ROOT / "data" / storage_dataset / "folds.pkl"
        if dataset else None
    )
    use_folds = folds_path is not None and folds_path.is_file()

    is_pheme = spec is not None and spec["name"] == "PHEME"
    if is_pheme or use_folds:
        if is_pheme:
            # Prepared graphs may exclude CSV threads (e.g. unusable trees).
            # Reuse only the prepared sample roster, never its old partition.
            if use_folds:
                with open(folds_path, "rb") as handle:
                    prepared_folds = pickle.load(handle)
                prepared_ids = {
                    str(event_id)
                    for pair in prepared_folds for part in pair for event_id in part
                }
                missing_rows = prepared_ids - set(row_by_id)
                if not prepared_ids or missing_rows:
                    raise ValueError("PHEME prepared roster is empty or has IDs absent from CSV: {}".format(
                        sorted(missing_rows)[:5]))
                eligible_rows = [row for row in rows if str(row["event_id"]) in prepared_ids]
                if len(eligible_rows) != len(rows):
                    print("PHEME sample roster: using {} prepared events; excluding {} CSV rows outside roster".format(
                        len(eligible_rows), len(rows) - len(eligible_rows)))
            else:
                eligible_rows = rows
            labeled_rows = apply_npz_labels(eligible_rows, dataset, graph_dir)
            folds = build_bigcn_folds(labeled_rows, split_seed + iteration)
        else:
            with open(folds_path, "rb") as handle:
                folds = pickle.load(handle)
        selected_fold = 0 if fold_id is None else int(fold_id)
        if selected_fold < 0 or selected_fold >= len(folds):
            raise ValueError(
                "fold {} is outside available fold range [0, {})".format(
                    selected_fold,
                    len(folds),
                )
            )
        fold_test_ids, fold_train_ids = folds[selected_fold]
        ordered_test_ids = [str(event_id) for event_id in fold_test_ids]
        ordered_train_ids = [str(event_id) for event_id in fold_train_ids]
        if split == "test":
            selected_ids = set(ordered_test_ids)
            return [
                row for row in rows
                if str(row["event_id"]) in selected_ids
            ]
    else:
        # Fallback for datasets without folds.pkl: the CSV is the sole
        # partition source, including an explicit Dev split when available.
        if split == "test":
            return [
                row for row in rows
                if str(row.get("split", "")).lower() == "test"
            ]
        explicit_dev = any(
            str(row.get("split", "")).lower() == "dev" for row in rows
        )
        if explicit_dev:
            return [
                row for row in rows
                if str(row.get("split", "")).lower() == split
            ]
        ordered_train_ids = [
            str(row["event_id"])
            for row in rows
            if str(row.get("split", "")).lower() == "train"
        ]

    class_to_ids = {}
    graph_dir = (
        Path(graph_dir)
        if graph_dir is not None
        else (
            PROJECT_ROOT / "data" / "{}graph".format(storage_dataset)
            if dataset else None
        )
    )
    for event_id in ordered_train_ids:
        row = row_by_id.get(event_id)
        label = int(row.get("label", -1)) if row is not None else -1
        graph_path = (
            graph_dir / "{}.npz".format(event_id)
            if graph_dir is not None else None
        )
        if graph_path is not None and graph_path.is_file():
            with np.load(graph_path, allow_pickle=True) as arrays:
                if "y" in arrays.files:
                    label = int(arrays["y"])
        if label >= 0:
            class_to_ids.setdefault(label, []).append(event_id)

    rng = random.Random(seed)
    dev_ids = set()
    for label in sorted(class_to_ids):
        ids = list(class_to_ids[label])
        rng.shuffle(ids)
        if len(ids) <= 1:
            n_dev = 0
        else:
            n_dev = max(1, int(round(len(ids) * dev_ratio)))
            n_dev = min(n_dev, len(ids) - 1)
        dev_ids.update(ids[:n_dev])

    selected_ids = (
        dev_ids
        if split == "dev"
        else set(ordered_train_ids) - dev_ids
    )
    return [
        row for row in rows if str(row["event_id"]) in selected_ids
    ]


def resolve_checkpoint(
    checkpoint,
    dataset,
    fusion_mode,
    ablation,
    iteration,
    fold_id,
):
    """Return an explicit checkpoint or construct the training-script name."""
    if checkpoint:
        return Path(checkpoint).expanduser().resolve()

    if fold_id is None:
        filename = "BiGCN_{}_{}_iter{}_{}.m".format(
            fusion_mode,
            ablation,
            iteration,
            dataset,
        )
    else:
        filename = "BiGCN_{}_{}_iter{}_fold{}_{}.m".format(
            fusion_mode,
            ablation,
            iteration,
            fold_id,
            dataset,
        )
    return (PROJECT_ROOT / "checkpoint" / filename).resolve()


def resolve_output(
    output,
    dataset,
    split,
    branch,
    fusion_mode,
    ablation,
    iteration,
    fold_id,
):
    """Return an explicit output or construct a collision-resistant NPZ name."""
    if output:
        return Path(output).expanduser().resolve()

    fields = [
        branch,
        fusion_mode,
        ablation,
        "iter{}".format(iteration),
    ]
    if fold_id is not None:
        fields.append("fold{}".format(fold_id))
    filename = "_".join(fields) + ".npz"
    return (
        PROJECT_ROOT
        / "results"
        / "UA-TAF"
        / "evidence"
        / dataset
        / split
        / filename
    ).resolve()


def resolve_input_csv(input_csv, dataset):
    if input_csv:
        return Path(input_csv).expanduser().resolve()
    spec = get_dataset_spec(dataset)
    return (PROJECT_ROOT / "data" / "csv" / spec["csv"]).resolve()


def resolve_graph_dir(graph_dir, dataset):
    if graph_dir:
        return Path(graph_dir).expanduser().resolve()
    spec = get_dataset_spec(dataset)
    return (
        PROJECT_ROOT / "data" / "{}graph".format(spec["storage"])
    ).resolve()
