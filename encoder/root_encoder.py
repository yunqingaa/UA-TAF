"""Extract Root logits from the precomputed source vector in graph NPZ files.

The graph file is the single source for the cached BERTweet representation
(``root``), event label (``y``), and graph identity. Only the Root projection,
normalization and classifier are trainable; no Transformer is loaded here.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


from path_utils import (
    apply_npz_labels,
    configure_dataset_args,
    resolve_checkpoint,
    resolve_graph_dir,
    resolve_input_csv,
    resolve_output,
    select_rows_for_split,
)


ROOT_STATE_PREFIXES = ("root_head.", "norm_root.", "fc_root.")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


class RootEncoder(th.nn.Module):
    """Trainable classifier over a cached 768-D source-post vector."""

    def __init__(
        self,
        input_dim=768,
        representation_dim=128,
        hidden_dim=256,
        num_classes=3,
        dropout=0.35,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.representation_dim = int(representation_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_classes = int(num_classes)
        self.root_head = th.nn.Sequential(
            th.nn.Linear(self.input_dim, self.hidden_dim),
            th.nn.LayerNorm(self.hidden_dim),
            th.nn.ReLU(),
            th.nn.Dropout(dropout),
            th.nn.Linear(self.hidden_dim, self.representation_dim),
            th.nn.ReLU(),
        )
        self.norm_root = th.nn.LayerNorm(self.representation_dim)
        self.fc_root = th.nn.Linear(self.representation_dim, self.num_classes)

    def encode(self, root_embedding):
        root_embedding = root_embedding.float()
        if root_embedding.dim() != 2 or root_embedding.size(1) != self.input_dim:
            raise ValueError(
                "root embedding shape {} does not match [B, {}]".format(
                    tuple(root_embedding.shape), self.input_dim
                )
            )
        return self.norm_root(self.root_head(root_embedding))

    def forward(self, root_embedding):
        return self.fc_root(self.encode(root_embedding))

    def load_full_checkpoint(self, checkpoint_path):
        """Load Root head tensors from a complete or branch-only checkpoint."""
        try:
            checkpoint = th.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except TypeError:
            checkpoint = th.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint does not contain a state dictionary")

        root_state = {}
        for raw_name, tensor in checkpoint.items():
            name = raw_name[7:] if raw_name.startswith("module.") else raw_name
            if name.startswith(ROOT_STATE_PREFIXES):
                root_state[name] = tensor
        expected = set(self.state_dict())
        available = set(root_state)
        missing = sorted(expected - available)
        unexpected = sorted(available - expected)
        if missing or unexpected:
            raise ValueError(
                "incompatible Root checkpoint; missing:{} unexpected:{}".format(
                    missing[:10], unexpected[:10]
                )
            )
        self.load_state_dict(root_state, strict=True)


def load_source_rows(
    csv_path,
    split="all",
    dataset=None,
    fold_id=None,
    graph_dir=None,
    iteration=0,
    split_seed=2026,
):
    """Read event IDs; split membership and labels are resolved elsewhere."""
    rows = []
    seen = set()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            event_id = ""
            for column in ("thread_id", "root_id", "source_id"):
                candidate = str(row.get(column, "")).strip()
                if candidate:
                    event_id = candidate
                    break
            if not event_id or event_id in seen:
                continue
            try:
                label = int(str(row.get("label_id", "")).strip())
            except (TypeError, ValueError):
                label = -1
            rows.append({
                "event_id": event_id,
                "label": label,
                "split": str(row.get("split", "")).strip().lower(),
                "event_group": str(row.get("event", "")).strip() or event_id,
                "platform": str(row.get("platform", "")).strip().lower(),
            })
            seen.add(event_id)

    rows = select_rows_for_split(
        rows,
        split,
        dataset=dataset,
        fold_id=fold_id,
        graph_dir=graph_dir,
        iteration=iteration,
        split_seed=split_seed,
    )
    rows = apply_npz_labels(rows, dataset=dataset, graph_dir=graph_dir)
    if not rows:
        raise ValueError(
            "no Root events found in {} for split {}".format(csv_path, split)
        )
    return rows


class RootNPZDataset(Dataset):
    """Read cached source vectors and authoritative labels from graph NPZ."""

    def __init__(self, rows, graph_dir, input_dim=768):
        self.rows = list(rows)
        self.graph_dir = Path(graph_dir)
        self.input_dim = int(input_dim)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.graph_dir / "{}.npz".format(row["event_id"])
        with np.load(path, allow_pickle=True) as arrays:
            root = np.asarray(arrays["root"], dtype=np.float32).reshape(-1)
            label = int(arrays["y"])
        if root.shape != (self.input_dim,):
            raise ValueError(
                "{} has root shape {}, expected ({},)".format(
                    row["event_id"], root.shape, self.input_dim
                )
            )
        return th.from_numpy(root), th.tensor(label, dtype=th.long)


def extract_z_root(model, dataset, device, batch_size=16):
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    model.eval()
    logits, labels = [], []
    with th.no_grad():
        for root, target in loader:
            logits.append(model(root.to(device)).cpu().float().numpy())
            labels.append(target.numpy())
    return (
        np.asarray([row["event_id"] for row in dataset.rows], dtype=str),
        np.concatenate(logits).astype(np.float32, copy=False),
        np.concatenate(labels).astype(np.int64, copy=False),
    )


def extract_mc_root(model, dataset, device, batch_size=16, samples=20, seed=2026):
    """Estimate frozen Root uncertainty with stochastic dropout passes."""
    from mc_uncertainty import enable_mc_dropout, seed_mc, summarize_mc_probabilities
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    passes = []
    seed_mc(seed)
    with th.no_grad():
        for _ in range(int(samples)):
            enable_mc_dropout(model)
            probabilities = [
                F.softmax(model(root.to(device)), dim=-1).cpu().numpy()
                for root, _ in loader
            ]
            passes.append(np.concatenate(probabilities, axis=0))
    model.eval()
    return summarize_mc_probabilities(np.stack(passes, axis=0))


def save_z_root(output_path, event_ids, z_root, labels, metadata, mc_stats=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        event_ids=event_ids,
        z_root=z_root,
        labels=labels,
    )
    if mc_stats:
        arrays.update(mc_stats)
    np.savez_compressed(output_path, **arrays)
    metadata_path = output_path.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    return output_path, metadata_path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--dataset", default="RumourEval2019")
    parser.add_argument(
        "--checkpoint-fusion-mode",
        choices=["class_conditional", "average"],
        default="class_conditional",
    )
    parser.add_argument(
        "--ablation",
        choices=[
            "full", "no_graph", "no_root", "no_stance",
            "graph_only", "root_only", "stance_only",
        ],
        default="full",
    )
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--fold-id", type=int, default=None)
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--split", choices=["all", "train", "dev", "test"], default="all"
    )
    parser.add_argument("--input-dim", type=int, default=768)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--representation-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_dataset_args(args)
    if args.iteration < 0 or (args.fold_id is not None and args.fold_id < 0):
        raise ValueError("iteration and fold-id must be non-negative")
    if min(
        args.batch_size, args.input_dim,
        args.num_classes, args.representation_dim, args.hidden_dim,
    ) <= 0:
        raise ValueError("batch-size and model dimensions must be positive")

    checkpoint_path = resolve_checkpoint(
        args.checkpoint,
        args.dataset,
        args.checkpoint_fusion_mode,
        args.ablation,
        args.iteration,
        args.fold_id,
    )
    input_csv = resolve_input_csv(args.input_csv, args.dataset)
    graph_dir = resolve_graph_dir(args.graph_dir, args.dataset)
    output_path = resolve_output(
        args.output, args.dataset, args.split, "root",
        args.checkpoint_fusion_mode, args.ablation,
        args.iteration, args.fold_id,
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError("checkpoint not found: {}".format(checkpoint_path))

    set_seed(args.seed)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model = RootEncoder(
        input_dim=args.input_dim,
        representation_dim=args.representation_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    )
    model.load_full_checkpoint(checkpoint_path)
    model.to(device)
    rows = load_source_rows(
        input_csv, args.split, args.dataset, args.fold_id, graph_dir,
        iteration=args.iteration, split_seed=args.seed,
    )
    dataset = RootNPZDataset(rows, graph_dir, input_dim=args.input_dim)
    event_ids, z_root, labels = extract_z_root(
        model, dataset, device, args.batch_size
    )
    metadata = {
        "checkpoint": str(checkpoint_path),
        "input_csv": str(input_csv),
        "graph_dir": str(graph_dir),
        "root_feature_source": "graph_npz.root",
        "label_source": "graph_npz.y",
        "split": args.split,
        "samples": int(len(event_ids)),
        "input_dim": int(args.input_dim),
        "representation_dim": int(args.representation_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_classes": int(args.num_classes),
        "seed": int(args.seed),
    }
    output_path, metadata_path = save_z_root(
        output_path, event_ids, z_root, labels, metadata
    )
    print("saved z_root:", output_path.resolve())
    print("shape:", z_root.shape)
    print("metadata:", metadata_path.resolve())


if __name__ == "__main__":
    main()
