"""Extract and save the BiGCN Graph branch logits ``z_graph``.

The architecture and parameter names match the Graph branch in
``model/BiGCN_ConflictFusion.py``:

    x -> TD-GCN -> mean --\
                                 concat + topology statistics
    x -> BU-GCN -> mean --/              |
                                      projection + MLP -> z_graph

The command-line entry point loads Graph parameters from a complete model
checkpoint and writes event IDs, raw logits and labels to a compressed NPZ.
"""

import argparse
import csv
import json
import random
from pathlib import Path
import math
import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
# from torch_geometric.nn import GCNConv
# from torch_scatter import scatter_mean
from torch_geometric.nn import (
    GCNConv,
    global_mean_pool,
    global_max_pool,
    global_add_pool,
)

from path_utils import (
    apply_npz_labels,
    configure_dataset_args,
    resolve_checkpoint,
    resolve_graph_dir,
    resolve_input_csv,
    resolve_output,
    select_rows_for_split,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_STATE_PREFIXES = (
    "TDrumorGCN.",
    "BUrumorGCN.",
    "graph_projection.",
    "norm_graph.",
    "fc_graph.",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


# class DirectionalGCN(th.nn.Module):
#     """Two-layer directional GCN used by both TD and BU branches."""

#     def __init__(
#         self, in_feats=768, hid_feats=64, out_feats=64, dropout=0.2
#     ):
#         super().__init__()
#         self.conv1 = GCNConv(in_feats, hid_feats)
#         self.conv2 = GCNConv(hid_feats, out_feats)
#         self.dropout = float(dropout)

#     def forward(self, data, edge_index):
#         h = self.conv1(data.x.float(), edge_index)
#         h = F.relu(h)
#         h = F.dropout(h, p=self.dropout, training=self.training)
#         h = self.conv2(h, edge_index)
#         h = F.relu(h)
#         return scatter_mean(h, data.batch, dim=0)
class DirectionalGCN(th.nn.Module):
    """Two-layer directional GCN with gated reply-only max pooling.

    The graph representation starts close to ordinary mean pooling.
    A learnable scalar gate controls how much reply-only max information
    is added. The source/root node is excluded from max pooling to reduce
    semantic overlap with the independent Root branch.
    """

    def __init__(
        self,
        in_feats=768,
        hid_feats=64,
        out_feats=64,
        dropout=0.2,
        initial_max_gate=0.1,
    ):
        super().__init__()

        self.in_feats = int(in_feats)
        self.hid_feats = int(hid_feats)
        self.out_feats = int(out_feats)
        self.dropout = float(dropout)

        if not 0.0 < initial_max_gate < 1.0:
            raise ValueError(
                "initial_max_gate must be between 0 and 1"
            )

        self.conv1 = GCNConv(
            self.in_feats,
            self.hid_feats,
        )

        self.conv2 = GCNConv(
            self.hid_feats,
            self.out_feats,
        )

        # sigmoid(-2.197...) is approximately 0.1.
        initial_logit = math.log(
            initial_max_gate
            / (1.0 - initial_max_gate)
        )

        self.max_gate_logit = th.nn.Parameter(
            th.tensor(
                initial_logit,
                dtype=th.float32,
            )
        )

    def forward(
        self,
        data,
        edge_index,
    ):
        x = data.x.float()

        h = self.conv1(
            x,
            edge_index,
        )
        h = F.relu(h)

        h = F.dropout(
            h,
            p=self.dropout,
            training=self.training,
        )

        h = self.conv2(
            h,
            edge_index,
        )
        h = F.relu(h)

        batch_size = int(data.num_graphs)

        # All-node mean preserves the original stable graph summary.
        mean_feature = global_mean_pool(
            h,
            data.batch,
            size=batch_size,
        )

        if hasattr(data, "root_mask"):
            root_mask = data.root_mask.bool()
        else:
            # Compatibility fallback. Formal experiments should always
            # provide root_mask from GraphNPZDataset.
            root_mask = th.zeros(
                h.size(0),
                dtype=th.bool,
                device=h.device,
            )

        reply_mask = ~root_mask

        if reply_mask.any():
            reply_max = global_max_pool(
                h[reply_mask],
                data.batch[reply_mask],
                size=batch_size,
            )

            # Count replies in every graph so graphs without replies can
            # safely fall back to mean pooling.
            reply_count = global_add_pool(
                reply_mask.to(h.dtype).unsqueeze(1),
                data.batch,
                size=batch_size,
            ).squeeze(1)

            reply_max = th.where(
                reply_count.unsqueeze(1) > 0,
                reply_max,
                mean_feature,
            )
        else:
            reply_max = mean_feature

        gate = th.sigmoid(
            self.max_gate_logit
        )

        # Convex interpolation:
        # gate=0 -> ordinary mean pooling
        # gate=1 -> reply-only max pooling
        pooled = (
            mean_feature
            + gate * (
                reply_max - mean_feature
            )
        )

        return pooled
class GraphEncoder(th.nn.Module):
    """Independent BiGCN graph classifier returning raw ``z_graph``."""

    def __init__(
        self,
        in_feats=768,
        hid_feats=64,
        out_feats=64,
        num_classes=3,
        representation_dim=128,
        classifier_hidden_dim=64,
        dropout=0.2,
    ):
        super().__init__()
        # Names intentionally match the complete model checkpoint.
        self.TDrumorGCN = DirectionalGCN(
            in_feats,
            hid_feats,
            out_feats,
            dropout,
        )
        self.BUrumorGCN = DirectionalGCN(
            in_feats,
            hid_feats,
            out_feats,
            dropout,
        )
        # TD and BU each return one out_feats-dimensional mean-pooled vector.
        pooled_dim = 2 * out_feats
        # pooled_dim = 4 * out_feats
        topology_dim = 4
        self.graph_projection = th.nn.Sequential(
            th.nn.Linear(pooled_dim + topology_dim, representation_dim),
            th.nn.ReLU(),
            th.nn.Dropout(dropout),
        )
        self.norm_graph = th.nn.LayerNorm(representation_dim)
        self.fc_graph = th.nn.Sequential(
            th.nn.Linear(representation_dim, classifier_hidden_dim),
            th.nn.ReLU(),
            th.nn.Dropout(dropout),
            th.nn.Linear(classifier_hidden_dim, num_classes),
        )

    def encode(self, data):
        td = self.TDrumorGCN(data, data.edge_index)
        bu = self.BUrumorGCN(data, data.BU_edge_index)
        if not hasattr(data, "topology_features"):
            raise ValueError("graph batch is missing topology_features")
        topology = data.topology_features.float().view(td.size(0), -1)
        graph_input = th.cat((bu, td, topology), dim=1)
        return self.norm_graph(self.graph_projection(graph_input))

    def forward(self, data):
        """Return raw class logits ``z_graph`` with shape ``[B, C]``."""
        return self.fc_graph(self.encode(data))

    def load_full_checkpoint(self, checkpoint_path):
        """Load only Graph-branch tensors from a complete checkpoint."""
        try:
            checkpoint = th.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            checkpoint = th.load(checkpoint_path, map_location="cpu")

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint does not contain a state dictionary")

        graph_state = {}
        for raw_name, tensor in checkpoint.items():
            name = raw_name[7:] if raw_name.startswith("module.") else raw_name
            if name.startswith(GRAPH_STATE_PREFIXES):
                graph_state[name] = tensor

        expected = set(self.state_dict())
        available = set(graph_state)
        missing = sorted(expected - available)
        unexpected = sorted(available - expected)
        if missing or unexpected:
            raise ValueError(
                "incompatible Graph checkpoint; missing:{} unexpected:{}".format(
                    missing[:10],
                    unexpected[:10],
                )
            )
        self.load_state_dict(graph_state, strict=True)


def load_event_rows(
    csv_path,
    split="all",
    dataset=None,
    fold_id=None,
    graph_dir=None,
    iteration=0,
    split_seed=2026,
):
    """Read unique event IDs and labels from the common thread CSV."""
    rows = []
    seen = set()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            row_split = str(row.get("split", "")).strip().lower()

            event_id = ""
            for column in ("thread_id", "root_id", "source_id"):
                candidate = str(row.get(column, "")).strip()
                if candidate:
                    event_id = candidate
                    break
            if not event_id or event_id in seen:
                continue

            label_text = str(row.get("label_id", "")).strip()
            try:
                label = int(label_text)
            except (TypeError, ValueError):
                label = -1
            rows.append({
                "event_id": event_id,
                "label": label,
                "split": row_split,
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
    rows = apply_npz_labels(
        rows,
        dataset=dataset,
        graph_dir=graph_dir,
    )
    if not rows:
        raise ValueError(
            "no event IDs found in {} for split {}".format(csv_path, split)
        )
    return rows


class GraphNPZDataset(Dataset):
    """Read propagation graphs without loading Root or Stance inputs."""

    def __init__(self, rows, graph_dir):
        self.graph_dir = Path(graph_dir)
        self.rows = []
        missing = []
        for row in rows:
            path = self.graph_dir / "{}.npz".format(row["event_id"])
            if path.exists():
                self.rows.append(row)
            else:
                missing.append(row["event_id"])
        if missing:
            print(
                "warning: skipped {} events without graph files; examples: {}".format(
                    len(missing),
                    missing[:5],
                )
            )
        if not self.rows:
            raise ValueError("no matching graph files found in {}".format(graph_dir))

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def topology_features(edge_index, num_nodes, root_index):
        """Return size, depth, branching and leaf statistics for one tree."""
        num_nodes = int(num_nodes)
        if num_nodes <= 0:
            raise ValueError("a propagation graph must contain at least one node")
        edges = edge_index.cpu().numpy()
        out_degree = np.zeros(num_nodes, dtype=np.float32)
        adjacency = [[] for _ in range(num_nodes)]
        if edges.size:
            for source, target in edges.T:
                source, target = int(source), int(target)
                if not (0 <= source < num_nodes and 0 <= target < num_nodes):
                    raise ValueError("edge index exceeds the number of nodes")
                out_degree[source] += 1.0
                # Undirected distance makes the statistic robust to the
                # stored TD/BU orientation.
                adjacency[source].append(target)
                adjacency[target].append(source)

        root_index = int(root_index)
        if root_index < 0 or root_index >= num_nodes:
            root_index = 0
        distances = [-1] * num_nodes
        distances[root_index] = 0
        queue = [root_index]
        for node in queue:
            for neighbour in adjacency[node]:
                if distances[neighbour] < 0:
                    distances[neighbour] = distances[node] + 1
                    queue.append(neighbour)
        max_depth = max(distances) if distances else 0
        non_leaf = int((out_degree > 0).sum())
        mean_branching = float(out_degree.sum()) / max(non_leaf, 1)
        leaf_ratio = float((out_degree == 0).mean())
        return th.tensor(
            [[
                np.log1p(num_nodes),
                np.log1p(max(max_depth, 0)),
                np.log1p(mean_branching),
                leaf_ratio,
            ]],
            dtype=th.float32,
        )

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.graph_dir / "{}.npz".format(row["event_id"])
        with np.load(path, allow_pickle=True) as arrays:
            x = th.as_tensor(arrays["x"], dtype=th.float32)
            edge_index = th.as_tensor(arrays["edgeindex"], dtype=th.long)
            label = int(arrays["y"]) if "y" in arrays.files else row["label"]
            root_index = (
                int(np.asarray(arrays["rootindex"]).reshape(-1)[0])
                if "rootindex" in arrays.files else 0
            )
            if root_index < 0 or root_index >= x.size(0):
                root_index = 0
            
            root_mask = th.zeros(
                x.size(0),
                dtype=th.bool,
            )
            
            root_mask[root_index] = True
        topology = self.topology_features(
            edge_index, x.size(0), root_index
        )
        # return Data(
        #     x=x,
        #     edge_index=edge_index,
        #     BU_edge_index=edge_index.flip(0),
        #     y=th.tensor([label], dtype=th.long),
        #     topology_features=topology,
        # )
        return Data(
            x=x,
            edge_index=edge_index,
            BU_edge_index=edge_index.flip(0),
            y=th.tensor(
                [label],
                dtype=th.long,
            ),
            topology_features=topology,
            root_mask=root_mask,
        )


def extract_z_graph(model, dataset, device, batch_size=16):
    """Run deterministic graph inference and preserve dataset event order."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()
    records = []
    dataset_offset = 0
    with th.no_grad():
        for data in loader:
            data = data.to(device)
            z_graph = model(data).detach().cpu().float().numpy()
            labels = data.y.view(-1).cpu().tolist()
            batch_samples = len(labels)
            if z_graph.shape[0] != batch_samples:
                raise ValueError(
                    "Graph encoder returned {} logits for {} labels".format(
                        z_graph.shape[0],
                        batch_samples,
                    )
                )
            for local_index in range(batch_samples):
                sample_index = dataset_offset + local_index
                records.append((
                    int(sample_index),
                    dataset.rows[int(sample_index)]["event_id"],
                    z_graph[local_index],
                    int(labels[local_index]),
                ))
            dataset_offset += batch_samples

    if dataset_offset != len(dataset):
        raise ValueError(
            "extracted {} Graph samples, expected {}".format(
                dataset_offset,
                len(dataset),
            )
        )

    records.sort(key=lambda item: item[0])
    return (
        np.asarray([item[1] for item in records], dtype=str),
        np.stack([item[2] for item in records]).astype(np.float32, copy=False),
        np.asarray([item[3] for item in records], dtype=np.int64),
    )


def extract_mc_graph(model, dataset, device, batch_size=16, samples=20, seed=2026):
    """Estimate frozen Graph uncertainty with stochastic dropout passes."""
    from mc_uncertainty import enable_mc_dropout, seed_mc, summarize_mc_probabilities
    passes = []
    seed_mc(seed)
    with th.no_grad():
        for _ in range(int(samples)):
            enable_mc_dropout(model)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
            probabilities = [
                F.softmax(model(data.to(device)), dim=-1).cpu().numpy()
                for data in loader
            ]
            passes.append(np.concatenate(probabilities, axis=0))
    model.eval()
    return summarize_mc_probabilities(np.stack(passes, axis=0))


def save_z_graph(output_path, event_ids, z_graph, labels, metadata, mc_stats=None):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        event_ids=event_ids,
        z_graph=z_graph,
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
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint override; otherwise constructed automatically",
    )
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
    parser.add_argument(
        "--input-csv",
        default=None,
    )
    parser.add_argument(
        "--graph-dir",
        default=None,
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output override; otherwise constructed automatically",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "dev", "test"],
        default="all",
    )
    parser.add_argument("--in-feats", type=int, default=768)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--out-dim", type=int, default=64)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_dataset_args(args)
    if args.iteration < 0 or (args.fold_id is not None and args.fold_id < 0):
        raise ValueError("iteration and fold-id must be non-negative")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if min(
        args.in_feats,
        args.hidden_dim,
        args.out_dim,
        args.num_classes,
    ) <= 0:
        raise ValueError("all model dimensions must be positive")

    checkpoint_path = resolve_checkpoint(
        args.checkpoint, args.dataset, args.checkpoint_fusion_mode,
        args.ablation, args.iteration, args.fold_id,
    )
    input_csv = resolve_input_csv(args.input_csv, args.dataset)
    graph_dir = resolve_graph_dir(args.graph_dir, args.dataset)
    output_path = resolve_output(
        args.output, args.dataset, args.split, "graph",
        args.checkpoint_fusion_mode, args.ablation,
        args.iteration, args.fold_id,
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError("checkpoint not found: {}".format(checkpoint_path))

    set_seed(args.seed)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model = GraphEncoder(
        in_feats=args.in_feats,
        hid_feats=args.hidden_dim,
        out_feats=args.out_dim,
        num_classes=args.num_classes,
    )
    model.load_full_checkpoint(checkpoint_path)
    model.to(device)

    rows = load_event_rows(
        input_csv,
        split=args.split,
        dataset=args.dataset,
        fold_id=args.fold_id,
        graph_dir=graph_dir,
        iteration=args.iteration,
        split_seed=args.seed,
    )
    dataset = GraphNPZDataset(rows, graph_dir)
    event_ids, z_graph, labels = extract_z_graph(
        model,
        dataset,
        device,
        batch_size=args.batch_size,
    )
    metadata = {
        "checkpoint": str(checkpoint_path),
        "input_csv": str(input_csv),
        "graph_dir": str(graph_dir),
        "split": args.split,
        "samples": int(len(event_ids)),
        "in_feats": int(args.in_feats),
        "hidden_dim": int(args.hidden_dim),
        "out_dim": int(args.out_dim),
        "pooling": "directional_mean",
        "topology_features": [
            "log_node_count",
            "log_max_depth",
            "log_mean_branching",
            "leaf_ratio",
        ],
        "representation_dim": 128,
        "classifier_hidden_dim": 64,
        "dropout": 0.2,
        "num_classes": int(args.num_classes),
        "seed": int(args.seed),
        "device": str(device),
        "saved_arrays": {
            "event_ids": list(event_ids.shape),
            "z_graph": list(z_graph.shape),
            "labels": list(labels.shape),
        },
    }
    output_path, metadata_path = save_z_graph(
        output_path,
        event_ids,
        z_graph,
        labels,
        metadata,
    )
    print("saved z_graph:", output_path.resolve())
    print("shape:", z_graph.shape)
    print("metadata:", metadata_path.resolve())


if __name__ == "__main__":
    main()
