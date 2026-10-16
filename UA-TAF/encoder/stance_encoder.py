"""Extract grouped stance-semantic evidence logits ``z_stance``.

The architecture and parameter names match the Stance branch in
``model/BiGCN_ConflictFusion.py``:

    reply BERT vector -> node projection -> soft S/D/Q/C means --\
    ratio + depth + log-count + stance-type embedding ------------ attention

The exported NPZ also stores ``stance_conf`` because the adaptive fusion module
uses it to modulate stance reliability.
"""

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


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
STANCE_STATE_PREFIXES = (
    "stance_evidence_encoder.",
    "norm_stance.",
    "fc_stance.",
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


class GroupedStanceEvidenceEncoder(th.nn.Module):
    """Grouped stance-semantic encoder kept under the legacy attribute name."""

    def __init__(
        self,
        stance_dim=16,
        text_dim=768,
        text_hidden_dim=128,
        stance_hidden_dim=32,
        out_dim=128,
        dropout=0.2,
        depth_bins=5,
        attention_heads=2,
        stance_type_dim=16,
    ):
        super().__init__()
        self.stance_dim = int(stance_dim)
        self.text_dim = int(text_dim)
        self.depth_bins = int(depth_bins)
        if self.stance_dim < 4:
            raise ValueError("stance_dim must contain S/D/Q/C probabilities")
        if out_dim % attention_heads != 0:
            raise ValueError("out_dim must be divisible by attention_heads")

        # Project each cached reply BERT vector before soft grouping.
        self.reply_projection = th.nn.Sequential(
            th.nn.Linear(self.text_dim, out_dim),
            th.nn.LayerNorm(out_dim),
            th.nn.ReLU(),
            th.nn.Dropout(dropout),
        )
        self.stance_type_embedding = th.nn.Embedding(4, stance_type_dim)
        # Token = group semantics + ratio + depth + log-count + type embedding.
        token_input_dim = out_dim + 1 + self.depth_bins + 1 + stance_type_dim
        self.token_projection = th.nn.Sequential(
            th.nn.Linear(token_input_dim, out_dim),
            th.nn.LayerNorm(out_dim),
            th.nn.ReLU(),
        )
        self.stance_attention = th.nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = th.nn.Sequential(
            th.nn.Linear(out_dim, text_hidden_dim),
            th.nn.ReLU(),
            th.nn.Dropout(dropout),
            th.nn.Linear(text_hidden_dim, out_dim),
            th.nn.LayerNorm(out_dim),
            th.nn.ReLU(),
        )
        

    @staticmethod
    def _soft_group_mean(values, stance_probability, reply_mask,
                         batch, batch_size, eps=1e-8):
        """Probability-weighted S/D/Q/C means and their soft masses."""
        feature_dim = values.size(1)
        weighted_sum = values.new_zeros((batch_size, 4, feature_dim))
        soft_mass = values.new_zeros((batch_size, 4))
        valid_probability = stance_probability * reply_mask.to(
            values.dtype
        ).unsqueeze(1)
        for stance_id in range(4):
            weight = valid_probability[:, stance_id]
            weighted_sum[:, stance_id].index_add_(
                0, batch, values * weight.unsqueeze(1)
            )
            soft_mass[:, stance_id].index_add_(0, batch, weight)
        mean = weighted_sum / soft_mass.unsqueeze(-1).clamp_min(eps)
        return mean, soft_mass

    def forward(self, text_x, stance_x, depth_x, batch):
        """Encode grouped S/D/Q/C evidence with empty-group masking.
    
        Empty stance groups are excluded from:
          1. attention keys and values;
          2. final stance-token aggregation.
    
        For threads with no valid reply groups, one zero-valued dummy token is
        temporarily unmasked only to keep MultiheadAttention numerically stable.
        Its output is removed again by the original valid-group mask.
        """
        text_x = text_x.float()
        stance_x = stance_x.float()
        depth_x = depth_x.float()
    
        if (
            text_x.size(0) != stance_x.size(0)
            or text_x.size(0) != depth_x.size(0)
        ):
            raise ValueError(
                "text, stance and depth features have different node counts"
            )
    
        batch_size = (
            int(batch.max().item()) + 1
            if batch.numel()
            else 0
        )
    
        # The first four dimensions are soft probabilities for:
        # Support, Deny, Query and Comment.
        stance_prob = stance_x[:, :4].clamp_min(0.0)
    
        stance_prob = stance_prob / stance_prob.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-8)
    
        # stance_x[:, 15] marks the source/root node.
        if stance_x.size(1) > 15:
            root_mask = stance_x[:, 15] >= 0.5
        else:
            root_mask = th.zeros(
                text_x.size(0),
                dtype=th.bool,
                device=text_x.device,
            )
    
        reply_mask = ~root_mask
    
        # Project every cached reply text vector before soft grouping.
        reply_semantics = self.reply_projection(
            text_x
        )
    
        # Soft S/D/Q/C semantic group representations.
        grouped_semantics, stance_mass = self._soft_group_mean(
            reply_semantics,
            stance_prob,
            reply_mask,
            batch,
            batch_size,
        )
    
        # Soft S/D/Q/C depth distributions.
        grouped_depth, _ = self._soft_group_mean(
            depth_x,
            stance_prob,
            reply_mask,
            batch,
            batch_size,
        )
    
        # Number of replies in each conversation thread.
        reply_count = text_x.new_zeros(
            batch_size
        )
    
        reply_count.index_add_(
            0,
            batch,
            reply_mask.to(text_x.dtype),
        )
    
        # Soft stance distribution:
        # d_s = (1 / m) * sum_i q_{i,s}
        stance_distribution = (
            stance_mass
            / reply_count.unsqueeze(1).clamp_min(1.0)
        )
    
        # A stance group is valid only when it receives non-negligible
        # probability mass from at least one reply.
        #
        # Shape: [batch_size, 4]
        valid_group = stance_mass > 1e-6
    
        log_stance_count = th.log1p(
            stance_mass
        ).unsqueeze(-1)
    
        type_ids = th.arange(
            4,
            device=text_x.device,
        )
    
        type_embedding = self.stance_type_embedding(
            type_ids
        ).unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )
    
        stance_tokens = th.cat(
            [
                grouped_semantics,
                stance_distribution.unsqueeze(-1),
                grouped_depth,
                log_stance_count,
                type_embedding,
            ],
            dim=-1,
        )
    
        stance_tokens = self.token_projection(
            stance_tokens
        )
    
        # Remove the type embedding and projection bias contributions of empty
        # groups before attention.
        valid_mask = valid_group.unsqueeze(-1).to(
            stance_tokens.dtype
        )
    
        stance_tokens = (
            stance_tokens * valid_mask
        )
    
        # MultiheadAttention produces NaN when every key in one sample is masked.
        # For a thread without any valid reply group, temporarily expose one
        # zero-valued dummy token. The original valid_mask below will remove it
        # from the final representation.
        safe_valid_group = valid_group.clone()
    
        no_valid_group = (
            safe_valid_group.sum(dim=1) == 0
        )
    
        if no_valid_group.any():
            safe_valid_group[
                no_valid_group,
                0,
            ] = True
    
            stance_tokens = stance_tokens.clone()
    
            stance_tokens[
                no_valid_group,
                0,
            ] = 0.0
    
        attended, _ = self.stance_attention(
            stance_tokens,
            stance_tokens,
            stance_tokens,
            key_padding_mask=~safe_valid_group,
            need_weights=False,
        )
    
        # Residual token representation.
        token_output = (
            stance_tokens + attended
        )
    
        # Empty query groups may still receive an attention output because
        # key_padding_mask masks keys/values but not queries. Mask them again.
        token_output = (
            token_output * valid_mask
        )
    
        # Masked mean over only the stance groups that are actually present.
        valid_count = valid_mask.sum(
            dim=1
        ).clamp_min(1.0)
    
        stance_evidence = token_output.sum(
            dim=1
        ) / valid_count
    
        return self.output_projection(
            stance_evidence
        )

class StanceEncoder(th.nn.Module):
    """Independent grouped-Stance classifier returning raw ``z_stance``."""

    def __init__(
        self,
        stance_dim=16,
        text_dim=768,
        text_hidden_dim=128,
        stance_hidden_dim=32,
        representation_dim=128,
        num_classes=3,
        dropout=0.2,
        depth_bins=5,
        attention_heads=2,
        stance_type_dim=16,
    ):
        super().__init__()
        # Attribute names intentionally match the complete model checkpoint.
        self.stance_evidence_encoder = GroupedStanceEvidenceEncoder(
            stance_dim=stance_dim,
            text_dim=text_dim,
            text_hidden_dim=text_hidden_dim,
            stance_hidden_dim=stance_hidden_dim,
            out_dim=representation_dim,
            dropout=dropout,
            depth_bins=depth_bins,
            attention_heads=attention_heads,
            stance_type_dim=stance_type_dim,
        )
        self.norm_stance = th.nn.LayerNorm(representation_dim)
        self.fc_stance = th.nn.Linear(representation_dim, num_classes)

    def encode(self, data):
        embedding = self.stance_evidence_encoder(
            data.x,
            data.stance_x,
            data.depth_x,
            data.batch,
        )
        return self.norm_stance(embedding)

    def forward(self, data):
        """Return raw class logits ``z_stance`` with shape ``[B, C]``."""
        return self.fc_stance(self.encode(data))

    @staticmethod
    def confidence(data):
        """Match the event-level stance confidence used by the full model."""
        stance_x = data.stance_x.float()
        batch = data.batch
        batch_size = int(data.num_graphs)
        conf_node = stance_x[:, 4].clamp(0.0, 1.0)
        if stance_x.size(1) > 15:
            # stance_x[:, 15] marks the source/root node.
            mask = stance_x[:, 15] < 0.5
        else:
            mask = th.ones_like(conf_node, dtype=th.bool)

        sum_conf = th.zeros(
            batch_size,
            dtype=conf_node.dtype,
            device=conf_node.device,
        )
        count = th.zeros_like(sum_conf)
        sum_conf.scatter_add_(0, batch[mask], conf_node[mask])
        count.scatter_add_(0, batch[mask], th.ones_like(conf_node[mask]))
        mean_conf = sum_conf / count.clamp_min(1.0)
        return th.where(
            count > 0,
            mean_conf,
            th.full_like(mean_conf, 0.5),
        )

    def load_full_checkpoint(self, checkpoint_path):
        """Load only Stance-branch tensors from a complete checkpoint."""
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

        stance_state = {}
        for raw_name, tensor in checkpoint.items():
            name = raw_name[7:] if raw_name.startswith("module.") else raw_name
            if name.startswith(STANCE_STATE_PREFIXES):
                stance_state[name] = tensor

        expected = set(self.state_dict())
        available = set(stance_state)
        missing = sorted(expected - available)
        unexpected = sorted(available - expected)
        if missing or unexpected:
            raise ValueError(
                "incompatible Stance checkpoint; "
                "missing:{} unexpected:{}".format(
                    missing[:10],
                    unexpected[:10],
                )
            )
        self.load_state_dict(stance_state, strict=True)


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


class StanceNPZDataset(Dataset):
    """Read reply text representations, stance features and TD edges."""

    def __init__(self, rows, graph_dir, stance_dim=16, text_dim=768,
                 depth_bins=5):
        self.graph_dir = Path(graph_dir)
        self.stance_dim = int(stance_dim)
        self.text_dim = int(text_dim)
        self.depth_bins = int(depth_bins)
        if self.depth_bins <= 0:
            raise ValueError("depth_bins must be positive")
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
    def depth_features(edge_index, num_nodes, root_index, depth_bins=5):
        """Return capped one-hot reply depths derived from TD edges."""
        adjacency = [[] for _ in range(num_nodes)]
        for parent, child in np.asarray(edge_index, dtype=np.int64).T:
            parent, child = int(parent), int(child)
            if 0 <= parent < num_nodes and 0 <= child < num_nodes:
                adjacency[parent].append(child)
        root_index = int(root_index)
        if root_index < 0 or root_index >= num_nodes:
            root_index = 0
        depths = np.full(num_nodes, -1, dtype=np.int64)
        depths[root_index] = 0
        queue = [root_index]
        for node in queue:
            for child in adjacency[node]:
                if depths[child] < 0:
                    depths[child] = depths[node] + 1
                    queue.append(child)
        # Disconnected reply nodes use the final (deep/unknown) bucket.
        bucket = np.where(depths >= 0, np.minimum(depths, depth_bins - 1), depth_bins - 1)
        output = np.zeros((num_nodes, depth_bins), dtype=np.float32)
        output[np.arange(num_nodes), bucket] = 1.0
        return output

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.graph_dir / "{}.npz".format(row["event_id"])
        with np.load(path, allow_pickle=True) as arrays:
            text_x = np.asarray(arrays["x"], dtype=np.float32)
            num_nodes = int(text_x.shape[0])
            edge_index = np.asarray(arrays["edgeindex"], dtype=np.int64)
            root_index = (
                int(np.asarray(arrays["rootindex"]).reshape(-1)[0])
                if "rootindex" in arrays.files else 0
            )
            if "stance_x" in arrays.files:
                stance_x = np.asarray(arrays["stance_x"], dtype=np.float32)
            else:
                stance_x = np.zeros(
                    (num_nodes, self.stance_dim), dtype=np.float32
                )
            label = int(arrays["y"]) if "y" in arrays.files else row["label"]
        if text_x.shape != (num_nodes, self.text_dim):
            raise ValueError(
                "{} has text x shape {}, expected ({}, {})".format(
                    row["event_id"],
                    text_x.shape,
                    num_nodes,
                    self.text_dim,
                )
            )
        if stance_x.shape != (num_nodes, self.stance_dim):
            raise ValueError(
                "{} has stance_x shape {}, expected ({}, {})".format(
                    row["event_id"],
                    stance_x.shape,
                    num_nodes,
                    self.stance_dim,
                )
            )

        depth_x = self.depth_features(
            edge_index, num_nodes, root_index, self.depth_bins
        )
        return Data(
            x=th.as_tensor(text_x, dtype=th.float32),
            stance_x=th.as_tensor(stance_x, dtype=th.float32),
            depth_x=th.as_tensor(depth_x, dtype=th.float32),
            edge_index=th.as_tensor(edge_index, dtype=th.long),
            y=th.tensor([label], dtype=th.long),
            num_nodes=num_nodes,
        )


def extract_z_stance(model, dataset, device, batch_size=16):
    """Run deterministic stance inference and preserve dataset event order."""
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
            z_stance = model(data).detach().cpu().float().numpy()
            stance_conf = (
                model.confidence(data).detach().cpu().float().numpy()
            )
            labels = data.y.view(-1).cpu().tolist()
            batch_samples = len(labels)
            if (
                z_stance.shape[0] != batch_samples
                or stance_conf.shape[0] != batch_samples
            ):
                raise ValueError(
                    "Stance encoder output sizes do not match {} labels".format(
                        batch_samples
                    )
                )
            for local_index in range(batch_samples):
                sample_index = dataset_offset + local_index
                records.append((
                    int(sample_index),
                    dataset.rows[int(sample_index)]["event_id"],
                    z_stance[local_index],
                    float(stance_conf[local_index]),
                    int(labels[local_index]),
                ))
            dataset_offset += batch_samples

    if dataset_offset != len(dataset):
        raise ValueError(
            "extracted {} Stance samples, expected {}".format(
                dataset_offset,
                len(dataset),
            )
        )

    records.sort(key=lambda item: item[0])
    return (
        np.asarray([item[1] for item in records], dtype=str),
        np.stack([item[2] for item in records]).astype(np.float32, copy=False),
        np.asarray([item[3] for item in records], dtype=np.float32),
        np.asarray([item[4] for item in records], dtype=np.int64),
    )


def extract_mc_stance(model, dataset, device, batch_size=16, samples=20, seed=2026):
    """Estimate frozen Stance uncertainty with stochastic dropout passes."""
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


def save_z_stance(
    output_path,
    event_ids,
    z_stance,
    stance_conf,
    labels,
    metadata,
    mc_stats=None,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        event_ids=event_ids,
        z_stance=z_stance,
        stance_conf=stance_conf,
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
    parser.add_argument("--stance-dim", type=int, default=16)
    parser.add_argument("--reply-text-dim", type=int, default=768)
    parser.add_argument("--reply-text-hidden-dim", type=int, default=128)
    parser.add_argument("--stance-hidden-dim", type=int, default=32)
    parser.add_argument("--representation-dim", type=int, default=128)
    parser.add_argument("--depth-bins", type=int, default=5)
    parser.add_argument("--attention-heads", type=int, default=2)
    parser.add_argument("--stance-type-dim", type=int, default=16)
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
        args.stance_dim,
        args.reply_text_dim,
        args.reply_text_hidden_dim,
        args.stance_hidden_dim,
        args.representation_dim,
        args.depth_bins,
        args.attention_heads,
        args.stance_type_dim,
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
        args.output, args.dataset, args.split, "stance",
        args.checkpoint_fusion_mode, args.ablation,
        args.iteration, args.fold_id,
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError("checkpoint not found: {}".format(checkpoint_path))

    set_seed(args.seed)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    model = StanceEncoder(
        stance_dim=args.stance_dim,
        text_dim=args.reply_text_dim,
        text_hidden_dim=args.reply_text_hidden_dim,
        stance_hidden_dim=args.stance_hidden_dim,
        representation_dim=args.representation_dim,
        num_classes=args.num_classes,
        depth_bins=args.depth_bins,
        attention_heads=args.attention_heads,
        stance_type_dim=args.stance_type_dim,
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
    dataset = StanceNPZDataset(
        rows,
        graph_dir,
        stance_dim=args.stance_dim,
        text_dim=args.reply_text_dim,
        depth_bins=args.depth_bins,
    )
    event_ids, z_stance, stance_conf, labels = extract_z_stance(
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
        "stance_dim": int(args.stance_dim),
        "reply_text_dim": int(args.reply_text_dim),
        "reply_text_hidden_dim": int(args.reply_text_hidden_dim),
        "stance_hidden_dim": int(args.stance_hidden_dim),
        "representation_dim": int(args.representation_dim),
        "depth_bins": int(args.depth_bins),
        "attention_heads": int(args.attention_heads),
        "stance_type_dim": int(args.stance_type_dim),
        "aggregation": "project_then_soft_group_with_two_head_stance_attention",
        "uses_source_post_semantics": False,
        "num_classes": int(args.num_classes),
        "seed": int(args.seed),
        "device": str(device),
        "saved_arrays": {
            "event_ids": list(event_ids.shape),
            "z_stance": list(z_stance.shape),
            "stance_conf": list(stance_conf.shape),
            "labels": list(labels.shape),
        },
    }
    output_path, metadata_path = save_z_stance(
        output_path,
        event_ids,
        z_stance,
        stance_conf,
        labels,
        metadata,
    )
    print("saved z_stance:", output_path.resolve())
    print("shape:", z_stance.shape)
    print("stance_conf shape:", stance_conf.shape)
    print("metadata:", metadata_path.resolve())


if __name__ == "__main__":
    main()
