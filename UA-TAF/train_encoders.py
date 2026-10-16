"""Independently train Graph, Root and Stance evidence encoders.

Each branch owns its optimizer, scheduler, early-stopping state and best
checkpoint. Dev Macro-F1 is the primary model-selection criterion; NLL is
used only as a tie-breaker. The best checkpoint of each branch is exported
for later fusion.
"""

import argparse
import copy
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_ROOT = Path(__file__).resolve().parent
ENCODER_ROOT = MODULE_ROOT / "encoder"
sys.path.insert(0, str(ENCODER_ROOT))

from graph_encoder import (  # noqa: E402
    GraphEncoder,
    GraphNPZDataset,
    extract_mc_graph,
    extract_z_graph,
    load_event_rows as load_graph_rows,
    save_z_graph,
)
from path_utils import (  # noqa: E402
    configure_dataset_args,
    resolve_graph_dir,
    resolve_input_csv,
    resolve_training_dir,
)
from root_encoder import (  # noqa: E402
    RootEncoder,
    RootNPZDataset,
    extract_mc_root,
    extract_z_root,
    load_source_rows,
    save_z_root,
)
from stance_encoder import (  # noqa: E402
    StanceEncoder,
    StanceNPZDataset,
    extract_mc_stance,
    extract_z_stance,
    load_event_rows as load_stance_rows,
    save_z_stance,
)


BRANCHES = ("graph", "root", "stance")
DEFAULT_SUPCON_TEMPERATURE = 0.20
EARLY_STOP_MACRO_TOLERANCE = 1e-4
EARLY_STOP_NLL_TOLERANCE = 1e-4


class ClassBalancedBatchSampler(Sampler):
    """Create full batches whose per-class counts differ by at most one."""

    def __init__(self, labels, batch_size, num_classes, seed):
        self.labels = np.asarray(labels, dtype=np.int64)
        self.batch_size = int(batch_size)
        self.num_classes = int(num_classes)
        self.seed = int(seed)
        self.epoch = 0

        if self.batch_size < self.num_classes:
            raise ValueError("batch-size must be at least num-classes")

        self.class_indices = [
            np.flatnonzero(self.labels == class_id).tolist()
            for class_id in range(self.num_classes)
        ]
        empty = [
            class_id
            for class_id, values in enumerate(self.class_indices)
            if not values
        ]
        if empty:
            raise ValueError(
                "balanced sampler has empty classes: {}".format(empty)
            )

        self.num_batches = int(
            math.ceil(len(self.labels) / self.batch_size)
        )

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1

        pools = [list(values) for values in self.class_indices]
        positions = [0] * self.num_classes
        for pool in pools:
            rng.shuffle(pool)

        def draw(class_id, count):
            selected = []
            while len(selected) < count:
                pool = pools[class_id]
                position = positions[class_id]
                if position >= len(pool):
                    rng.shuffle(pool)
                    position = 0
                take = min(
                    count - len(selected),
                    len(pool) - position,
                )
                selected.extend(pool[position:position + take])
                positions[class_id] = position + take
            return selected

        base, remainder = divmod(
            self.batch_size,
            self.num_classes,
        )

        for batch_id in range(self.num_batches):
            counts = [base] * self.num_classes
            for offset in range(remainder):
                counts[
                    (batch_id + offset) % self.num_classes
                ] += 1

            batch = []
            for class_id, count in enumerate(counts):
                batch.extend(draw(class_id, count))

            rng.shuffle(batch)
            yield batch


def dataset_labels(dataset):
    labels = []
    for index in range(len(dataset)):
        sample = dataset[index]
        target = (
            sample[1]
            if isinstance(sample, tuple)
            else sample.y.view(-1)[0]
        )
        labels.append(int(target))
    return np.asarray(labels, dtype=np.int64)


def supervised_contrastive_loss(
    features,
    labels,
    temperature=DEFAULT_SUPCON_TEMPERATURE,
):
    """Supervised contrastive loss over event representations."""
    if temperature <= 0:
        raise ValueError("SupCon temperature must be positive")

    features = F.normalize(
        features,
        p=2,
        dim=1,
        eps=1e-8,
    )
    similarity = (
        features @ features.t()
    ) / float(temperature)

    sample_count = similarity.size(0)
    self_mask = th.eye(
        sample_count,
        dtype=th.bool,
        device=features.device,
    )
    positive_mask = (
        labels.view(-1, 1).eq(labels.view(1, -1))
        & ~self_mask
    )

    valid = positive_mask.sum(dim=1) > 0
    if not valid.any():
        return features.sum() * 0.0

    similarity = (
        similarity
        - similarity.max(dim=1, keepdim=True)[0].detach()
    )
    exp_similarity = th.exp(similarity).masked_fill(
        self_mask,
        0.0,
    )
    log_probability = similarity - th.log(
        exp_similarity.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)
    )
    positive_count = positive_mask.sum(
        dim=1
    ).clamp_min(1)

    mean_positive_log_probability = (
        (log_probability * positive_mask).sum(dim=1)
        / positive_count
    )
    return -mean_positive_log_probability[valid].mean()


def current_supcon_weight(
    epoch,
    target_weight,
    warmup_epochs,
):
    """Linearly warm up the SupCon coefficient."""
    target_weight = float(target_weight)

    if target_weight <= 0:
        return 0.0
    if warmup_epochs <= 0:
        return target_weight

    progress = min(
        float(epoch + 1) / float(warmup_epochs),
        1.0,
    )
    return target_weight * progress


def sqrt_inverse_frequency_weights(
    labels,
    num_classes,
    device,
):
    """Return mild class weights whose mean is one."""
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(
        labels,
        minlength=num_classes,
    ).astype(np.float32)

    if np.any(counts <= 0):
        raise ValueError(
            "cannot calculate class weights with empty classes: {}".format(
                counts.tolist()
            )
        )

    total = float(counts.sum())
    weights = np.sqrt(
        total / (float(num_classes) * counts)
    )
    weights = weights / weights.mean()

    return th.as_tensor(
        weights,
        dtype=th.float32,
        device=device,
    )


def branch_training_loss(
    logits,
    features,
    labels,
    class_weights,
    label_smoothing,
    supcon_weight,
    supcon_temperature,
):
    """Cross entropy plus optional supervised contrastive loss."""
    ce_loss = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        label_smoothing=float(label_smoothing),
    )

    if supcon_weight <= 0:
        return (
            ce_loss,
            ce_loss.detach(),
            logits.new_zeros(()),
        )

    contrastive_loss = supervised_contrastive_loss(
        features,
        labels,
        temperature=supcon_temperature,
    )
    total_loss = (
        ce_loss
        + float(supcon_weight) * contrastive_loss
    )

    return (
        total_loss,
        ce_loss.detach(),
        contrastive_loss.detach(),
    )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed(seed)
        th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False


def metrics_from_logits(
    logits,
    labels,
    num_classes,
):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    if logits.ndim != 2:
        raise ValueError(
            "logits must have shape [N,C], got {}".format(
                logits.shape
            )
        )
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            "logits and labels have inconsistent sample counts"
        )

    predictions = logits.argmax(axis=1)
    classes = list(range(num_classes))

    shifted = logits - logits.max(
        axis=1,
        keepdims=True,
    )
    log_probabilities = shifted - np.log(
        np.exp(shifted).sum(
            axis=1,
            keepdims=True,
        )
    )
    nll = -log_probabilities[
        np.arange(labels.shape[0]),
        labels,
    ].mean()

    return {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        ),
        "class_f1": f1_score(
            labels,
            predictions,
            labels=classes,
            average=None,
            zero_division=0,
        ).astype(float).tolist(),
        "nll": float(nll),
    }


def save_best(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    th.save(model.state_dict(), path)


def load_state(model, path):
    try:
        state = th.load(
            path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        state = th.load(
            path,
            map_location="cpu",
        )
    model.load_state_dict(state, strict=True)


def update_early_stopping(
    state,
    metrics,
    model,
    checkpoint,
    patience,
):
    """Select by Macro-F1 and use NLL only as a tie-breaker."""
    macro_f1 = float(metrics["macro_f1"])
    nll = float(metrics["nll"])

    best_macro_f1 = state.get("best_macro_f1")
    best_nll = state.get("best_nll")

    macro_improved = (
        best_macro_f1 is None
        or macro_f1
        > best_macro_f1 + EARLY_STOP_MACRO_TOLERANCE
    )
    macro_tied_nll_improved = (
        best_macro_f1 is not None
        and abs(
            macro_f1 - best_macro_f1
        ) <= EARLY_STOP_MACRO_TOLERANCE
        and (
            best_nll is None
            or nll
            < best_nll - EARLY_STOP_NLL_TOLERANCE
        )
    )

    if macro_improved or macro_tied_nll_improved:
        state.update(
            best=float(macro_f1),
            best_macro_f1=float(macro_f1),
            best_nll=float(nll),
            counter=0,
            best_epoch=int(state["epoch"]),
        )
        save_best(model, checkpoint)
        return False

    state["counter"] += 1
    return state["counter"] >= patience


def make_root_train_loader(
    dataset,
    labels,
    args,
    seed,
):
    pin_memory = th.cuda.is_available()

    if args.balance_mode == "batch":
        return th.utils.data.DataLoader(
            dataset,
            batch_sampler=ClassBalancedBatchSampler(
                labels,
                args.batch_size,
                args.num_classes,
                seed,
            ),
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    generator = th.Generator()
    generator.manual_seed(seed)

    return th.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=generator,
        drop_last=False,
    )


def make_graph_train_loader(
    dataset,
    labels,
    args,
    seed,
):
    pin_memory = th.cuda.is_available()

    if args.balance_mode == "batch":
        return DataLoader(
            dataset,
            batch_sampler=ClassBalancedBatchSampler(
                labels,
                args.batch_size,
                args.num_classes,
                seed,
            ),
            num_workers=args.num_workers,
            pin_memory=pin_memory,
        )

    generator = th.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        generator=generator,
        drop_last=False,
    )


def make_training_class_weights(
    labels,
    args,
    device,
):
    if args.balance_mode != "loss":
        return None

    return sqrt_inverse_frequency_weights(
        labels,
        args.num_classes,
        device,
    )


def root_epoch(
    model,
    loader,
    optimizer,
    device,
    args,
    class_weights,
    epoch,
):
    model.train()

    total_losses = []
    ce_losses = []
    supcon_losses = []

    supcon_weight = current_supcon_weight(
        epoch,
        args.root_supcon_weight,
        args.supcon_warmup_epochs,
    )

    for root, labels in loader:
        root = root.to(
            device,
            non_blocking=True,
        )
        labels = labels.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        features = model.encode(root)
        logits = model.fc_root(features)

        loss, ce_loss, supcon_loss = branch_training_loss(
            logits=logits,
            features=features,
            labels=labels,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            supcon_weight=supcon_weight,
            supcon_temperature=args.supcon_temperature,
        )

        loss.backward()
        th.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )
        optimizer.step()

        total_losses.append(
            float(loss.detach().cpu())
        )
        ce_losses.append(
            float(ce_loss.cpu())
        )
        supcon_losses.append(
            float(supcon_loss.cpu())
        )

    return {
        "loss": float(np.mean(total_losses)),
        "ce_loss": float(np.mean(ce_losses)),
        "supcon_loss": float(np.mean(supcon_losses)),
        "supcon_weight": float(supcon_weight),
    }


def evaluate_root(
    model,
    dataset,
    device,
    args,
):
    _, logits, labels = extract_z_root(
        model,
        dataset,
        device,
        batch_size=args.batch_size,
    )
    return metrics_from_logits(
        logits,
        labels,
        args.num_classes,
    )


def graph_like_epoch(
    branch,
    model,
    loader,
    optimizer,
    device,
    args,
    class_weights,
    epoch,
):
    model.train()

    total_losses = []
    ce_losses = []
    supcon_losses = []

    if branch == "graph":
        target_supcon_weight = args.graph_supcon_weight
    elif branch == "stance":
        target_supcon_weight = args.stance_supcon_weight
    else:
        raise ValueError(
            "unsupported graph-like branch: {}".format(branch)
        )

    supcon_weight = current_supcon_weight(
        epoch,
        target_supcon_weight,
        args.supcon_warmup_epochs,
    )

    for data in loader:
        data = data.to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)

        labels = data.y.view(-1)
        features = model.encode(data)
        classifier = getattr(
            model,
            "fc_{}".format(branch),
        )
        logits = classifier(features)

        loss, ce_loss, supcon_loss = branch_training_loss(
            logits=logits,
            features=features,
            labels=labels,
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            supcon_weight=supcon_weight,
            supcon_temperature=args.supcon_temperature,
        )

        loss.backward()
        th.nn.utils.clip_grad_norm_(
            model.parameters(),
            args.grad_clip,
        )
        optimizer.step()

        total_losses.append(
            float(loss.detach().cpu())
        )
        ce_losses.append(
            float(ce_loss.cpu())
        )
        supcon_losses.append(
            float(supcon_loss.cpu())
        )

    return {
        "loss": float(np.mean(total_losses)),
        "ce_loss": float(np.mean(ce_losses)),
        "supcon_loss": float(np.mean(supcon_losses)),
        "supcon_weight": float(supcon_weight),
    }


def evaluate_graph_like(
    model,
    loader,
    device,
    num_classes,
):
    model.eval()
    logits = []
    labels = []

    with th.no_grad():
        for data in loader:
            data = data.to(device)
            logits.append(
                model(data).detach().cpu().numpy()
            )
            labels.append(
                data.y.view(-1).cpu().numpy()
            )

    return metrics_from_logits(
        np.concatenate(logits),
        np.concatenate(labels),
        num_classes,
    )


def extract_root_features(
    model,
    dataset,
    device,
    batch_size,
):
    loader = th.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()

    features = []
    with th.no_grad():
        for root, _ in loader:
            features.append(
                model.encode(
                    root.to(device)
                ).cpu().numpy()
            )

    return np.concatenate(features).astype(
        np.float32,
        copy=False,
    )


def extract_graph_like_features(
    model,
    dataset,
    device,
    batch_size,
):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    model.eval()

    features = []
    with th.no_grad():
        for data in loader:
            features.append(
                model.encode(
                    data.to(device)
                ).cpu().numpy()
            )

    return np.concatenate(features).astype(
        np.float32,
        copy=False,
    )


def train_root(
    args,
    paths,
    device,
):
    train_rows = load_source_rows(
        paths["csv"],
        "train",
        args.dataset,
        args.fold_id,
        paths["graphs"],
        iteration=args.iteration,
        split_seed=args.seed,
    )
    dev_rows = load_source_rows(
        paths["csv"],
        "dev",
        args.dataset,
        args.fold_id,
        paths["graphs"],
        iteration=args.iteration,
        split_seed=args.seed,
    )

    train_data = RootNPZDataset(
        train_rows,
        paths["graphs"],
        input_dim=args.in_feats,
    )
    dev_data = RootNPZDataset(
        dev_rows,
        paths["graphs"],
        input_dim=args.in_feats,
    )

    train_labels = dataset_labels(train_data)
    train_loader = make_root_train_loader(
        train_data,
        train_labels,
        args,
        args.seed + args.iteration + 1000,
    )
    class_weights = make_training_class_weights(
        train_labels,
        args,
        device,
    )

    print(
        "Root training counts:",
        np.bincount(
            train_labels,
            minlength=args.num_classes,
        ).tolist(),
    )
    if class_weights is not None:
        print(
            "Root class weights:",
            class_weights.detach().cpu().tolist(),
        )

    model = RootEncoder(
        input_dim=args.in_feats,
        representation_dim=args.representation_dim,
        hidden_dim=args.root_hidden_dim,
        num_classes=args.num_classes,
        dropout=args.root_dropout,
    ).to(device)

    optimizer = th.optim.AdamW(
        model.parameters(),
        lr=args.root_lr,
        weight_decay=args.root_weight_decay,
    )
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    checkpoint = paths["checkpoints"] / "root_best.m"
    state = {
        "best": None,
        "best_macro_f1": None,
        "best_nll": None,
        "counter": 0,
        "best_epoch": -1,
    }
    history = []

    for epoch in range(args.epochs):
        state["epoch"] = epoch

        train_stats = root_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args,
            class_weights,
            epoch,
        )
        dev = evaluate_root(
            model,
            dev_data,
            device,
            args,
        )

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "ce_loss": train_stats["ce_loss"],
            "supcon_loss": train_stats["supcon_loss"],
            "supcon_weight": train_stats["supcon_weight"],
            "learning_rate": current_lr,
            **dev,
        })

        print(
            "Root epoch {:03d} | loss {:.4f} | CE {:.4f} | "
            "SupCon {:.4f} (w={:.3f}) | lr {:.2e} | "
            "Dev Macro-F1 {:.2f}% | NLL {:.4f}".format(
                epoch,
                train_stats["loss"],
                train_stats["ce_loss"],
                train_stats["supcon_loss"],
                train_stats["supcon_weight"],
                current_lr,
                100.0 * dev["macro_f1"],
                dev["nll"],
            )
        )

        scheduler.step(dev["macro_f1"])

        if update_early_stopping(
            state,
            dev,
            model,
            checkpoint,
            args.patience,
        ):
            break

    load_state(model, checkpoint)
    export_root(model, args, paths, device)
    return checkpoint, state, history


def export_root(
    model,
    args,
    paths,
    device,
):
    splits = ["train", "dev"] + (
        ["test"] if args.export_test else []
    )

    for split in splits:
        rows = load_source_rows(
            paths["csv"],
            split,
            args.dataset,
            args.fold_id,
            paths["graphs"],
            iteration=args.iteration,
            split_seed=args.seed,
        )
        dataset = RootNPZDataset(
            rows,
            paths["graphs"],
            input_dim=args.in_feats,
        )
        event_ids, logits, labels = extract_z_root(
            model,
            dataset,
            device,
            batch_size=args.batch_size,
        )

        mc_stats = extract_mc_root(
            model,
            dataset,
            device,
            args.batch_size,
            args.mc_samples,
            args.seed
            + args.iteration
            + {"train": 10, "dev": 20, "test": 30}[split],
        )
        mc_stats["h_root"] = extract_root_features(
            model,
            dataset,
            device,
            args.batch_size,
        )

        save_z_root(
            paths["evidence"] / "{}_root.npz".format(split),
            event_ids,
            logits,
            labels,
            {
                "branch": "root",
                "split": split,
                "best_checkpoint": "root_best.m",
                "mc_samples": args.mc_samples,
            },
            mc_stats=mc_stats,
        )


def train_graph(
    args,
    paths,
    device,
):
    train_rows = load_graph_rows(
        paths["csv"],
        "train",
        args.dataset,
        args.fold_id,
        paths["graphs"],
        iteration=args.iteration,
        split_seed=args.seed,
    )
    dev_rows = load_graph_rows(
        paths["csv"],
        "dev",
        args.dataset,
        args.fold_id,
        paths["graphs"],
        iteration=args.iteration,
        split_seed=args.seed,
    )

    train_data = GraphNPZDataset(
        train_rows,
        paths["graphs"],
    )
    dev_data = GraphNPZDataset(
        dev_rows,
        paths["graphs"],
    )

    train_labels = dataset_labels(train_data)
    train_loader = make_graph_train_loader(
        train_data,
        train_labels,
        args,
        args.seed + args.iteration,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=th.cuda.is_available(),
    )
    class_weights = make_training_class_weights(
        train_labels,
        args,
        device,
    )

    print(
        "Graph training counts:",
        np.bincount(
            train_labels,
            minlength=args.num_classes,
        ).tolist(),
    )
    if class_weights is not None:
        print(
            "Graph class weights:",
            class_weights.detach().cpu().tolist(),
        )

    model = GraphEncoder(
        in_feats=args.in_feats,
        hid_feats=args.graph_hidden_dim,
        out_feats=args.graph_out_dim,
        num_classes=args.num_classes,
        dropout=args.graph_dropout,
    ).to(device)

    bu_parameters = list(
        model.BUrumorGCN.parameters()
    )
    bu_parameter_ids = {
        id(parameter)
        for parameter in bu_parameters
    }
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in bu_parameter_ids
    ]

    optimizer = th.optim.AdamW(
        [
            {
                "params": other_parameters,
                "lr": args.graph_lr,
            },
            {
                "params": bu_parameters,
                "lr": (
                    args.graph_lr
                    * args.graph_bu_lr_ratio
                ),
            },
        ],
        weight_decay=args.graph_weight_decay,
    )

    return train_graph_like(
        branch="graph",
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        class_weights=class_weights,
        args=args,
        paths=paths,
        device=device,
    )


def train_stance(
    args,
    paths,
    device,
):
    train_rows = load_stance_rows(
        paths["csv"],
        "train",
        args.dataset,
        args.fold_id,
        paths["graphs"],
        iteration=args.iteration,
        split_seed=args.seed,
    )
    dev_rows = load_stance_rows(
        paths["csv"],
        "dev",
        args.dataset,
        args.fold_id,
        paths["graphs"],
        iteration=args.iteration,
        split_seed=args.seed,
    )

    train_data = StanceNPZDataset(
        train_rows,
        paths["graphs"],
        stance_dim=args.stance_dim,
        text_dim=args.reply_text_dim,
    )
    dev_data = StanceNPZDataset(
        dev_rows,
        paths["graphs"],
        stance_dim=args.stance_dim,
        text_dim=args.reply_text_dim,
    )

    train_labels = dataset_labels(train_data)
    train_loader = make_graph_train_loader(
        train_data,
        train_labels,
        args,
        args.seed + args.iteration + 2000,
    )
    dev_loader = DataLoader(
        dev_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=th.cuda.is_available(),
    )
    class_weights = make_training_class_weights(
        train_labels,
        args,
        device,
    )

    print(
        "Stance training counts:",
        np.bincount(
            train_labels,
            minlength=args.num_classes,
        ).tolist(),
    )
    if class_weights is not None:
        print(
            "Stance class weights:",
            class_weights.detach().cpu().tolist(),
        )

    model = StanceEncoder(
        stance_dim=args.stance_dim,
        text_dim=args.reply_text_dim,
        text_hidden_dim=args.reply_text_hidden_dim,
        stance_hidden_dim=args.stance_hidden_dim,
        representation_dim=args.representation_dim,
        num_classes=args.num_classes,
        dropout=args.stance_dropout,
    ).to(device)

    optimizer = th.optim.AdamW(
        model.parameters(),
        lr=args.stance_lr,
        weight_decay=args.stance_weight_decay,
    )

    return train_graph_like(
        branch="stance",
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        optimizer=optimizer,
        class_weights=class_weights,
        args=args,
        paths=paths,
        device=device,
    )


def train_graph_like(
    branch,
    model,
    train_loader,
    dev_loader,
    optimizer,
    class_weights,
    args,
    paths,
    device,
):
    checkpoint = (
        paths["checkpoints"]
        / "{}_best.m".format(branch)
    )
    state = {
        "best": None,
        "best_macro_f1": None,
        "best_nll": None,
        "counter": 0,
        "best_epoch": -1,
    }
    scheduler = th.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )
    history = []

    for epoch in range(args.epochs):
        state["epoch"] = epoch

        train_stats = graph_like_epoch(
            branch=branch,
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            args=args,
            class_weights=class_weights,
            epoch=epoch,
        )
        dev = evaluate_graph_like(
            model,
            dev_loader,
            device,
            args.num_classes,
        )

        current_lr = float(
            optimizer.param_groups[0]["lr"]
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "ce_loss": train_stats["ce_loss"],
            "supcon_loss": train_stats["supcon_loss"],
            "supcon_weight": train_stats["supcon_weight"],
            "learning_rate": current_lr,
            **dev,
        })

        print(
            "{} epoch {:03d} | loss {:.4f} | CE {:.4f} | "
            "SupCon {:.4f} (w={:.3f}) | lr {:.2e} | "
            "Dev Macro-F1 {:.2f}% | NLL {:.4f}".format(
                branch.capitalize(),
                epoch,
                train_stats["loss"],
                train_stats["ce_loss"],
                train_stats["supcon_loss"],
                train_stats["supcon_weight"],
                current_lr,
                100.0 * dev["macro_f1"],
                dev["nll"],
            )
        )

        scheduler.step(dev["macro_f1"])

        if update_early_stopping(
            state,
            dev,
            model,
            checkpoint,
            args.patience,
        ):
            break

    load_state(model, checkpoint)
    export_graph_like(
        branch,
        model,
        args,
        paths,
        device,
    )
    return checkpoint, state, history


def export_graph_like(
    branch,
    model,
    args,
    paths,
    device,
):
    splits = ["train", "dev"] + (
        ["test"] if args.export_test else []
    )

    for split in splits:
        if branch == "graph":
            rows = load_graph_rows(
                paths["csv"],
                split,
                args.dataset,
                args.fold_id,
                paths["graphs"],
                iteration=args.iteration,
                split_seed=args.seed,
            )
            dataset = GraphNPZDataset(
                rows,
                paths["graphs"],
            )
            event_ids, logits, labels = extract_z_graph(
                model,
                dataset,
                device,
                args.batch_size,
            )
            mc_stats = extract_mc_graph(
                model,
                dataset,
                device,
                args.batch_size,
                args.mc_samples,
                args.seed
                + args.iteration
                + {"train": 110, "dev": 120, "test": 130}[split],
            )
            mc_stats["h_graph"] = extract_graph_like_features(
                model,
                dataset,
                device,
                args.batch_size,
            )
            save_z_graph(
                paths["evidence"] / "{}_graph.npz".format(split),
                event_ids,
                logits,
                labels,
                {
                    "branch": branch,
                    "split": split,
                    "best_checkpoint": "graph_best.m",
                    "mc_samples": args.mc_samples,
                },
                mc_stats=mc_stats,
            )
        else:
            rows = load_stance_rows(
                paths["csv"],
                split,
                args.dataset,
                args.fold_id,
                paths["graphs"],
                iteration=args.iteration,
                split_seed=args.seed,
            )
            dataset = StanceNPZDataset(
                rows,
                paths["graphs"],
                stance_dim=args.stance_dim,
                text_dim=args.reply_text_dim,
            )
            event_ids, logits, confidence, labels = extract_z_stance(
                model,
                dataset,
                device,
                args.batch_size,
            )
            mc_stats = extract_mc_stance(
                model,
                dataset,
                device,
                args.batch_size,
                args.mc_samples,
                args.seed
                + args.iteration
                + {"train": 210, "dev": 220, "test": 230}[split],
            )
            mc_stats["h_stance"] = extract_graph_like_features(
                model,
                dataset,
                device,
                args.batch_size,
            )
            save_z_stance(
                paths["evidence"] / "{}_stance.npz".format(split),
                event_ids,
                logits,
                confidence,
                labels,
                {
                    "branch": branch,
                    "split": split,
                    "best_checkpoint": "stance_best.m",
                    "mc_samples": args.mc_samples,
                },
                mc_stats=mc_stats,
            )


def write_history(path, rows):
    fieldnames = [
        "branch",
        "epoch",
        "train_loss",
        "ce_loss",
        "supcon_loss",
        "supcon_weight",
        "learning_rate",
        "accuracy",
        "macro_f1",
        "class_f1",
        "nll",
    ]

    with open(
        path,
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset",
        default="RumourEval2019",
    )
    parser.add_argument(
        "--branch",
        choices=["all", *BRANCHES],
        default="all",
    )
    parser.add_argument(
        "--iteration",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--fold-id",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--input-csv",
        default=None,
    )
    parser.add_argument(
        "--graph-dir",
        default=None,
    )
    parser.add_argument(
        "--output-dir",
        default=None,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
    )

    # Backward-compatible global settings.
    parser.add_argument(
        "--encoder-lr",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
    )

    # Branch-specific optimization.
    parser.add_argument(
        "--graph-lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--root-lr",
        type=float,
        default=3e-4,
    )
    parser.add_argument(
        "--stance-lr",
        type=float,
        default=1e-4,
    )
    parser.add_argument(
        "--graph-weight-decay",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--root-weight-decay",
        type=float,
        default=1e-2,
    )
    parser.add_argument(
        "--stance-weight-decay",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--graph-bu-lr-ratio",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--balance-mode",
        choices=["batch", "loss", "none"],
        default="loss",
        help=(
            "batch: balanced sampler; "
            "loss: natural sampling with sqrt inverse-frequency weights; "
            "none: natural sampling with unweighted CE"
        ),
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--graph-supcon-weight",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--root-supcon-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--stance-supcon-weight",
        type=float,
        default=0.03,
    )
    parser.add_argument(
        "--supcon-temperature",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--supcon-warmup-epochs",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--scheduler-factor",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--scheduler-patience",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-6,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--grad-clip",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--representation-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--in-feats",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--graph-hidden-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--graph-out-dim",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--graph-dropout",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--root-hidden-dim",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--root-dropout",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--stance-dim",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--reply-text-dim",
        type=int,
        default=768,
    )
    parser.add_argument(
        "--reply-text-hidden-dim",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--stance-hidden-dim",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--stance-dropout",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=20,
        help=(
            "Stochastic dropout passes used only when exporting uncertainty"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )
    parser.add_argument(
        "--export-test",
        action="store_true",
        help=(
            "Export test logits only after all model choices are finalized"
        ),
    )

    return parser.parse_args()


def validate_and_resolve_args(args):
    configure_dataset_args(args)

    if args.encoder_lr is not None:
        args.graph_lr = args.encoder_lr
        args.root_lr = args.encoder_lr
        args.stance_lr = args.encoder_lr

    if args.weight_decay is not None:
        args.graph_weight_decay = args.weight_decay
        args.root_weight_decay = args.weight_decay
        args.stance_weight_decay = args.weight_decay

    if min(
        args.epochs,
        args.patience,
        args.batch_size,
    ) <= 0:
        raise ValueError(
            "epochs, patience and batch-size must be positive"
        )
    if args.mc_samples < 2:
        raise ValueError(
            "mc-samples must be at least 2"
        )
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError(
            "label-smoothing must be in [0,1)"
        )
    if args.supcon_temperature <= 0:
        raise ValueError(
            "supcon-temperature must be positive"
        )
    if args.supcon_warmup_epochs < 0:
        raise ValueError(
            "supcon-warmup-epochs must be non-negative"
        )
    if not 0.0 < args.scheduler_factor < 1.0:
        raise ValueError(
            "scheduler-factor must be in (0,1)"
        )
    if args.scheduler_patience < 1:
        raise ValueError(
            "scheduler-patience must be positive"
        )
    if args.graph_bu_lr_ratio <= 0:
        raise ValueError(
            "graph-bu-lr-ratio must be positive"
        )
    if args.num_workers < 0:
        raise ValueError(
            "num-workers must be non-negative"
        )

    for name, value in {
        "graph_lr": args.graph_lr,
        "root_lr": args.root_lr,
        "stance_lr": args.stance_lr,
        "graph_weight_decay": args.graph_weight_decay,
        "root_weight_decay": args.root_weight_decay,
        "stance_weight_decay": args.stance_weight_decay,
        "graph_supcon_weight": args.graph_supcon_weight,
        "root_supcon_weight": args.root_supcon_weight,
        "stance_supcon_weight": args.stance_supcon_weight,
    }.items():
        if value < 0:
            raise ValueError(
                "{} must be non-negative".format(name)
            )

    if (
        args.balance_mode == "batch"
        and args.batch_size < args.num_classes
    ):
        raise ValueError(
            "batch-size must be at least num-classes "
            "when balance-mode=batch"
        )

    return args


def main(args=None):
    args = validate_and_resolve_args(parse_args() if args is None else args)
    dataset_spec = configure_dataset_args(args)

    if dataset_spec["name"] == "PHEME" and args.fold_id is None:
        for fold_id in range(5):
            fold_args = copy.deepcopy(args)
            fold_args.fold_id = fold_id
            if args.output_dir:
                fold_args.output_dir = str(Path(args.output_dir) / "fold{}".format(fold_id))
            main(fold_args)
        return 0

    set_seed(args.seed + args.iteration)

    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else resolve_training_dir(
            args.dataset,
            args.iteration,
            args.fold_id,
        )
    )
    paths = {
        "csv": resolve_input_csv(
            args.input_csv,
            args.dataset,
        ),
        "graphs": resolve_graph_dir(
            args.graph_dir,
            args.dataset,
        ),
        "checkpoints": output / "checkpoints",
        "evidence": output / "evidence",
    }
    paths["checkpoints"].mkdir(
        parents=True,
        exist_ok=True,
    )
    paths["evidence"].mkdir(
        parents=True,
        exist_ok=True,
    )

    if dataset_spec["name"] == "PHEME":
        partition = {
            split: [row["event_id"] for row in load_graph_rows(
                paths["csv"], split, args.dataset, args.fold_id, paths["graphs"],
                iteration=args.iteration, split_seed=args.seed,
            )]
            for split in ("train", "dev", "test")
        }
        with open(output / "split_manifest.json", "w", encoding="utf-8") as handle:
            json.dump({"protocol": "original-bigcn-classwise-five-fold",
                       "split_seed": args.seed + args.iteration,
                       "iteration": args.iteration, "fold_id": args.fold_id,
                       **partition}, handle, indent=2)

    device = th.device(
        "cuda:0"
        if th.cuda.is_available()
        else "cpu"
    )
    print("Device:", device)
    if device.type == "cuda":
        print(
            "GPU:",
            th.cuda.get_device_name(0),
        )

    selected = (
        BRANCHES
        if args.branch == "all"
        else (args.branch,)
    )

    summaries = {}
    histories = []

    for branch in selected:
        branch_seed = (
            args.seed
            + args.iteration
            + {
                "graph": 0,
                "root": 1000,
                "stance": 2000,
            }[branch]
        )
        set_seed(branch_seed)

        print("\n" + "=" * 80)
        print(
            "Training independent {} encoder".format(branch)
        )
        print("Branch seed:", branch_seed)
        print("=" * 80)

        trainer = {
            "graph": train_graph,
            "root": train_root,
            "stance": train_stance,
        }[branch]

        checkpoint, state, history = trainer(
            args,
            paths,
            device,
        )
        summaries[branch] = {
            "checkpoint": str(
                checkpoint.resolve()
            ),
            "best_dev_macro_f1": state["best"],
            "best_dev_nll": state.get("best_nll"),
            "best_epoch": state["best_epoch"],
        }

        for row in history:
            histories.append({
                "branch": branch,
                **row,
            })

        if th.cuda.is_available():
            th.cuda.empty_cache()

    write_history(
        output / "encoder_history.csv",
        histories,
    )

    with open(
        output / "encoder_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "dataset": args.dataset,
                "storage_dataset": dataset_spec["storage"],
                "iteration": args.iteration,
                "fold_id": args.fold_id,
                "test_exported": args.export_test,
                "training_configuration": {
                    "balance_mode": args.balance_mode,
                    "label_smoothing": args.label_smoothing,
                    "graph_lr": args.graph_lr,
                    "root_lr": args.root_lr,
                    "stance_lr": args.stance_lr,
                    "graph_weight_decay": args.graph_weight_decay,
                    "root_weight_decay": args.root_weight_decay,
                    "stance_weight_decay": args.stance_weight_decay,
                    "graph_bu_lr_ratio": args.graph_bu_lr_ratio,
                    "graph_supcon_weight": args.graph_supcon_weight,
                    "root_supcon_weight": args.root_supcon_weight,
                    "stance_supcon_weight": args.stance_supcon_weight,
                    "supcon_temperature": args.supcon_temperature,
                    "supcon_warmup_epochs": args.supcon_warmup_epochs,
                },
                "branch_seeds": {
                    branch: (
                        args.seed
                        + args.iteration
                        + {
                            "graph": 0,
                            "root": 1000,
                            "stance": 2000,
                        }[branch]
                    )
                    for branch in selected
                },
                "branches": summaries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )

    print(
        "\nIndependent encoder training completed:",
        output.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
