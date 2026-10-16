# -*- coding: utf-8 -*-
"""Train and calibrate a BERTweet RumourEval2019 reply-stance classifier.

RumourEval train/dev stance labels are used for fitting and model selection.
The official test labels are evaluated exactly once per seed.  The saved
classifiers are later consumed by the separate PHEME pseudo-label/stance_x
generation script; this training script never reads or modifies graph files.
"""

import argparse
import csv
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL2ID = {"support": 0, "deny": 1, "query": 2, "comment": 3}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
LABEL_NAMES = ["Support", "Deny", "Query", "Comment"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_ready(value), handle, ensure_ascii=False, indent=2)


def save_prediction_results(
    output_dir, split, examples, logits, probabilities, predictions, labels
):
    """Save reply-level predictions in both paper-friendly CSV and NPZ."""
    output_dir = Path(output_dir)
    if not (
        len(examples) == len(logits) == len(probabilities)
        == len(predictions) == len(labels)
    ):
        raise ValueError("{} prediction arrays are not aligned".format(split))

    rows = []
    for index, (example, prediction, label) in enumerate(
        zip(examples, predictions, labels)
    ):
        row = {
            "sample_index": index,
            "split": split,
            "event_id": example["event_id"],
            "reply_id": example["node_id"],
            "true_label_id": int(label),
            "true_label": LABEL_NAMES[int(label)],
            "predicted_label_id": int(prediction),
            "predicted_label": LABEL_NAMES[int(prediction)],
            "confidence": float(probabilities[index].max()),
            "correct": bool(int(prediction) == int(label)),
            "reply_text": example["reply"],
        }
        for class_id, class_name in enumerate(LABEL_NAMES):
            key = class_name.lower()
            row["{}_logit".format(key)] = float(logits[index, class_id])
            row["{}_probability".format(key)] = float(
                probabilities[index, class_id]
            )
        rows.append(row)

    csv_path = output_dir / "{}_predictions.csv".format(split)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    npz_path = output_dir / "{}_predictions.npz".format(split)
    np.savez_compressed(
        npz_path,
        event_ids=np.asarray([item["event_id"] for item in examples]),
        reply_ids=np.asarray([item["node_id"] for item in examples]),
        labels=np.asarray(labels, dtype=np.int64),
        predictions=np.asarray(predictions, dtype=np.int64),
        logits=np.asarray(logits, dtype=np.float32),
        probabilities=np.asarray(probabilities, dtype=np.float32),
        class_names=np.asarray(LABEL_NAMES),
    )
    return csv_path, npz_path


def save_normalized_confusion_heatmap(output_dir, split, labels, predictions):
    """Save a row-normalized confusion matrix (y=true, x=predicted)."""
    output_dir = Path(output_dir)
    matrix = confusion_matrix(
        labels, predictions, labels=[0, 1, 2, 3], normalize="true"
    )
    matrix = np.nan_to_num(matrix, nan=0.0)

    csv_path = output_dir / "{}_confusion_matrix_normalized.csv".format(split)
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["True / Predicted"] + LABEL_NAMES)
        for class_name, values in zip(LABEL_NAMES, matrix):
            writer.writerow([class_name] + [float(value) for value in values])

    figure, axis = plt.subplots(figsize=(7.2, 6.0))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Proportion")
    axis.set(
        xticks=np.arange(len(LABEL_NAMES)),
        yticks=np.arange(len(LABEL_NAMES)),
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title="{} normalized confusion matrix".format(split.capitalize()),
    )
    threshold = 0.5
    for true_id in range(matrix.shape[0]):
        for predicted_id in range(matrix.shape[1]):
            value = matrix[true_id, predicted_id]
            axis.text(
                predicted_id, true_id, "{:.2f}".format(value),
                ha="center", va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    png_path = output_dir / "{}_confusion_matrix_normalized.png".format(split)
    pdf_path = output_dir / "{}_confusion_matrix_normalized.pdf".format(split)
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return matrix, csv_path, png_path, pdf_path


def read_json(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return json.load(handle)


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def find_one_json(folder):
    folder = Path(folder)
    if not folder.exists():
        return None
    files = sorted(
        path for path in folder.glob("*.json")
        if not path.name.startswith((".", "._"))
    )
    return files[0] if files else None


def get_text_from_json(obj):
    if not isinstance(obj, dict):
        return ""
    for container in (obj, obj.get("data", {})):
        if not isinstance(container, dict):
            continue
        for key in ("text", "full_text", "body", "title", "selftext", "content"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return normalize_text(value)
    return ""


def normalize_stance(label):
    label = str(label).strip().lower() if label is not None else ""
    aliases = {
        "support": {"support", "supporting", "supports", "agree", "agrees"},
        "deny": {"deny", "denying", "denies", "refute", "refutes", "disagree", "contradict"},
        "query": {"query", "querying", "question", "questions", "ask", "asks"},
        "comment": {"comment", "commenting", "other", "neutral", "none", "unclear"},
    }
    for canonical, values in aliases.items():
        if label in values:
            return canonical
    return None


def load_key_file(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError("missing stance key: {}".format(path))
    raw = read_json(path)
    if "subtaskaenglish" in raw:
        raw = raw["subtaskaenglish"]
    elif "subtaska" in raw:
        raw = raw["subtaska"]
    result = {}
    for node_id, label in raw.items():
        normalized = normalize_stance(label)
        if normalized is not None:
            result[str(node_id)] = normalized
    return result


def resolve_key_path(explicit_path, search_root, filename):
    if explicit_path:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError("missing explicit key path: {}".format(path))
        return path
    direct = Path(search_root) / filename
    if direct.exists():
        return direct
    matches = sorted(Path(search_root).glob("**/{}".format(filename)))
    if len(matches) != 1:
        raise FileNotFoundError(
            "expected one {} below {}, found {}".format(filename, search_root, len(matches))
        )
    return matches[0]


def flatten_structure(structure):
    edges = []

    def visit(parent, children):
        if not isinstance(children, dict):
            return
        for child, descendants in children.items():
            edges.append((str(parent), str(child)))
            visit(str(child), descendants)

    if isinstance(structure, dict):
        for root, children in structure.items():
            visit(str(root), children)
    return edges


def collect_event_dirs(root_dir):
    root_dir = Path(root_dir)
    return sorted(
        path.parent for path in root_dir.glob("**/structure.json")
        if not path.name.startswith((".", "._"))
    )


def find_source_json(event_dir):
    for folder_name in (
        "source-tweet", "source-tweets", "source-post", "source-posts", "source"
    ):
        path = find_one_json(Path(event_dir) / folder_name)
        if path is not None:
            return path
    return None


def read_event(event_dir, split):
    event_dir = Path(event_dir)
    structure = read_json(event_dir / "structure.json")
    if not isinstance(structure, dict) or not structure:
        return None
    root_id = str(next(iter(structure)))
    source_path = find_source_json(event_dir)
    if source_path is None:
        return None
    source_obj = read_json(source_path)
    source_id = str(source_obj.get("id_str", source_obj.get("id", root_id)))
    root_text = get_text_from_json(source_obj)
    if not root_text:
        return None

    id_to_text = {root_id: root_text, source_id: root_text}
    id_to_order = {}
    order = 1
    for folder_name in ("replies", "reactions", "comments"):
        folder = event_dir / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json")):
            if path.name.startswith((".", "._")):
                continue
            try:
                obj = read_json(path)
                node_id = str(obj.get("id_str", obj.get("id", obj.get("tweet_id", path.stem))))
                text = get_text_from_json(obj)
                if text:
                    id_to_text[node_id] = text
                    id_to_order[node_id] = order
                    order += 1
            except Exception:
                continue

    edges = flatten_structure(structure)
    parent_map = {child: parent for parent, child in edges}
    node_ids = {root_id}
    for parent, child in edges:
        node_ids.add(parent)
        node_ids.add(child)
    node_order = [root_id] + sorted(node_ids - {root_id})
    return {
        "event_id": event_dir.name,
        "split": split,
        "root_id": root_id,
        "source_id": source_id,
        "root_text": root_text,
        "id_to_text": id_to_text,
        "id_to_order": id_to_order,
        "parent_map": parent_map,
        "node_order": node_order,
    }


def load_events(train_root, test_root):
    events = {}
    for split, root in (("train_pool", train_root), ("test", test_root)):
        for event_dir in collect_event_dirs(root):
            event = read_event(event_dir, split)
            if event is not None:
                events[event["event_id"]] = event
    return events


def build_labeled_examples(events, labels, split_name):
    examples = []
    for event in events.values():
        if split_name == "test" and event["split"] != "test":
            continue
        if split_name != "test" and event["split"] == "test":
            continue
        for node_id in event["node_order"]:
            if node_id == event["root_id"] or node_id == event["source_id"]:
                continue
            label_name = labels.get(node_id)
            reply = event["id_to_text"].get(node_id, "")
            if label_name not in LABEL2ID or not reply:
                continue
            parent_id = event["parent_map"].get(node_id, event["root_id"])
            examples.append({
                "event_id": event["event_id"],
                "node_id": node_id,
                "source": event["root_text"],
                "parent": event["id_to_text"].get(parent_id, event["root_text"]),
                "reply": reply,
                "label": LABEL2ID[label_name],
                "split": split_name,
            })
    return examples


class StanceDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        item = self.examples[index]
        # BERTweet/RoBERTa does not use token-type embeddings. Encode the
        # structured context as one sequence instead of a text pair. This
        # avoids the slow-tokenizer warning that overflowing tokens cannot be
        # returned for paired inputs with longest-first truncation.
        text = (
            # Put the reply first because right-side truncation should retain
            # the primary stance-bearing text before its conversational
            # context.
            "Reply: {}\n"
            "Parent post: {}\n"
            "Source claim: {}"
        ).format(
            item["reply"][:300],
            item["parent"][:300],
            item["source"][:400],
        )
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
        )
        result = {key: torch.tensor(value, dtype=torch.long) for key, value in encoded.items()}
        if item.get("label") is not None:
            result["labels"] = torch.tensor(item["label"], dtype=torch.long)
        return result


def compute_class_weights(examples, max_weight):
    counts = np.bincount([item["label"] for item in examples], minlength=4).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    effective_num = 1.0 - np.power(0.999, counts)
    weights = (1.0 - 0.999) / effective_num
    weights /= weights.mean()
    if max_weight is not None:
        weights = np.minimum(weights, max_weight)
        weights /= weights.mean()
    return counts, torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    logits_all = []
    labels_all = []
    for batch in loader:
        labels = batch.pop("labels").to(device)
        inputs = {key: value.to(device) for key, value in batch.items()}
        logits = model(**inputs).logits
        logits_all.append(logits.cpu())
        labels_all.append(labels.cpu())
    return (
        torch.cat(logits_all, dim=0).numpy(),
        torch.cat(labels_all, dim=0).numpy(),
    )


def softmax_np(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64) / max(float(temperature), 1e-6)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return (exp_logits / exp_logits.sum(axis=1, keepdims=True)).astype(np.float32)


def fit_temperature(logits, labels):
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=100)

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 10.0)
        loss = F.cross_entropy(logits_tensor / temperature, labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.exp().clamp(0.05, 10.0).detach().item())


def probability_nll(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    return float(-np.log(np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())


def predict_with_thresholds(probabilities, thresholds):
    thresholds = np.asarray(thresholds, dtype=np.float32).reshape(1, -1)
    return np.argmax(probabilities / thresholds, axis=1)


def search_thresholds(probabilities, labels):
    # Coarser and bounded grid reduces threshold overfitting on the small dev set.
    grids = (
        np.arange(0.7, 1.31, 0.1),
        np.arange(0.5, 1.21, 0.1),
        np.arange(0.5, 1.21, 0.1),
    )
    best = {"score": -1.0, "thresholds": [1.0, 1.0, 1.0, 1.0]}
    for support in grids[0]:
        for deny in grids[1]:
            for query in grids[2]:
                thresholds = [support, deny, query, 1.0]
                predictions = predict_with_thresholds(probabilities, thresholds)
                score = f1_score(
                    labels, predictions, labels=[0, 1, 2, 3], average="macro", zero_division=0
                )
                if score > best["score"]:
                    best = {"score": float(score), "thresholds": [float(x) for x in thresholds]}
    return best


def metric_bundle(labels, predictions):
    precision, recall, f1_values, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1, 2, 3], average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "per_class": {
            LABEL_NAMES[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1_values[i]),
                "support": int(support[i]),
            }
            for i in range(4)
        },
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1, 2, 3]).tolist(),
    }


def train_seed(seed, args, tokenizer, train_examples, dev_examples, test_examples):
    set_seed(seed)
    seed_dir = Path(args.output_dir) / "seed_{}".format(seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=4, id2label=ID2LABEL, label2id=LABEL2ID
    )
    embedding_size = model.get_input_embeddings().num_embeddings
    tokenizer_size = len(tokenizer)
    if tokenizer_size != embedding_size:
        print(
            "Resizing token embeddings: {} -> {}".format(
                embedding_size, tokenizer_size
            )
        )
        model.resize_token_embeddings(tokenizer_size)

    # RoBERTa/BERTweet reserves two position indices because position IDs
    # start after the padding index. A config value of 130 therefore supports
    # at most 128 input tokens. Catch an invalid length on CPU before CUDA
    # reports an opaque device-side assertion.
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None:
        max_input_length = int(max_positions)
        if getattr(model.config, "model_type", "") == "roberta":
            max_input_length -= 2
        if args.max_length > max_input_length:
            raise ValueError(
                "--max-length {} exceeds the {} model limit {}; "
                "use --max-length {} or smaller".format(
                    args.max_length,
                    getattr(model.config, "model_type", "transformer"),
                    max_input_length,
                    max_input_length,
                )
            )
    train_set = StanceDataset(train_examples, tokenizer, args.max_length)
    dev_set = StanceDataset(dev_examples, tokenizer, args.max_length)
    test_set = StanceDataset(test_examples, tokenizer, args.max_length)
    counts, class_weights = compute_class_weights(train_examples, args.class_weight_max)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    dev_loader = DataLoader(dev_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    class_weights = class_weights.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    total_steps = max(len(train_loader) * args.epochs, 1)
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_factor(step):
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return max(
            0.0,
            float(total_steps - step) / float(max(total_steps - warmup_steps, 1)),
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    best_dev_macro_f1 = -1.0
    best_epoch = -1
    best_state = None
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            inputs = {key: value.to(device) for key, value in batch.items()}
            logits = model(**inputs).logits
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item())
            global_step += 1
            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                print(
                    "seed {} epoch {}/{} step {} train_loss {:.4f}".format(
                        seed, epoch + 1, args.epochs, global_step,
                        running_loss / max(global_step - epoch * len(train_loader), 1),
                    )
                )

        dev_logits_epoch, dev_labels_epoch = predict_logits(model, dev_loader, device)
        dev_predictions_epoch = np.argmax(dev_logits_epoch, axis=1)
        dev_macro_f1_epoch = f1_score(
            dev_labels_epoch,
            dev_predictions_epoch,
            labels=[0, 1, 2, 3],
            average="macro",
            zero_division=0,
        )
        print(
            "seed {} epoch {}/{} train_loss {:.4f} dev_macro_f1 {:.4f}".format(
                seed, epoch + 1, args.epochs,
                running_loss / max(len(train_loader), 1),
                dev_macro_f1_epoch,
            )
        )
        if dev_macro_f1_epoch > best_dev_macro_f1:
            best_dev_macro_f1 = float(dev_macro_f1_epoch)
            best_epoch = epoch + 1
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training did not produce a best model")
    model.load_state_dict(best_state)
    model.to(device)
    best_model_dir = seed_dir / "best_model"
    model.save_pretrained(best_model_dir)
    tokenizer.save_pretrained(seed_dir / "best_model")

    # Dev is used for calibration and threshold/model selection only.
    dev_logits, dev_labels = predict_logits(model, dev_loader, device)
    temperature = fit_temperature(dev_logits, dev_labels)
    dev_probabilities = softmax_np(dev_logits, temperature)
    threshold_result = search_thresholds(dev_probabilities, dev_labels)
    dev_predictions = predict_with_thresholds(dev_probabilities, threshold_result["thresholds"])

    # Official test is evaluated exactly once after model selection,
    # temperature calibration, and threshold selection are complete.
    test_logits, test_labels = predict_logits(model, test_loader, device)
    test_probabilities = softmax_np(test_logits, temperature)
    test_predictions = predict_with_thresholds(test_probabilities, threshold_result["thresholds"])

    metrics = {
        "seed": seed,
        "best_epoch": best_epoch,
        "best_dev_macro_f1_before_calibration": best_dev_macro_f1,
        "train_class_counts": counts,
        "class_weights": class_weights.detach().cpu().numpy(),
        "temperature": temperature,
        "thresholds": threshold_result["thresholds"],
        "calibration": {
            "dev_nll_before": probability_nll(
                softmax_np(dev_logits, 1.0), dev_labels
            ),
            "dev_nll_after": probability_nll(dev_probabilities, dev_labels),
        },
        "dev": metric_bundle(dev_labels, dev_predictions),
        "test": metric_bundle(test_labels, test_predictions),
    }
    save_json(seed_dir / "metrics.json", metrics)
    save_json(seed_dir / "calibration.json", {
        "temperature": temperature,
        "thresholds": threshold_result["thresholds"],
        "dev_threshold_macro_f1": threshold_result["score"],
    })
    saved_prediction_files = {}
    for split, examples, logits, probabilities, predictions, split_labels in (
        (
            "dev", dev_examples, dev_logits, dev_probabilities,
            dev_predictions, dev_labels,
        ),
        (
            "test", test_examples, test_logits, test_probabilities,
            test_predictions, test_labels,
        ),
    ):
        prediction_csv, prediction_npz = save_prediction_results(
            seed_dir, split, examples, logits, probabilities,
            predictions, split_labels,
        )
        matrix, matrix_csv, matrix_png, matrix_pdf = (
            save_normalized_confusion_heatmap(
                seed_dir, split, split_labels, predictions
            )
        )
        saved_prediction_files[split] = {
            "predictions_csv": str(prediction_csv),
            "predictions_npz": str(prediction_npz),
            "normalized_confusion_matrix": matrix,
            "confusion_csv": str(matrix_csv),
            "confusion_png": str(matrix_png),
            "confusion_pdf": str(matrix_pdf),
        }
        print("Saved {} predictions: {}".format(split, prediction_csv))
        print("Saved {} normalized confusion heatmap: {}".format(
            split, matrix_png
        ))
    save_json(seed_dir / "prediction_artifacts.json", saved_prediction_files)
    with open(seed_dir / "test_report.txt", "w", encoding="utf-8") as handle:
        handle.write(classification_report(
            test_labels,
            test_predictions,
            labels=[0, 1, 2, 3],
            target_names=LABEL_NAMES,
            digits=4,
            zero_division=0,
        ))
    return metrics


def summarize_metrics(seed_metrics):
    summary = {"num_seeds": len(seed_metrics), "seeds": [item["seed"] for item in seed_metrics]}
    for split in ("dev", "test"):
        for metric in ("accuracy", "macro_f1"):
            values = np.array([item[split][metric] for item in seed_metrics], dtype=np.float64)
            summary["{}_{}_mean".format(split, metric)] = float(values.mean())
            summary["{}_{}_std".format(split, metric)] = float(values.std(ddof=0))
        for class_name in LABEL_NAMES:
            values = np.array([
                item[split]["per_class"][class_name]["f1"] for item in seed_metrics
            ], dtype=np.float64)
            summary["{}_{}_f1_mean".format(split, class_name.lower())] = float(values.mean())
            summary["{}_{}_f1_std".format(split, class_name.lower())] = float(values.std(ddof=0))
    temperatures = np.array([item["temperature"] for item in seed_metrics], dtype=np.float64)
    summary["temperature_mean"] = float(temperatures.mean())
    summary["temperature_std"] = float(temperatures.std(ddof=0))
    return summary


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "rumoureval2019"))
    parser.add_argument(
        "--model-name",
        default=str(PROJECT_ROOT / "pretrained" / "bertweet-base"),
        help="Pretrained BERTweet model directory (default: pretrained/bertweet-base)",
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "checkpoint" / "rumoureval_stance"))
    parser.add_argument("--train-key", default=None)
    parser.add_argument("--dev-key", default=None)
    parser.add_argument("--test-key", default=None)
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Maximum token length; BERTweet supports at most 128 tokens",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--class-weight-max", type=float, default=1.5)
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_root = data_dir / "rumoureval-2019-training-data"
    test_root = data_dir / "rumoureval-2019-test-data"
    train_key = resolve_key_path(args.train_key, train_root, "train-key.json")
    dev_key = resolve_key_path(args.dev_key, train_root, "dev-key.json")
    test_key = resolve_key_path(args.test_key, data_dir, "final-eval-key.json")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")

    config = vars(args).copy()
    config.update({
        "project_root": str(PROJECT_ROOT),
        "train_root": str(train_root),
        "test_root": str(test_root),
        "resolved_train_key": str(train_key),
        "resolved_dev_key": str(dev_key),
        "resolved_test_key": str(test_key),
    })
    save_json(output_dir / "training_config.json", config)

    labels = {
        "train": load_key_file(train_key),
        "dev": load_key_file(dev_key),
        "test": load_key_file(test_key),
    }
    events = load_events(train_root, test_root)
    train_examples = build_labeled_examples(events, labels["train"], "train")
    dev_examples = build_labeled_examples(events, labels["dev"], "dev")
    test_examples = build_labeled_examples(events, labels["test"], "test")
    for name, examples in (
        ("train", train_examples), ("dev", dev_examples), ("test", test_examples)
    ):
        if not examples:
            raise RuntimeError("no {} examples found".format(name))
        print(name, "examples:", len(examples), "distribution:", Counter(x["label"] for x in examples))
    save_json(output_dir / "data_statistics.json", {
        "events": len(events),
        "train": Counter(x["label"] for x in train_examples),
        "dev": Counter(x["label"] for x in dev_examples),
        "test": Counter(x["label"] for x in test_examples),
    })

    # BERTweet's tokenizer performs tweet-specific tokenization and
    # normalization. Use its faithful slow-tokenizer implementation.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    seed_metrics = []
    for seed in seeds:
        print("=" * 80)
        print("Training stance classifier with seed", seed)
        metrics = train_seed(
            seed, args, tokenizer,
            train_examples, dev_examples, test_examples,
        )
        seed_metrics.append(metrics)
    summary = summarize_metrics(seed_metrics)
    best_seed_metrics = max(seed_metrics, key=lambda item: item["dev"]["macro_f1"])
    best_seed = int(best_seed_metrics["seed"])
    best_source = output_dir / "seed_{}".format(best_seed) / "best_model"
    best_dir = output_dir / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    shutil.copytree(best_source, best_dir)
    shutil.copy2(
        output_dir / "seed_{}".format(best_seed) / "calibration.json",
        best_dir / "calibration.json",
    )
    best_seed_dir = output_dir / "seed_{}".format(best_seed)
    for pattern in (
        "*_predictions.csv",
        "*_predictions.npz",
        "*_confusion_matrix_normalized.csv",
        "*_confusion_matrix_normalized.png",
        "*_confusion_matrix_normalized.pdf",
        "prediction_artifacts.json",
        "metrics.json",
        "test_report.txt",
    ):
        for source in best_seed_dir.glob(pattern):
            shutil.copy2(source, best_dir / source.name)
    summary["selected_best_seed"] = best_seed
    summary["selected_best_dev_macro_f1"] = best_seed_metrics["dev"]["macro_f1"]
    save_json(output_dir / "metrics_summary.json", summary)
    print("Metrics summary:", output_dir / "metrics_summary.json")
    print("Best classifier for PHEME pseudo-label generation:", best_dir)


if __name__ == "__main__":
    main()
