# -*- coding: utf-8 -*-
"""
Train a source-only DeBERTa/RoBERTa classifier for RumourEval2019 and export
its logits/probabilities/hidden states as root evidence.

Example:
  python Process/train_source_root_evidence.py \
    --model-name microsoft/deberta-v3-base \
    --epochs 5 \
    --batch-size 8

The output pickle has the following structure:
  {
    "model_name": ...,
    "label_names": ["FR", "TR", "UR"],
    "train_ids": [...],
    "test_ids": [...],
    "evidence": {
      event_id: {
        "logits": np.ndarray[num_classes],
        "prob": np.ndarray[num_classes],
        "hidden": np.ndarray[hidden_size],
        "pred": int,
        "label": int,
        "source_text": str
      }
    }
  }
"""

import argparse
import csv
import os
import pickle
import random
from collections import Counter
from pathlib import Path

if not os.environ.get("OMP_NUM_THREADS", "1").strip().isdigit():
    os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


CWD = Path(os.getcwd())
LABEL2ID = {
    "false": 0,
    "true": 1,
    "unverified": 2,
}
ID2LABEL = {
    0: "FR",
    1: "TR",
    2: "UR",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def read_json(path):
    import json

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def find_one_json(folder):
    folder = Path(folder)
    if not folder.exists():
        return None

    files = [
        p for p in folder.glob("*.json")
        if not p.name.startswith(".") and not p.name.startswith("._")
    ]

    files = sorted(files)
    if len(files) == 0:
        return None

    return files[0]


def get_text_from_json(obj):
    if not isinstance(obj, dict):
        return ""

    for key in ["text", "full_text", "body", "title", "selftext", "content"]:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value

    data = obj.get("data")
    if isinstance(data, dict):
        for key in ["text", "full_text", "body", "title", "selftext", "content"]:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value

    return ""


def load_key(path):
    key = read_json(path)
    veracity = key.get("subtaskbenglish", {})
    return {str(k): str(v).lower() for k, v in veracity.items()}


def load_training_veracity(raw_dir):
    train_root = Path(raw_dir) / "rumoureval-2019-training-data"
    veracity = {}

    for key_path in sorted(train_root.glob("**/*key*.json")):
        veracity.update(load_key(key_path))

    return veracity


def load_test_veracity(raw_dir):
    key_path = Path(raw_dir) / "final-eval-key.json"
    if not key_path.exists():
        return {}
    return load_key(key_path)


def collect_structure_dirs(root):
    root = Path(root)
    if not root.exists():
        return []

    return sorted([
        path.parent
        for path in root.glob("**/structure.json")
        if not path.name.startswith(".")
    ])


def source_json_path(thread_dir):
    path = find_one_json(Path(thread_dir) / "source-tweet")
    if path is None:
        path = find_one_json(Path(thread_dir) / "source-tweets")
    if path is None:
        path = find_one_json(Path(thread_dir) / "source-post")
    if path is None:
        path = find_one_json(Path(thread_dir) / "source-posts")
    return path


def structure_root_id(thread_dir):
    structure_path = Path(thread_dir) / "structure.json"
    if not structure_path.exists():
        return Path(thread_dir).name

    try:
        structure = read_json(structure_path)
        if isinstance(structure, dict) and len(structure) > 0:
            return str(next(iter(structure.keys())))
    except Exception:
        pass

    return Path(thread_dir).name


def read_records_from_raw(raw_dir):
    raw_dir = Path(raw_dir)
    train_root = raw_dir / "rumoureval-2019-training-data"
    test_root = raw_dir / "rumoureval-2019-test-data"

    train_veracity = load_training_veracity(raw_dir)
    test_veracity = load_test_veracity(raw_dir)

    records = {}
    for split_name, split_root, veracity in [
        ("train", train_root, train_veracity),
        ("test", test_root, test_veracity),
    ]:
        for thread_dir in collect_structure_dirs(split_root):
            eid = Path(thread_dir).name
            root_id = structure_root_id(thread_dir)
            label_name = veracity.get(root_id, veracity.get(eid))
            if label_name not in LABEL2ID:
                continue

            src_path = source_json_path(thread_dir)
            if src_path is None:
                continue

            try:
                source_obj = read_json(src_path)
            except Exception:
                continue

            source_text = normalize_text(get_text_from_json(source_obj))
            if not source_text:
                source_text = "[NO_TEXT]"

            records[eid] = {
                "eid": eid,
                "label": LABEL2ID[label_name],
                "label_name": label_name,
                "source_text": source_text,
                "split": split_name,
            }

    return records


def read_threads_csv(path):
    records = {}

    with open(path, "r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dataset = str(row.get("dataset", "")).strip().lower()
            if dataset and dataset != "rumoureval2019":
                continue

            eid = str(row.get("thread_id", "")).strip()
            label_name = str(row.get("label", "")).strip().lower()
            source_text = normalize_text(row.get("source_text", ""))

            if not eid or label_name not in LABEL2ID:
                continue
            if not source_text:
                source_text = "[NO_TEXT]"

            records[eid] = {
                "eid": eid,
                "label": LABEL2ID[label_name],
                "label_name": label_name,
                "source_text": source_text,
                "split": str(row.get("split", "")).strip().lower(),
            }

    return records


def load_official_split(meta_dir):
    folds_path = meta_dir / "folds.pkl"
    if not folds_path.exists():
        raise FileNotFoundError("missing folds.pkl: {}".format(folds_path))

    with open(folds_path, "rb") as f:
        folds = pickle.load(f)

    test_ids, train_ids = folds[0]
    return list(train_ids), list(test_ids)


class SourceOnlyDataset(Dataset):
    def __init__(self, ids, records, tokenizer, max_length):
        self.examples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        missing = []
        for eid in ids:
            eid = str(eid)
            item = records.get(eid)
            if item is None:
                missing.append(eid)
                continue
            self.examples.append(item)

        if missing:
            print("missing csv records:", len(missing))
            print("first missing ids:", missing[:10])

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        item = self.examples[idx]
        enc = self.tokenizer(
            item["source_text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "eid": item["eid"],
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
            "source_text": item["source_text"],
        }


def collate_fn(batch):
    return {
        "eid": [x["eid"] for x in batch],
        "input_ids": torch.stack([x["input_ids"] for x in batch], dim=0),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch], dim=0),
        "label": torch.stack([x["label"] for x in batch], dim=0),
        "source_text": [x["source_text"] for x in batch],
    }


def compute_class_weights(dataset, num_classes=3, max_weight=1.5):
    labels = [int(x["label"]) for x in dataset.examples]
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = np.minimum(weights, max_weight)
    weights = weights / weights.mean()

    print("train label counts:", counts)
    print("source-only class weights:", weights)

    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(model, loader, optimizer, device, class_weights=None):
    model.train()
    total_loss = 0.0
    all_pred = []
    all_true = []

    for batch in tqdm(loader, desc="train"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits

        loss = F.cross_entropy(
            logits,
            labels,
            weight=class_weights.to(device) if class_weights is not None else None,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        all_pred.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        all_true.extend(labels.detach().cpu().numpy().tolist())

    acc = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    return total_loss / max(len(loader), 1), acc, macro_f1


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_pred = []
    all_true = []
    total_loss = 0.0

    for batch in tqdm(loader, desc="eval"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        loss = F.cross_entropy(logits, labels)

        total_loss += float(loss.item())
        all_pred.extend(logits.argmax(dim=1).cpu().numpy().tolist())
        all_true.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_true, all_pred)
    macro_f1 = f1_score(all_true, all_pred, labels=[0, 1, 2], average="macro", zero_division=0)
    print(classification_report(
        all_true,
        all_pred,
        labels=[0, 1, 2],
        target_names=["FR", "TR", "UR"],
        zero_division=0,
    ))
    return total_loss / max(len(loader), 1), acc, macro_f1


@torch.no_grad()
def export_evidence(model, loader, device):
    model.eval()
    evidence = {}

    for batch in tqdm(loader, desc="export"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        logits = out.logits
        prob = F.softmax(logits, dim=1)
        hidden = out.hidden_states[-1][:, 0, :]
        pred = logits.argmax(dim=1)

        logits_np = logits.detach().cpu().numpy().astype(np.float32)
        prob_np = prob.detach().cpu().numpy().astype(np.float32)
        hidden_np = hidden.detach().cpu().numpy().astype(np.float32)
        pred_np = pred.detach().cpu().numpy().astype(np.int64)
        label_np = labels.detach().cpu().numpy().astype(np.int64)

        for i, eid in enumerate(batch["eid"]):
            evidence[str(eid)] = {
                "logits": logits_np[i],
                "prob": prob_np[i],
                "hidden": hidden_np[i],
                "pred": int(pred_np[i]),
                "label": int(label_np[i]),
                "source_text": batch["source_text"][i],
            }

    return evidence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="RumourEval2019")
    parser.add_argument("--csv-dir", default=str(CWD / "data" / "csv"))
    parser.add_argument("--raw-dir", default=str(CWD / "data" / "rumoureval2019"))
    parser.add_argument("--meta-dir", default=str(CWD / "data" / "RumourEval2019"))
    parser.add_argument("--model-name", default="microsoft/deberta-v3-base")
    parser.add_argument("--out-path", default=str(CWD / "data" / "RumourEval2019" / "source_root_evidence.pkl"))
    parser.add_argument("--save-model-dir", default=str(CWD / "checkpoint" / "source_root_plm"))
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--class-weight-max", type=float, default=1.5)
    parser.add_argument("--no-class-weight", action="store_true")
    args = parser.parse_args()

    if args.dataset != "RumourEval2019":
        raise ValueError("Only RumourEval2019 is supported by this script.")

    set_seed(args.seed)

    csv_path = Path(args.csv_dir) / "RumourEval2019_threads.csv"
    meta_dir = Path(args.meta_dir)
    out_path = Path(args.out_path)
    save_model_dir = Path(args.save_model_dir)
    save_model_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = {}
    if csv_path.exists():
        records = read_threads_csv(csv_path)
        print("records from csv:", len(records))
    else:
        print("csv not found:", csv_path)

    if len(records) == 0:
        print("CSV records are empty. Falling back to raw RumourEval2019 data.")
        records = read_records_from_raw(args.raw_dir)

    if len(records) == 0:
        raise RuntimeError(
            "No RumourEval2019 source records found. Check --csv-dir or --raw-dir."
        )

    train_ids, test_ids = load_official_split(meta_dir)

    print("records:", len(records))
    print("csv label count:", Counter(x["label_name"] for x in records.values()))
    print("train ids:", len(train_ids))
    print("test ids:", len(test_ids))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=3,
    )

    train_set = SourceOnlyDataset(train_ids, records, tokenizer, args.max_length)
    test_set = SourceOnlyDataset(test_ids, records, tokenizer, args.max_length)
    all_set = SourceOnlyDataset(train_ids + test_ids, records, tokenizer, args.max_length)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    all_loader = DataLoader(
        all_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    class_weights = None
    if not args.no_class_weight:
        class_weights = compute_class_weights(train_set, max_weight=args.class_weight_max)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_macro_f1 = -1.0
    best_state = None

    for epoch in range(args.epochs):
        print("Epoch {}/{}".format(epoch + 1, args.epochs))
        train_loss, train_acc, train_macro_f1 = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            class_weights=class_weights,
        )
        print("train loss {:.4f} acc {:.4f} macro_f1 {:.4f}".format(
            train_loss,
            train_acc,
            train_macro_f1,
        ))

        val_loss, val_acc, val_macro_f1 = evaluate(model, test_loader, device)
        print("eval loss {:.4f} acc {:.4f} macro_f1 {:.4f}".format(
            val_loss,
            val_acc,
            val_macro_f1,
        ))

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            model.save_pretrained(save_model_dir)
            tokenizer.save_pretrained(save_model_dir)
            print("saved best source-only model:", save_model_dir)

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    evidence = export_evidence(model, all_loader, device)
    output = {
        "model_name": args.model_name,
        "label_names": ["FR", "TR", "UR"],
        "label2id": LABEL2ID,
        "id2label": ID2LABEL,
        "train_ids": train_ids,
        "test_ids": test_ids,
        "best_macro_f1": float(best_macro_f1),
        "evidence": evidence,
    }

    with open(out_path, "wb") as f:
        pickle.dump(output, f)

    print("saved source root evidence:", out_path)
    print("evidence items:", len(evidence))


if __name__ == "__main__":
    main()
