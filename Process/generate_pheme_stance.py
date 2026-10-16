# -*- coding: utf-8 -*-
"""Generate calibrated 16-dimensional stance features for PHEME graphs.

The classifier is trained by ``train_rumoureval_stance_classifier.py``.  This
script only performs PHEME inference and writes a new graph directory; source
graphs are never modified.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABELS = ["support", "deny", "query", "comment"]
STANCE_DIM = 16


def read_json(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return json.load(handle)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def text_from_json(obj):
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


def find_one_json(folder):
    folder = Path(folder)
    if not folder.exists():
        return None
    files = sorted(
        path for path in folder.glob("*.json")
        if not path.name.startswith((".", "._"))
    )
    return files[0] if files else None


def find_source_json(thread_dir):
    for name in ("source-tweets", "source-tweet", "source-posts", "source-post", "source"):
        result = find_one_json(Path(thread_dir) / name)
        if result is not None:
            return result
    return None


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


def node_sort_key(node_id):
    """Tweet IDs are numeric; numeric ordering matches the usual graph builder."""
    value = str(node_id)
    try:
        return 0, int(value)
    except ValueError:
        return 1, value


def build_thread_index(raw_root):
    raw_root = Path(raw_root)
    if not raw_root.exists():
        raise FileNotFoundError("PHEME raw directory does not exist: {}".format(raw_root))
    index = {}
    collisions = set()
    for structure_path in raw_root.glob("**/structure.json"):
        thread_dir = structure_path.parent
        if thread_dir.name.startswith((".", "._")):
            continue
        candidate_ids = {thread_dir.name}
        try:
            structure = read_json(structure_path)
            if isinstance(structure, dict) and structure:
                candidate_ids.add(str(next(iter(structure))))
        except Exception:
            pass
        source_path = find_source_json(thread_dir)
        if source_path is not None:
            candidate_ids.add(source_path.stem)
            try:
                source = read_json(source_path)
                source_id = source.get("id_str", source.get("id"))
                if source_id is not None:
                    candidate_ids.add(str(source_id))
            except Exception:
                pass
        for candidate in candidate_ids:
            candidate = str(candidate).strip()
            if candidate in index and index[candidate] != thread_dir:
                collisions.add(candidate)
            elif candidate:
                index[candidate] = thread_dir
    for candidate in collisions:
        index.pop(candidate, None)
    return index, sorted(collisions)


def read_thread(thread_dir):
    thread_dir = Path(thread_dir)
    source_path = find_source_json(thread_dir)
    structure_path = thread_dir / "structure.json"
    if source_path is None or not structure_path.exists():
        return None
    source = read_json(source_path)
    root_text = text_from_json(source)
    structure = read_json(structure_path)
    if not root_text or not isinstance(structure, dict) or not structure:
        return None
    root_id = str(next(iter(structure)))
    source_id = str(source.get("id_str", source.get("id", root_id)))
    id_to_text = {root_id: root_text, source_id: root_text}
    id_to_order = {root_id: 0, source_id: 0}
    order = 1
    for folder_name in ("reactions", "replies", "comments"):
        folder = thread_dir / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("*.json"), key=lambda p: node_sort_key(p.stem)):
            if path.name.startswith((".", "._")):
                continue
            try:
                obj = read_json(path)
                node_id = str(obj.get("id_str", obj.get("id", obj.get("tweet_id", path.stem))))
                text = text_from_json(obj)
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
        node_ids.update((parent, child))
    node_order = [root_id] + sorted(node_ids - {root_id}, key=node_sort_key)
    return {
        "root_id": root_id,
        "root_text": root_text,
        "id_to_text": id_to_text,
        "id_to_order": id_to_order,
        "parent_map": parent_map,
        "node_order": node_order,
    }


def load_calibration(model_dir):
    path = Path(model_dir) / "calibration.json"
    if not path.exists():
        return 1.0, [1.0] * 4, None
    value = read_json(path)
    temperature = max(float(value.get("temperature", 1.0)), 1e-6)
    thresholds = value.get("thresholds", [1.0] * 4)
    if not isinstance(thresholds, list) or len(thresholds) != 4:
        raise ValueError("calibration thresholds must contain four values: {}".format(path))
    thresholds = [max(float(item), 1e-6) for item in thresholds]
    return temperature, thresholds, str(path)


def build_prediction_items(thread):
    items = []
    for node_id in thread["node_order"]:
        if node_id == thread["root_id"]:
            continue
        parent_id = thread["parent_map"].get(node_id, thread["root_id"])
        items.append({
            "node_id": node_id,
            "source": thread["root_text"],
            "parent": thread["id_to_text"].get(parent_id, thread["root_text"]),
            "reply": thread["id_to_text"].get(node_id, ""),
        })
    return items


@torch.inference_mode()
def predict_items(items, model, tokenizer, device, batch_size, max_length, temperature):
    results = {}
    empty = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    valid = [item for item in items if item["reply"]]
    for item in items:
        if not item["reply"]:
            results[item["node_id"]] = empty.copy()
    for start in range(0, len(valid), batch_size):
        batch = valid[start:start + batch_size]
        text_a = [
            "Source claim: {}\nParent post: {}".format(x["source"][:400], x["parent"][:300])
            for x in batch
        ]
        text_b = ["Reply: {}".format(x["reply"][:300]) for x in batch]
        encoded = tokenizer(
            text_a, text_b, truncation="longest_first", padding=True,
            max_length=max_length, return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits / temperature
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float32)
        for item, probability in zip(batch, probabilities):
            results[item["node_id"]] = probability / max(float(probability.sum()), 1e-8)
    return results


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator >= 1e-8 else 0.0


def build_stance_x(thread, probabilities, node_features, root_index, phase_count=5):
    node_order = thread["node_order"]
    num_nodes = len(node_order)
    if node_features.shape[0] != num_nodes:
        raise ValueError("raw nodes {} != graph nodes {}".format(num_nodes, node_features.shape[0]))
    if root_index < 0 or root_index >= num_nodes:
        raise ValueError("invalid rootindex {} for {} nodes".format(root_index, num_nodes))
    if root_index != 0:
        raise ValueError(
            "rootindex is {}; raw node-order reconstruction assumes rootindex 0".format(root_index)
        )
    stance_x = np.zeros((num_nodes, STANCE_DIM), dtype=np.float32)
    root_probability = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    all_probabilities = {thread["root_id"]: root_probability}
    all_probabilities.update(probabilities)
    reaction_orders = [thread["id_to_order"].get(node_id, i) for i, node_id in enumerate(node_order[1:], 1)]
    max_order = max(reaction_orders, default=1)
    root_vector = node_features[root_index]
    for index, node_id in enumerate(node_order):
        probability = all_probabilities.get(node_id, root_probability)
        if index == root_index:
            delta = np.zeros(4, dtype=np.float32)
            similarity = 1.0
            phase_index = 0
            root_flag = 1.0
            confidence = 1.0
        else:
            parent_id = thread["parent_map"].get(node_id, thread["root_id"])
            parent_probability = all_probabilities.get(parent_id, root_probability)
            delta = probability - parent_probability
            similarity = cosine_similarity(root_vector, node_features[index])
            order = thread["id_to_order"].get(node_id, max_order)
            phase_index = min(int((max(order, 1) - 1) * phase_count / max(max_order, 1)), phase_count - 1)
            root_flag = 0.0
            confidence = float(probability.max())
        phase = np.zeros(phase_count, dtype=np.float32)
        phase[phase_index] = 1.0
        stance_x[index] = np.concatenate((
            probability,
            np.array([confidence], dtype=np.float32),
            delta,
            np.array([similarity], dtype=np.float32),
            phase,
            np.array([root_flag], dtype=np.float32),
        ))
    if stance_x.shape != (num_nodes, STANCE_DIM) or not np.isfinite(stance_x).all():
        raise ValueError("invalid stance_x generated: {}".format(stance_x.shape))
    return stance_x


def copy_metadata(source_dir, output_dir):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    if not source_dir.exists():
        return copied
    for path in source_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)
            copied.append(path.name)
    return copied


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default=str(PROJECT_ROOT / "data" / "PHEME_raw" / "all-rnr-annotated-threads"))
    parser.add_argument("--source-graph-dir", default=str(PROJECT_ROOT / "data" / "PHEME4PLMgraph"))
    parser.add_argument("--output-graph-dir", default=str(PROJECT_ROOT / "data" / "PHEME4Stancegraph"))
    parser.add_argument("--source-meta-dir", default=str(PROJECT_ROOT / "data" / "PHEME4"))
    parser.add_argument("--output-meta-dir", default=str(PROJECT_ROOT / "data" / "PHEME4Stance"))
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "checkpoint" / "rumoureval_stance" / "best"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--max-threads", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--allow-node-mismatch", action="store_true", help="Pad/truncate reconstructed raw nodes; unsafe and disabled by default.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    source_graph_dir = Path(args.source_graph_dir)
    output_graph_dir = Path(args.output_graph_dir)
    model_dir = Path(args.model_dir)
    if not source_graph_dir.exists():
        raise FileNotFoundError("source graph directory does not exist: {}".format(source_graph_dir))
    if not model_dir.exists():
        raise FileNotFoundError("trained stance model does not exist: {}".format(model_dir))
    output_graph_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    temperature, thresholds, calibration_path = load_calibration(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
    thread_index, collisions = build_thread_index(raw_root)
    graph_files = sorted(source_graph_dir.glob("*.npz"))
    if args.max_threads is not None:
        graph_files = graph_files[:max(args.max_threads, 0)]
    copied_meta = copy_metadata(args.source_meta_dir, args.output_meta_dir)
    counters = Counter()
    probability_sum = np.zeros(4, dtype=np.float64)
    hard_counts = np.zeros(4, dtype=np.int64)
    confidence_sum = 0.0
    non_root_count = 0
    errors = []
    print("device:", device)
    print("model:", model_dir)
    print("temperature:", temperature, "thresholds:", thresholds)
    print("source graphs:", len(graph_files), "thread index:", len(thread_index))
    for graph_path in tqdm(graph_files, desc="Generating calibrated PHEME stance_x"):
        output_path = output_graph_dir / graph_path.name
        if output_path.exists() and not args.overwrite:
            counters["skipped_existing"] += 1
            continue
        thread_dir = thread_index.get(graph_path.stem)
        if thread_dir is None:
            counters["missing_thread"] += 1
            errors.append({"event_id": graph_path.stem, "error": "thread not found"})
            continue
        try:
            thread = read_thread(thread_dir)
            if thread is None:
                raise ValueError("could not read raw thread")
            with np.load(graph_path, allow_pickle=True) as source:
                arrays = {key: source[key] for key in source.files}
            node_features = arrays["x"]
            graph_nodes = int(node_features.shape[0])
            raw_nodes = len(thread["node_order"])
            if raw_nodes != graph_nodes:
                counters["node_mismatch"] += 1
                if not args.allow_node_mismatch:
                    raise ValueError("raw nodes {} != graph nodes {}".format(raw_nodes, graph_nodes))
                # Compatibility mode mirrors the legacy behavior, but reports it.
                if raw_nodes < graph_nodes:
                    missing = ["__missing_{}".format(i) for i in range(graph_nodes - raw_nodes)]
                    thread["node_order"].extend(missing)
                else:
                    thread["node_order"] = thread["node_order"][:graph_nodes]
            probabilities = predict_items(
                build_prediction_items(thread), model, tokenizer, device,
                args.batch_size, args.max_length, temperature,
            )
            root_index = int(np.asarray(arrays["rootindex"]).reshape(-1)[0])
            stance_x = build_stance_x(thread, probabilities, node_features, root_index)
            arrays["stance_x"] = stance_x
            np.savez(output_path, **arrays)
            comment_probabilities = stance_x[:, :4][stance_x[:, 15] < 0.5]
            probability_sum += comment_probabilities.sum(axis=0)
            confidence_sum += float(comment_probabilities.max(axis=1).sum())
            hard = np.argmax(comment_probabilities / np.asarray(thresholds).reshape(1, 4), axis=1)
            hard_counts += np.bincount(hard, minlength=4)
            non_root_count += len(comment_probabilities)
            counters["saved"] += 1
        except Exception as exc:
            counters["failed"] += 1
            errors.append({"event_id": graph_path.stem, "error": str(exc)})
    report = {
        "config": vars(args),
        "model_dir": str(model_dir),
        "calibration_path": calibration_path,
        "temperature": temperature,
        "thresholds": thresholds,
        "counts": dict(counters),
        "index_collisions": collisions,
        "copied_metadata": copied_meta,
        "non_root_nodes": non_root_count,
        "mean_probabilities": {
            LABELS[i]: float(probability_sum[i] / max(non_root_count, 1)) for i in range(4)
        },
        "mean_confidence": float(confidence_sum / max(non_root_count, 1)),
        "thresholded_prediction_counts": {LABELS[i]: int(hard_counts[i]) for i in range(4)},
        "errors": errors,
    }
    report_path = output_graph_dir / "generation_report.json"
    save_json(report_path, report)
    print("generation counts:", dict(counters))
    print("mean stance probabilities:", report["mean_probabilities"])
    print("report:", report_path)
    if counters["failed"] or counters["missing_thread"]:
        print("warning: some graphs were not generated; inspect the report before training")


if __name__ == "__main__":
    main()
