# -*- coding: utf-8 -*-
import argparse
import os
import csv
import json
import pickle
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

try:
    from Process.graph_connectivity import (
        connect_orphan_components,
        reachable_nodes,
    )
except ModuleNotFoundError:
    from graph_connectivity import connect_orphan_components, reachable_nodes


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
CWD = str(PROJECT_ROOT)


# =========================
# Basic utils
# =========================

def read_json_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def get_text_from_json(obj):
    """
    兼容 Twitter / Reddit 的 json 字段。
    """
    if not isinstance(obj, dict):
        return ""

    # Twitter common fields
    for key in ["text", "full_text"]:
        if key in obj and isinstance(obj[key], str):
            return obj[key]

    # Reddit common fields
    for key in ["body", "title", "selftext"]:
        if key in obj and isinstance(obj[key], str):
            return obj[key]

    # nested data field
    if "data" in obj and isinstance(obj["data"], dict):
        data = obj["data"]
        for key in ["body", "title", "selftext", "text", "full_text"]:
            if key in data and isinstance(data[key], str):
                return data[key]

    return ""


def find_one_json(folder):
    folder = Path(folder)
    if not folder.exists():
        return None

    files = [
        p for p in folder.glob("*.json")
        if not p.name.startswith("._") and not p.name.startswith(".")
    ]

    files = sorted(files)

    if len(files) == 0:
        return None

    return files[0]


def flatten_structure(structure):
    """
    RumourEval2019 / PHEME 的 structure.json 通常都是嵌套 dict:
    {
      "source_id": {
         "reply_id_1": {...},
         "reply_id_2": {...}
      }
    }

    返回边列表: [(parent_id, child_id), ...]
    """
    edges = []

    def visit(parent, children):
        if not isinstance(children, dict):
            return

        for child, sub in children.items():
            edges.append((str(parent), str(child)))
            visit(str(child), sub)

    if not isinstance(structure, dict):
        return edges

    for root, children in structure.items():
        visit(str(root), children)

    return edges


def batch_encode(texts, tokenizer, model, device, batch_size=16, max_length=128):
    """
    与 PHEME 转换代码保持一致：使用本地 PLM 对节点文本进行 mean pooling。
    """
    all_vecs = []
    model.eval()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_texts = [
                t if isinstance(t, str) and len(t.strip()) > 0 else "[NO_TEXT]"
                for t in batch_texts
            ]

            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )

            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            hidden = outputs.last_hidden_state

            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            all_vecs.append(pooled.cpu().numpy())

    return np.vstack(all_vecs).astype(np.float32)


# =========================
# RumourEval labels
# =========================

VERACITY_MAP = {
    "false": 0,        # FR
    "true": 1,         # TR
    "unverified": 2    # UR
}

LABEL_NAME_MAP = {
    0: "false",
    1: "true",
    2: "unverified"
}

STANCE_MAP = {
    "support": 0,
    "supporting": 0,
    "deny": 1,
    "denying": 1,
    "query": 2,
    "querying": 2,
    "comment": 3,
    "commenting": 3
}


def load_rumoureval_key(path):
    """
    读取 RumourEval2019 key 文件。

    subtaskaenglish: stance labels
    subtaskbenglish: veracity labels
    """
    key = read_json_file(path)

    stance_labels = key.get("subtaskaenglish", {})
    veracity_labels = key.get("subtaskbenglish", {})

    stance_labels = {str(k): str(v).lower() for k, v in stance_labels.items()}
    veracity_labels = {str(k): str(v).lower() for k, v in veracity_labels.items()}

    return stance_labels, veracity_labels


def load_all_train_keys(train_root):
    """
    training data 中可能有 train-key.json / dev-key.json。
    这里递归读取所有 *key*.json，并合并 subtaskaenglish 与 subtaskbenglish。
    """
    train_root = Path(train_root)

    stance_all = {}
    veracity_all = {}

    key_files = sorted(train_root.glob("**/*key*.json"))

    print("training key files:")
    for kp in key_files:
        print(" ", kp)

        stance, veracity = load_rumoureval_key(kp)
        stance_all.update(stance)
        veracity_all.update(veracity)

    print("train stance labels:", len(stance_all))
    print("train veracity labels:", len(veracity_all))
    print("train veracity count:", Counter(veracity_all.values()))

    return stance_all, veracity_all


# =========================
# Stance features
# =========================

def stance_onehot(label):
    p = np.zeros(4, dtype=np.float32)

    if label is None:
        p[3] = 1.0
        return p

    label = str(label).lower()

    if label in STANCE_MAP:
        p[STANCE_MAP[label]] = 1.0
    else:
        p[3] = 1.0

    return p


def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)


def compute_depths(root_id, edges):
    children = {}

    for p, c in edges:
        p, c = str(p), str(c)
        children.setdefault(p, []).append(c)

    depth = {str(root_id): 0}
    queue = [str(root_id)]

    while len(queue) > 0:
        cur = queue.pop(0)

        for child in children.get(cur, []):
            depth[child] = depth[cur] + 1
            queue.append(child)

    return depth


def build_stance_feature(
    node_id,
    root_id,
    stance_labels,
    parent_map,
    root_emb,
    node_emb,
    depth,
    max_depth
):
    """
    stance_x 维度保持 16：
    0-3: stance probability [S, D, Q, C]
    4: stance confidence
    5-8: parent-child stance delta
    9: root-comment semantic similarity
    10-14: propagation phase one-hot
    15: root flag
    """
    node_id = str(node_id)
    root_id = str(root_id)
    is_root = (node_id == root_id)

    if is_root:
        p = np.zeros(4, dtype=np.float32)
        conf = 0.0
    else:
        label = stance_labels.get(node_id, "comment")
        p = stance_onehot(label)
        # RumourEval 官方 stance 标签，置信度设为 1
        conf = 1.0

    parent_id = parent_map.get(node_id, None)

    if parent_id is None or is_root:
        parent_p = np.zeros(4, dtype=np.float32)
    else:
        parent_label = stance_labels.get(str(parent_id), "comment")
        parent_p = stance_onehot(parent_label)

    delta_p = p - parent_p

    sim = 1.0 if is_root else cosine_similarity(root_emb, node_emb)

    phase = np.zeros(5, dtype=np.float32)

    if max_depth <= 0:
        phase_idx = 0
    else:
        phase_idx = int(min(4, np.floor(5 * depth / (max_depth + 1e-6))))

    phase[phase_idx] = 1.0

    root_flag = 1.0 if is_root else 0.0

    feat = np.concatenate(
        [
            p,
            np.array([conf], dtype=np.float32),
            delta_p,
            np.array([sim], dtype=np.float32),
            phase,
            np.array([root_flag], dtype=np.float32)
        ],
        axis=0
    )

    return feat.astype(np.float32)


# =========================
# Event reading
# =========================

def collect_event_dirs(root_dir):
    """
    递归寻找所有包含 structure.json 的事件目录。
    """
    root_dir = Path(root_dir)

    event_dirs = []
    for sp in root_dir.glob("**/structure.json"):
        if sp.name.startswith(".") or sp.name.startswith("._"):
            continue
        event_dirs.append(sp.parent)

    event_dirs = sorted(event_dirs)

    return event_dirs


def build_one_rumoureval_thread(thread_dir, label, stance_labels, tokenizer, model, device):
    """
    按 PHEME build_one_thread 的输出格式构建 RumourEval2019 单个事件。
    额外保存 stance_x。
    """
    thread_dir = Path(thread_dir)
    eid = thread_dir.name

    source_json_path = find_one_json(thread_dir / "source-tweet")
    if source_json_path is None:
        source_json_path = find_one_json(thread_dir / "source-tweets")

    structure_path = thread_dir / "structure.json"

    if source_json_path is None:
        return None

    if not structure_path.exists():
        return None

    source_obj = read_json_file(source_json_path)
    source_text = get_text_from_json(source_obj)

    structure = read_json_file(structure_path)

    if isinstance(structure, dict) and len(structure) > 0:
        root_id = str(list(structure.keys())[0])
    else:
        root_id = str(source_obj.get("id_str", source_obj.get("id", eid)))

    source_id = str(source_obj.get("id_str", source_obj.get("id", root_id)))

    # id -> text
    id_to_text = {}
    id_to_text[source_id] = source_text
    id_to_text[root_id] = source_text

    # RumourEval2019 一般是 replies
    replies_dir = thread_dir / "replies"
    reactions_dir = thread_dir / "reactions"

    for folder in [replies_dir, reactions_dir]:
        if folder.exists():
            for rp in folder.glob("*.json"):
                if rp.name.startswith("._") or rp.name.startswith("."):
                    continue

                try:
                    r_obj = read_json_file(rp)
                    rid = str(r_obj.get("id_str", r_obj.get("id", rp.stem)))
                    text = get_text_from_json(r_obj)
                    id_to_text[rid] = text
                except Exception:
                    continue

    edges_raw = flatten_structure(structure)

    # 保留 structure 中所有节点，不因缺文本而删除
    node_ids = set([root_id])

    for p, c in edges_raw:
        node_ids.add(str(p))
        node_ids.add(str(c))

    if root_id not in node_ids:
        node_ids.add(root_id)

    # root 固定放第 0 个
    node_ids = [root_id] + sorted([nid for nid in node_ids if nid != root_id])
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    row, col = [], []
    parent_map = {}

    for p, c in edges_raw:
        p, c = str(p), str(c)

        if p in node_to_idx and c in node_to_idx:
            row.append(node_to_idx[p])
            col.append(node_to_idx[c])
            parent_map[c] = p

    # 没有边的事件跳过，保持和 PHEME 一致
    if len(row) == 0:
        return None

    texts = []
    for nid in node_ids:
        text = id_to_text.get(nid, "")
        if not isinstance(text, str) or len(text.strip()) == 0:
            text = "[NO_TEXT]"
        texts.append(text)

    x = batch_encode(texts, tokenizer, model, device)

    edgeindex = np.array([row, col], dtype=np.int64)
    edgeindex, added_edges = connect_orphan_components(
        edgeindex, root_index=0, node_count=len(node_ids)
    )
    effective_edges = list(edges_raw)
    for parent_idx, child_idx in added_edges:
        parent_id = node_ids[parent_idx]
        child_id = node_ids[child_idx]
        effective_edges.append((parent_id, child_id))
        parent_map[child_id] = parent_id
    rootindex = np.array(0, dtype=np.int64)
    root = x[0:1]
    y = np.array(label, dtype=np.int64)

    # stance_x
    depth_dict = compute_depths(root_id, effective_edges)
    max_depth = max(depth_dict.values()) if len(depth_dict) > 0 else 0

    root_emb = x[0]

    stance_feats = []
    for nid in node_ids:
        idx = node_to_idx[nid]
        depth = depth_dict.get(nid, 0)

        feat = build_stance_feature(
            node_id=nid,
            root_id=root_id,
            stance_labels=stance_labels,
            parent_map=parent_map,
            root_emb=root_emb,
            node_emb=x[idx],
            depth=depth,
            max_depth=max_depth
        )

        stance_feats.append(feat)

    stance_x = np.stack(stance_feats, axis=0).astype(np.float32)

    return {
        "eid": eid,
        "root_id": root_id,
        "x": x,
        "edgeindex": edgeindex,
        "root": root,
        "rootindex": rootindex,
        "y": y,
        "node_num": len(node_ids),
        "source_bert": root.reshape(-1),
        "stance_x": stance_x
    }


# =========================
# CSV reading
# =========================

def read_csv_rows(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize_text(text):
    if not isinstance(text, str):
        return ""
    return text.strip()


def load_rumoureval_csv(csv_dir):
    """
    Read aligned CSV files generated by export_raw_datasets_to_csv.py.

    Expected files:
      RumourEval2019_threads.csv
      RumourEval2019_nodes.csv
    """
    csv_dir = Path(csv_dir)
    threads_path = csv_dir / "RumourEval2019_threads.csv"
    nodes_path = csv_dir / "RumourEval2019_nodes.csv"

    if not threads_path.exists():
        raise FileNotFoundError(f"missing csv file: {threads_path}")
    if not nodes_path.exists():
        raise FileNotFoundError(f"missing csv file: {nodes_path}")

    thread_rows = read_csv_rows(threads_path)
    node_rows = read_csv_rows(nodes_path)

    nodes_by_thread = {}
    for row in node_rows:
        if row.get("dataset") != "RumourEval2019":
            continue
        tid = str(row.get("thread_id", "")).strip()
        if tid == "":
            continue
        nodes_by_thread.setdefault(tid, []).append(row)

    records = []
    for row in thread_rows:
        if row.get("dataset") != "RumourEval2019":
            continue

        tid = str(row.get("thread_id", "")).strip()
        if tid == "":
            continue

        label_name = str(row.get("label", "")).strip().lower()
        if label_name not in VERACITY_MAP:
            continue

        records.append({
            "split": str(row.get("split", "")).strip().lower(),
            "eid": tid,
            "root_id": str(row.get("root_id", "") or tid).strip(),
            "source_id": str(row.get("source_id", "") or row.get("root_id", "") or tid).strip(),
            "label": VERACITY_MAP[label_name],
            "label_name": label_name,
            "source_text": normalize_text(row.get("source_text", "")),
            "nodes": nodes_by_thread.get(tid, []),
        })

    records_by_split = {
        "train": [r for r in records if r["split"] == "train"],
        "test": [r for r in records if r["split"] == "test"],
    }

    print("csv threads:", len(records))
    print("csv train threads:", len(records_by_split["train"]))
    print("csv test threads:", len(records_by_split["test"]))
    print("csv label count:", Counter(r["label_name"] for r in records))

    return records_by_split


def build_one_rumoureval_thread_from_csv(
    record,
    tokenizer,
    model,
    device,
    batch_size=16,
    max_length=128
):
    """
    Build one RumourEval2019 graph from CSV rows instead of raw JSON files.
    """
    eid = str(record["eid"])
    root_id = str(record["root_id"] or eid)
    source_id = str(record["source_id"] or root_id)
    label = int(record["label"])

    id_to_text = {}
    stance_labels = {}
    parent_map = {}
    edges_raw = []

    source_text = normalize_text(record.get("source_text", ""))
    if source_text:
        id_to_text[source_id] = source_text
        id_to_text[root_id] = source_text

    for row in record["nodes"]:
        node_id = str(row.get("node_id", "")).strip()
        if node_id == "":
            continue

        text = normalize_text(row.get("text", ""))
        if text:
            id_to_text[node_id] = text

        stance = str(row.get("stance", "")).strip().lower()
        if stance:
            stance_labels[node_id] = stance

        parent_id = str(row.get("parent_id", "")).strip()
        if parent_id:
            edges_raw.append((parent_id, node_id))
            parent_map[node_id] = parent_id

    node_ids = set([root_id, source_id])
    for node_id in id_to_text.keys():
        node_ids.add(str(node_id))
    for p, c in edges_raw:
        node_ids.add(str(p))
        node_ids.add(str(c))

    node_ids = [root_id] + sorted([nid for nid in node_ids if nid != root_id])
    node_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    row_idx, col_idx = [], []
    clean_edges = []
    clean_parent_map = {}
    for p, c in edges_raw:
        p, c = str(p), str(c)
        if p in node_to_idx and c in node_to_idx:
            row_idx.append(node_to_idx[p])
            col_idx.append(node_to_idx[c])
            clean_edges.append((p, c))
            clean_parent_map[c] = p

    if len(row_idx) == 0:
        return None

    texts = []
    for nid in node_ids:
        text = id_to_text.get(nid, "")
        if not isinstance(text, str) or len(text.strip()) == 0:
            text = "[NO_TEXT]"
        texts.append(text)

    x = batch_encode(
        texts,
        tokenizer,
        model,
        device,
        batch_size=batch_size,
        max_length=max_length
    )

    edgeindex = np.array([row_idx, col_idx], dtype=np.int64)
    edgeindex, added_edges = connect_orphan_components(
        edgeindex, root_index=0, node_count=len(node_ids)
    )
    for parent_idx, child_idx in added_edges:
        parent_id = node_ids[parent_idx]
        child_id = node_ids[child_idx]
        clean_edges.append((parent_id, child_id))
        clean_parent_map[child_id] = parent_id
    rootindex = np.array(0, dtype=np.int64)
    root = x[0:1]
    y = np.array(label, dtype=np.int64)

    depth_dict = compute_depths(root_id, clean_edges)
    max_depth = max(depth_dict.values()) if len(depth_dict) > 0 else 0
    root_emb = x[0]

    stance_feats = []
    for nid in node_ids:
        idx = node_to_idx[nid]
        depth = depth_dict.get(nid, 0)

        feat = build_stance_feature(
            node_id=nid,
            root_id=root_id,
            stance_labels=stance_labels,
            parent_map=clean_parent_map,
            root_emb=root_emb,
            node_emb=x[idx],
            depth=depth,
            max_depth=max_depth
        )
        stance_feats.append(feat)

    stance_x = np.stack(stance_feats, axis=0).astype(np.float32)

    return {
        "eid": eid,
        "root_id": root_id,
        "x": x,
        "edgeindex": edgeindex,
        "root": root,
        "rootindex": rootindex,
        "y": y,
        "node_num": len(node_ids),
        "node_ids": np.array(node_ids, dtype=str),
        "source_bert": root.reshape(-1),
        "stance_x": stance_x
    }


# =========================
# Split processing
# =========================

def process_split(split_name, event_dirs, stance_labels, veracity_labels,
                  tokenizer, model, device,
                  out_graph,
                  label_lines,
                  tree_lines,
                  source_bert,
                  ids_by_label):
    saved_ids = []
    saved = 0
    skipped = 0

    for thread_dir in tqdm(event_dirs, desc=f"Converting {split_name}"):
        try:
            thread_dir = Path(thread_dir)
            eid = thread_dir.name

            # label 优先用 structure root id
            structure_path = thread_dir / "structure.json"
            if not structure_path.exists():
                skipped += 1
                continue

            structure = read_json_file(structure_path)

            if isinstance(structure, dict) and len(structure) > 0:
                root_id = str(list(structure.keys())[0])
            else:
                root_id = eid

            label_name = veracity_labels.get(root_id, None)

            # 有些情况下 key 用文件夹名作为 id
            if label_name is None:
                label_name = veracity_labels.get(eid, None)

            if label_name is None:
                skipped += 1
                continue

            label_name = str(label_name).lower()

            if label_name not in VERACITY_MAP:
                skipped += 1
                continue

            label = VERACITY_MAP[label_name]

            item = build_one_rumoureval_thread(
                thread_dir=thread_dir,
                label=label,
                stance_labels=stance_labels,
                tokenizer=tokenizer,
                model=model,
                device=device
            )

            if item is None:
                skipped += 1
                continue

            eid = item["eid"]

            np.savez(
                out_graph / f"{eid}.npz",
                x=item["x"],
                edgeindex=item["edgeindex"],
                root=item["root"],
                rootindex=item["rootindex"],
                y=item["y"],
                stance_x=item["stance_x"]
            )

            source_bert[eid] = item["source_bert"]
            ids_by_label[label].append(eid)

            label_lines.append(f"{LABEL_NAME_MAP[label]}\t0\t{eid}\n")

            # 写 tree 文件，indexC 使用 1-based
            tree_lines.append(f"{eid}\tNone\t1\t0\t0\t0:0\n")

            for p, c in item["edgeindex"].T:
                tree_lines.append(f"{eid}\t{int(p) + 1}\t{int(c) + 1}\t0\t0\t0:0\n")

            saved_ids.append(eid)
            saved += 1

        except Exception as e:
            skipped += 1
            print("skip:", thread_dir, e)

    print(f"\n========== {split_name} ==========")
    print("saved:", saved)
    print("skipped:", skipped)

    split_labels = []
    for eid in saved_ids:
        for label, ids in ids_by_label.items():
            if eid in ids:
                split_labels.append(label)
                break

    print("label count:", Counter(split_labels))

    return saved_ids


def process_split_from_csv(split_name, records,
                           tokenizer, model, device,
                           out_graph,
                           label_lines,
                           tree_lines,
                           source_bert,
                           ids_by_label,
                           batch_size=16,
                           max_length=128,
                           resume=False):
    saved_ids = []
    saved = 0
    skipped = 0
    split_labels = []

    for record in tqdm(records, desc=f"Converting {split_name}"):
        try:
            eid = str(record["eid"])
            graph_path = out_graph / f"{eid}.npz"

            if resume and graph_path.exists():
                with np.load(graph_path, allow_pickle=True) as graph_data:
                    item = {
                        "eid": eid,
                        "edgeindex": graph_data["edgeindex"],
                        "root": graph_data["root"],
                        "y": graph_data["y"],
                        "source_bert": graph_data["root"].reshape(-1),
                    }
            else:
                item = build_one_rumoureval_thread_from_csv(
                    record=record,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    batch_size=batch_size,
                    max_length=max_length
                )

            if item is None:
                skipped += 1
                continue

            eid = item["eid"]
            label = int(item["y"].item())

            if not (resume and graph_path.exists()):
                np.savez(
                    graph_path,
                    x=item["x"],
                    edgeindex=item["edgeindex"],
                    root=item["root"],
                    rootindex=item["rootindex"],
                    y=item["y"],
                    stance_x=item["stance_x"],
                    node_ids=item["node_ids"]
                )

            source_bert[eid] = item["source_bert"]
            ids_by_label[label].append(eid)

            label_lines.append(f"{LABEL_NAME_MAP[label]}\t0\t{eid}\n")
            tree_lines.append(f"{eid}\tNone\t1\t0\t0\t0:0\n")

            for p, c in item["edgeindex"].T:
                tree_lines.append(f"{eid}\t{int(p) + 1}\t{int(c) + 1}\t0\t0\t0:0\n")

            saved_ids.append(eid)
            split_labels.append(label)
            saved += 1

        except Exception as e:
            skipped += 1
            print("skip:", record.get("eid", ""), e)

    print(f"\n========== {split_name} ==========")
    print("saved:", saved)
    print("skipped:", skipped)
    print("label count:", Counter(split_labels))

    return saved_ids


def save_id_file(path, ids):
    with open(path, "w", encoding="utf-8") as f:
        for eid in ids:
            f.write(str(eid) + "\n")


def validate_generated_outputs(out_graph, train_records, test_records,
                               train_ids, test_ids, tree_lines):
    """Require CSV, NPZ, tree and folds source IDs to agree exactly."""
    expected_train = {str(record["eid"]) for record in train_records}
    expected_test = {str(record["eid"]) for record in test_records}
    saved_train = set(map(str, train_ids))
    saved_test = set(map(str, test_ids))
    graph_ids = {path.stem for path in Path(out_graph).glob("*.npz")}
    tree_ids = {
        line.split("\t", 1)[0]
        for line in tree_lines
        if isinstance(line, str) and line.strip()
    }
    expected_all = expected_train | expected_test
    saved_all = saved_train | saved_test
    errors = []

    expected_node_counts = {}
    for record in list(train_records) + list(test_records):
        node_ids = {
            str(record.get("root_id", "")),
            str(record.get("source_id", "")),
        }
        for row in record.get("nodes", []):
            node_id = str(row.get("node_id", "")).strip()
            parent_id = str(row.get("parent_id", "")).strip()
            if node_id:
                node_ids.add(node_id)
            if parent_id:
                node_ids.add(parent_id)
        node_ids.discard("")
        expected_node_counts[str(record["eid"])] = len(node_ids)

    node_count_mismatches = []
    disconnected_graphs = []
    for event_id in sorted(expected_all & graph_ids):
        with np.load(Path(out_graph) / (event_id + ".npz"), allow_pickle=True) as graph:
            graph_node_count = int(graph["x"].shape[0])
            root_index = int(np.asarray(graph["rootindex"]).reshape(-1)[0])
            reached = reachable_nodes(
                graph["edgeindex"], root_index, graph_node_count
            )
        expected_node_count = expected_node_counts[event_id]
        if graph_node_count != expected_node_count:
            node_count_mismatches.append(
                (event_id, expected_node_count, graph_node_count)
            )
        if len(reached) != graph_node_count:
            disconnected_graphs.append(
                (event_id, graph_node_count - len(reached))
            )

    if expected_train & expected_test:
        errors.append("CSV train/test overlap: {}".format(
            sorted(expected_train & expected_test)[:10]
        ))
    if saved_train != expected_train:
        errors.append("train missing: {}; train extra: {}".format(
            sorted(expected_train - saved_train)[:10],
            sorted(saved_train - expected_train)[:10]
        ))
    if saved_test != expected_test:
        errors.append("test missing: {}; test extra: {}".format(
            sorted(expected_test - saved_test)[:10],
            sorted(saved_test - expected_test)[:10]
        ))
    if tree_ids != saved_all:
        errors.append("tree missing: {}; tree extra: {}".format(
            sorted(saved_all - tree_ids)[:10], sorted(tree_ids - saved_all)[:10]
        ))
    if graph_ids != expected_all:
        errors.append("graph missing: {}; stale graph files: {}".format(
            sorted(expected_all - graph_ids)[:10],
            sorted(graph_ids - expected_all)[:10]
        ))
    if node_count_mismatches:
        errors.append(
            "graph node-count mismatches (event, CSV, NPZ): {}. "
            "Regenerate without --resume.".format(node_count_mismatches[:10])
        )
    if disconnected_graphs:
        errors.append(
            "disconnected graphs (event, unreachable nodes): {}. "
            "Regenerate without --resume.".format(disconnected_graphs[:10])
        )
    if errors:
        raise RuntimeError(
            "RumourEval2019 output consistency validation failed:\n- "
            + "\n- ".join(errors)
        )
    print(
        "output consistency: OK | train:{} | test:{} | total:{}".format(
            len(saved_train), len(saved_test), len(saved_all)
        )
    )


# =========================
# Main
# =========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default=Path(CWD) / "data" / "csv")
    parser.add_argument("--out-graph", default=Path(CWD) / "data" / "RumourEval2019graph")
    parser.add_argument("--out-meta", default=Path(CWD) / "data" / "RumourEval2019")
    parser.add_argument("--model-path", default=Path(CWD) / "pretrained" / "bertweet-base")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    args = parser.parse_args()

    csv_dir = Path(args.csv_dir)
    print("csv dir:", csv_dir)

    out_graph = Path(args.out_graph)
    out_meta = Path(args.out_meta)

    out_graph.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    local_deberta_path = Path(args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    print("model path:", local_deberta_path)
    print("batch size:", args.batch_size)
    print("max length:", args.max_length)
    print("resume:", args.resume)

    tokenizer = AutoTokenizer.from_pretrained(
        str(local_deberta_path),
        local_files_only=True,
        use_fast=args.use_fast_tokenizer
    )

    model = AutoModel.from_pretrained(
        str(local_deberta_path),
        local_files_only=True
    ).to(device)

    # 如果 tokenizer 新增特殊符号，保证 embedding 尺寸一致
    model.resize_token_embeddings(len(tokenizer))

    records_by_split = load_rumoureval_csv(csv_dir)
    train_records = records_by_split["train"]
    test_records = records_by_split["test"]

    label_lines = []
    tree_lines = []
    source_bert = {}

    # RumourEval2019 三分类：
    # 0: false, 1: true, 2: unverified
    ids_by_label = {0: [], 1: [], 2: []}

    train_ids = process_split_from_csv(
        split_name="train",
        records=train_records,
        tokenizer=tokenizer,
        model=model,
        device=device,
        out_graph=out_graph,
        label_lines=label_lines,
        tree_lines=tree_lines,
        source_bert=source_bert,
        ids_by_label=ids_by_label,
        batch_size=args.batch_size,
        max_length=args.max_length,
        resume=args.resume
    )

    test_ids = process_split_from_csv(
        split_name="test",
        records=test_records,
        tokenizer=tokenizer,
        model=model,
        device=device,
        out_graph=out_graph,
        label_lines=label_lines,
        tree_lines=tree_lines,
        source_bert=source_bert,
        ids_by_label=ids_by_label,
        batch_size=args.batch_size,
        max_length=args.max_length,
        resume=args.resume
    )

    validate_generated_outputs(
        out_graph=out_graph,
        train_records=train_records,
        test_records=test_records,
        train_ids=train_ids,
        test_ids=test_ids,
        tree_lines=tree_lines
    )

    # 保存 meta，文件命名与 PHEME 风格一致
    with open(out_meta / "RumourEval2019_label_All.txt", "w", encoding="utf-8") as f:
        f.writelines(label_lines)

    with open(out_meta / "data.TD_RvNN.vol_5000.txt", "w", encoding="utf-8") as f:
        f.writelines(tree_lines)

    with open(out_meta / "source_bertweet.pkl", "wb") as f:
        pickle.dump(source_bert, f)

    with open(out_meta / "source_deberta.pkl", "wb") as f:
        pickle.dump(source_bert, f)

    # 官方 train/test 划分
    save_id_file(out_meta / "train_ids.txt", train_ids)
    save_id_file(out_meta / "test_ids.txt", test_ids)

    official_split = {
        "train": train_ids,
        "test": test_ids
    }

    with open(out_meta / "official_split.pkl", "wb") as f:
        pickle.dump(official_split, f)

    # 为了兼容原有 folds.pkl 读取方式，保存为一个 fold: (test, train)
    folds = [(test_ids, train_ids)]
    with open(out_meta / "folds.pkl", "wb") as f:
        pickle.dump(folds, f)

    print("\ndone.")
    print("train ids:", len(train_ids))
    print("test ids:", len(test_ids))
    print("label 0 FR false:", len(ids_by_label[0]))
    print("label 1 TR true:", len(ids_by_label[1]))
    print("label 2 UR unverified:", len(ids_by_label[2]))
    print("graph dir:", out_graph)
    print("meta dir:", out_meta)


if __name__ == "__main__":
    main()
