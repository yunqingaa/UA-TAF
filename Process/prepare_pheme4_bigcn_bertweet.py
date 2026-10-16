# 将原始数据利用BERT编码
import os
import json
import pickle
import random
from pathlib import Path

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


CWD = os.getcwd()


def read_json_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def get_text_from_tweet_json(obj):
    if "text" in obj and isinstance(obj["text"], str):
        return obj["text"]
    if "full_text" in obj and isinstance(obj["full_text"], str):
        return obj["full_text"]
    return ""


def find_one_json(folder):
    folder = Path(folder)

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
    PHEME 的 structure.json 通常是嵌套 dict:
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

    for root, children in structure.items():
        visit(str(root), children)

    return edges

def safe_int(v):
    if v is None:
        return None
    if isinstance(v, bool):
        return int(v)
    try:
        return int(v)
    except Exception:
        return None


def get_pheme4_label(thread_dir, is_rumour):
    """
    PHEME four-class labels:
    0: non-rumour
    1: false rumour
    2: true rumour
    3: unverified rumour
    """
    thread_dir = Path(thread_dir)

    if not is_rumour:
        return 0

    ann_path = thread_dir / "annotation.json"
    if not ann_path.exists():
        return None

    try:
        ann = read_json_file(ann_path)
    except Exception:
        return None

    misinfo = safe_int(ann.get("misinformation", None))
    truth = safe_int(ann.get("true", None))

    # false rumour
    if misinfo == 1 and truth == 0:
        return 1

    # true rumour
    if misinfo == 0 and truth == 1:
        return 2

    # unverified rumour
    if misinfo == 0 and truth == 0:
        return 3

    # ambiguous or abnormal annotation
    return None

def batch_encode(texts, tokenizer, model, device, batch_size=16, max_length=96):
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


def build_one_thread(thread_dir, label, tokenizer, model, device):
    thread_dir = Path(thread_dir)
    eid = thread_dir.name

    # 兼容 source-tweets / source-tweet 两种目录名
    source_json_path = find_one_json(thread_dir / "source-tweets")
    if source_json_path is None:
        source_json_path = find_one_json(thread_dir / "source-tweet")

    structure_path = thread_dir / "structure.json"

    if source_json_path is None:
        return None

    if not structure_path.exists():
        return None

    source_obj = read_json_file(source_json_path)
    source_id = str(source_obj.get("id_str", source_obj.get("id", eid)))
    source_text = get_text_from_tweet_json(source_obj)

    structure = read_json_file(structure_path)

    # structure.json 中的根节点通常最可靠
    if isinstance(structure, dict) and len(structure) > 0:
        root_id = str(list(structure.keys())[0])
    else:
        root_id = source_id

    # id -> text
    id_to_text = {}
    id_to_text[source_id] = source_text
    id_to_text[root_id] = source_text

    reactions_dir = thread_dir / "reactions"
    if reactions_dir.exists():
        for rp in reactions_dir.glob("*.json"):
            if rp.name.startswith("._") or rp.name.startswith("."):
                continue

            try:
                r_obj = read_json_file(rp)
                rid = str(r_obj.get("id_str", r_obj.get("id", rp.stem)))
                text = get_text_from_tweet_json(r_obj)
                id_to_text[rid] = text
            except Exception:
                continue

    edges_raw = flatten_structure(structure)

    # 关键修改：
    # 不再只保留有文本的节点，而是保留 structure 中出现的所有节点
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
    for p, c in edges_raw:
        p, c = str(p), str(c)
        if p in node_to_idx and c in node_to_idx:
            row.append(node_to_idx[p])
            col.append(node_to_idx[c])

    # 对于没有边的 thread，先跳过，避免 Bi-GCN 训练不稳定
    if len(row) == 0:
        return None

    # 没有文本的节点使用 [NO_TEXT]，不要删除节点
    texts = []
    for nid in node_ids:
        text = id_to_text.get(nid, "")
        if not isinstance(text, str) or len(text.strip()) == 0:
            text = "[NO_TEXT]"
        texts.append(text)

    x = batch_encode(texts, tokenizer, model, device)

    edgeindex = np.array([row, col], dtype=np.int64)
    edgeindex, _ = connect_orphan_components(
        edgeindex, root_index=0, node_count=len(node_ids)
    )
    rootindex = np.array(0, dtype=np.int64)
    root = x[0:1]
    y = np.array(label, dtype=np.int64)

    return {
        "eid": eid,
        "x": x,
        "edgeindex": edgeindex,
        "root": root,
        "rootindex": rootindex,
        "y": y,
        "node_num": len(node_ids),
        "node_ids": np.asarray(node_ids, dtype=str),
        "source_bert": root.reshape(-1)
    }

def make_folds(ids_by_label, seed=2026):
    random.seed(seed)

    label_ids = {}

    for label, ids in ids_by_label.items():
        ids = list(ids)
        random.shuffle(ids)
        label_ids[label] = ids

    folds = []

    for k in range(5):
        test = []
        train = []

        for label, arr in label_ids.items():
            n = len(arr)
            fold_size = int(n * 0.2)

            start = k * fold_size
            end = (k + 1) * fold_size if k < 4 else n

            test_part = arr[start:end]
            train_part = arr[:start] + arr[end:]

            test.extend(test_part)
            train.extend(train_part)

        random.shuffle(test)
        random.shuffle(train)
        folds.append((test, train))

    return folds

def main():
    raw_root = Path(CWD) / "data" / "PHEME_raw" / "all-rnr-annotated-threads"
    out_graph = Path(CWD) / "data" / "PHEME4graph"
    out_meta = Path(CWD) / "data" / "PHEME4"

    out_graph.mkdir(parents=True, exist_ok=True)
    out_meta.mkdir(parents=True, exist_ok=True)

    local_bertweet_path = Path(CWD) / "pretrained" / "bertweet-base"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    tokenizer = AutoTokenizer.from_pretrained(
        str(local_bertweet_path),
        local_files_only=True
    )

    model = AutoModel.from_pretrained(
        str(local_bertweet_path),
        local_files_only=True
    ).to(device)

    thread_dirs = []

    for event_dir in raw_root.iterdir():
        if not event_dir.is_dir():
            continue

        rumour_dir = event_dir / "rumours"
        nonrumour_dir = event_dir / "non-rumours"

        if nonrumour_dir.exists():
            for td in nonrumour_dir.iterdir():
                if td.is_dir() and not td.name.startswith("."):
                    label = get_pheme4_label(td, is_rumour=False)
                    thread_dirs.append((td, label))

        if rumour_dir.exists():
            for td in rumour_dir.iterdir():
                if td.is_dir() and not td.name.startswith("."):
                    label = get_pheme4_label(td, is_rumour=True)
                    if label is not None:
                        thread_dirs.append((td, label))

    print("thread candidates:", len(thread_dirs))

    label_lines = []
    tree_lines = []
    source_bert = {}
    ids_by_label = {0: [], 1: [], 2: [], 3: []}

    saved = 0
    skipped = 0

    for thread_dir, label in tqdm(thread_dirs):
        try:
            item = build_one_thread(thread_dir, label, tokenizer, model, device)

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
                node_ids=item["node_ids"]
            )

            source_bert[eid] = item["source_bert"]
            ids_by_label[label].append(eid)

            label_map = {
                0: "non-rumour",
                1: "false",
                2: "true",
                3: "unverified"
            }

            label_name = label_map[label]
            label_lines.append(f"{label_name}\t0\t{eid}\n")

            # 给 loadTree 用：indexC 使用 1-based，npz edgeindex 使用 0-based
            # parent 为 None 的 root 节点
            node_num = item["node_num"]
            tree_lines.append(f"{eid}\tNone\t1\t0\t0\t0:0\n")

            # 根据 edgeindex 写树边
            # edgeindex: parent_idx -> child_idx, both 0-based
            for p, c in item["edgeindex"].T:
                tree_lines.append(f"{eid}\t{int(p) + 1}\t{int(c) + 1}\t0\t0\t0:0\n")

            saved += 1

        except Exception as e:
            skipped += 1
            print("skip:", thread_dir, e)

    saved_id_set = set(source_bert)
    graph_id_set = {path.stem for path in out_graph.glob("*.npz")}
    if len(saved_id_set) != saved:
        raise RuntimeError(
            "duplicate PHEME event IDs were generated: saved rows={}, unique IDs={}".format(
                saved, len(saved_id_set)
            )
        )
    if graph_id_set != saved_id_set:
        raise RuntimeError(
            "PHEME graph/meta mismatch; missing graphs:{} stale graphs:{}".format(
                sorted(saved_id_set - graph_id_set)[:10],
                sorted(graph_id_set - saved_id_set)[:10]
            )
        )
    for event_id in sorted(saved_id_set):
        with np.load(out_graph / (event_id + ".npz"), allow_pickle=True) as graph:
            node_count = int(graph["x"].shape[0])
            root_index = int(np.asarray(graph["rootindex"]).reshape(-1)[0])
            reached = reachable_nodes(graph["edgeindex"], root_index, node_count)
        if len(reached) != node_count:
            raise RuntimeError(
                "disconnected generated PHEME graph: {} ({} unreachable)".format(
                    event_id, node_count - len(reached)
                )
            )
    print("output consistency/connectivity: OK | events:", len(saved_id_set))

    with open(out_meta / "PHEME_label_All.txt", "w", encoding="utf-8") as f:
        f.writelines(label_lines)

    with open(out_meta / "data.TD_RvNN.vol_5000.txt", "w", encoding="utf-8") as f:
        f.writelines(tree_lines)

    with open(out_meta / "source_bertweet.pkl", "wb") as f:
        pickle.dump(source_bert, f)

    folds = make_folds(ids_by_label)
    with open(out_meta / "folds.pkl", "wb") as f:
        pickle.dump(folds, f)

    print("done.")
    print("saved:", saved)
    print("skipped:", skipped)
    print("label 0 NR non-rumour:", len(ids_by_label[0]))
    print("label 1 FR false:", len(ids_by_label[1]))
    print("label 2 TR true:", len(ids_by_label[2]))
    print("label 3 UR unverified:", len(ids_by_label[3]))
    print("graph dir:", out_graph)
    print("meta dir:", out_meta)


if __name__ == "__main__":
    main()
