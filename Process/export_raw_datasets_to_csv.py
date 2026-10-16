# -*- coding: utf-8 -*-
"""
Export the repository's raw conversation datasets into common CSV files.

Outputs:
  data/csv/PHEME_threads.csv
  data/csv/PHEME_nodes.csv
  data/csv/PHEME_RumourScheme_threads.csv
  data/csv/PHEME_RumourScheme_nodes.csv
  data/csv/RumourEval2019_threads.csv
  data/csv/RumourEval2019_nodes.csv
  data/csv/PHEME_RNR_threads.csv
  data/csv/PHEME_RNR_nodes.csv
  data/csv/SemEval2017Task8_threads.csv
  data/csv/SemEval2017Task8_nodes.csv
"""

import argparse
import csv
import json
import os
import tarfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


PHEME_LABEL_MAP = {
    0: "non-rumour",
    1: "false",
    2: "true",
    3: "unverified",
}

RUMOUREVAL_LABEL_MAP = {
    "false": 0,
    "true": 1,
    "unverified": 2,
}

PHEME_RNR_LABEL_MAP = {
    "non-rumours": (0, "non-rumour"),
    "rumours": (1, "rumour"),
}

THREAD_FIELDNAMES = [
    "dataset",
    "split",
    "platform",
    "event",
    "thread_id",
    "root_id",
    "source_id",
    "label_id",
    "label",
    "is_rumour_dir",
    "node_count",
    "edge_count",
    "source_text",
]

NODE_FIELDNAMES = [
    "dataset",
    "split",
    "platform",
    "event",
    "thread_id",
    "node_id",
    "parent_id",
    "node_type",
    "label_id",
    "label",
    "stance",
    "text",
]

SCHEME_THREAD_FIELDNAMES = THREAD_FIELDNAMES + [
    "language",
    "category",
    "is_turnaround",
]

SCHEME_NODE_FIELDNAMES = NODE_FIELDNAMES + [
    "language",
    "support",
    "responsetype_vs_source",
    "responsetype_vs_previous",
    "certainty",
    "evidentiality",
]

PHEME_RESPONSE_TO_STANCE = {
    "agreed": "support",
    "disagreed": "deny",
    "appeal-for-more-information": "query",
    "comment": "comment",
}


def read_json_file(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return json.load(f)


def get_text_from_json(obj):
    """Extract Twitter text or Reddit post/comment text from API wrappers."""
    if isinstance(obj, list):
        for item in obj:
            text = get_text_from_json(item)
            if text:
                return text
        return ""

    if not isinstance(obj, dict):
        return ""

    for key in ["text", "full_text", "body"]:
        value = obj.get(key)
        if isinstance(value, str):
            return value

    # A Reddit source post often has both a headline and a self-text body.
    title = obj.get("title")
    selftext = obj.get("selftext")
    reddit_parts = [
        value.strip()
        for value in (title, selftext)
        if isinstance(value, str) and value.strip()
        and value.strip() not in {"[deleted]", "[removed]"}
    ]
    if reddit_parts:
        return " ".join(reddit_parts)

    data = obj.get("data")
    if isinstance(data, dict):
        text = get_text_from_json(data)
        if text:
            return text

        # Reddit Listing objects wrap posts/comments in data.children.
        children = data.get("children")
        if isinstance(children, list):
            text = get_text_from_json(children)
            if text:
                return text

    children = obj.get("children")
    if isinstance(children, list):
        return get_text_from_json(children)

    return ""


def clean_text(text):
    if not isinstance(text, str):
        return ""
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def safe_int(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except Exception:
        return None


def find_one_json(folder):
    folder = Path(folder)
    if not folder.exists():
        return None

    files = [
        p for p in folder.glob("*.json")
        if not p.name.startswith(".") and not p.name.startswith("._")
    ]
    return sorted(files)[0] if files else None


def flatten_structure(structure):
    edges = []

    def visit(parent, children):
        if not isinstance(children, dict):
            return
        for child, sub in children.items():
            edges.append((str(parent), str(child)))
            visit(str(child), sub)

    if isinstance(structure, dict):
        for root, children in structure.items():
            visit(str(root), children)

    return edges


def collect_structure_dirs(root_dir):
    root_dir = Path(root_dir)
    return sorted(
        p.parent for p in root_dir.glob("**/structure.json")
        if not p.name.startswith(".") and not p.name.startswith("._")
    )


def get_pheme_label(thread_dir, is_rumour):
    if not is_rumour:
        return 0

    ann_path = Path(thread_dir) / "annotation.json"
    if not ann_path.exists():
        return None

    try:
        ann = read_json_file(ann_path)
    except Exception:
        return None

    misinfo = safe_int(ann.get("misinformation"))
    truth = safe_int(ann.get("true"))

    if misinfo == 1 and truth == 0:
        return 1
    if misinfo == 0 and truth == 1:
        return 2
    if misinfo == 0 and truth == 0:
        return 3
    return None


def get_pheme_label_from_annotation(annotation):
    """Convert a PHEME thread annotation object to the existing four labels."""
    if not isinstance(annotation, dict):
        return None

    is_rumour = str(annotation.get("is_rumour", "rumour")).strip().lower()
    if is_rumour in {"non-rumour", "nonrumour", "false", "0"}:
        return 0

    misinfo = safe_int(annotation.get("misinformation"))
    truth = safe_int(annotation.get("true"))
    if misinfo == 1 and truth in {None, 0}:
        return 1
    if misinfo == 0 and truth == 1:
        return 2
    if misinfo == 0 and truth in {None, 0}:
        return 3
    return None


def read_tar_json(member_data):
    return json.loads(member_data.decode("utf-8", errors="ignore"))


def twitter_id(obj, fallback=""):
    if not isinstance(obj, dict):
        return str(fallback)
    return str(obj.get("id_str", obj.get("id", fallback)))


def twitter_parent_id(obj):
    if not isinstance(obj, dict):
        return ""
    parent = obj.get(
        "in_reply_to_status_id_str",
        obj.get("in_reply_to_status_id"),
    )
    return "" if parent is None else str(parent)


def export_pheme_rnr_archive(archive_path, out_dir):
    """Export the binary PHEME RNR archive without extracting it."""
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"PHEME RNR archive not found: {archive_path}")

    threads = {}
    with tarfile.open(archive_path, mode="r:bz2") as archive:
        for member in archive:
            if (
                not member.isfile()
                or member.name.startswith("._")
                or not member.name.endswith(".json")
            ):
                continue

            parts = member.name.replace("\\", "/").split("/")
            if len(parts) < 6:
                continue
            label_dir = parts[2]
            if label_dir not in PHEME_RNR_LABEL_MAP:
                continue
            event, thread_id, node_folder = parts[1], parts[3], parts[4]
            if node_folder not in {"source-tweet", "source-tweets", "reactions"}:
                continue

            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            try:
                obj = read_tar_json(extracted.read())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

            label_id, label = PHEME_RNR_LABEL_MAP[label_dir]
            record = threads.setdefault(
                thread_id,
                {
                    "event": event,
                    "thread_id": thread_id,
                    "label_id": label_id,
                    "label": label,
                    "source": None,
                    "reactions": [],
                },
            )
            item = {
                "id": twitter_id(obj, Path(member.name).stem),
                "parent_id": twitter_parent_id(obj),
                "text": clean_text(get_text_from_json(obj)),
            }
            if node_folder in {"source-tweet", "source-tweets"}:
                record["source"] = item
            else:
                record["reactions"].append(item)

    thread_rows = []
    node_rows = []
    for thread_id in sorted(threads):
        record = threads[thread_id]
        source = record["source"]
        if source is None:
            continue
        root_id = source["id"] or thread_id
        unique_reactions = {}
        for item in record["reactions"]:
            node_id = item["id"]
            if node_id and node_id != root_id:
                unique_reactions.setdefault(node_id, item)
        node_ids = {root_id} | set(unique_reactions)
        edges = set()

        node_rows.append({
            "dataset": "PHEME_RNR",
            "split": "",
            "platform": "twitter",
            "event": record["event"],
            "thread_id": thread_id,
            "node_id": root_id,
            "parent_id": "",
            "node_type": "source",
            "label_id": record["label_id"],
            "label": record["label"],
            "stance": "",
            "text": source["text"],
        })
        for item in sorted(
            unique_reactions.values(),
            key=lambda value: value["id"],
        ):
            parent_id = item["parent_id"]
            if not parent_id or parent_id not in node_ids:
                parent_id = root_id
            edges.add((parent_id, item["id"]))
            node_rows.append({
                "dataset": "PHEME_RNR",
                "split": "",
                "platform": "twitter",
                "event": record["event"],
                "thread_id": thread_id,
                "node_id": item["id"],
                "parent_id": parent_id,
                "node_type": "reaction",
                "label_id": record["label_id"],
                "label": record["label"],
                "stance": "",
                "text": item["text"],
            })

        thread_rows.append({
            "dataset": "PHEME_RNR",
            "split": "",
            "platform": "twitter",
            "event": record["event"],
            "thread_id": thread_id,
            "root_id": root_id,
            "source_id": root_id,
            "label_id": record["label_id"],
            "label": record["label"],
            "is_rumour_dir": record["label_id"],
            "node_count": len(node_ids),
            "edge_count": len(edges),
            "source_text": source["text"],
        })

    write_csv(
        Path(out_dir) / "PHEME_RNR_threads.csv",
        thread_rows,
        THREAD_FIELDNAMES,
    )
    write_csv(
        Path(out_dir) / "PHEME_RNR_nodes.csv",
        node_rows,
        NODE_FIELDNAMES,
    )
    return len(thread_rows), len(node_rows)


def export_semeval2017_archive(archive_path, out_dir):
    """Export SemEval-2017 Task 8 veracity and SDQC labels."""
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"SemEval 2017 archive not found: {archive_path}")

    json_files = {}
    with tarfile.open(archive_path, mode="r:bz2") as archive:
        for member in archive:
            if (
                not member.isfile()
                or member.name.startswith("._")
                or not member.name.endswith(".json")
            ):
                continue
            name = member.name.replace("\\", "/")
            wanted = (
                "/source-tweet/" in name
                or "/replies/" in name
                or name.endswith("/structure.json")
                or "/traindev/rumoureval-subtask" in name
            )
            if not wanted:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            try:
                json_files[name] = read_tar_json(extracted.read())
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    def find_key(suffix):
        matches = sorted(name for name in json_files if name.endswith(suffix))
        return json_files[matches[0]] if matches else {}

    stance_by_split = {
        "train": find_key("rumoureval-subtaskA-train.json"),
        "dev": find_key("rumoureval-subtaskA-dev.json"),
    }
    veracity_by_split = {
        "train": find_key("rumoureval-subtaskB-train.json"),
        "dev": find_key("rumoureval-subtaskB-dev.json"),
    }
    root_to_split = {
        str(root_id): split
        for split, labels in veracity_by_split.items()
        for root_id in labels
    }

    thread_rows = []
    node_rows = []
    for structure_name in sorted(
        name for name in json_files
        if "/rumoureval-data/" in name and name.endswith("/structure.json")
    ):
        prefix = structure_name[: -len("/structure.json")]
        parts = prefix.split("/")
        try:
            data_index = parts.index("rumoureval-data")
            event, thread_id = parts[data_index + 1:data_index + 3]
        except (ValueError, IndexError):
            continue

        structure = json_files[structure_name]
        root_id = (
            str(next(iter(structure)))
            if isinstance(structure, dict) and structure
            else thread_id
        )
        split = root_to_split.get(root_id, root_to_split.get(thread_id))
        if split is None:
            continue
        label = str(
            veracity_by_split[split].get(
                root_id,
                veracity_by_split[split].get(thread_id, ""),
            )
        ).lower()
        label_id = RUMOUREVAL_LABEL_MAP.get(label)
        if label_id is None:
            continue

        source_names = sorted(
            name for name in json_files
            if name.startswith(f"{prefix}/source-tweet/")
            and name.endswith(".json")
        )
        if not source_names:
            continue
        source = json_files[source_names[0]]
        source_id = twitter_id(source, root_id)
        source_text = clean_text(get_text_from_json(source))
        edges = flatten_structure(structure)
        id_to_parent = {child: parent for parent, child in edges}
        node_ids = {root_id} | {node for edge in edges for node in edge}

        thread_rows.append({
            "dataset": "SemEval2017Task8",
            "split": split,
            "platform": "twitter",
            "event": event,
            "thread_id": thread_id,
            "root_id": root_id,
            "source_id": source_id,
            "label_id": label_id,
            "label": label,
            "is_rumour_dir": 1,
            "node_count": len(node_ids),
            "edge_count": len(edges),
            "source_text": source_text,
        })
        node_rows.append({
            "dataset": "SemEval2017Task8",
            "split": split,
            "platform": "twitter",
            "event": event,
            "thread_id": thread_id,
            "node_id": source_id,
            "parent_id": "",
            "node_type": "source",
            "label_id": label_id,
            "label": label,
            "stance": str(stance_by_split[split].get(source_id, "")).lower(),
            "text": source_text,
        })

        reply_prefix = f"{prefix}/replies/"
        for reply_name in sorted(
            name for name in json_files
            if name.startswith(reply_prefix) and name.endswith(".json")
        ):
            obj = json_files[reply_name]
            node_id = twitter_id(obj, Path(reply_name).stem)
            node_rows.append({
                "dataset": "SemEval2017Task8",
                "split": split,
                "platform": "twitter",
                "event": event,
                "thread_id": thread_id,
                "node_id": node_id,
                "parent_id": id_to_parent.get(node_id, root_id),
                "node_type": "reply",
                "label_id": label_id,
                "label": label,
                "stance": str(stance_by_split[split].get(node_id, "")).lower(),
                "text": clean_text(get_text_from_json(obj)),
            })

    write_csv(
        Path(out_dir) / "SemEval2017Task8_threads.csv",
        thread_rows,
        THREAD_FIELDNAMES,
    )
    write_csv(
        Path(out_dir) / "SemEval2017Task8_nodes.csv",
        node_rows,
        NODE_FIELDNAMES,
    )
    return len(thread_rows), len(node_rows)


def export_pheme_rumour_scheme(archive_path, out_dir):
    """Export phemerumourschemedataset.tar.bz2 without extracting it to disk."""
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"PHEME Rumour Scheme archive not found: {archive_path}")

    json_files = {}
    scheme_annotations = {}

    # Read the compressed archive once. Repeated random extraction from a bz2
    # tar file is extremely slow because bzip2 has to seek by decompression.
    with tarfile.open(archive_path, mode="r:bz2") as archive:
        for member in archive:
            if not member.isfile() or member.name.startswith("._"):
                continue
            name = member.name.replace("\\", "/")
            wanted = (
                name.endswith("/structure.json")
                or name.endswith("/annotation.json")
                or "/source-tweets/" in name
                or "/reactions/" in name
                or name.endswith("-scheme-annotations.json")
            )
            if not wanted:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            raw = extracted.read()

            if name.endswith("-scheme-annotations.json"):
                for raw_line in raw.decode("utf-8", errors="ignore").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = (str(item.get("threadid", "")), str(item.get("tweetid", "")))
                    scheme_annotations[key] = item
            else:
                try:
                    json_files[name] = read_tar_json(raw)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue

    thread_rows = []
    node_rows = []
    structure_suffix = "/structure.json"

    for structure_name in sorted(n for n in json_files if n.endswith(structure_suffix)):
        thread_prefix = structure_name[: -len(structure_suffix)]
        parts = thread_prefix.split("/")
        try:
            threads_index = parts.index("threads")
            language, event, thread_id = parts[threads_index + 1:threads_index + 4]
        except (ValueError, IndexError):
            continue

        annotation = json_files.get(f"{thread_prefix}/annotation.json", {})
        label_id = get_pheme_label_from_annotation(annotation)
        if label_id is None:
            continue

        structure = json_files[structure_name]
        edges = flatten_structure(structure)
        root_id = str(next(iter(structure), thread_id)) if isinstance(structure, dict) else thread_id
        source_candidates = sorted(
            name for name in json_files
            if name.startswith(f"{thread_prefix}/source-tweets/") and name.endswith(".json")
        )
        if not source_candidates:
            continue
        source_obj = json_files[source_candidates[0]]
        source_id = str(source_obj.get("id_str", source_obj.get("id", root_id)))
        source_text = clean_text(get_text_from_json(source_obj))
        label = PHEME_LABEL_MAP[label_id]
        all_node_ids = {root_id} | {node for edge in edges for node in edge}

        thread_rows.append({
            "dataset": "PHEME_RumourScheme",
            "split": "",
            "platform": "twitter",
            "event": event,
            "thread_id": thread_id,
            "root_id": root_id,
            "source_id": source_id,
            "label_id": label_id,
            "label": label,
            "is_rumour_dir": int(label_id != 0),
            "node_count": len(all_node_ids),
            "edge_count": len(edges),
            "source_text": source_text,
            "language": language,
            "category": annotation.get("category", ""),
            "is_turnaround": annotation.get("is_turnaround", ""),
        })

        id_to_parent = {child: parent for parent, child in edges}

        def add_node(node_id, parent_id, node_type, text):
            manual = scheme_annotations.get((thread_id, node_id), {})
            support = str(manual.get("support", ""))
            response_source = str(manual.get("responsetype-vs-source", ""))
            response_previous = str(manual.get("responsetype-vs-previous", ""))
            # Source tweets use supporting/denying/underspecified. Replies use
            # agreed/disagreed/appeal-for-more-information/comment; expose a
            # normalized SDQC-style stance while retaining the raw annotation.
            stance = support or PHEME_RESPONSE_TO_STANCE.get(response_source, "")
            node_rows.append({
                "dataset": "PHEME_RumourScheme",
                "split": "",
                "platform": "twitter",
                "event": event,
                "thread_id": thread_id,
                "node_id": node_id,
                "parent_id": parent_id,
                "node_type": node_type,
                "label_id": label_id,
                "label": label,
                "stance": stance,
                "text": text,
                "language": language,
                "support": support,
                "responsetype_vs_source": response_source,
                "responsetype_vs_previous": response_previous,
                "certainty": manual.get("certainty", ""),
                "evidentiality": manual.get("evidentiality", ""),
            })

        add_node(source_id, "", "source", source_text)
        reaction_prefix = f"{thread_prefix}/reactions/"
        for reaction_name in sorted(
            name for name in json_files
            if name.startswith(reaction_prefix) and name.endswith(".json")
        ):
            obj = json_files[reaction_name]
            fallback_id = Path(reaction_name).stem
            node_id = str(obj.get("id_str", obj.get("id", fallback_id)))
            add_node(
                node_id,
                id_to_parent.get(node_id, ""),
                "reaction",
                clean_text(get_text_from_json(obj)),
            )

    write_csv(
        Path(out_dir) / "PHEME_RumourScheme_threads.csv",
        thread_rows,
        SCHEME_THREAD_FIELDNAMES,
    )
    write_csv(
        Path(out_dir) / "PHEME_RumourScheme_nodes.csv",
        node_rows,
        SCHEME_NODE_FIELDNAMES,
    )
    return len(thread_rows), len(node_rows)


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_completed_thread_ids(path):
    path = Path(path)
    if not path.exists():
        return set()

    completed = set()
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            thread_id = row.get("thread_id")
            if thread_id:
                completed.add(str(thread_id))
    return completed


def open_csv_writer(path, fieldnames, append=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    f = open(path, "a" if append else "w", encoding="utf-8-sig", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not append:
        writer.writeheader()
    return f, writer


def export_pheme(raw_root, out_dir, resume=False):
    raw_root = Path(raw_root)
    thread_path = Path(out_dir) / "PHEME_threads.csv"
    node_path = Path(out_dir) / "PHEME_nodes.csv"
    completed_threads = read_completed_thread_ids(thread_path) if resume else set()
    append = resume and thread_path.exists() and node_path.exists()

    thread_file, thread_writer = open_csv_writer(thread_path, THREAD_FIELDNAMES, append=append)
    node_file, node_writer = open_csv_writer(node_path, NODE_FIELDNAMES, append=append)
    thread_count = 0
    node_count = 0

    try:
        for event_dir in sorted(raw_root.iterdir()):
            if not event_dir.is_dir() or event_dir.name.startswith("."):
                continue

            for split_name, is_rumour in [("rumours", True), ("non-rumours", False)]:
                split_dir = event_dir / split_name
                if not split_dir.exists():
                    continue

                for thread_dir in sorted(split_dir.iterdir()):
                    if not thread_dir.is_dir() or thread_dir.name.startswith("."):
                        continue
                    if thread_dir.name in completed_threads:
                        continue

                    label_id = get_pheme_label(thread_dir, is_rumour)
                    if label_id is None:
                        continue

                    structure_path = thread_dir / "structure.json"
                    source_path = find_one_json(thread_dir / "source-tweets")
                    if source_path is None:
                        source_path = find_one_json(thread_dir / "source-tweet")
                    if not structure_path.exists() or source_path is None:
                        continue

                    structure = read_json_file(structure_path)
                    edges = flatten_structure(structure)
                    root_id = str(list(structure.keys())[0]) if isinstance(structure, dict) and structure else thread_dir.name

                    source_obj = read_json_file(source_path)
                    source_id = str(source_obj.get("id_str", source_obj.get("id", root_id)))
                    source_text = clean_text(get_text_from_json(source_obj))

                    thread_row = {
                        "dataset": "PHEME",
                        "split": "",
                        "platform": "twitter",
                        "event": event_dir.name.replace("-all-rnr-threads", ""),
                        "thread_id": thread_dir.name,
                        "root_id": root_id,
                        "source_id": source_id,
                        "label_id": label_id,
                        "label": PHEME_LABEL_MAP[label_id],
                        "is_rumour_dir": int(is_rumour),
                        "node_count": len({root_id} | {x for edge in edges for x in edge}),
                        "edge_count": len(edges),
                        "source_text": source_text,
                    }

                    id_to_parent = {child: parent for parent, child in edges}
                    node_rows = [{
                        "dataset": "PHEME",
                        "split": "",
                        "platform": "twitter",
                        "event": event_dir.name.replace("-all-rnr-threads", ""),
                        "thread_id": thread_dir.name,
                        "node_id": source_id,
                        "parent_id": "",
                        "node_type": "source",
                        "label_id": label_id,
                        "label": PHEME_LABEL_MAP[label_id],
                        "stance": "",
                        "text": source_text,
                    }]

                    reactions_dir = thread_dir / "reactions"
                    if reactions_dir.exists():
                        for reply_path in sorted(reactions_dir.glob("*.json")):
                            if reply_path.name.startswith(".") or reply_path.name.startswith("._"):
                                continue
                            try:
                                obj = read_json_file(reply_path)
                            except Exception:
                                continue
                            node_id = str(obj.get("id_str", obj.get("id", reply_path.stem)))
                            node_rows.append({
                                "dataset": "PHEME",
                                "split": "",
                                "platform": "twitter",
                                "event": event_dir.name.replace("-all-rnr-threads", ""),
                                "thread_id": thread_dir.name,
                                "node_id": node_id,
                                "parent_id": id_to_parent.get(node_id, ""),
                                "node_type": "reaction",
                                "label_id": label_id,
                                "label": PHEME_LABEL_MAP[label_id],
                                "stance": "",
                                "text": clean_text(get_text_from_json(obj)),
                            })

                    node_writer.writerows(node_rows)
                    thread_writer.writerow(thread_row)
                    thread_count += 1
                    node_count += len(node_rows)
    finally:
        thread_file.close()
        node_file.close()

    return thread_count, node_count


def load_rumoureval_key(path):
    key = read_json_file(path)
    stance = {str(k): str(v).lower() for k, v in key.get("subtaskaenglish", {}).items()}
    veracity = {str(k): str(v).lower() for k, v in key.get("subtaskbenglish", {}).items()}
    return stance, veracity


def load_required_rumoureval_key(key_path):
    """Load one RumourEval key file and fail clearly when it is absent."""
    key_path = Path(key_path)
    if not key_path.is_file():
        raise FileNotFoundError(
            "RumourEval key file not found: {}".format(key_path)
        )
    return load_rumoureval_key(key_path)


def first_matching_label(identifiers, labels):
    """Return the first label found for root/thread/source identifiers."""
    for identifier in identifiers:
        identifier = str(identifier)
        if identifier in labels:
            return labels[identifier]
    return None


def infer_training_split_from_path(thread_dir, train_root):
    """Fallback split inference from a path relative to the training root.

    Label-key membership is the authoritative source. This path-based
    inference is used only when a released thread identifier is absent from
    both train-key.json and dev-key.json.
    """
    try:
        relative_parts = [
            part.lower()
            for part in Path(thread_dir).relative_to(train_root).parts
        ]
    except ValueError:
        relative_parts = [
            part.lower() for part in Path(thread_dir).parts
        ]

    dev_markers = {
        "dev",
        "dev-data",
        "reddit-dev-data",
        "twitter-dev-data",
        "development",
    }
    train_markers = {
        "train",
        "train-data",
        "training-data",
        "reddit-training-data",
        "twitter-training-data",
    }

    if any(part in dev_markers or part.startswith("dev-")
           or part.endswith("-dev-data") for part in relative_parts):
        return "dev"

    if any(part in train_markers or part.startswith("train-")
           or part.endswith("-training-data") for part in relative_parts):
        return "train"

    return None


def export_rumoureval(raw_root, out_dir):
    """Export RumourEval2019 with distinct train/dev/test CSV splits.

    The original release stores development conversations inside
    rumoureval-2019-training-data. Therefore, directory location alone is not
    sufficient. train-key.json and dev-key.json are loaded separately, and
    root/thread/source identifier membership in those key files determines the
    CSV split.
    """
    raw_root = Path(raw_root)
    train_root = raw_root / "rumoureval-2019-training-data"
    test_root = raw_root / "rumoureval-2019-test-data"

    train_key_path = train_root / "train-key.json"
    dev_key_path = train_root / "dev-key.json"
    final_key_path = raw_root / "final-eval-key.json"

    train_stance, train_veracity = load_required_rumoureval_key(
        train_key_path
    )
    dev_stance, dev_veracity = load_required_rumoureval_key(
        dev_key_path
    )
    test_stance, test_veracity = load_required_rumoureval_key(
        final_key_path
    )

    thread_rows = []
    node_rows = []
    split_counts = {"train": 0, "dev": 0, "test": 0}
    skipped_without_veracity = []

    # The training root contains both official Train and official Dev.
    sources = [
        (
            train_root,
            None,
        ),
        (
            raw_root / "rumoureval-2019-test-data",
            "test",
        ),
    ]

    for split_root, fixed_split in sources:
        for thread_dir in collect_structure_dirs(split_root):
            structure_path = thread_dir / "structure.json"
            source_path = find_one_json(thread_dir / "source-tweet")
            if source_path is None:
                source_path = find_one_json(thread_dir / "source-tweets")
            if source_path is None:
                continue

            structure = read_json_file(structure_path)
            edges = flatten_structure(structure)
            root_id = (
                str(list(structure.keys())[0])
                if isinstance(structure, dict) and structure
                else thread_dir.name
            )

            source_obj = read_json_file(source_path)
            source_id = str(
                source_obj.get(
                    "id_str",
                    source_obj.get("id", root_id),
                )
            )
            source_text = clean_text(
                get_text_from_json(source_obj)
            )

            identifiers = (
                root_id,
                thread_dir.name,
                source_id,
            )

            if fixed_split == "test":
                split_name = "test"
                stance_labels = test_stance
                veracity_labels = test_veracity
                label = first_matching_label(
                    identifiers,
                    veracity_labels,
                )
            else:
                train_label = first_matching_label(
                    identifiers,
                    train_veracity,
                )
                dev_label = first_matching_label(
                    identifiers,
                    dev_veracity,
                )

                if train_label is not None and dev_label is not None:
                    raise ValueError(
                        "thread {} appears in both train-key.json and "
                        "dev-key.json".format(thread_dir)
                    )

                if dev_label is not None:
                    split_name = "dev"
                    stance_labels = dev_stance
                    veracity_labels = dev_veracity
                    label = dev_label
                elif train_label is not None:
                    split_name = "train"
                    stance_labels = train_stance
                    veracity_labels = train_veracity
                    label = train_label
                else:
                    # Fallback for unusual release layouts. The label must
                    # still be found in the corresponding key file.
                    inferred = infer_training_split_from_path(
                        thread_dir,
                        train_root,
                    )
                    if inferred == "dev":
                        split_name = "dev"
                        stance_labels = dev_stance
                        veracity_labels = dev_veracity
                    elif inferred == "train":
                        split_name = "train"
                        stance_labels = train_stance
                        veracity_labels = train_veracity
                    else:
                        skipped_without_veracity.append(
                            str(thread_dir)
                        )
                        continue

                    label = first_matching_label(
                        identifiers,
                        veracity_labels,
                    )

            if label is None:
                skipped_without_veracity.append(
                    str(thread_dir)
                )
                continue

            label = str(label).lower()
            label_id = RUMOUREVAL_LABEL_MAP.get(label)
            if label_id is None:
                skipped_without_veracity.append(
                    "{} [unsupported label={!r}]".format(
                        thread_dir,
                        label,
                    )
                )
                continue

            platform = (
                "reddit"
                if "reddit" in str(thread_dir).lower()
                else "twitter"
            )
            event = thread_dir.parent.name

            thread_rows.append({
                "dataset": "RumourEval2019",
                "split": split_name,
                "platform": platform,
                "event": event,
                "thread_id": thread_dir.name,
                "root_id": root_id,
                "source_id": source_id,
                "label_id": label_id,
                "label": label,
                "is_rumour_dir": "",
                "node_count": len(
                    {root_id} | {
                        node for edge in edges for node in edge
                    }
                ),
                "edge_count": len(edges),
                "source_text": source_text,
            })
            split_counts[split_name] += 1

            id_to_parent = {
                child: parent for parent, child in edges
            }
            node_rows.append({
                "dataset": "RumourEval2019",
                "split": split_name,
                "platform": platform,
                "event": event,
                "thread_id": thread_dir.name,
                "node_id": source_id,
                "parent_id": "",
                "node_type": "source",
                "label_id": label_id,
                "label": label,
                "stance": stance_labels.get(source_id, ""),
                "text": source_text,
            })

            # Load available reply files first, then emit rows in structure
            # order. Structure nodes without JSON are retained.
            reply_objects = {}
            for folder_name in ["replies", "reactions"]:
                folder = thread_dir / folder_name
                if not folder.exists():
                    continue
                for reply_path in sorted(folder.glob("*.json")):
                    if (
                        reply_path.name.startswith(".")
                        or reply_path.name.startswith("._")
                    ):
                        continue
                    try:
                        obj = read_json_file(reply_path)
                    except Exception:
                        continue
                    node_id = str(
                        obj.get(
                            "id_str",
                            obj.get("id", reply_path.stem),
                        )
                    )
                    reply_objects[node_id] = (
                        folder_name,
                        obj,
                    )

            structure_node_ids = []
            seen_node_ids = {
                root_id,
                source_id,
            }
            for parent_id, child_id in edges:
                for node_id in (
                    str(parent_id),
                    str(child_id),
                ):
                    if node_id not in seen_node_ids:
                        structure_node_ids.append(node_id)
                        seen_node_ids.add(node_id)

            # Keep unreferenced reply files for audit after topology nodes.
            ordered_reply_ids = (
                structure_node_ids
                + sorted(
                    set(reply_objects) - seen_node_ids
                )
            )

            for node_id in ordered_reply_ids:
                available = reply_objects.get(node_id)
                if available is None:
                    folder_name, obj = (
                        "missing_replies",
                        {},
                    )
                else:
                    folder_name, obj = available

                parent_id = id_to_parent.get(
                    node_id,
                    "",
                )

                # Attach secondary structure roots to the official source.
                if (
                    not parent_id
                    and node_id in structure_node_ids
                ):
                    parent_id = root_id

                node_rows.append({
                    "dataset": "RumourEval2019",
                    "split": split_name,
                    "platform": platform,
                    "event": event,
                    "thread_id": thread_dir.name,
                    "node_id": node_id,
                    "parent_id": parent_id,
                    "node_type": {
                        "replies": "reply",
                        "reactions": "reaction",
                        "missing_replies": "missing_reply",
                    }.get(folder_name, folder_name),
                    "label_id": label_id,
                    "label": label,
                    "stance": stance_labels.get(node_id, ""),
                    "text": clean_text(
                        get_text_from_json(obj)
                    ),
                })

    if dev_veracity and split_counts["dev"] == 0:
        raise RuntimeError(
            "dev-key.json contains labels, but no Dev threads were exported. "
            "Check thread/root identifiers and the raw directory structure."
        )

    thread_output = (
        Path(out_dir) / "RumourEval2019_threads.csv"
    )
    node_output = (
        Path(out_dir) / "RumourEval2019_nodes.csv"
    )

    write_csv(
        thread_output,
        thread_rows,
        THREAD_FIELDNAMES,
    )
    write_csv(
        node_output,
        node_rows,
        NODE_FIELDNAMES,
    )

    print(
        "RumourEval2019 split counts: "
        "train={train}, dev={dev}, test={test}".format(
            **split_counts
        )
    )

    if skipped_without_veracity:
        print(
            "warning: skipped {} threads without a matching veracity "
            "label; examples: {}".format(
                len(skipped_without_veracity),
                skipped_without_veracity[:5],
            )
        )

    return len(thread_rows), len(node_rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=PROJECT_ROOT / "data")
    parser.add_argument("--out-dir", default=PROJECT_ROOT / "data" / "csv")
    parser.add_argument(
        "--dataset",
        choices=[
            "all",
            "pheme",
            "pheme_rnr",
            "pheme_scheme",
            "rumoureval2019",
            "semeval2017",
        ],
        default="all",
    )
    parser.add_argument("--resume", action="store_true", help="append missing PHEME threads instead of overwriting")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    if args.dataset in ["all", "pheme"]:
        threads, nodes = export_pheme(
            data_dir / "PHEME_raw" / "all-rnr-annotated-threads",
            out_dir,
            resume=args.resume,
        )
        print(f"PHEME: wrote {threads} threads and {nodes} nodes")

    if args.dataset in ["all", "pheme_scheme"]:
        threads, nodes = export_pheme_rumour_scheme(
            data_dir / "phemerumourschemedataset.tar.bz2",
            out_dir,
        )
        print(f"PHEME Rumour Scheme: wrote {threads} threads and {nodes} nodes")

    if args.dataset in ["all", "pheme_rnr"]:
        threads, nodes = export_pheme_rnr_archive(
            data_dir / "phemernrdataset.tar.bz2",
            out_dir,
        )
        print(f"PHEME RNR: wrote {threads} threads and {nodes} nodes")

    if args.dataset in ["all", "rumoureval2019"]:
        threads, nodes = export_rumoureval(
            data_dir / "rumoureval2019",
            out_dir,
        )
        print(f"RumourEval2019: wrote {threads} threads and {nodes} nodes")

    if args.dataset in ["all", "semeval2017"]:
        threads, nodes = export_semeval2017_archive(
            data_dir / "semeval2017-task8-dataset.tar.bz2",
            out_dir,
        )
        print(f"SemEval 2017 Task 8: wrote {threads} threads and {nodes} nodes")

    print(f"CSV output dir: {out_dir}")


if __name__ == "__main__":
    main()