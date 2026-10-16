"""Compare raw RumourEval2019 conversations with processed graph NPZ files.

The script distinguishes three commonly confused counts:

1. node identifiers recorded in ``structure.json``;
2. source/reply JSON files that physically exist;
3. nodes in the processed ``RumourEval2019graph/*.npz`` files.

Detailed CSV files are written so missing events and posts can be audited.
"""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def structure_nodes(tree, depth=0):
    """Return ``[(post_id, depth), ...]`` from a nested structure object."""
    result = []
    if not isinstance(tree, dict):
        return result
    for post_id, children in tree.items():
        result.append((str(post_id), depth))
        result.extend(structure_nodes(children, depth + 1))
    return result


def infer_raw_split(path):
    parts = [part.lower() for part in path.parts]
    if any("test-data" in part for part in parts):
        return "test"
    if any("dev-data" in part or part.startswith("dev") for part in parts):
        return "dev"
    return "train"


def infer_platform(path):
    text = str(path).lower()
    if "reddit" in text:
        return "reddit"
    if "twitter" in text:
        return "twitter"
    return "unknown"


def event_post_files(event_dir):
    """Return IDs backed by an actual source/reply JSON file."""
    source_ids = {
        path.stem for path in (event_dir / "source-tweet").glob("*.json")
    }
    reply_ids = {
        path.stem for path in (event_dir / "replies").glob("*.json")
    }
    return source_ids, reply_ids


def load_raw_events(raw_dir):
    events = {}
    duplicate_ids = []
    for structure_path in raw_dir.rglob("structure.json"):
        event_id = structure_path.parent.name
        tree = json.loads(structure_path.read_text(encoding="utf-8"))
        node_depth_pairs = structure_nodes(tree)
        structure_ids = {post_id for post_id, _ in node_depth_pairs}
        source_ids, reply_ids = event_post_files(structure_path.parent)
        file_ids = source_ids | reply_ids
        record = {
            "event_id": event_id,
            "split": infer_raw_split(structure_path),
            "platform": infer_platform(structure_path),
            "structure_path": str(structure_path),
            "structure_ids": structure_ids,
            "file_ids": file_ids,
            "missing_file_ids": structure_ids - file_ids,
            "extra_file_ids": file_ids - structure_ids,
            "structure_nodes": len(node_depth_pairs),
            "post_files": len(file_ids),
            "depth_sum": sum(depth for _, depth in node_depth_pairs),
            "max_depth": max((depth for _, depth in node_depth_pairs), default=0),
        }
        if event_id in events:
            duplicate_ids.append(event_id)
        events[event_id] = record
    if duplicate_ids:
        raise ValueError(
            "duplicate raw event IDs found: {}".format(sorted(set(duplicate_ids))[:10])
        )
    return events


def load_graph_events(graph_dir):
    events = {}
    for graph_path in graph_dir.glob("*.npz"):
        with np.load(graph_path, allow_pickle=True) as data:
            node_count = int(data["x"].shape[0])
            label = int(np.asarray(data["y"]).reshape(-1)[0])
            keys = set(data.files)
            saved_node_ids = None
            for key in ("node_ids", "post_ids", "tweet_ids"):
                if key in keys:
                    saved_node_ids = {
                        str(value) for value in np.asarray(data[key]).reshape(-1)
                    }
                    break
        events[graph_path.stem] = {
            "event_id": graph_path.stem,
            "graph_path": str(graph_path),
            "graph_nodes": node_count,
            "label": label,
            "saved_node_ids": saved_node_ids,
        }
    return events


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(raw_events, graph_events):
    rows = []
    groups = [("total", list(raw_events.values()))]
    for split in ("train", "dev", "test"):
        records = [record for record in raw_events.values() if record["split"] == split]
        if records:
            groups.append((split, records))
    for name, records in groups:
        event_ids = {record["event_id"] for record in records}
        graph_records = [graph_events[event_id] for event_id in event_ids if event_id in graph_events]
        structure_count = sum(record["structure_nodes"] for record in records)
        post_file_count = sum(record["post_files"] for record in records)
        graph_node_count = sum(record["graph_nodes"] for record in graph_records)
        depth_sum = sum(record["depth_sum"] for record in records)
        rows.append({
            "split": name,
            "raw_threads": len(records),
            "structure_nodes": structure_count,
            "existing_post_json": post_file_count,
            "graph_threads": len(graph_records),
            "graph_nodes": graph_node_count,
            "missing_graph_threads": len(event_ids - set(graph_events)),
            "structure_without_json": structure_count - post_file_count,
            "structure_minus_graph_nodes": structure_count - graph_node_count,
            "avg_post_depth": depth_sum / structure_count if structure_count else 0.0,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Audit raw and processed RumourEval2019 event graphs"
    )
    parser.add_argument(
        "--raw-dir", default="data/rumoureval2019", help="official raw dataset directory"
    )
    parser.add_argument(
        "--graph-dir", default="data/RumourEval2019graph", help="processed NPZ graph directory"
    )
    parser.add_argument(
        "--output-dir",
        default="results/RumourEval2019_data_audit",
        help="directory for detailed audit CSV files",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    graph_dir = Path(args.graph_dir)
    output_dir = Path(args.output_dir)
    if not raw_dir.exists():
        raise FileNotFoundError("raw directory not found: {}".format(raw_dir))
    if not graph_dir.exists():
        raise FileNotFoundError("graph directory not found: {}".format(graph_dir))

    raw_events = load_raw_events(raw_dir)
    graph_events = load_graph_events(graph_dir)
    raw_ids, graph_ids = set(raw_events), set(graph_events)

    comparison_rows = []
    missing_post_rows = []
    for event_id in sorted(raw_ids):
        raw = raw_events[event_id]
        graph = graph_events.get(event_id)
        graph_nodes = graph["graph_nodes"] if graph else None
        comparison_rows.append({
            "event_id": event_id,
            "split": raw["split"],
            "platform": raw["platform"],
            "structure_nodes": raw["structure_nodes"],
            "existing_post_json": raw["post_files"],
            "graph_exists": int(graph is not None),
            "graph_nodes": "" if graph_nodes is None else graph_nodes,
            "structure_minus_graph": "" if graph_nodes is None else raw["structure_nodes"] - graph_nodes,
            "json_minus_graph": "" if graph_nodes is None else raw["post_files"] - graph_nodes,
            "missing_post_json_count": len(raw["missing_file_ids"]),
            "extra_post_json_count": len(raw["extra_file_ids"]),
            "max_depth": raw["max_depth"],
        })
        saved_ids = graph["saved_node_ids"] if graph else None
        graph_missing_ids = raw["structure_ids"] - saved_ids if saved_ids is not None else None
        for post_id in sorted(raw["missing_file_ids"]):
            missing_post_rows.append({
                "event_id": event_id,
                "split": raw["split"],
                "platform": raw["platform"],
                "post_id": post_id,
                "reason": "listed_in_structure_but_json_missing",
            })
        if graph_missing_ids is not None:
            for post_id in sorted(graph_missing_ids - raw["missing_file_ids"]):
                missing_post_rows.append({
                    "event_id": event_id,
                    "split": raw["split"],
                    "platform": raw["platform"],
                    "post_id": post_id,
                    "reason": "raw_post_not_saved_in_graph",
                })

    summary_rows = aggregate(raw_events, graph_events)
    write_csv(output_dir / "summary.csv", list(summary_rows[0]), summary_rows)
    write_csv(
        output_dir / "thread_node_comparison.csv",
        list(comparison_rows[0]),
        comparison_rows,
    )
    write_csv(
        output_dir / "missing_post_files.csv",
        ["event_id", "split", "platform", "post_id", "reason"],
        missing_post_rows,
    )
    write_csv(
        output_dir / "missing_graph_threads.csv",
        ["event_id", "split", "platform", "structure_path"],
        [
            {
                "event_id": event_id,
                "split": raw_events[event_id]["split"],
                "platform": raw_events[event_id]["platform"],
                "structure_path": raw_events[event_id]["structure_path"],
            }
            for event_id in sorted(raw_ids - graph_ids)
        ],
    )
    write_csv(
        output_dir / "extra_graph_threads.csv",
        ["event_id", "graph_nodes", "label", "graph_path"],
        [
            {
                "event_id": event_id,
                "graph_nodes": graph_events[event_id]["graph_nodes"],
                "label": graph_events[event_id]["label"],
                "graph_path": graph_events[event_id]["graph_path"],
            }
            for event_id in sorted(graph_ids - raw_ids)
        ],
    )

    print("\nRaw/graph comparison")
    for row in summary_rows:
        print(
            "{split}: raw threads={raw_threads}, structure nodes={structure_nodes}, "
            "post JSON={existing_post_json}, graph threads={graph_threads}, "
            "graph nodes={graph_nodes}, missing graphs={missing_graph_threads}".format(**row)
        )
    print("Missing graph event IDs:", sorted(raw_ids - graph_ids)[:20])
    print("Extra graph event IDs:", sorted(graph_ids - raw_ids)[:20])
    print("Detailed audit files:", output_dir.resolve())
    if graph_events and all(record["saved_node_ids"] is None for record in graph_events.values()):
        print(
            "Note: NPZ files do not contain node/post IDs. Exact posts omitted while "
            "building a graph cannot be named; use thread_node_comparison.csv to locate "
            "the affected event, and missing_post_files.csv for raw JSON omissions."
        )


if __name__ == "__main__":
    main()
