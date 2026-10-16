"""Export paper-ready statistics for RumourEval2019/PHEME datasets."""

import argparse
import csv
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np


DATASETS = [
    {
        "dataset": "Dataset 1",
        "name": "RumourEval2019",
        "graph_dirs": ["RumourEval2019graph"],
        "csv_prefix": "RumourEval2019",
        "classes": "F/T/U",
        "stance": "√",
    },
    {
        "dataset": "Dataset 2",
        "name": "PHEME",
        "graph_dirs": ["PHEME4Lgraph", "PHEME4PLMgraph", "PHEME4graph"],
        "csv_prefix": "PHEME",
        "classes": "N/F/T/U",
        "stance": "Generated",
    },
    {
        "dataset": "Dataset 3",
        "name": "PHEME-SCHEME",
        "graph_dirs": ["PHEMESchemegraph", "PHEME_Schemegraph"],
        "csv_prefix": "PHEME_RumourScheme",
        "classes": "F/T/U",
        "stance": "Generated",
    },
]


def graph_depths(edge_index, root_index, node_count):
    children = [[] for _ in range(node_count)]
    for parent, child in np.asarray(edge_index).T:
        parent, child = int(parent), int(child)
        if 0 <= parent < node_count and 0 <= child < node_count:
            children[parent].append(child)
    depth = [-1] * node_count
    if not 0 <= root_index < node_count:
        raise ValueError("invalid root index {} for {} nodes".format(root_index, node_count))
    depth[root_index] = 0
    queue = deque([root_index])
    while queue:
        parent = queue.popleft()
        for child in children[parent]:
            if depth[child] < 0:
                depth[child] = depth[parent] + 1
                queue.append(child)
    # Disconnected nodes are invalid for depth statistics and reported by the
    # caller instead of silently treating them as roots.
    return depth


def statistics_from_graphs(graph_dir):
    paths = sorted(Path(graph_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError("no NPZ graphs in {}".format(graph_dir))
    posts = 0
    reply_depth_sum = 0
    reply_count = 0
    labels = Counter()
    disconnected_events = []
    stance_count = 0
    for path in paths:
        with np.load(path, allow_pickle=True) as graph:
            node_count = int(graph["x"].shape[0])
            edge_index = np.asarray(graph["edgeindex"])
            root_index = int(np.asarray(graph["rootindex"]).reshape(-1)[0])
            label = int(np.asarray(graph["y"]).reshape(-1)[0])
            has_stance = "stance_x" in graph.files
        depths = graph_depths(edge_index, root_index, node_count)
        disconnected = sum(value < 0 for value in depths)
        if disconnected:
            disconnected_events.append((path.stem, disconnected))
        valid_reply_depths = [value for value in depths if value > 0]
        posts += node_count
        reply_depth_sum += sum(valid_reply_depths)
        reply_count += len(valid_reply_depths)
        labels[label] += 1
        stance_count += int(has_stance)
    return {
        "threads": len(paths),
        "posts": posts,
        "avg_depth": reply_depth_sum / reply_count if reply_count else 0.0,
        "label_counts": dict(sorted(labels.items())),
        "stance_graphs": stance_count,
        "disconnected_events": disconnected_events,
        "source": str(Path(graph_dir)),
    }


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def statistics_from_csv(csv_dir, prefix):
    thread_path = Path(csv_dir) / (prefix + "_threads.csv")
    node_path = Path(csv_dir) / (prefix + "_nodes.csv")
    if not thread_path.exists() or not node_path.exists():
        raise FileNotFoundError("missing CSV pair for {}".format(prefix))
    threads = read_csv(thread_path)
    nodes = read_csv(node_path)
    valid_thread_ids = {
        str(row.get("thread_id", "")).strip()
        for row in threads
        if str(row.get("thread_id", "")).strip()
        and str(row.get("label", "")).strip()
    }
    nodes_by_thread = defaultdict(list)
    for row in nodes:
        event_id = str(row.get("thread_id", "")).strip()
        if event_id in valid_thread_ids:
            nodes_by_thread[event_id].append(row)

    posts = 0
    depth_sum = 0
    reply_count = 0
    disconnected_events = []
    for event_id in sorted(valid_thread_ids):
        rows = nodes_by_thread[event_id]
        node_ids = {
            str(row.get("node_id", "")).strip()
            for row in rows
            if str(row.get("node_id", "")).strip()
        }
        parent = {}
        roots = []
        for row in rows:
            node_id = str(row.get("node_id", "")).strip()
            parent_id = str(row.get("parent_id", "")).strip()
            if not node_id:
                continue
            if parent_id:
                parent[node_id] = parent_id
            else:
                roots.append(node_id)
        memo = {}

        def depth(node_id, visiting=None):
            if node_id in memo:
                return memo[node_id]
            visiting = set() if visiting is None else visiting
            parent_id = parent.get(node_id)
            if not parent_id or parent_id not in node_ids:
                memo[node_id] = 0
            elif node_id in visiting:
                memo[node_id] = -1
            else:
                visiting.add(node_id)
                parent_depth = depth(parent_id, visiting)
                visiting.discard(node_id)
                memo[node_id] = -1 if parent_depth < 0 else parent_depth + 1
            return memo[node_id]

        depths = [depth(node_id) for node_id in node_ids]
        disconnected = sum(value < 0 for value in depths)
        if disconnected:
            disconnected_events.append((event_id, disconnected))
        reply_depths = [value for value in depths if value > 0]
        posts += len(node_ids)
        depth_sum += sum(reply_depths)
        reply_count += len(reply_depths)
    labels = Counter(
        str(row.get("label", "")).strip() for row in threads
        if str(row.get("thread_id", "")).strip() in valid_thread_ids
    )
    return {
        "threads": len(valid_thread_ids),
        "posts": posts,
        "avg_depth": depth_sum / reply_count if reply_count else 0.0,
        "label_counts": dict(labels),
        "stance_graphs": None,
        "disconnected_events": disconnected_events,
        "source": "{} + {}".format(thread_path, node_path),
    }


def resolve_graph_dir(data_dir, candidates):
    for name in candidates:
        path = Path(data_dir) / name
        if path.is_dir() and any(path.glob("*.npz")):
            return path
    return None


def write_output(path, rows):
    fieldnames = [
        "Dataset", "DatasetName", "Threads", "Posts",
        "Avg. Depth", "Classes", "Stance",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--csv-dir", default="data/csv")
    parser.add_argument(
        "--output", default="results/dataset_table_statistics.csv"
    )
    parser.add_argument(
        "--no-csv-fallback", action="store_true",
        help="fail instead of using CSV when a graph directory is absent",
    )
    args = parser.parse_args()

    table_rows = []
    for spec in DATASETS:
        graph_dir = resolve_graph_dir(args.data_dir, spec["graph_dirs"])
        if graph_dir is not None:
            stats = statistics_from_graphs(graph_dir)
        elif not args.no_csv_fallback:
            stats = statistics_from_csv(args.csv_dir, spec["csv_prefix"])
        else:
            raise FileNotFoundError(
                "no graph directory for {}".format(spec["name"])
            )
        row = {
            "Dataset": spec["dataset"],
            "DatasetName": spec["name"],
            "Threads": stats["threads"],
            "Posts": stats["posts"],
            "Avg. Depth": "{:.1f}".format(stats["avg_depth"]),
            "Classes": spec["classes"],
            "Stance": spec["stance"],
        }
        table_rows.append(row)
        print(
            "{}: Threads={} | Posts={} | Avg. Depth={} | source={}".format(
                row["DatasetName"],
                row["Threads"],
                row["Posts"],
                row["Avg. Depth"],
                stats["source"],
            )
        )
        print("  label counts:", stats["label_counts"])
        if stats["disconnected_events"]:
            print(
                "  warning: disconnected nodes:",
                stats["disconnected_events"][:10],
            )

    output = Path(args.output)
    write_output(output, table_rows)
    print("\nPaper-ready table:", output.resolve())
    print("\t".join(table_rows[0].keys()))
    for row in table_rows:
        print("\t".join(str(value) for value in row.values()))


if __name__ == "__main__":
    main()
