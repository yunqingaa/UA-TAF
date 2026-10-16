"""Utilities for enforcing root reachability in directed propagation graphs."""

from collections import deque

import numpy as np


def reachable_nodes(edge_index, root_index, node_count):
    children = [[] for _ in range(node_count)]
    for parent, child in np.asarray(edge_index).T:
        parent, child = int(parent), int(child)
        if 0 <= parent < node_count and 0 <= child < node_count:
            children[parent].append(child)
    reached = {int(root_index)}
    queue = deque([int(root_index)])
    while queue:
        for child in children[queue.popleft()]:
            if child not in reached:
                reached.add(child)
                queue.append(child)
    return reached


def connect_orphan_components(edge_index, root_index, node_count):
    """Attach each unreachable weak component to the event root.

    The anchor is an in-degree-zero node within that component, corresponding
    to an extra top-level root in malformed ``structure.json``. If malformed
    cycles provide no such node, the smallest node index is used.
    """
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.size == 0:
        edge_index = np.empty((2, 0), dtype=np.int64)
    edge_index = edge_index.reshape(2, -1)
    reached = reachable_nodes(edge_index, root_index, node_count)
    unreachable = set(range(node_count)) - reached
    if not unreachable:
        return edge_index, []

    undirected = {node: set() for node in unreachable}
    indegree = {node: 0 for node in unreachable}
    for parent, child in edge_index.T:
        parent, child = int(parent), int(child)
        if parent in unreachable and child in unreachable:
            undirected[parent].add(child)
            undirected[child].add(parent)
            indegree[child] += 1

    added = []
    remaining = set(unreachable)
    while remaining:
        start = min(remaining)
        component = {start}
        queue = deque([start])
        while queue:
            for neighbor in undirected[queue.popleft()]:
                if neighbor not in component:
                    component.add(neighbor)
                    queue.append(neighbor)
        remaining -= component
        anchors = sorted(node for node in component if indegree[node] == 0)
        if not anchors:
            anchors = [min(component)]
        for anchor in anchors:
            added.append((int(root_index), anchor))

    repaired = np.concatenate(
        [edge_index, np.asarray(added, dtype=np.int64).T], axis=1
    )
    residual = set(range(node_count)) - reachable_nodes(
        repaired, root_index, node_count
    )
    if residual:
        raise ValueError(
            "root-connectivity repair left unreachable nodes: {}".format(
                sorted(residual)[:20]
            )
        )
    return repaired, added
