# 构造 PyG 图数据
import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data


def _topology_features(edge_index, num_nodes, root_index):
    """Match the four propagation statistics used by UA-TAF GraphEncoder."""
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        raise ValueError("a propagation graph must contain at least one node")
    edges = np.asarray(edge_index)
    out_degree = np.zeros(num_nodes, dtype=np.float32)
    adjacency = [[] for _ in range(num_nodes)]
    if edges.size:
        for source, target in edges.T:
            source, target = int(source), int(target)
            if not (0 <= source < num_nodes and 0 <= target < num_nodes):
                raise ValueError("edge index exceeds the number of nodes")
            out_degree[source] += 1.0
            adjacency[source].append(target)
            adjacency[target].append(source)
    root_index = int(root_index)
    if root_index < 0 or root_index >= num_nodes:
        root_index = 0
    distances = [-1] * num_nodes
    distances[root_index] = 0
    queue = [root_index]
    for node in queue:
        for neighbour in adjacency[node]:
            if distances[neighbour] < 0:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)
    max_depth = max(distances) if distances else 0
    non_leaf = int((out_degree > 0).sum())
    mean_branching = float(out_degree.sum()) / max(non_leaf, 1)
    leaf_ratio = float((out_degree == 0).mean())
    return torch.tensor([[
        np.log1p(num_nodes),
        np.log1p(max(max_depth, 0)),
        np.log1p(mean_branching),
        leaf_ratio,
    ]], dtype=torch.float32)

class GraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.droprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

def collate_fn(data):
    return data

class BiGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, tddroprate=0,budroprate=0,
                 data_path=os.path.join('..','..', 'data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.tddroprate = tddroprate
        self.budroprate = budroprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]

        source_bert = torch.tensor(data['root'], dtype=torch.float32)

        if source_bert.dim() == 1:
            source_bert = source_bert.view(1, -1)

        if "stance_x" in data.files:
            stance_x = torch.tensor(data["stance_x"], dtype=torch.float32)
        else:
            stance_x = torch.zeros((data["x"].shape[0], 16), dtype=torch.float32)

        root_index = int(np.asarray(data['rootindex']).reshape(-1)[0])
        topology_features = _topology_features(
            edgeindex, data["x"].shape[0], root_index
        )

        return Data(
            x=torch.tensor(data['x'], dtype=torch.float32),
            edge_index=torch.LongTensor(new_edgeindex),
            BU_edge_index=torch.LongTensor(bunew_edgeindex),
            y=torch.LongTensor([int(data['y'])]),
            root=torch.tensor(data['root'], dtype=torch.float32),
            rootindex=torch.LongTensor([root_index]),
            source_bert=source_bert,
            stance_x=stance_x,
            topology_features=topology_features,
        )


class UdGraphDataset(Dataset):
    def __init__(self, fold_x, treeDic,lower=2, upper=100000, droprate=0,
                 data_path=os.path.join('..','..','data', 'Weibograph')):
        self.fold_x = list(filter(lambda id: id in treeDic and len(treeDic[id]) >= lower and len(treeDic[id]) <= upper, fold_x))
        self.treeDic = treeDic
        self.data_path = data_path
        self.droprate = droprate

    def __len__(self):
        return len(self.fold_x)

    def __getitem__(self, index):
        id =self.fold_x[index]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row.extend(burow)
        col.extend(bucol)
        if self.droprate > 0:
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.droprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]

        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.tensor(data['root'], dtype=torch.float32),
             rootindex=torch.LongTensor([int(data['rootindex'])]))
