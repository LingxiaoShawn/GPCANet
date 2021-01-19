import torch, os
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Planetoid
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected

"""
Here we don't do any normalization for input node features
    graph.adj: SparseTensor (use matmul) 
"""
def load_data(data_name, mode='DA'):
    if data_name in ['cora', 'citeseer', 'pubmed']:
        return load_ptg_data(data_name, mode)
    if data_name in ['products', 'arxiv']:
        return load_ogb_data(data_name, mode)

def load_ptg_data(data_name, mode='DA'):
    # pytorch geometric data
    DATA_ROOT = 'dataset'
    dataset = Planetoid(os.path.join(DATA_ROOT, data_name), data_name)
    graph = dataset[0]
    graph.num_classes = dataset.num_classes
    # masks
    graph.valid_mask = graph.val_mask
    graph.train_idx = graph.train_mask.nonzero(as_tuple=False).squeeze()
    graph.valid_idx = graph.val_mask.nonzero(as_tuple=False).squeeze()
    graph.test_idx = graph.test_mask.nonzero(as_tuple=False).squeeze()
    # process adj
    adj = to_sparsetensor(graph)
    graph.adj = normalize_adj(adj, mode)
    return graph
    
def load_ogb_data(data_name, mode='DA'):
    # open graph benchmark data
    dataset = PygNodePropPredDataset(name='ogbn-'+data_name) 
    graph = dataset[0]
    graph.num_classes = dataset.num_classes
    # create train mask
    split_idx = dataset.get_idx_split()
    graph.train_idx, graph.valid_idx, graph.test_idx = split_idx["train"], split_idx["valid"], split_idx["test"] 
    
    # Convert split indices to boolean masks and add them to `data`.
    for key, idx in split_idx.items():
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[idx] = True
        graph[f'{key}_mask'] = mask
    
    # process adj
    adj = to_sparsetensor(graph)
    graph.adj = normalize_adj(adj, mode)
    graph.y = graph.y.squeeze()
    return graph

def normalize_adj(adj, mode='DA'):
    deg = adj.sum(dim=1).to(torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    if mode == 'DA':
        return deg_inv_sqrt.view(-1,1) * deg_inv_sqrt.view(-1,1)*adj
    if mode == 'DAD':
        return deg_inv_sqrt.view(-1,1)*adj*deg_inv_sqrt.view(1,-1)
    return adj
    
def to_sparsetensor(data):
    N = data.num_nodes
    data.edge_index = to_undirected(data.edge_index, N)
    row, col = data.edge_index
    adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
    adj = adj.set_diag()
    return adj
