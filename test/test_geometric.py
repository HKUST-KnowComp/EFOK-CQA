import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, subgraph



edge_index = torch.tensor([[0, 0, 1, 2, 3], [0, 1, 0, 3, 0]])
edge_attr = torch.Tensor([1, 2, 3, 4, 5])
subset = torch.tensor([0,1])
print(to_dense_adj(edge_index, edge_attr=edge_attr))
print(subgraph(subset,edge_index))
