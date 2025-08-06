"""
Graph convolutional layer
"""

import torch
import torch.nn as nn

from ..utils import DEBUG

class GCNLayer(nn.Module):
    """
    Applies a graph convolutional layer as described in Kipf et al. (2016)
    """
    def __init__(self, in_channels: int, out_channels: int, add_self_loops: bool = False, add_bias: bool = False) -> None:
        super(GCNLayer, self).__init__()
        self.add_self_loops = add_self_loops
        self.add_bias = add_bias
        self.linear = nn.Linear(in_channels, out_channels)

        if self.add_bias:
            self.bias = nn.Parameter(torch.randn(out_channels))

    def __call__(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): node feature matrix of shape [N, in_channels]
            edge_index (torch.Tensor): graph connectivity, shape [2, E],
                where edge_index[0] = dst, edge_index[1] = src

        Returns:
            Updated node features of shape [N, out_channels]
        """
        N = x.size(0)  # number of nodes

        # 1 - transform node features
        x = self.linear(x)  # [N, in_channels] -> [N, out_channels]

        # 2 - add self-loops to edge_index
        if self.add_self_loops:
            loop = torch.arange(0, N, device=x.device, dtype=edge_index.dtype)
            loop_edge = torch.stack([loop, loop], dim=0)  # [2, N]
            edge_index = torch.cat([edge_index, loop_edge], dim=1)  # [2, E + N]

        row = edge_index[0]  # destination nodes (i)
        col = edge_index[1]  # source nodes (j)

        # 3 - compute node degrees (with self-loops), thus deg[i] = number of incoming edges to node i
        deg = torch.bincount(row, minlength=N)

        # 4 - compute normalization: norm = 1 / sqrt(deg[i]) / sqrt(deg[j])
        deg_i = deg[row]  # degree of destination node i
        deg_j = deg[col]  # degree of source node j
        norm = 1.0 / (deg_i.sqrt() * deg_j.sqrt())  # [E + N]

        # 5 - message passing
        x_j = x[col]  # sender features, [E + N, out_channels]

        # scale messages
        messages = norm.view(-1, 1) * x_j  # [E + N, out_channels]

        # aggregate messages at each node
        out = torch.zeros(N, x.size(1), device=x.device)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(messages), messages)

        # 6 - add bias
        if self.add_bias:
            out += self.bias

        if DEBUG > 2:
            print(f"<x {x.dtype} Tensor with shape = {x.shape} on {x.device}>")
            print(f"<edge_index {edge_index.dtype} Tensor with shape = {edge_index.shape} on {edge_index.device}>")
            print(f"<norm {norm.dtype} Tensor with shape = {norm.shape} on {norm.device}>")
            print(f"<out {out.dtype} Tensor with shape = {out.shape} on {out.device}>")

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}: ({self.linear.in_features}, {self.linear.out_features})"
