"""
Simple graph neural network
"""

from typing import Any, Dict
from ..nn import GCNLayer

import torch
import torch.nn as nn

def get_config() -> Dict:
    return {
        "name": "Simple Graph neural network on an undirected graph",
        "description": "Example of a GCN neural network",
        "model": "GCN",
        "task": "node_classification",
        "num_nodes": 4,
        "num_edges": 8,
        "features_dim": 3,
        "activation": "relu",
        "epochs": 2,
    }

def create_dataset(config: Dict) -> Dict:
    N: int = config.get("num_nodes", -1)
    n_feats: int = config.get("features_dim", -1)
    # check for >= 0 values
    if N <= 0 or n_feats <= 0:
        raise ValueError(f"Either N={N} or n_feats={n_feats} is <= 0!")

    # input tensor of dims [N, # features]
    x = torch.randn(N, n_feats)
    # output tensor of dim [N,]
    y = torch.randint(0, 4, (N,))
    # edge indices with shape [2, # edges]
    edge_index = torch.tensor([
        [1, 0, 2, 3, 1, 3, 1, 2],
        [0, 1, 1, 1, 2, 2, 3, 3],
    ], dtype=torch.int64)
    return {
        "x": x,
        "y": y,
        "edge_index": edge_index,
    }

def infer(model, ds: Dict) -> Any:
    z = model.forward(x=ds.get("x"), edge_index=ds.get("edge_index"))
    print(f"out = {z}, shape = {z.shape}")

def train(model, ds: Dict) -> Any:
    raise NotImplementedError("training method not ready yet")

def run(config: Dict) -> None:
    ds = create_dataset(config=config)
    graph_nn = SimpleGNN(input_features=config["features_dim"],  output_features=config["features_dim"], activation=config["activation"])
    infer(graph_nn, ds)

class SimpleGNN(nn.Module):
    """
    Simple GNN based on the GCN model of Kipf et al. (2016)
    """
    def __init__(self, input_features, output_features, activation="", **kwargs) -> None:
        super(SimpleGNN, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        if activation.lower() == "relu":
            self.activation_fnc = nn.ReLU
        elif activation.lower() == "sigmoid":
            self.activation_fnc = nn.Sigmoid
        else:
            raise ValueError(f"Activation function ({activation}) not supported")

        self.conv = GCNLayer(in_channels=input_features, out_channels=output_features, **kwargs)
        self.act = self.activation_fnc()

    def forward(self, x, edge_index):
        """Forward pass

        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)

        Returns:
            A tensor (torch.Tensor) with the label estimated for each node in the graph, computed
            with the argmax function on the output features of the nodes

        """
        x = self.conv(x, edge_index)
        return torch.argmax(self.act(x), dim=1)

    def __call__(self, x, edge_index):
        return self.forward(x, edge_index)
