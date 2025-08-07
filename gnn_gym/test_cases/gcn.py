"""
Graph convolutional network example
"""

from typing import Any, Dict

import torch.nn as nn
import torch.optim as optim

from gnn_gym.nn import GCNLayer
from gnn_gym.datasets import karateclub

def get_config() -> Dict:
    return {
        "name": "Graph convolutional network (GCN) on an undirected graph",
        "description": "Example of a GCN neural network",
        "model": "GCN",
        "dataset": "",
        "task": "node_classification",
        "features_dim": -1,
        "hidden_features": -1,
        "activation": "relu",
        "epochs": 1,
    }

def load_dataset(config: Dict) -> Dict:
    dataset_name: str = config.get("dataset", "unknown")
    if dataset_name == "karate":
        ds = karateclub()
    else:
        raise ValueError("only dataset available is KarateClub")
    return ds

def infer(model, ds: Dict) -> Any:
    z = model.forward(x=ds.get("x"), edge_index=ds.get("edge_index"))
    print(f"out = {z}, shape = {z.shape}")

def accuracy(pred, y):
    return (pred == y).sum()/len(y)

def train(ds, model, epochs) -> Any:

    optimizer = optim.Adam(model.parameters(), lr=0.01)
    lossfn = nn.CrossEntropyLoss()

    losses, accuracies, outputs = [], [], []
    for epoch in range(epochs):
        # clear grads
        optimizer.zero_grad()

        # forward pass
        z = model(x=ds.get("x"), edge_index=ds.get("edge_index"))

        # loss function
        loss = lossfn(z, ds.get("y"))

        # accuracy
        pred = z.argmax(dim=1)
        acc = accuracy(pred, ds.get("y"))

        # gradients
        loss.backward()

        # tune parameters
        optimizer.step()

        # store data
        losses.append(loss)
        accuracies.append(acc)
        outputs.append(z)

        print(f"Epoch {epoch+1:>3} | loss: {loss:.2f} | acc: {acc*100:.2f}%")

def run(config: Dict) -> None:
    # create dataset
    ds = load_dataset(config=config)

    # update config values
    for key, value in ds.items():
        if key == "features_dim":
            config[key] = value

    # create graph neural network
    model = GCN(input_features=config["features_dim"], hidden_features=config["hidden_features"], output_features=config["features_dim"], activation=config["activation"])
    print(f"GCN initialized with structure:\n{model}\n")

    # train gnn
    print("- Training model...")
    train(ds.get("data"), model, config["epochs"])

    # make inference
    # infer(model, ds)

class GCN(nn.Module):
    """
    Simple GNN based on the GCN model of Kipf et al. (2016)
    """
    def __init__(self, input_features, hidden_features, output_features, activation="", **kwargs) -> None:
        super(GCN, self).__init__()
        self.input_features = input_features
        self.hidden_features = hidden_features
        self.output_features = output_features
        if activation.lower() == "relu":
            self.activation_fnc = nn.ReLU
        elif activation.lower() == "sigmoid":
            self.activation_fnc = nn.Sigmoid
        else:
            raise ValueError(f"Activation function ({activation}) not supported")

        self.conv1 = GCNLayer(in_channels=input_features, out_channels=hidden_features, **kwargs)
        self.conv2 = GCNLayer(in_channels=hidden_features, out_channels=output_features, **kwargs)
        self.act = self.activation_fnc()

    def __call__(self, x, edge_index):
        """Forward pass

        Args:
            x: Input features per node
            edge_index: List of vertex index pairs representing the edges in the graph (PyTorch geometric notation)

        Returns:
            A tensor (torch.Tensor) with the label estimated for each node in the graph, computed
            with the argmax function on the output features of the nodes

        """
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        logits = self.act(x)
        return logits

    def __repr__(self):
        extra_lines = []
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            child_lines.append("  (" + key + "): " + mod_str)
        lines = extra_lines + child_lines
        main_str = "  " + self._get_name() + ":"
        main_str += "\n  " + "\n  ".join(lines)
        return main_str
