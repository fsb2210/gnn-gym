"""
Graph convolutional network example
"""

from typing import Dict

import torch.nn as nn

from gnn_gym.utils import DEBUG
from gnn_gym.nn import GCNLayer
from gnn_gym.test_cases.utils import create_criterion, create_optimizer, load_dataset, train

def get_config() -> Dict:
    return {
        "name": "Graph neural network - GCN",
        "description": "Train a GNN on the Karate Club dataset",
        "dataset": "karate",
        "model": {
            "type": "GCN",
            "hidden_dim": 32,
            "activation": "relu",
            "out_channels": 4,  # number of classes
        },
        "training": {
            "num_epochs": 1,
            "verbose": True,
        },
        "optimizer": {
            "type": "Adam",
            "lr": 0.01,
            "weight_decay": 5e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
        },
        "criterion": {
            "type": "CrossEntropyLoss",
            "weight": None,
            "reduction": "mean"
        },
    }

def run(config_override: Dict) -> None:
    # load config
    config = get_config()

    # create dataset
    ds = load_dataset(dataset_name=config.get("dataset", ""))

    # Apply overrides (supports nested keys like 'optimizer.lr')
    if config_override:
        apply_nested_override(config, config_override)

    if DEBUG >= 1:
        print("- Configuration Summary:")
        _print_config(config)

    # create graph neural network
    model = GCN(input_features=ds["features_dim"], hidden_features=config["hidden_features"], output_features=ds["features_dim"], activation=config["model"]["activation"])
    print(f"- GCN initialized with structure:\n{model}\n")

    # (c) Create optimizer and criterion
    print("- Initializing optimizer and loss function...")
    optimizer = create_optimizer(model.parameters(), config["optimizer"])
    criterion = create_criterion(config["criterion"])

    print(f"  Optimizer config: {dict(config['optimizer'])}")
    print(f"  Criterion config: {dict(config['criterion'])}", end="\n\n")

    # train gnn
    print("- Training model...")
    train(ds.get("data"), model, config["training"]["num_epochs"], config["training"]["verbose"], optimizer, criterion)


def apply_nested_override(config: dict, overrides: dict):
    """
    Apply overrides that may include dot-separated keys like 'optimizer.lr'.
    Example: {'optimizer.lr': 0.005, 'training.num_epochs': 50}
    """
    if isinstance(overrides, list):
        # Convert ['key=value', ...] to dict
        override_dict = {}
        for item in overrides:
            if '=' not in item:
                continue
            k, v = item.split('=', 1)
            try:
                v = float(v) if '.' in v else int(v)
            except ValueError:
                pass
            override_dict[k] = v
    else:
        override_dict = overrides

    for key, value in override_dict.items():
        keys = key.split('.')
        d = config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

def _print_config(config: dict, indent=2):
    """
    Pretty print the config dictionary.
    """
    import json
    print(json.dumps(config, indent=indent, ensure_ascii=False), end="\n\n")

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
