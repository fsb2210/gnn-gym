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

class SimpleGNN():
    def __init__(self) -> None:
        pass
