"""
Simple graph neural network
"""

from typing import Dict

def get_config() -> Dict:

    config = {
        "name": "Simple Graph",
        "description": "4-node simple graph",
        "task": "node_classification",
        "num_nodes": 4,
        "num_edges": 8,
        "feature_dim": 3,
        "model": "GCN",
        "epochs": 4,
    }
    return config

def run(config: Dict):
    raise NotImplementedError("run method not implemented yet")

class SimpleGNN():
    def __init__(self) -> None:
        pass
