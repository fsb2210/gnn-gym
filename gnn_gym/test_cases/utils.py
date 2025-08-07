
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.optim as optim

from gnn_gym.datasets import karateclub

def load_dataset(dataset_name: str) -> Dict:
    if dataset_name == "karate":
        ds = karateclub()
    else:
        raise ValueError("only dataset available is KarateClub")
    return ds

def create_optimizer(model_params, optim_config):
    """
    Create a PyTorch optimizer from config

    Supports: Adam, SGD, etc.
    """

    optim_type = optim_config["type"]
    args = {k: v for k, v in optim_config.items() if k != "type"}

    if optim_type.lower() == "adam":
        return torch.optim.Adam(model_params, **args)
    elif optim_type.lwer() == "sgd":
        return torch.optim.SGD(model_params, **args)
    elif optim_type.lower() == "adamw":
        return torch.optim.AdamW(model_params, **args)
    else:
        raise ValueError(f"unsupported optimizer: {optim_type}")

def create_criterion(criterion_config):
    """
    Create a PyTorch loss function from config
    """

    crit_type = criterion_config["type"]
    args = {k: v for k, v in criterion_config.items() if k != "type"}

    if crit_type.lower() == "crossentropyloss":
        return nn.CrossEntropyLoss(**args)
    elif crit_type.lower() == "mseloss":
        return nn.MSELoss(**args)
    elif crit_type.lower() == "bcewithlogitsloss":
        return nn.BCEWithLogitsLoss(**args)
    else:
        raise ValueError(f"unsupported criterion: {crit_type}")

def accuracy(pred, y):
    return (pred == y).sum()/len(y)

def train(ds: Dict, model: nn.Module, epochs: int, verbose: bool, optimizer: optim.Optimizer, loss_fn: Callable) -> None:

    losses, accuracies, outputs = [], [], []
    for epoch in range(epochs):
        # clear grads
        optimizer.zero_grad()

        # forward pass
        z = model(x=ds.get("x"), edge_index=ds.get("edge_index"))

        # loss function
        loss = loss_fn(z, ds.get("y"))

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

        if verbose:
            print(f"Epoch {epoch+1:>3} | loss: {loss:.2f} | acc: {acc*100:.2f}%")
