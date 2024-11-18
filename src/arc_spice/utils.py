import json
import os
import random
from datetime import datetime

import numpy as np
import torch
import yaml


def flatten(xss):
    """
    flattens a list

    Args:
        xss: list of nested lists

    Returns:
        flattened list
    """
    return [x for xs in xss for x in xs]


def open_json_path(path: str) -> dict:
    with open(path) as file:
        return json.load(file)


def open_yaml_path(path: str) -> dict:
    with open(path) as file:
        return yaml.safe_load(file)


def seed_everything(seed: int) -> None:
    """Set random seeds for torch, numpy, random, and python.

    Args:
        seed: Seed to set.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_datetime_str() -> str:
    """
    Returns:
        The current datetime as a string in the format %Y%m%d-%H%M%S-%f, e.g.
        20240528-105332-123456 (yearmonthday-hourminutesseconds-milliseconds).
    """
    return datetime.strftime(datetime.now(), "%Y%m%d-%H%M%S-%f")


def get_device() -> torch.device:
    """Gets the best available device for pytorch to use.
    (According to: gpu -> mps -> cpu) Currently only works for one GPU.

    Returns:
        torch.device: available torch device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
