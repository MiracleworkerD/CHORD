from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"bf16", "fp16"}
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def cycle_loader(loader: Iterable):
    while True:
        for batch in loader:
            yield batch


def save_checkpoint(path: str | Path, model: torch.nn.Module, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model": model.state_dict(), **payload}
    torch.save(state, path)


def load_checkpoint(path: str | Path, model: torch.nn.Module, map_location="cpu") -> dict:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state["model"])
    return {k: v for k, v in state.items() if k != "model"}
