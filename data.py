from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset

from .config import DataConfig, TrainConfig


def _natural_int_key(name: str) -> int:
    hits = re.findall(r"\d+", name)
    return int(hits[-1]) if hits else -1


def _read_bf16_h5_dataset(ds) -> torch.Tensor:
    arr = ds[()]
    if isinstance(arr, bytes):
        raw = np.frombuffer(arr, dtype=np.int16)
    else:
        arr = np.asarray(arr)
        if arr.dtype == np.uint8:
            raw = np.frombuffer(arr.tobytes(), dtype=np.int16)
        elif arr.dtype == np.int16:
            raw = arr.reshape(-1)
        else:
            return torch.as_tensor(arr)
    return torch.from_numpy(raw.copy()).view(torch.bfloat16)


def _label_to_float(category: str, cfg: DataConfig) -> float | None:
    if category == cfg.positive_label:
        return 1.0
    if category in cfg.negative_labels:
        return 0.0
    return None


@dataclass(frozen=True)
class SampleIndex:
    sample_id: str
    label: float


class HiddenStateDomainDataset(Dataset):
    def __init__(self, domain_name: str, data_dir: str | os.PathLike, cfg: DataConfig):
        self.domain_name = domain_name
        self.data_dir = Path(data_dir)
        self.cfg = cfg
        self.h5_path = self.data_dir / cfg.hidden_filename
        self.metadata_path = self.data_dir / cfg.metadata_filename
        self.samples = self._read_index()
        self.cached_features: List[torch.Tensor] | None = None
        self.cached_labels: torch.Tensor | None = None
        if cfg.cache_in_memory:
            xs, ys = [], []
            for idx in range(len(self.samples)):
                x, y, _ = self._read_item(idx)
                xs.append(x)
                ys.append(y)
            self.cached_features = xs
            self.cached_labels = torch.tensor(ys, dtype=torch.float32).view(-1, 1)

    def _read_index(self) -> List[SampleIndex]:
        if not self.h5_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(f"{self.domain_name}: missing {self.h5_path} or {self.metadata_path}")
        out = []
        with h5py.File(self.h5_path, "r") as f_h5, open(self.metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                sample_id = str(item[self.cfg.sample_id_key])
                label = _label_to_float(item.get(self.cfg.label_key), self.cfg)
                if label is None or sample_id not in f_h5:
                    continue
                out.append(SampleIndex(sample_id, label))
        if not out:
            raise RuntimeError(f"{self.domain_name}: no valid samples found")
        return out

    def _select_last_tokens_layers(self, sample_tensor: torch.Tensor) -> torch.Tensor:
        t_total, l_total, h = sample_tensor.shape
        if l_total >= self.cfg.last_l_layers:
            x = sample_tensor[:, -self.cfg.last_l_layers:, :]
        else:
            pad = torch.zeros(t_total, self.cfg.last_l_layers - l_total, h, dtype=sample_tensor.dtype)
            x = torch.cat([pad, sample_tensor], dim=1)
        if t_total >= self.cfg.last_k_tokens:
            x = x[-self.cfg.last_k_tokens:, :, :]
        else:
            pad = torch.zeros(self.cfg.last_k_tokens - t_total, self.cfg.last_l_layers, h, dtype=x.dtype)
            x = torch.cat([pad, x], dim=0)
        return x.contiguous()

    def _read_item(self, idx: int):
        rec = self.samples[idx]
        with h5py.File(self.h5_path, "r") as f_h5:
            token_group = f_h5[rec.sample_id][self.cfg.token_group_name]
            token_tensors = []
            for tok_key in sorted(token_group.keys(), key=_natural_int_key):
                layer_group = token_group[tok_key]
                layer_keys = [k for k in layer_group.keys() if k.startswith(self.cfg.layer_prefix)]
                layer_keys = sorted(layer_keys, key=_natural_int_key)
                layers = []
                for layer_key in layer_keys:
                    layers.append(_read_bf16_h5_dataset(layer_group[layer_key]))
                if layers:
                    token_tensors.append(torch.stack(layers, dim=0))
            if not token_tensors:
                raise RuntimeError(f"{self.domain_name}: empty token tensor for sample {rec.sample_id}")
            x = torch.stack(token_tensors, dim=0)
            x = self._select_last_tokens_layers(x)
            return x, rec.label, rec.sample_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if self.cached_features is not None and self.cached_labels is not None:
            return {
                "x": self.cached_features[idx],
                "y": self.cached_labels[idx],
                "domain": self.domain_name,
                "sample_id": self.samples[idx].sample_id,
            }
        x, y, sample_id = self._read_item(idx)
        return {
            "x": x,
            "y": torch.tensor([y], dtype=torch.float32),
            "domain": self.domain_name,
            "sample_id": sample_id,
        }


def collate_domain_batch(batch):
    xs = torch.stack([b["x"] for b in batch], dim=0)
    ys = torch.stack([b["y"] for b in batch], dim=0).float()
    return {
        "x": xs,
        "y": ys,
        "domain": batch[0]["domain"],
        "sample_id": [b["sample_id"] for b in batch],
    }


def build_domain_loaders(data_cfg: DataConfig, train_cfg: TrainConfig):
    loaders = {}
    for name, path in data_cfg.source_llm_dirs.items():
        dataset = HiddenStateDomainDataset(name, path, data_cfg)
        labels = np.array([s.label for s in dataset.samples])
        idx = np.arange(len(dataset))
        stratify = labels if len(np.unique(labels)) == 2 else None
        train_idx, val_idx = train_test_split(
            idx,
            test_size=data_cfg.val_ratio,
            random_state=train_cfg.seed,
            stratify=stratify,
        )
        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())
        loaders[name] = {
            "train": DataLoader(
                train_subset,
                batch_size=train_cfg.batch_size,
                shuffle=True,
                num_workers=data_cfg.num_workers,
                pin_memory=data_cfg.pin_memory,
                collate_fn=collate_domain_batch,
                drop_last=True,
            ),
            "val": DataLoader(
                val_subset,
                batch_size=train_cfg.eval_batch_size,
                shuffle=False,
                num_workers=data_cfg.num_workers,
                pin_memory=data_cfg.pin_memory,
                collate_fn=collate_domain_batch,
                drop_last=False,
            ),
            "num_train": len(train_subset),
            "num_val": len(val_subset),
        }
    return loaders
