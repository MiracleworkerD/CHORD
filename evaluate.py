from __future__ import annotations

from pathlib import Path

from .config import cfg
from .data import build_domain_loaders
from .model import RelationalHallucinationDetector
from .train_meta import evaluate_all
from .utils import load_checkpoint, resolve_device


def main():
    device = resolve_device(cfg.train.device)
    loaders = build_domain_loaders(cfg.data, cfg.train)
    model = RelationalHallucinationDetector(cfg.data, cfg.model).to(device)
    load_checkpoint(Path(cfg.train.output_dir) / cfg.train.checkpoint_name, model, map_location=device)
    results = evaluate_all(model, loaders, device, cfg.train.precision)
    for name, metrics in results.items():
        print(name, metrics)


if __name__ == "__main__":
    main()
