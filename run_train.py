from __future__ import annotations

from .config import cfg
from .data import build_domain_loaders
from .model import RelationalHallucinationDetector
from .train_meta import train_meta
from .utils import set_seed


def main():
    set_seed(cfg.train.seed)
    loaders = build_domain_loaders(cfg.data, cfg.train)
    if len(loaders) < cfg.train.min_meta_train_domains + 1:
        raise RuntimeError(
            f"Need at least {cfg.train.min_meta_train_domains + 1} source LLMs, got {len(loaders)}"
        )
    model = RelationalHallucinationDetector(cfg.data, cfg.model)
    train_meta(model, loaders, cfg)


if __name__ == "__main__":
    main()
