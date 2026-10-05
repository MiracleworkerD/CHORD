from __future__ import annotations

import copy
import random
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim

from .config import Config
from .metrics import binary_metrics
from .utils import autocast_context, cycle_loader, resolve_device, save_checkpoint, set_seed


def smooth_targets(y: torch.Tensor, smoothing: float) -> torch.Tensor:
    if smoothing <= 0:
        return y
    return y * (1.0 - smoothing) + 0.5 * smoothing


def batch_loss(model: nn.Module, batch: dict, criterion, device: torch.device, precision: str, smoothing: float = 0.0):
    x = batch["x"].to(device, non_blocking=True).float()
    y = batch["y"].to(device, non_blocking=True).float()
    with autocast_context(device, precision):
        logits = model(x)
        loss = criterion(logits.float(), smooth_targets(y, smoothing))
    return loss, logits.float(), y


def evaluate_domain(model: nn.Module, loader, device: torch.device, precision: str) -> dict:
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    losses, logits_all, y_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            loss, logits, y = batch_loss(model, batch, criterion, device, precision, 0.0)
            losses.append(loss.detach().float().cpu())
            logits_all.append(logits.detach().cpu())
            y_all.append(y.detach().cpu())
    metrics = binary_metrics(torch.cat(logits_all, dim=0), torch.cat(y_all, dim=0))
    metrics["loss"] = float(torch.stack(losses).mean())
    return metrics


def evaluate_all(model: nn.Module, loaders: dict, device: torch.device, precision: str) -> dict:
    return {name: evaluate_domain(model, pack["val"], device, precision) for name, pack in loaders.items()}


def choose_episode_domains(domain_names: List[str], min_train: int, max_train: int):
    if len(domain_names) < min_train + 1:
        raise ValueError(f"meta-learning needs at least {min_train + 1} source LLM domains")
    heldout = random.choice(domain_names)
    pool = [x for x in domain_names if x != heldout]
    k = min(len(pool), random.randint(min_train, max(min_train, min(max_train, len(pool)))))
    train_domains = random.sample(pool, k)
    return train_domains, heldout


def train_inner_model(inner_model, train_domains, train_iters, cfg: Config, device: torch.device):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        inner_model.parameters(),
        lr=cfg.train.inner_lr,
        weight_decay=cfg.train.weight_decay,
        eps=1e-5,
    )
    inner_model.train()
    running = []
    for _ in range(cfg.train.inner_steps):
        for domain in train_domains:
            batch = next(train_iters[domain])
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = batch_loss(
                inner_model,
                batch,
                criterion,
                device,
                cfg.train.precision,
                cfg.train.label_smoothing,
            )
            loss.backward()
            if cfg.train.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(inner_model.parameters(), cfg.train.grad_clip_norm)
            optimizer.step()
            running.append(loss.detach().float().cpu())
    return float(torch.stack(running).mean()) if running else 0.0


def reptile_update(model: nn.Module, inner_model: nn.Module, meta_lr: float):
    with torch.no_grad():
        for p, q in zip(model.parameters(), inner_model.parameters()):
            if p.requires_grad:
                p.add_(q.detach() - p, alpha=meta_lr)


def train_meta(model: nn.Module, loaders: dict, cfg: Config):
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    model.to(device)
    domain_names = list(loaders.keys())
    train_iters = {name: cycle_loader(pack["train"]) for name, pack in loaders.items()}
    best_mean_auc = -1.0
    history = []
    criterion = nn.BCEWithLogitsLoss()

    for episode in range(1, cfg.train.episodes + 1):
        train_domains, heldout = choose_episode_domains(
            domain_names,
            cfg.train.min_meta_train_domains,
            cfg.train.max_meta_train_domains,
        )
        inner_model = copy.deepcopy(model).to(device)
        inner_loss = train_inner_model(inner_model, train_domains, train_iters, cfg, device)

        heldout_batch = next(train_iters[heldout])
        heldout_loss, heldout_logits, heldout_y = batch_loss(
            inner_model,
            heldout_batch,
            criterion,
            device,
            cfg.train.precision,
            0.0,
        )
        heldout_metrics = binary_metrics(heldout_logits.detach(), heldout_y.detach())
        reptile_update(model, inner_model, cfg.train.meta_lr)

        record = {
            "episode": episode,
            "heldout": heldout,
            "train_domains": train_domains,
            "inner_loss": inner_loss,
            "heldout_loss": float(heldout_loss.detach().float().cpu()),
            **{f"heldout_{k}": v for k, v in heldout_metrics.items()},
        }
        history.append(record)

        if episode % cfg.train.eval_every == 0:
            evals = evaluate_all(model, loaders, device, cfg.train.precision)
            mean_auc = sum(v["auc"] for v in evals.values()) / len(evals)
            line = f"Ep {episode:05d} | meta_train={','.join(train_domains)} | heldout={heldout} | hold_auc={record['heldout_auc']:.4f} | val_mean_auc={mean_auc:.4f}"
            print(line, flush=True)
            if mean_auc > best_mean_auc:
                best_mean_auc = mean_auc
                save_checkpoint(
                    Path(cfg.train.output_dir) / cfg.train.checkpoint_name,
                    model,
                    {"episode": episode, "best_mean_auc": best_mean_auc, "evals": evals, "config": cfg},
                )

        if episode % cfg.train.save_every == 0:
            save_checkpoint(
                Path(cfg.train.output_dir) / f"episode_{episode}.pt",
                model,
                {"episode": episode, "history": history[-cfg.train.save_every :], "config": cfg},
            )

    return history
