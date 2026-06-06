"""Checkpoint save/load helpers."""

from __future__ import annotations

import os
from typing import Any

import torch

from fm_bc.datasets.pointmaze import PointMazeStats
from fm_bc.models.flow_policy import FlowPolicy, FlowPolicyConfig


def checkpoint_payload(
    model: FlowPolicy,
    cfg: FlowPolicyConfig,
    stats: PointMazeStats,
    *,
    epoch: int,
    global_step: int,
    train_loss: float,
    best_loss: float,
    args: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "stats": stats.to_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "train_loss": train_loss,
        "best_loss": best_loss,
        "args": args,
    }


def save_checkpoint(path: str, payload: dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    return path


def load_checkpoint(
    path: str,
    *,
    device: torch.device,
) -> tuple[FlowPolicy, FlowPolicyConfig, PointMazeStats, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg: FlowPolicyConfig = payload["config"]
    stats = PointMazeStats.from_dict(payload["stats"])
    model = FlowPolicy(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    return model, cfg, stats, payload
