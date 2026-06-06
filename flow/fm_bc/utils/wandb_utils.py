"""Weights & Biases helpers."""

from __future__ import annotations

from typing import Any, Optional


def init_wandb(
    *,
    project: str,
    name: str,
    config: dict[str, Any],
    entity: Optional[str] = None,
    group: Optional[str] = None,
    run_id: Optional[str] = None,
    resume: bool = False,
):
    import wandb

    init_kwargs: dict[str, Any] = {
        "project": project,
        "name": name,
        "config": config,
    }
    if entity is not None:
        init_kwargs["entity"] = entity
    if group is not None:
        init_kwargs["group"] = group
    if resume and run_id is not None:
        init_kwargs["id"] = run_id
        init_kwargs["resume"] = "allow"

    return wandb.init(**init_kwargs)
