"""Trainer for residual flow matching on base-policy action differences."""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from fm_bc.datasets.pointmaze import PointMazeStats, build_pointmaze_dataset
from fm_bc.models.flow_policy import FlowPolicy
from fm_bc.utils.checkpoint import load_checkpoint
from fm_bc.utils.device import resolve_device
from fm_bc.utils.wandb_utils import init_wandb
from models.residual_flow_policy import (
    ResidualFlowConfig,
    ResidualFlowPolicy,
    residual_flow_matching_loss,
)


def build_pair_indices(k: int) -> tuple[torch.Tensor, torch.Tensor]:
    pairs_i: list[int] = []
    pairs_j: list[int] = []
    for i in range(k):
        for j in range(k):
            if i != j:
                pairs_i.append(i)
                pairs_j.append(j)
    return torch.tensor(pairs_i, dtype=torch.long), torch.tensor(pairs_j, dtype=torch.long)


def residual_checkpoint_payload(
    model: ResidualFlowPolicy,
    cfg: ResidualFlowConfig,
    stats: PointMazeStats,
    *,
    global_step: int,
    train_loss: float,
    best_loss: float,
    args: dict[str, Any],
    base_checkpoint: str,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "stats": stats.to_dict(),
        "global_step": global_step,
        "train_loss": train_loss,
        "best_loss": best_loss,
        "args": args,
        "base_checkpoint": base_checkpoint,
    }


def save_residual_checkpoint(path: str, payload: dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(payload, path)
    return path


class ResidualFlowTrainer:
    def __init__(self, args):
        self.args = args
        self.device = resolve_device(use_cpu=args.cpu)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.base_policy, self.base_cfg, self.stats, _ = load_checkpoint(
            args.base_checkpoint,
            device=self.device,
        )
        self.base_policy.eval()
        for param in self.base_policy.parameters():
            param.requires_grad_(False)

        self.dataset, _, self.n_samples = build_pointmaze_dataset(
            dataset_id=args.dataset_id,
            ac_chunk=self.base_cfg.ac_chunk,
            download=True,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )
        self.loader_iter = iter(self.loader)

        self.cfg = ResidualFlowConfig(
            action_dim=self.base_cfg.action_dim,
            ac_chunk=self.base_cfg.ac_chunk,
            obs_dim=self.base_cfg.obs_dim,
            goal_dim=self.base_cfg.goal_dim,
            mode=args.mode,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            time_emb_dim=args.time_emb_dim,
        )
        self.model = ResidualFlowPolicy(self.cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        self.pair_i, self.pair_j = build_pair_indices(args.k)
        self.n_pairs = len(self.pair_i)

        self.global_step = 0
        self.best_loss = float("inf")
        self.wandb = None

        os.makedirs(args.checkpoint_dir, exist_ok=True)

        if args.resume:
            self._load_checkpoint(args.resume)

    def _load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.global_step = payload.get("global_step", 0)
        self.best_loss = payload.get("best_loss", float("inf"))
        self.stats = PointMazeStats.from_dict(payload["stats"])
        print(f"Resumed from {path} at step={self.global_step}")

    def _maybe_init_wandb(self) -> None:
        if self.args.no_wandb:
            return
        run_name = (
            self.args.wandb_run_name
            or f"{self.args.mode}_k{self.args.k}_seed{self.args.seed}"
        )
        config = vars(self.args).copy()
        config.update(
            {
                "n_samples": self.n_samples,
                "cond_dim": self.cfg.cond_dim,
                "action_dim": self.cfg.action_dim,
                "ac_chunk": self.cfg.ac_chunk,
                "flow_dim": self.cfg.flow_dim,
                "obs_dim": self.cfg.obs_dim,
                "n_pairs_per_obs": self.n_pairs,
                "effective_batch_size": self.args.batch_size * self.n_pairs,
                "base_mode": self.base_cfg.mode,
            }
        )
        self.wandb = init_wandb(
            project=self.args.wandb_project,
            entity=self.args.wandb_entity,
            name=run_name,
            group=self.args.wandb_group,
            config=config,
            run_id=self.args.wandb_run_id,
            resume=bool(self.args.resume),
        )

    def _batch_goal(self, batch: dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if self.base_cfg.mode == "goal_conditioned":
            return batch["goal"].to(self.device)
        return None

    def _next_batch(self) -> dict[str, torch.Tensor]:
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.loader_iter = iter(self.loader)
            batch = next(self.loader_iter)
        return batch

    @torch.no_grad()
    def _sample_base_actions(
        self,
        obs: torch.Tensor,
        goal: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = obs.shape[0]
        k = self.args.k
        obs_rep = obs.repeat_interleave(k, dim=0)
        goal_rep = goal.repeat_interleave(k, dim=0) if goal is not None else None

        actions = self.base_policy.sample(
            obs_rep,
            goal=goal_rep,
            n_steps=self.args.base_sample_steps,
        )
        flow_dim = self.cfg.flow_dim
        return actions.reshape(batch_size, k, flow_dim)

    def train_step(self, batch: dict[str, torch.Tensor]) -> float:
        obs = batch["obs"].to(self.device)
        goal = self._batch_goal(batch)
        batch_size = obs.shape[0]

        actions_k = self._sample_base_actions(obs, goal)

        pair_i = self.pair_i.to(self.device)
        pair_j = self.pair_j.to(self.device)

        anchor = actions_k[:, pair_i, :]
        other = actions_k[:, pair_j, :]

        anchor_flat = anchor.reshape(batch_size * self.n_pairs, self.cfg.flow_dim)
        other_flat = other.reshape(batch_size * self.n_pairs, self.cfg.flow_dim)
        obs_flat = (
            obs[:, None, :]
            .expand(batch_size, self.n_pairs, -1)
            .reshape(batch_size * self.n_pairs, self.cfg.obs_dim)
        )

        if self.cfg.mode in ("unconditioned", "obs_conditioned"):
            diff_target = anchor_flat - other_flat
            anchor_for_cond = None
        else:
            diff_target = other_flat - anchor_flat
            anchor_for_cond = anchor_flat

        self.model.train()
        loss = residual_flow_matching_loss(
            self.model,
            diff_target,
            obs=obs_flat,
            anchor_action=anchor_for_cond,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())

    def _save(self, step_loss: float, is_best: bool) -> None:
        payload = residual_checkpoint_payload(
            self.model,
            self.cfg,
            self.stats,
            global_step=self.global_step,
            train_loss=step_loss,
            best_loss=self.best_loss,
            args=vars(self.args),
            base_checkpoint=self.args.base_checkpoint,
        )
        payload["optimizer_state_dict"] = self.optimizer.state_dict()

        latest_path = os.path.join(self.args.checkpoint_dir, "latest.pt")
        save_residual_checkpoint(latest_path, payload)

        if is_best:
            save_residual_checkpoint(
                os.path.join(self.args.checkpoint_dir, "best.pt"),
                payload,
            )

        if self.args.save_every > 0 and self.global_step % self.args.save_every == 0:
            save_residual_checkpoint(
                os.path.join(
                    self.args.checkpoint_dir,
                    f"step_{self.global_step:07d}.pt",
                ),
                payload,
            )

        if self.wandb is not None:
            self.wandb.save(latest_path, base_path=self.args.checkpoint_dir)
            if is_best:
                self.wandb.save(
                    os.path.join(self.args.checkpoint_dir, "best.pt"),
                    base_path=self.args.checkpoint_dir,
                )

    def train(self) -> None:
        self._maybe_init_wandb()

        print(f"Base checkpoint: {self.args.base_checkpoint}")
        print(f"Base mode:       {self.base_cfg.mode}")
        print(f"Residual mode:   {self.args.mode}")
        print(f"Dataset:         {self.args.dataset_id}")
        print(f"Action chunk:    {self.cfg.ac_chunk}")
        print(f"K samples:       {self.args.k}")
        print(f"Pairs per obs:   {self.n_pairs}")
        print(f"Samples:         {self.n_samples}")
        print(f"Device:          {self.device}")
        print(f"Training steps:  {self.args.n_steps}")
        print(f"Checkpoint dir:  {self.args.checkpoint_dir}")

        running_loss = 0.0
        log_count = 0

        while self.global_step < self.args.n_steps:
            batch = self._next_batch()
            loss = self.train_step(batch)
            self.global_step += 1

            running_loss += loss
            log_count += 1

            if self.global_step % self.args.log_interval == 0:
                avg_loss = running_loss / log_count
                print(f"step {self.global_step:7d}  loss={avg_loss:.6f}")
                if self.wandb is not None:
                    self.wandb.log(
                        {"train/batch_loss": avg_loss},
                        step=self.global_step,
                    )
                running_loss = 0.0
                log_count = 0

            is_best = loss < self.best_loss
            if is_best:
                self.best_loss = loss

            if self.args.save_every > 0 and self.global_step % self.args.save_every == 0:
                self._save(loss, is_best)

        final_loss = running_loss / max(log_count, 1)
        self._save(final_loss, final_loss < self.best_loss)

        if self.wandb is not None:
            self.wandb.finish()

        print(f"Done. best_loss={self.best_loss:.6f}")
