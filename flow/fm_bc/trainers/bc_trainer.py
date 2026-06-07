"""Flow matching BC trainer with action-chunk masking."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from fm_bc.datasets.pointmaze import PointMazeStats, build_pointmaze_dataset
from fm_bc.eval.plot import plot_policy_rollouts
from fm_bc.eval.rollout import rollout_policy
from fm_bc.models.flow_policy import FlowPolicy, FlowPolicyConfig, flow_matching_loss
from fm_bc.utils.checkpoint import checkpoint_payload, save_checkpoint
from fm_bc.utils.device import resolve_device
from fm_bc.utils.wandb_utils import init_wandb


class BCTrainer:
    def __init__(self, args):
        self.args = args
        self.device = resolve_device(use_cpu=args.cpu)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        self.dataset, self.stats, self.n_samples = build_pointmaze_dataset(
            dataset_id=args.dataset_id,
            ac_chunk=args.ac_chunk,
            download=True,
            data_fraction=args.data_fraction,
            seed=args.seed,
        )
        self.loader = DataLoader(
            self.dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            drop_last=True,
        )

        self.cfg = FlowPolicyConfig(
            action_dim=2,
            ac_chunk=args.ac_chunk,
            obs_dim=4,
            goal_dim=2,
            mode=args.mode,
            hidden_dim=args.hidden_dim,
            n_layers=args.n_layers,
            time_emb_dim=args.time_emb_dim,
        )
        self.model = FlowPolicy(self.cfg).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)

        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")
        self.wandb = None
        self._eval_env = None

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(args.checkpoint_dir, "rollouts"), exist_ok=True)

        if args.resume:
            self._load_checkpoint(args.resume)

    def _get_eval_env(self):
        if self._eval_env is None:
            import minari

            dataset = minari.load_dataset(self.args.dataset_id, download=False)
            self._eval_env = dataset.recover_environment()
        return self._eval_env

    def _load_checkpoint(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(payload["model_state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        self.epoch = payload.get("epoch", 0)
        self.global_step = payload.get("global_step", 0)
        self.best_loss = payload.get("best_loss", float("inf"))
        self.stats = PointMazeStats.from_dict(payload["stats"])
        print(f"Resumed from {path} at epoch={self.epoch}, step={self.global_step}")

    def _maybe_init_wandb(self) -> None:
        if self.args.no_wandb:
            return
        fraction_tag = (
            f"_{self.args.data_fraction:g}" if self.args.data_fraction < 1.0 else ""
        )
        run_name = (
            self.args.wandb_run_name
            or f"{self.args.mode}_ac{self.args.ac_chunk}{fraction_tag}_seed{self.args.seed}"
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
        if self.cfg.mode == "goal_conditioned":
            return batch["goal"].to(self.device)
        return None

    @torch.no_grad()
    def evaluate_policy(self) -> dict[str, float]:
        env = self._get_eval_env()
        rollouts, metrics = rollout_policy(
            self.model,
            env,
            self.stats,
            self.cfg,
            n_episodes=self.args.eval_rollouts,
            seed=self.args.seed + self.global_step,
            sample_steps=self.args.sample_steps,
            device=self.device,
            episodic_eval=True,
        )

        plot_path = os.path.join(
            self.args.checkpoint_dir,
            "rollouts",
            f"step_{self.global_step:06d}.png",
        )
        plot_policy_rollouts(
            rollouts,
            dataset_id=self.args.dataset_id,
            title=f"BC Policy Rollouts ({self.cfg.mode})",
            save_path=plot_path,
            show=False,
        )

        print(
            f"  eval step={self.global_step} "
            f"success={metrics['eval/success_rate']:.3f} "
            f"return={metrics['eval/mean_return']:.3f} "
            f"plot={plot_path}"
        )

        if self.wandb is not None:
            import wandb

            log_dict = dict(metrics)
            log_dict["eval/rollout_plot"] = wandb.Image(plot_path)
            self.wandb.log(log_dict, step=self.global_step)

        return metrics

    def _maybe_evaluate(self) -> None:
        if self.args.eval_every <= 0:
            return
        if self.global_step == 0 or self.global_step % self.args.eval_every != 0:
            return

        was_training = self.model.training
        self.model.eval()
        self.evaluate_policy()
        if was_training:
            self.model.train()

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.loader:
            obs = batch["obs"].to(self.device)
            actions = batch["action"].to(self.device)
            mask = batch["mask"].to(self.device)
            goal = self._batch_goal(batch)

            loss = flow_matching_loss(self.model, actions, mask, obs=obs, goal=goal)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_val = float(loss.item())
            total_loss += loss_val
            n_batches += 1
            self.global_step += 1

            if self.wandb is not None and self.global_step % self.args.log_interval == 0:
                self.wandb.log(
                    {"train/batch_loss": loss_val, "epoch": self.epoch + 1},
                    step=self.global_step,
                )

            self._maybe_evaluate()

        return total_loss / max(n_batches, 1)

    def _save(self, epoch_loss: float, is_best: bool) -> None:
        payload = checkpoint_payload(
            self.model,
            self.cfg,
            self.stats,
            epoch=self.epoch,
            global_step=self.global_step,
            train_loss=epoch_loss,
            best_loss=self.best_loss,
            args=vars(self.args),
        )
        payload["optimizer_state_dict"] = self.optimizer.state_dict()

        latest_path = os.path.join(self.args.checkpoint_dir, "latest.pt")
        save_checkpoint(latest_path, payload)

        if is_best:
            save_checkpoint(os.path.join(self.args.checkpoint_dir, "best.pt"), payload)

        if self.args.save_every > 0 and (
            self.epoch % self.args.save_every == 0 or self.epoch == self.args.epochs
        ):
            save_checkpoint(
                os.path.join(self.args.checkpoint_dir, f"epoch_{self.epoch:04d}.pt"),
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

        print(f"Dataset:        {self.args.dataset_id}")
        print(f"Mode:           {self.args.mode}")
        print(f"Action chunk:   {self.args.ac_chunk}")
        print(f"Data fraction:  {self.args.data_fraction}")
        print(f"Samples:        {self.n_samples}")
        print(f"Device:         {self.device}")
        print(f"Eval every:     {self.args.eval_every} steps")
        print(f"Eval rollouts:  {self.args.eval_rollouts}")
        print(f"Checkpoint dir: {self.args.checkpoint_dir}")

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.args.epochs):
            self.epoch = epoch + 1
            epoch_loss = self.train_epoch()
            print(f"epoch {self.epoch:4d}  loss={epoch_loss:.6f}")

            is_best = epoch_loss < self.best_loss
            if is_best:
                self.best_loss = epoch_loss

            if self.wandb is not None:
                self.wandb.log(
                    {"train/epoch_loss": epoch_loss, "epoch": self.epoch},
                    step=self.global_step,
                )

            self._save(epoch_loss, is_best)

        if self._eval_env is not None:
            self._eval_env.close()

        if self.wandb is not None:
            self.wandb.finish()

        print(f"Done. best_loss={self.best_loss:.6f}")
