"""Training utilities for residual RL on frozen flow policies."""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from stable_baselines3.common.callbacks import BaseCallback

from eval_viz import collect_eval_rollouts, plot_eval_paths, plot_eval_rollouts


class WandbCallback(BaseCallback):
    def __init__(
        self,
        *,
        log_freq: int = 1000,
        use_wandb: bool = True,
        eval_env: Any = None,
        max_steps: int = 1000,
        eval_episodes: int = 20,
        eval_plot_trajectories: int = 5,
        eval_plot_dir: str | None = None,
        dataset_id: str = "D4RL/pointmaze/large-dense-v2",
        eval_interval: int = 20000,
        plot_arrow_stride: int = 20,
        arrow_length_scale: float = 0.5,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self.use_wandb = use_wandb
        self.eval_env = eval_env
        self.max_steps = max_steps
        self.eval_episodes = eval_episodes
        self.eval_plot_trajectories = eval_plot_trajectories
        self.eval_plot_dir = eval_plot_dir
        self.dataset_id = dataset_id
        self.eval_interval = eval_interval
        self.plot_arrow_stride = plot_arrow_stride
        self.arrow_length_scale = arrow_length_scale

        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.success_rate = 0.0
        self.success_count = 0
        self.timesteps = 0

    def _on_training_start(self) -> None:
        if self.eval_env is None or self.eval_episodes <= 0:
            return
        saved_timesteps = self.timesteps
        self.timesteps = 0
        self.evaluate(self.model, metric_key="eval/success_rate")
        self.timesteps = saved_timesteps

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for env_idx, info in enumerate(infos):
            if "episode" not in info:
                continue
            self.episode_lengths.append(int(info["episode"]["l"]))
            self.episode_rewards.append(float(info["episode"]["r"]))
            self.success_count += 1
            episode_success = bool(info.get("success", rewards[env_idx] >= 0.0))
            self.success_rate += float(episode_success)

        self.timesteps += 1

        if self.n_calls % self.log_freq == 0 and self.success_count > 0:
            if self.use_wandb:
                log_dict = {
                    "train/ep_len_mean": float(np.mean(self.episode_lengths)),
                    "train/success_rate": self.success_rate / self.success_count,
                    "train/ep_rew_mean": float(np.mean(self.episode_rewards)),
                    "train/timesteps": self.num_timesteps,
                }
                logger = self.locals["self"].logger.name_to_value
                for key in ("train/ent_coef", "train/actor_loss", "train/critic_loss", "train/ent_coef_loss"):
                    if key in logger:
                        log_dict[key] = logger[key]
                wandb.log(log_dict, step=self.timesteps)

            self.episode_rewards = []
            self.episode_lengths = []
            self.success_rate = 0.0
            self.success_count = 0

        if self.eval_env is not None and self.timesteps > 0 and self.timesteps % self.eval_interval == 0:
            self.evaluate(self.locals["self"])
        return True

    def set_initial_timesteps(self, timesteps: int) -> None:
        self.timesteps = timesteps

    def evaluate(self, agent: Any, metric_key: str = "eval/success_rate") -> float:
        if self.eval_env is None or self.eval_episodes <= 0:
            return 0.0

        with torch.no_grad():
            rollouts = collect_eval_rollouts(
                self.eval_env,
                agent,
                self.eval_episodes,
                self.max_steps,
                deterministic=False,
                print_action_stride=self.plot_arrow_stride,
            )

        successes = [float(r["success"]) for r in rollouts]
        success_rate = float(np.sum(successes) / max(len(rollouts), 1))
        print(
            f"{metric_key}: {success_rate:.3f} "
            f"({int(sum(successes))}/{len(rollouts)} successes)"
        )

        wandb_payload: dict[str, Any] = {metric_key: success_rate}

        paths_fig = plot_eval_paths(
            rollouts,
            dataset_id=self.dataset_id,
            title=f"Eval paths @ step {self.timesteps}",
            save_path=None,
            show=False,
            close_fig=False,
        )
        if self.use_wandb:
            wandb_payload["eval/paths"] = wandb.Image(
                paths_fig,
                caption=f"All trajectories @ step {self.timesteps}",
            )

        if self.eval_plot_trajectories > 0:
            plot_rollouts = rollouts[: self.eval_plot_trajectories]
            arrows_fig = plot_eval_rollouts(
                plot_rollouts,
                dataset_id=self.dataset_id,
                title=f"Eval action arrows @ step {self.timesteps}",
                save_path=None,
                show=False,
                close_fig=False,
                arrow_stride=self.plot_arrow_stride,
                arrow_length_scale=self.arrow_length_scale,
            )
            if self.use_wandb:
                wandb_payload["eval/arrows"] = wandb.Image(
                    arrows_fig,
                    caption=f"Action arrows (first {len(plot_rollouts)}) @ step {self.timesteps}",
                )

        if self.eval_plot_dir is not None:
            os.makedirs(self.eval_plot_dir, exist_ok=True)
            step_tag = f"eval_step_{self.timesteps:07d}"
            paths_save_path = os.path.join(self.eval_plot_dir, f"{step_tag}_paths.png")
            arrows_save_path = os.path.join(self.eval_plot_dir, f"{step_tag}_arrows.png")
            paths_fig.savefig(paths_save_path, dpi=150, bbox_inches="tight")
            if self.eval_plot_trajectories > 0:
                arrows_fig.savefig(arrows_save_path, dpi=150, bbox_inches="tight")

        if self.use_wandb:
            wandb.log(wandb_payload, step=self.timesteps)

        plt.close(paths_fig)
        if self.eval_plot_trajectories > 0:
            plt.close(arrows_fig)

        return success_rate


def collect_rollouts(
    model: Any,
    env: Any,
    num_rollouts: int,
    max_steps: int,
    *,
    log_data: bool = False,
) -> tuple[float, list[dict[str, Any]], list[dict[str, Any]]]:
    """Collect initial rollouts via the policy and optionally seed the replay buffer."""
    n_envs = getattr(env, "num_envs", getattr(env, "n_envs", 1))
    n_rounds = int(np.ceil(num_rollouts / n_envs))
    success_transitions: list[dict[str, Any]] = []
    failed_transitions: list[dict[str, Any]] = []
    num_success = 0
    total_trials = 0

    is_vec = hasattr(env, "step_async")

    for round_idx in range(n_rounds):
        print(f"Collecting rollout round {round_idx + 1}/{n_rounds}")
        if is_vec:
            obs = env.reset()
        else:
            obs, _ = env.reset()
        active = np.ones(n_envs, dtype=bool)

        for _ in range(max_steps):
            if not active.any():
                break

            if is_vec:
                next_obs, rewards, dones, infos, action_batch = env.step_rand(
                    obs,
                    model=model,
                    deterministic=False,
                )
                obs_batch = obs
                next_obs_batch = next_obs
            else:
                next_obs, reward, terminated, truncated, info, action_out = env.step_rand(
                    model=model,
                    deterministic=False,
                )
                rewards = np.array([reward], dtype=np.float32)
                dones = np.array([terminated or truncated or reward >= 0.0], dtype=bool)
                infos = [info]
                action_batch = action_out.reshape(1, -1)
                obs_batch = obs[None] if obs.ndim == 1 else obs
                next_obs_batch = next_obs[None] if next_obs.ndim == 1 else next_obs

            if log_data:
                scaled_actions = model.policy.scale_action(
                    np.asarray(action_batch, dtype=np.float32).reshape(n_envs, -1)
                )
                model.replay_buffer.add(
                    obs=obs_batch,
                    next_obs=next_obs_batch,
                    action=scaled_actions,
                    reward=np.asarray(rewards, dtype=np.float32),
                    done=np.asarray(dones, dtype=np.float32),
                    infos=infos,
                )

            for env_idx in range(n_envs):
                if not active[env_idx]:
                    continue

                done = bool(dones[env_idx])
                reward = float(rewards[env_idx])
                info = infos[env_idx]
                action_scaled = model.policy.scale_action(
                    action_batch[env_idx].reshape(1, -1)
                )[0]

                transition = {
                    "obs": obs_batch[env_idx].copy(),
                    "action": action_scaled.copy(),
                    "reward": reward,
                    "next_obs": next_obs_batch[env_idx].copy(),
                    "done": done,
                }
                if info.get("success", reward >= 0.0):
                    success_transitions.append(transition)
                else:
                    failed_transitions.append(transition)

                if done:
                    total_trials += 1
                    if info.get("success", reward >= 0.0):
                        num_success += 1
                    active[env_idx] = False

            obs = next_obs

        for env_idx in range(n_envs):
            if active[env_idx]:
                total_trials += 1

    if log_data and hasattr(model.replay_buffer, "final_offline_step"):
        model.replay_buffer.final_offline_step()

    success_rate = float(num_success / max(total_trials, 1))
    return success_rate, success_transitions, failed_transitions
