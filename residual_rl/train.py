#!/usr/bin/env python3
"""Train a residual SAC policy on top of a frozen flow matching BC policy."""

from __future__ import annotations

import argparse
import os
import random
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FLOW_ROOT = os.path.join(ROOT, "..", "flow")
DSRL_SB3_ROOT = os.path.join(ROOT, "..", "..", "dsrl", "stable-baselines3")

for path in (FLOW_ROOT, DSRL_SB3_ROOT, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import numpy as np
import torch
import wandb
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback

from fm_bc.eval.rollout import format_eval_cell, resolve_eval_reset_cells
from train_utils import WandbCallback, collect_rollouts
from wrapper import PointMazeResidualVecEnv, PointMazeResidualWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to frozen flow policy checkpoint")
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="D4RL/pointmaze/large-dense-v2",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-envs", type=int, default=4, help="Parallel training environments")
    parser.add_argument("--sample-steps", type=int, default=20, help="Flow policy denoising steps")
    parser.add_argument("--residual-scale", type=float, default=0.1)
    parser.add_argument("--action-mag", type=float, default=2.0, help="Residual action bound")
    parser.add_argument("--max-episode-steps", type=int, default=1000)

    parser.add_argument(
        "--init-rollouts",
        type=int,
        default=20,
        help="Initial rollouts to seed the replay buffer before training",
    )
    parser.add_argument(
        "--standard-gauss-init",
        action="store_true",
        help="Initialize SAC actor to output a standard Gaussian (DSRL-style policy init)",
    )
    parser.add_argument("--total-timesteps", type=int, default=500_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--train-freq", type=int, default=2)
    parser.add_argument("--utd", type=int, default=20, help="Gradient steps per env step")
    parser.add_argument("--learning-starts", type=int, default=1)
    parser.add_argument("--ent-coef", type=float, default=-1, help="Use -1 for automatic entropy tuning")
    parser.add_argument("--target-ent", type=float, default=-1, help="Use -1 for automatic target entropy")
    parser.add_argument("--actor-gradient-steps", type=int, default=-1)

    parser.add_argument("--net-arch", type=int, nargs="+", default=None)
    parser.add_argument("--n-critics", type=int, default=2)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--eval-plot-trajectories", type=int, default=5)
    parser.add_argument(
        "--save-eval-plots",
        action="store_true",
        help="Also save eval plot PNGs to disk (wandb logging is always on when wandb is enabled)",
    )
    parser.add_argument(
        "--eval-plot-dir",
        type=str,
        default=os.path.join(ROOT, "eval_plots"),
        help="Directory for eval plot PNGs when --save-eval-plots is set",
    )
    parser.add_argument("--eval-interval", type=int, default=20_000)
    parser.add_argument("--plot-arrow-stride", type=int, default=20)
    parser.add_argument("--arrow-length-scale", type=float, default=0.5)
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--save-freq", type=int, default=50_000)

    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join(ROOT, "checkpoints"),
    )
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--wandb-project", type=str, default="d4rl-residual-rl")
    parser.add_argument("--wandb-group", type=str, default="")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--fixed-eval", action="store_true", help="Use fixed start/goal cells for training env")
    parser.add_argument("--eval-seed", type=int, default=42)
    parser.add_argument("--goal-cell", type=int, nargs=2, metavar=("ROW", "COL"), default=None)
    parser.add_argument("--reset-cell", type=int, nargs=2, metavar=("ROW", "COL"), default=None)
    parser.add_argument("--start-cell", type=int, nargs=2, metavar=("ROW", "COL"), default=None)
    return parser.parse_args()


def resolve_reset_options(args: argparse.Namespace, env) -> dict[str, np.ndarray] | None:
    if not args.fixed_eval and args.goal_cell is None and args.reset_cell is None and args.start_cell is None:
        return None

    goal_cell = tuple(args.goal_cell) if args.goal_cell is not None else None
    reset_cell = tuple(args.reset_cell) if args.reset_cell is not None else None
    if args.start_cell is not None:
        reset_cell = tuple(args.start_cell)

    goal_cell, reset_cell = resolve_eval_reset_cells(
        env,
        seed=args.eval_seed,
        goal_cell=goal_cell,
        reset_cell=reset_cell,
    )
    return {
        "goal_cell": np.asarray(goal_cell, dtype=np.int64),
        "reset_cell": np.asarray(reset_cell, dtype=np.int64),
    }


def default_net_arch(ac_chunk: int) -> list[int]:
    return [512, 512] if ac_chunk >= 5 else [256, 256]


def make_vec_env(args: argparse.Namespace, reset_options: dict[str, np.ndarray] | None) -> PointMazeResidualVecEnv:
    return PointMazeResidualVecEnv(
        args.n_envs,
        dataset_id=args.dataset_id,
        flow_checkpoint=args.checkpoint,
        sample_steps=args.sample_steps,
        residual_scale=args.residual_scale,
        action_mag=args.action_mag,
        max_episode_steps=args.max_episode_steps,
        use_cpu=args.cpu,
        reset_options=reset_options,
        base_seed=args.seed,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    probe_env = PointMazeResidualWrapper(
        dataset_id=args.dataset_id,
        flow_checkpoint=args.checkpoint,
        sample_steps=args.sample_steps,
        residual_scale=args.residual_scale,
        action_mag=args.action_mag,
        max_episode_steps=args.max_episode_steps,
        use_cpu=args.cpu,
    )
    ac_chunk = probe_env.cfg.ac_chunk
    reset_options = resolve_reset_options(args, probe_env.env)
    if reset_options is not None:
        goal_cell = tuple(int(v) for v in reset_options["goal_cell"])
        reset_cell = tuple(int(v) for v in reset_options["reset_cell"])
        print(f"Fixed eval reset cell: {reset_cell} ({format_eval_cell(probe_env.env, reset_cell)})")
        print(f"Fixed eval goal cell:  {goal_cell} ({format_eval_cell(probe_env.env, goal_cell)})")
    probe_env.close()

    run_name = args.run_name or os.path.splitext(os.path.basename(args.checkpoint))[0]
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            group=args.wandb_group or None,
            config=vars(args),
            monitor_gym=True,
        )

    train_env = make_vec_env(args, reset_options)
    eval_env = make_vec_env(args, reset_options)

    net_arch = args.net_arch if args.net_arch is not None else default_net_arch(ac_chunk)
    policy_kwargs = dict(
        log_std_init=0.0,
        net_arch=dict(pi=net_arch, qf=net_arch),
        activation_fn=torch.nn.Tanh,
        post_linear_modules=[torch.nn.LayerNorm],
        n_critics=args.n_critics,
        standard_gauss_init=args.standard_gauss_init,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.utd,
        ent_coef="auto" if args.ent_coef < 0 else args.ent_coef,
        target_entropy="auto" if args.target_ent < 0 else args.target_ent,
        target_update_interval=1,
        verbose=1,
        seed=args.seed,
        policy_kwargs=policy_kwargs,
        actor_gradient_steps=args.actor_gradient_steps,
    )
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // train_env.num_envs, 1),
        save_path=args.checkpoint_dir,
        name_prefix=run_name,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    wandb_callback = WandbCallback(
        log_freq=args.log_freq,
        use_wandb=not args.no_wandb,
        eval_env=eval_env,
        max_steps=args.max_episode_steps,
        eval_episodes=args.eval_episodes,
        eval_plot_trajectories=args.eval_plot_trajectories,
        eval_plot_dir=args.eval_plot_dir if args.save_eval_plots else None,
        dataset_id=args.dataset_id,
        eval_interval=args.eval_interval,
        plot_arrow_stride=args.plot_arrow_stride,
        arrow_length_scale=args.arrow_length_scale,
    )

    seed_env = make_vec_env(args, reset_options)

    if args.init_rollouts > 0:
        print(f"Collecting {args.init_rollouts} initial rollouts...")
        init_success = collect_rollouts(
            model,
            seed_env,
            args.init_rollouts,
            args.max_episode_steps,
            log_data=True,
        )[0]
        print(f"Initial rollout success rate: {init_success:.3f}")
        wandb_callback.set_initial_timesteps(args.init_rollouts * args.max_episode_steps)
    seed_env.close()

    model.learn(
        total_timesteps=args.total_timesteps,
        callback=[checkpoint_callback, wandb_callback],
        progress_bar=False,
    )

    final_path = os.path.join(args.checkpoint_dir, f"{run_name}_final.zip")
    model.save(final_path)
    print(f"Saved final SAC model to {final_path}")

    train_env.close()
    eval_env.close()
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
