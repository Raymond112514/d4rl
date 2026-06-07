#!/usr/bin/env python3
"""Evaluate a residual flow policy with flow/residual/combined action visualization."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Optional

ROOT = os.path.dirname(os.path.abspath(__file__))
FLOW_ROOT = os.path.join(ROOT, "..", "flow")

for path in (FLOW_ROOT, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

import minari
import numpy as np
import torch

from eval_viz import plot_eval_paths, plot_eval_rollouts
from fm_bc.datasets.pointmaze import PointMazeStats
from fm_bc.eval.rollout import (
    GOAL_TOLERANCE,
    EvalResetConfig,
    configure_episodic_eval_env,
    denormalize,
    normalize,
    resolve_eval_reset_cells,
    restore_episodic_eval_env,
)
from fm_bc.models.flow_policy import FlowPolicy, FlowPolicyConfig
from fm_bc.utils.checkpoint import load_checkpoint
from fm_bc.utils.device import resolve_device
from models.residual_flow_policy import ResidualFlowConfig, ResidualFlowPolicy


def load_residual_checkpoint(
    path: str,
    *,
    device: torch.device,
) -> tuple[ResidualFlowPolicy, ResidualFlowConfig, PointMazeStats, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg: ResidualFlowConfig = payload["config"]
    stats = PointMazeStats.from_dict(payload["stats"])
    model = ResidualFlowPolicy(cfg).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model, cfg, stats, payload


def combine_actions(
    flow_norm: np.ndarray,
    residual_norm: np.ndarray,
    *,
    stats: PointMazeStats,
    residual_cfg: ResidualFlowConfig,
    residual_scale: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
) -> dict[str, np.ndarray]:
    """Combine base flow and residual in normalized space, return physical chunks."""
    ac_chunk = residual_cfg.ac_chunk
    action_dim = residual_cfg.action_dim
    flow_chunk = np.asarray(flow_norm, dtype=np.float64).reshape(ac_chunk, action_dim)
    residual_chunk = np.asarray(residual_norm, dtype=np.float64).reshape(ac_chunk, action_dim)

    if residual_cfg.mode in ("unconditioned", "obs_conditioned"):
        combined_norm = flow_chunk - residual_scale * residual_chunk
    else:
        combined_norm = flow_chunk + residual_scale * residual_chunk

    flow_phys = denormalize(flow_chunk, stats.action_mean, stats.action_std)
    combined_phys = denormalize(combined_norm, stats.action_mean, stats.action_std)
    combined_phys = np.clip(combined_phys, action_low, action_high)
    residual_phys = combined_phys - flow_phys

    return {
        "flow": flow_phys,
        "residual": residual_phys,
        "combined": combined_phys,
    }


@torch.no_grad()
def sample_residual_norm(
    residual_model: Optional[ResidualFlowPolicy],
    *,
    obs_norm: np.ndarray,
    flow_norm: np.ndarray,
    residual_cfg: ResidualFlowConfig,
    baseline: bool,
    sample_steps: int,
    device: torch.device,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a normalized residual action-difference vector."""
    flow_dim = residual_cfg.flow_dim
    if baseline:
        return rng.standard_normal(flow_dim, dtype=np.float32)

    obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)
    anchor_t = None
    if residual_cfg.uses_action_conditioning:
        anchor_flat = np.asarray(flow_norm, dtype=np.float32).reshape(-1)
        anchor_t = torch.as_tensor(anchor_flat, dtype=torch.float32, device=device).unsqueeze(0)

    residual = residual_model.sample(
        obs_t,
        anchor_action=anchor_t,
        n_steps=sample_steps,
    )
    return residual.cpu().numpy()[0].astype(np.float32)


@torch.no_grad()
def rollout_episode(
    base_policy: FlowPolicy,
    base_cfg: FlowPolicyConfig,
    stats: PointMazeStats,
    residual_model: Optional[ResidualFlowPolicy],
    residual_cfg: ResidualFlowConfig,
    env,
    *,
    seed: int,
    base_sample_steps: int,
    residual_sample_steps: int,
    residual_scale: float,
    baseline: bool,
    device: torch.device,
    max_steps: int = 1000,
    eval_reset: Optional[EvalResetConfig] = None,
) -> dict[str, Any]:
    reset_seed = seed
    reset_options: Optional[dict] = None
    rng = np.random.default_rng(seed)

    if eval_reset is not None and eval_reset.fixed:
        goal_cell, reset_cell = resolve_eval_reset_cells(
            env,
            seed=eval_reset.seed,
            goal_cell=eval_reset.goal_cell,
            reset_cell=eval_reset.reset_cell,
        )
        reset_options = {
            "goal_cell": np.asarray(goal_cell, dtype=np.int64),
            "reset_cell": np.asarray(reset_cell, dtype=np.int64),
        }
        reset_seed = eval_reset.seed
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if reset_options is None:
        obs_dict, _ = env.reset(seed=reset_seed)
    else:
        obs_dict, _ = env.reset(seed=reset_seed, options=reset_options)

    positions = [np.asarray(obs_dict["achieved_goal"], dtype=np.float64).copy()]
    action_queue: list[dict[str, np.ndarray]] = []
    total_return = 0.0
    done = False
    steps = 0
    step_records: list[dict[str, Any]] = []

    initial_goal = np.asarray(obs_dict["desired_goal"], dtype=np.float64).copy()
    reached_initial_goal = np.linalg.norm(
        obs_dict["achieved_goal"] - initial_goal
    ) <= GOAL_TOLERANCE

    action_low = env.action_space.low
    action_high = env.action_space.high

    while not done and steps < max_steps:
        start_pos = np.asarray(obs_dict["achieved_goal"], dtype=np.float64).copy()

        if not action_queue:
            obs_norm = normalize(
                np.asarray(obs_dict["observation"], dtype=np.float64),
                stats.obs_mean,
                stats.obs_std,
            )
            obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

            goal_t = None
            if base_cfg.mode == "goal_conditioned":
                goal_norm = normalize(initial_goal, stats.goal_mean, stats.goal_std)
                goal_t = torch.as_tensor(
                    goal_norm, dtype=torch.float32, device=device
                ).unsqueeze(0)

            flow_norm_chunk = base_policy.sample(
                obs_t,
                goal=goal_t,
                n_steps=base_sample_steps,
            )
            flow_norm = flow_norm_chunk.cpu().numpy()[0]

            residual_norm = sample_residual_norm(
                residual_model,
                obs_norm=obs_norm,
                flow_norm=flow_norm,
                residual_cfg=residual_cfg,
                baseline=baseline,
                sample_steps=residual_sample_steps,
                device=device,
                rng=rng,
            )

            decomp = combine_actions(
                flow_norm,
                residual_norm,
                stats=stats,
                residual_cfg=residual_cfg,
                residual_scale=residual_scale,
                action_low=action_low,
                action_high=action_high,
            )

            action_queue = [
                {
                    "flow": decomp["flow"][idx],
                    "residual": decomp["residual"][idx],
                    "combined": decomp["combined"][idx],
                }
                for idx in range(residual_cfg.ac_chunk)
            ]

        record = action_queue.pop(0)
        action = record["combined"]
        obs_dict, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        positions.append(np.asarray(obs_dict["achieved_goal"], dtype=np.float64).copy())

        step_records.append(
            {
                "step_idx": steps,
                "start_pos": start_pos,
                "flow": record["flow"][None, :],
                "residual": record["residual"][None, :],
                "combined": record["combined"][None, :],
                "reward": float(reward),
            }
        )

        if not reached_initial_goal:
            reached_initial_goal = np.linalg.norm(
                obs_dict["achieved_goal"] - initial_goal
            ) <= GOAL_TOLERANCE
        done = bool(terminated or truncated) or reached_initial_goal
        steps += 1

    return {
        "positions": np.stack(positions, axis=0),
        "goal": initial_goal,
        "success": reached_initial_goal,
        "total_return": total_return,
        "steps": step_records,
    }


@torch.no_grad()
def collect_rollouts(
    base_policy: FlowPolicy,
    base_cfg: FlowPolicyConfig,
    stats: PointMazeStats,
    residual_model: Optional[ResidualFlowPolicy],
    residual_cfg: ResidualFlowConfig,
    env,
    *,
    n_episodes: int,
    seed: int,
    base_sample_steps: int,
    residual_sample_steps: int,
    residual_scale: float,
    baseline: bool,
    device: torch.device,
    max_steps: int = 1000,
    eval_reset: Optional[EvalResetConfig] = None,
    verbose: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    previous_episodic_settings = configure_episodic_eval_env(env)
    rollouts: list[dict[str, Any]] = []
    try:
        for ep_idx in range(n_episodes):
            rollout = rollout_episode(
                base_policy,
                base_cfg,
                stats,
                residual_model,
                residual_cfg,
                env,
                seed=seed + ep_idx,
                base_sample_steps=base_sample_steps,
                residual_sample_steps=residual_sample_steps,
                residual_scale=residual_scale,
                baseline=baseline,
                device=device,
                max_steps=max_steps,
                eval_reset=eval_reset,
            )
            rollouts.append(rollout)
            if verbose:
                status = "success" if rollout["success"] else "fail"
                print(
                    f"  episode {ep_idx + 1}/{n_episodes} done: "
                    f"{status}, steps={len(rollout['steps'])}, "
                    f"return={rollout['total_return']:.1f}"
                )
    finally:
        restore_episodic_eval_env(env, previous_episodic_settings)

    success_rate = sum(r["success"] for r in rollouts) / max(n_episodes, 1)
    mean_return = float(np.mean([r["total_return"] for r in rollouts]))
    metrics = {
        "eval/success_rate": success_rate,
        "eval/mean_return": mean_return,
    }
    return rollouts, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        required=True,
        help="Path to frozen base FlowPolicy checkpoint (.pt)",
    )
    parser.add_argument(
        "--residual-checkpoint",
        type=str,
        default=None,
        help="Path to trained ResidualFlowPolicy checkpoint (.pt)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Sample residuals from standard Gaussian instead of residual flow policy",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=(
            "unconditioned",
            "obs_conditioned",
            "action_conditioned",
            "obs_action_conditioned",
        ),
        default=None,
        help="Residual conditioning mode (required for --baseline; inferred from checkpoint otherwise)",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="D4RL/pointmaze/large-dense-v2",
    )
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--base-sample-steps", type=int, default=20)
    parser.add_argument("--residual-sample-steps", type=int, default=20)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--save-dir",
        type=str,
        default=os.path.join(ROOT, "eval_plots"),
    )
    parser.add_argument("--show", action="store_true", help="Open plot windows")
    parser.add_argument("--cpu", action="store_true", help="Force CPU instead of CUDA")
    parser.add_argument(
        "--arrow-stride",
        type=int,
        default=20,
        help="Plot action arrows every N env steps",
    )
    parser.add_argument(
        "--arrow-length-scale",
        type=float,
        default=0.5,
        help="Visual scale for action arrows",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Seed for env reset noise (defaults to --seed)",
    )
    parser.add_argument(
        "--goal-cell",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        default=[7, 1],
        help="Fixed maze goal cell (row, col); default: 7 1",
    )
    parser.add_argument(
        "--reset-cell",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        default=[3, 10],
        help="Fixed maze start/reset cell (row, col); default: 3 10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.baseline and args.residual_checkpoint is None:
        raise ValueError("Provide --residual-checkpoint or pass --baseline")

    device = resolve_device(use_cpu=args.cpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_policy, base_cfg, stats, _ = load_checkpoint(
        args.base_checkpoint,
        device=device,
    )
    base_policy.eval()

    if args.baseline:
        if args.mode is None:
            raise ValueError("--mode is required when using --baseline")
        residual_model = None
        residual_cfg = ResidualFlowConfig(
            action_dim=base_cfg.action_dim,
            ac_chunk=base_cfg.ac_chunk,
            obs_dim=base_cfg.obs_dim,
            goal_dim=base_cfg.goal_dim,
            mode=args.mode,
        )
        mode_label = f"baseline_gaussian_{args.mode}"
    else:
        residual_model, residual_cfg, stats, payload = load_residual_checkpoint(
            args.residual_checkpoint,
            device=device,
        )
        if args.mode is not None and args.mode != residual_cfg.mode:
            print(
                f"Warning: overriding checkpoint mode {residual_cfg.mode} with --mode {args.mode}"
            )
            residual_cfg.mode = args.mode
        mode_label = residual_cfg.mode
        if payload.get("base_checkpoint") and payload["base_checkpoint"] != args.base_checkpoint:
            print(
                "Warning: residual checkpoint base policy differs from --base-checkpoint\n"
                f"  checkpoint: {payload['base_checkpoint']}\n"
                f"  provided:   {args.base_checkpoint}"
            )

    dataset = minari.load_dataset(args.dataset_id, download=False)
    env = dataset.recover_environment()

    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed
    goal_cell = tuple(args.goal_cell)
    reset_cell = tuple(args.reset_cell)
    eval_reset = EvalResetConfig(
        fixed=True,
        seed=eval_seed,
        goal_cell=goal_cell,
        reset_cell=reset_cell,
    )
    print(
        f"Fixed eval: reset_cell={reset_cell}, goal_cell={goal_cell}, seed={eval_seed}"
    )

    print(f"Collecting {args.episodes} rollouts (mode={mode_label}, scale={args.residual_scale})...")
    rollouts, metrics = collect_rollouts(
        base_policy,
        base_cfg,
        stats,
        residual_model,
        residual_cfg,
        env,
        n_episodes=args.episodes,
        seed=args.seed,
        base_sample_steps=args.base_sample_steps,
        residual_sample_steps=args.residual_sample_steps,
        residual_scale=args.residual_scale,
        baseline=args.baseline,
        device=device,
        max_steps=args.max_steps,
        eval_reset=eval_reset,
    )
    env.close()

    print("Rollouts complete. Generating plots...")
    suffix = "baseline" if args.baseline else mode_label
    title = (
        f"Residual Flow Eval ({suffix}, scale={args.residual_scale:g})"
        if not args.baseline
        else f"Gaussian Baseline Eval (scale={args.residual_scale:g})"
    )

    os.makedirs(args.save_dir, exist_ok=True)
    rollouts_path = os.path.join(args.save_dir, f"rollouts_{suffix}.png")
    paths_path = os.path.join(args.save_dir, f"paths_{suffix}.png")

    plot_eval_rollouts(
        rollouts,
        dataset_id=args.dataset_id,
        title=title,
        save_path=rollouts_path,
        show=args.show,
        arrow_stride=args.arrow_stride,
        arrow_length_scale=args.arrow_length_scale,
    )
    plot_eval_paths(
        rollouts,
        dataset_id=args.dataset_id,
        title=title,
        save_path=paths_path,
        show=args.show,
    )

    print(f"Mode:           {mode_label}")
    print(f"Residual scale: {args.residual_scale}")
    print(f"Success rate:   {metrics['eval/success_rate']:.3f}")
    print(f"Mean return:    {metrics['eval/mean_return']:.3f}")
    print(f"Saved rollouts: {rollouts_path}")
    print(f"Saved paths:    {paths_path}")


if __name__ == "__main__":
    main()
