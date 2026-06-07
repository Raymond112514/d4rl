#!/usr/bin/env python3
"""Evaluate success rate across all four residual flow conditioning modes."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FLOW_ROOT = os.path.join(ROOT, "..", "flow")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
if FLOW_ROOT not in sys.path:
    sys.path.insert(1, FLOW_ROOT)

import minari
import numpy as np
import torch

from eval import collect_rollouts, load_residual_checkpoint
from fm_bc.eval.rollout import EvalResetConfig
from fm_bc.utils.checkpoint import load_checkpoint
from fm_bc.utils.device import resolve_device

MODES = (
    "unconditioned",
    "obs_conditioned",
    "action_conditioned",
    "obs_action_conditioned",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        required=True,
        help="Path to frozen base FlowPolicy checkpoint (.pt)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join(ROOT, "checkpoints", "goal_conditioned_ac1_k8"),
        help="Directory containing per-mode residual checkpoints as {mode}/best.pt",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="D4RL/pointmaze/large-dense-v2",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--base-sample-steps", type=int, default=20)
    parser.add_argument("--residual-sample-steps", type=int, default=20)
    parser.add_argument("--residual-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--cpu", action="store_true", help="Force CPU instead of CUDA")
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
    )
    parser.add_argument(
        "--reset-cell",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        default=[3, 10],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(use_cpu=args.cpu)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    base_policy, base_cfg, stats, _ = load_checkpoint(
        args.base_checkpoint,
        device=device,
    )
    base_policy.eval()

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

    print(f"Base checkpoint:  {args.base_checkpoint}")
    print(f"Checkpoint dir:   {args.checkpoint_dir}")
    print(f"Fixed eval:       reset_cell={reset_cell}, goal_cell={goal_cell}")
    print(f"Episodes/mode:    {args.episodes}")
    print(f"Residual scale:   {args.residual_scale}")
    print()

    results: list[tuple[str, float, float, int]] = []

    for mode in MODES:
        residual_path = os.path.join(args.checkpoint_dir, mode, "best.pt")
        if not os.path.isfile(residual_path):
            raise FileNotFoundError(f"Missing residual checkpoint for {mode}: {residual_path}")

        residual_model, residual_cfg, stats, _ = load_residual_checkpoint(
            residual_path,
            device=device,
        )
        if residual_cfg.mode != mode:
            raise ValueError(
                f"Checkpoint mode mismatch for {residual_path}: "
                f"expected {mode}, got {residual_cfg.mode}"
            )

        print(f"Evaluating {mode}...")
        _, metrics = collect_rollouts(
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
            baseline=False,
            device=device,
            max_steps=args.max_steps,
            eval_reset=eval_reset,
            verbose=False,
        )

        success_rate = metrics["eval/success_rate"]
        mean_return = metrics["eval/mean_return"]
        n_success = int(round(success_rate * args.episodes))
        results.append((mode, success_rate, mean_return, n_success))
        print(
            f"  {mode}: success={n_success}/{args.episodes} "
            f"({success_rate:.3f}), mean_return={mean_return:.1f}"
        )
        print()

    env.close()

    print("=" * 60)
    print(f"{'Mode':<24} {'Success':>12} {'Rate':>8} {'Return':>10}")
    print("-" * 60)
    for mode, success_rate, mean_return, n_success in results:
        print(
            f"{mode:<24} "
            f"{n_success:>4}/{args.episodes:<7} "
            f"{success_rate:>7.3f} "
            f"{mean_return:>10.1f}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
