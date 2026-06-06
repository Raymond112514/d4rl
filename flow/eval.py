#!/usr/bin/env python3
"""Evaluate a trained flow matching BC policy and plot rollout trajectories."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import minari

from fm_bc.eval.plot import plot_policy_rollouts
from fm_bc.eval.rollout import (
    EvalResetConfig,
    find_hardest_eval_cells,
    format_eval_cell,
    get_eval_cell_candidates,
    resolve_eval_reset_cells,
    rollout_policy,
)
from fm_bc.utils.checkpoint import load_checkpoint
from fm_bc.utils.device import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="D4RL/pointmaze/large-dense-v2",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of rollout episodes")
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save rollout trajectory plot",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open plot window")
    parser.add_argument("--cpu", action="store_true", help="Force CPU instead of CUDA")
    parser.add_argument(
        "--fixed-eval",
        action="store_true",
        help="Use the same fixed goal and start state for every rollout episode",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help="Seed for fixed goal/start selection and env reset noise (defaults to --seed)",
    )
    parser.add_argument(
        "--goal-cell",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        default=None,
        help="Explicit maze goal cell (row, col)",
    )
    parser.add_argument(
        "--reset-cell",
        "--start-cell",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        default=None,
        dest="reset_cell",
        help="Explicit maze start/reset cell (row, col)",
    )
    parser.add_argument(
        "--list-eval-cells",
        action="store_true",
        help="Print valid reset and goal cells, then exit",
    )
    parser.add_argument(
        "--hardest-eval",
        action="store_true",
        help="Use the most challenging fixed start/goal pair for this maze",
    )
    parser.add_argument(
        "--hardest-metric",
        choices=("path", "euclidean"),
        default="path",
        help="How to rank start/goal difficulty when using --hardest-eval",
    )
    return parser.parse_args()


def print_eval_cells(env) -> None:
    reset_cells, goal_cells = get_eval_cell_candidates(env)
    print(f"Valid reset/start cells ({len(reset_cells)}):")
    for cell in reset_cells:
        print(f"  {format_eval_cell(env, cell)}")
    print(f"\nValid goal cells ({len(goal_cells)}):")
    for cell in goal_cells:
        print(f"  {format_eval_cell(env, cell)}")

    hardest_path = find_hardest_eval_cells(env, metric="path")
    hardest_eucl = find_hardest_eval_cells(env, metric="euclidean")
    print("\nSuggested challenging cases:")
    print(
        "  Longest maze path: "
        f"reset {format_eval_cell(env, hardest_path[0])} -> "
        f"goal {format_eval_cell(env, hardest_path[1])} "
        f"(path={int(hardest_path[2]['path_length'])}, "
        f"euclidean={hardest_path[2]['euclidean']:.3f})"
    )
    print(
        "  Farthest Euclidean: "
        f"reset {format_eval_cell(env, hardest_eucl[0])} -> "
        f"goal {format_eval_cell(env, hardest_eucl[1])} "
        f"(path={int(hardest_eucl[2]['path_length'])}, "
        f"euclidean={hardest_eucl[2]['euclidean']:.3f})"
    )


def main() -> None:
    args = parse_args()

    dataset = minari.load_dataset(args.dataset_id, download=False)
    env = dataset.recover_environment()

    if args.list_eval_cells:
        print_eval_cells(env)
        env.close()
        return

    if args.checkpoint is None:
        env.close()
        raise SystemExit("--checkpoint is required unless --list-eval-cells is used.")

    device = resolve_device(use_cpu=args.cpu)
    model, cfg, stats, _payload = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    fixed_eval = (
        args.fixed_eval
        or args.hardest_eval
        or args.goal_cell is not None
        or args.reset_cell is not None
    )
    if (args.goal_cell is not None or args.reset_cell is not None) and not args.fixed_eval:
        print("Note: --goal-cell/--reset-cell enabled fixed eval mode.")
    eval_seed = args.seed if args.eval_seed is None else args.eval_seed

    goal_cell = tuple(args.goal_cell) if args.goal_cell is not None else None
    reset_cell = tuple(args.reset_cell) if args.reset_cell is not None else None

    if args.hardest_eval:
        reset_cell, goal_cell, hardest_info = find_hardest_eval_cells(
            env,
            metric=args.hardest_metric,
        )
        print(
            "Using hardest eval pair "
            f"({args.hardest_metric} metric): "
            f"reset {format_eval_cell(env, reset_cell)} -> "
            f"goal {format_eval_cell(env, goal_cell)} "
            f"(path={int(hardest_info['path_length'])}, "
            f"euclidean={hardest_info['euclidean']:.3f})"
        )

    eval_reset = None
    if fixed_eval:
        eval_reset = EvalResetConfig(
            fixed=True,
            seed=eval_seed,
            goal_cell=goal_cell,
            reset_cell=reset_cell,
        )

    rollouts, metrics = rollout_policy(
        model,
        env,
        stats,
        cfg,
        n_episodes=args.episodes,
        seed=args.seed,
        sample_steps=args.sample_steps,
        device=device,
        max_steps=args.max_steps,
        eval_reset=eval_reset,
    )

    resolved_goal_cell = None
    resolved_reset_cell = None
    resolved_goal_label = None
    resolved_reset_label = None
    if fixed_eval and eval_reset is not None:
        resolved_goal_cell, resolved_reset_cell = resolve_eval_reset_cells(
            env,
            seed=eval_seed,
            goal_cell=eval_reset.goal_cell,
            reset_cell=eval_reset.reset_cell,
        )
        resolved_goal_label = format_eval_cell(env, resolved_goal_cell)
        resolved_reset_label = format_eval_cell(env, resolved_reset_cell)

    env.close()

    for ep_idx, rollout in enumerate(rollouts):
        print(
            f"episode {ep_idx:3d}: return={rollout.return_:.3f} "
            f"success={rollout.success} steps={rollout.length}"
        )

    print("\nSummary")
    print(f"  mode:          {cfg.mode}")
    print(f"  ac_chunk:      {cfg.ac_chunk}")
    print(f"  device:        {device}")
    print(f"  checkpoint:    {args.checkpoint}")
    if fixed_eval and resolved_goal_cell is not None and resolved_reset_cell is not None:
        print(f"  fixed eval:    True (eval_seed={eval_seed})")
        print(f"  goal cell:     {resolved_goal_cell} ({resolved_goal_label})")
        print(f"  reset cell:    {resolved_reset_cell} ({resolved_reset_label})")
        if rollouts:
            print(f"  start:         {rollouts[0].positions[0].tolist()}")
            print(f"  goal:          {rollouts[0].goal.tolist()}")
    print(f"  success rate:  {metrics['eval/success_rate']:.3f}")
    print(f"  mean return:   {metrics['eval/mean_return']:.3f}")

    save_path = args.save_path
    if save_path is None:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        save_path = os.path.join(ROOT, "rollouts", f"{ckpt_name}_seed{args.seed}.png")
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    plot_policy_rollouts(
        rollouts,
        dataset_id=args.dataset_id,
        title=f"BC Policy Rollouts ({cfg.mode})",
        save_path=save_path,
        show=not args.no_show,
    )
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    main()
