#!/usr/bin/env python3
"""Train a flow matching BC policy on D4RL point maze."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fm_bc.trainers.bc_trainer import BCTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        type=str,
        choices=("unconditional", "goal_conditioned"),
        default="unconditional",
        help=(
            "unconditional: pi(a|obs); "
            "goal_conditioned: pi(a|obs, goal)"
        ),
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="D4RL/pointmaze/large-dense-v2",
    )
    parser.add_argument("--ac-chunk", type=int, default=10, help="Action chunk size")
    parser.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset transitions to use for training (e.g. 0.5 for 50%%)",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--time-emb-dim", type=int, default=64)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1000,
        help="Run policy rollout eval every N training steps (0 disables)",
    )
    parser.add_argument(
        "--eval-rollouts",
        type=int,
        default=5,
        help="Number of trajectories to rollout during eval",
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join(ROOT, "checkpoints", "{mode}"),
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--cpu", action="store_true", help="Force CPU instead of CUDA")

    parser.add_argument("--wandb-project", type=str, default="flow-pointmaze-bc")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.data_fraction <= 1.0:
        raise ValueError("--data-fraction must be in (0, 1]")
    fraction_tag = f"_{args.data_fraction:g}" if args.data_fraction < 1.0 else ""
    args.checkpoint_dir = args.checkpoint_dir.format(
        mode=f"{args.mode}_ac{args.ac_chunk}{fraction_tag}"
    )
    trainer = BCTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
