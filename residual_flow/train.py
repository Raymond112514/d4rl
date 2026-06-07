#!/usr/bin/env python3
"""Train a residual flow model on action differences from a frozen base flow policy."""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
FLOW_ROOT = os.path.join(ROOT, "..", "flow")

for path in (FLOW_ROOT, ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from trainers.trainer import ResidualFlowTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        required=True,
        help="Path to frozen base FlowPolicy checkpoint (.pt)",
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
        default="unconditioned",
        help=(
            "unconditioned: denoise w -> a-a'; "
            "obs_conditioned: denoise w -> a-a' given obs; "
            "action_conditioned: denoise w -> a'-a given anchor a; "
            "obs_action_conditioned: denoise w -> a'-a given obs and anchor a"
        ),
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="D4RL/pointmaze/large-dense-v2",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=4,
        help="Number of base-policy actions sampled per observation",
    )
    parser.add_argument(
        "--base-sample-steps",
        type=int,
        default=20,
        help="Euler integration steps for base policy sampling",
    )
    parser.add_argument("--n-steps", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--time-emb-dim", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=os.path.join(ROOT, "checkpoints", "{mode}"),
    )
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    parser.add_argument("--cpu", action="store_true", help="Force CPU instead of CUDA")

    parser.add_argument("--wandb-project", type=str, default="residual-flow-pointmaze")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.k < 2:
        raise ValueError("--k must be at least 2 to form distinct action pairs")
    args.checkpoint_dir = args.checkpoint_dir.format(mode=args.mode)
    trainer = ResidualFlowTrainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
