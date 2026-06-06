#!/usr/bin/env python3
"""Visualize action-chunk distributions over the point maze."""

from __future__ import annotations

import argparse
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import matplotlib.pyplot as plt
import minari
import numpy as np
from matplotlib.animation import FFMpegWriter

from fm_bc.eval.action_viz import (
    AnchorState,
    _subsample_anchors,
    build_dataset_anchors,
    build_grid_anchors,
    chain_action_chunk,
    plot_action_chunk_distribution,
    sample_action_chunk_trajectories,
)
from fm_bc.eval.plot import draw_maze, get_maze_map
from fm_bc.eval.rollout import resolve_eval_reset_cells
from fm_bc.utils.checkpoint import load_checkpoint
from fm_bc.utils.device import resolve_device


def resolve_ffmpeg_path() -> str:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError(
            "MP4 export requires ffmpeg on PATH or the imageio-ffmpeg package. "
            "Install with: pip install imageio-ffmpeg"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="D4RL/pointmaze/large-dense-v2",
    )
    parser.add_argument(
        "--source",
        choices=("grid", "dataset"),
        default="grid",
        help="Anchor states from maze grid cells or dataset observations",
    )
    parser.add_argument(
        "--grid-stride",
        type=int,
        default=3,
        help="Use every Nth free maze cell for fan anchors when --source=grid",
    )
    parser.add_argument(
        "--flow-grid-stride",
        type=int,
        default=1,
        help="Use every Nth free maze cell for the mean flow field (1 = all cells)",
    )
    parser.add_argument(
        "--n-anchors",
        type=int,
        default=120,
        help="Number of anchors when --source=dataset",
    )
    parser.add_argument(
        "--n-fan-anchors",
        type=int,
        default=12,
        help="How many anchors get full stochastic fan plots in the left panel",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=8,
        help="Stochastic action chunks sampled per anchor",
    )
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument(
        "--vector-scale",
        type=float,
        default=0.12,
        help="Visualization scale for chaining action vectors tail-to-tail",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--goal-cell",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        default=None,
        help="Fixed maze goal cell for goal-conditioned policies",
    )
    parser.add_argument(
        "--use-dataset-goals",
        action="store_true",
        help="Use each dataset state's own goal instead of a single fixed goal",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Path to save the figure",
    )
    parser.add_argument(
        "--denoising-video",
        type=str,
        default=None,
        help="Path to save one MP4 showing flow denoising (noise -> final action)",
    )
    parser.add_argument(
        "--denoising-fps",
        type=int,
        default=4,
        help="Frame rate for --denoising-video",
    )
    parser.add_argument(
        "--denoising-n-samples",
        type=int,
        default=8,
        help="Number of stochastic samples shown evolving in each denoising frame",
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open plot window")
    parser.add_argument("--cpu", action="store_true", help="Force CPU instead of CUDA")
    return parser.parse_args()


def resolve_goal_xy(env, args: argparse.Namespace) -> np.ndarray | None:
    if args.use_dataset_goals:
        return None

    goal_cell = tuple(args.goal_cell) if args.goal_cell is not None else None
    goal_cell, reset_cell = resolve_eval_reset_cells(
        env,
        seed=args.seed,
        goal_cell=goal_cell,
    )
    obs_dict, _ = env.reset(
        seed=args.seed,
        options={
            "goal_cell": np.asarray(goal_cell, dtype=np.int64),
            "reset_cell": np.asarray(reset_cell, dtype=np.int64),
        },
    )
    return np.asarray(obs_dict["desired_goal"], dtype=np.float64)


def _render_denoising_frame(
    ax: plt.Axes,
    *,
    maze_map,
    fan_anchors: list[AnchorState],
    all_trajectories: list[np.ndarray],
    frame_idx: int,
    n_samples: int,
    sample_steps: int,
    vector_scale: float,
    goal: np.ndarray | None,
    dataset_id: str,
    cfg,
    title: str,
) -> None:
    ax.clear()
    draw_maze(ax, maze_map)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, zorder=1)

    if goal is not None:
        ax.scatter(
            goal[0],
            goal[1],
            s=140,
            marker="*",
            c="gold",
            edgecolors="black",
            linewidths=0.6,
            zorder=6,
        )

    cmap = plt.cm.viridis
    for anchor_idx, (anchor, trajectories) in enumerate(zip(fan_anchors, all_trajectories)):
        color = cmap(anchor_idx / max(len(fan_anchors) - 1, 1))
        for sample_idx in range(n_samples):
            chunk = trajectories[frame_idx, sample_idx]
            path = chain_action_chunk(chunk, start=anchor.position, scale=vector_scale)
            ax.plot(
                path[:, 0],
                path[:, 1],
                color=color,
                alpha=0.35,
                linewidth=1.2,
                zorder=2,
            )
            deltas = path[1:] - path[:-1]
            ax.quiver(
                path[:-1, 0],
                path[:-1, 1],
                deltas[:, 0],
                deltas[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1.0,
                color=color,
                alpha=0.55,
                width=0.0025,
                zorder=3,
            )
        ax.scatter(
            anchor.position[0],
            anchor.position[1],
            s=28,
            color=color,
            edgecolors="black",
            linewidths=0.4,
            zorder=4,
        )

    if frame_idx == 0:
        step_label = "noise"
    elif frame_idx == sample_steps:
        step_label = "final"
    else:
        step_label = f"step {frame_idx}/{sample_steps}"

    subtitle = (
        f"{dataset_id} | chunk={cfg.ac_chunk} | {step_label} "
        f"| anchors={len(fan_anchors)} | samples/anchor={n_samples} | scale={vector_scale:g}"
    )
    if goal is not None:
        subtitle += f" | goal=({goal[0]:.2f}, {goal[1]:.2f})"
    ax.set_title(f"{title}\n{subtitle}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def save_denoising_video(
    model,
    cfg,
    stats,
    anchors: list[AnchorState],
    *,
    dataset_id: str,
    goal: np.ndarray | None,
    n_fan_anchors: int,
    n_samples: int,
    sample_steps: int,
    vector_scale: float,
    device,
    seed: int,
    output_path: str,
    fps: int,
    title: str = "Flow denoising",
) -> str:
    """Save one MP4 showing parallel sample evolution through flow denoising."""
    dataset = minari.load_dataset(dataset_id, download=False)
    maze_map = get_maze_map(dataset)
    fan_anchors = _subsample_anchors(anchors, n_fan_anchors, seed=seed)

    all_trajectories = [
        sample_action_chunk_trajectories(
            model,
            stats,
            cfg,
            anchor,
            n_samples=n_samples,
            sample_steps=sample_steps,
            device=device,
            seed=seed + anchor_idx,
        )
        for anchor_idx, anchor in enumerate(fan_anchors)
    ]

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    n_frames = sample_steps + 1
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.tight_layout()

    ffmpeg_path = resolve_ffmpeg_path()
    plt.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    writer = FFMpegWriter(fps=fps)
    with writer.saving(fig, output_path, dpi=150):
        for frame_idx in range(n_frames):
            _render_denoising_frame(
                ax,
                maze_map=maze_map,
                fan_anchors=fan_anchors,
                all_trajectories=all_trajectories,
                frame_idx=frame_idx,
                n_samples=n_samples,
                sample_steps=sample_steps,
                vector_scale=vector_scale,
                goal=goal,
                dataset_id=dataset_id,
                cfg=cfg,
                title=title,
            )
            writer.grab_frame()

    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    device = resolve_device(use_cpu=args.cpu)

    model, cfg, stats, _payload = load_checkpoint(args.checkpoint, device=device)
    model.eval()

    dataset = minari.load_dataset(args.dataset_id, download=False)
    env = dataset.recover_environment()

    goal = resolve_goal_xy(env, args)
    if cfg.mode == "goal_conditioned" and goal is None and not args.use_dataset_goals:
        raise ValueError(
            "goal_conditioned checkpoint requires --goal-cell or --use-dataset-goals"
        )

    if args.source == "grid":
        anchors = build_grid_anchors(env, stride=args.grid_stride, goal=goal)
        flow_anchors = build_grid_anchors(env, stride=args.flow_grid_stride, goal=goal)
    else:
        anchors = build_dataset_anchors(
            args.dataset_id,
            n_anchors=args.n_anchors,
            seed=args.seed,
            goal=goal,
            use_episode_goals=args.use_dataset_goals,
        )
        flow_anchors = build_grid_anchors(env, stride=args.flow_grid_stride, goal=goal)

    env.close()

    save_path = args.save_path
    if save_path is None:
        ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
        save_path = os.path.join(ROOT, "rollouts", f"{ckpt_name}_action_chunks.png")

    print(f"Fan anchors:     {len(anchors)} ({args.source}, stride={args.grid_stride})")
    print(f"Flow anchors:    {len(flow_anchors)} (stride={args.flow_grid_stride})")
    print(f"Fan shown:       {min(args.n_fan_anchors, len(anchors))}")
    print(f"Samples/anchor:  {args.n_samples}")
    print(f"Vector scale:    {args.vector_scale}")
    if goal is not None:
        print(f"Fixed goal:      {goal.tolist()}")

    if args.denoising_video is not None:
        denoising_video = args.denoising_video
        if not denoising_video:
            ckpt_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
            denoising_video = os.path.join(ROOT, "rollouts", f"{ckpt_name}_denoising.mp4")
        elif not os.path.isabs(denoising_video):
            denoising_video = os.path.join(ROOT, denoising_video)
        if not denoising_video.lower().endswith(".mp4"):
            denoising_video = f"{denoising_video}.mp4"

        video_path = save_denoising_video(
            model,
            cfg,
            stats,
            anchors,
            dataset_id=args.dataset_id,
            goal=goal,
            n_fan_anchors=args.n_fan_anchors,
            n_samples=args.denoising_n_samples,
            sample_steps=args.sample_steps,
            vector_scale=args.vector_scale,
            device=device,
            seed=args.seed,
            output_path=denoising_video,
            fps=args.denoising_fps,
            title=f"Flow Denoising ({cfg.mode})",
        )
        print(
            f"Saved denoising video ({min(args.n_fan_anchors, len(anchors))} anchors, "
            f"{args.denoising_n_samples} samples/anchor, "
            f"{args.sample_steps + 1} frames @ {args.denoising_fps} fps) to {video_path}"
        )
        return

    plot_action_chunk_distribution(
        model,
        cfg,
        stats,
        anchors,
        flow_anchors=flow_anchors,
        dataset_id=args.dataset_id,
        goal=goal,
        n_samples=args.n_samples,
        n_fan_anchors=args.n_fan_anchors,
        sample_steps=args.sample_steps,
        vector_scale=args.vector_scale,
        device=device,
        seed=args.seed,
        title=f"Action Chunk Distribution ({cfg.mode})",
        save_path=save_path,
        show=not args.no_show,
    )
    print(f"Saved plot to {save_path}")


if __name__ == "__main__":
    main()
