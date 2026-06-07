"""Evaluation visualization for residual flow rollouts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import minari
import numpy as np

from fm_bc.eval.plot import draw_maze, get_maze_map


FLOW_COLOR = "#1f77b4"
RESIDUAL_COLOR = "#ff7f0e"
COMBINED_COLOR = "#2ca02c"
TRAJECTORY_COLOR = "#9467bd"
SUCCESS_COLOR = "#2ca02c"
FAILURE_COLOR = "#d62728"


def _draw_action_arrows(
    ax: plt.Axes,
    start_pos: np.ndarray,
    flow: np.ndarray,
    residual: np.ndarray,
    combined: np.ndarray,
    *,
    alpha: float = 0.95,
    arrow_width: float = 0.0035,
    arrow_length_scale: float = 0.5,
) -> None:
    """Draw flow + residual from start_pos; residual starts at flow tip."""
    cursor = np.asarray(start_pos, dtype=np.float64)
    quiver_kwargs = dict(
        angles="xy",
        scale_units="xy",
        scale=1.0,
        alpha=alpha,
        width=arrow_width,
        headwidth=2.0,
        headlength=2.5,
    )

    for flow_delta, residual_delta, combined_delta in zip(flow, residual, combined):
        flow_vis = flow_delta * arrow_length_scale
        residual_vis = residual_delta * arrow_length_scale
        combined_vis = combined_delta * arrow_length_scale
        flow_tip = cursor + flow_vis

        ax.quiver(
            cursor[0],
            cursor[1],
            flow_vis[0],
            flow_vis[1],
            color=FLOW_COLOR,
            zorder=4,
            **quiver_kwargs,
        )
        ax.quiver(
            flow_tip[0],
            flow_tip[1],
            residual_vis[0],
            residual_vis[1],
            color=RESIDUAL_COLOR,
            zorder=5,
            **quiver_kwargs,
        )
        ax.quiver(
            cursor[0],
            cursor[1],
            combined_vis[0],
            combined_vis[1],
            color=COMBINED_COLOR,
            zorder=6,
            **quiver_kwargs,
        )
        cursor = cursor + combined_delta


def plot_eval_rollouts(
    rollouts: list[dict[str, Any]],
    *,
    dataset_id: str,
    title: str,
    save_path: str | Path | None = None,
    show: bool = False,
    arrow_stride: int = 1,
    arrow_length_scale: float = 0.5,
) -> plt.Figure:
    """Plot eval rollouts with flow/residual/combined action arrows."""
    n_rollouts = len(rollouts)
    n_cols = min(3, n_rollouts)
    n_rows = int(np.ceil(n_rollouts / n_cols))

    dataset = minari.load_dataset(dataset_id, download=False)
    maze_map = get_maze_map(dataset)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)

    for idx, rollout in enumerate(rollouts):
        row, col = divmod(idx, n_cols)
        ax = axes[row][col]
        draw_maze(ax, maze_map)

        positions = rollout["positions"]
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            color=TRAJECTORY_COLOR,
            linewidth=1.5,
            alpha=0.8,
            zorder=2,
        )
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            s=50,
            color="black",
            marker="o",
            zorder=7,
        )
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            s=60,
            color=COMBINED_COLOR if rollout["success"] else FAILURE_COLOR,
            marker="x",
            linewidths=2.0,
            zorder=7,
        )

        goal = rollout["goal"]
        ax.scatter(
            goal[0],
            goal[1],
            s=120,
            marker="*",
            c="gold",
            edgecolors="black",
            linewidths=0.6,
            zorder=7,
        )

        for step in rollout["steps"]:
            if step["step_idx"] % arrow_stride != 0:
                continue
            _draw_action_arrows(
                ax,
                step["start_pos"],
                step["flow"],
                step["residual"],
                step["combined"],
                arrow_length_scale=arrow_length_scale,
            )

        status = "success" if rollout["success"] else "fail"
        ax.set_title(
            f"Rollout {idx + 1}: {status}\n"
            f"steps={len(rollout['steps'])} return={rollout['total_return']:.1f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, zorder=1)

    for idx in range(n_rollouts, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    handles = [
        plt.Line2D([0], [0], color=FLOW_COLOR, lw=2.5, label="flow (base policy)"),
        plt.Line2D([0], [0], color=RESIDUAL_COLOR, lw=2.5, label="residual (from flow tip)"),
        plt.Line2D([0], [0], color=COMBINED_COLOR, lw=2.5, label="combined (executed)"),
        plt.Line2D([0], [0], color=TRAJECTORY_COLOR, lw=2, label="actual path"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, fontsize=13, y=1.05)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_eval_paths(
    rollouts: list[dict[str, Any]],
    *,
    dataset_id: str,
    title: str,
    save_path: str | Path | None = None,
    show: bool = False,
) -> plt.Figure:
    """Plot all eval trajectories on one maze, green=success and red=failure."""
    dataset = minari.load_dataset(dataset_id, download=False)
    maze_map = get_maze_map(dataset)

    fig, ax = plt.subplots(figsize=(8, 7))
    draw_maze(ax, maze_map)

    for rollout in rollouts:
        color = SUCCESS_COLOR if rollout["success"] else FAILURE_COLOR
        positions = rollout["positions"]
        ax.plot(
            positions[:, 0],
            positions[:, 1],
            color=color,
            linewidth=1.5,
            alpha=0.9,
            zorder=2,
        )
        ax.scatter(
            positions[0, 0],
            positions[0, 1],
            color=color,
            s=40,
            marker="o",
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        ax.scatter(
            positions[-1, 0],
            positions[-1, 1],
            color=color,
            s=40,
            marker="x",
            linewidths=1.5,
            zorder=3,
        )
        goal = rollout["goal"]
        ax.scatter(
            goal[0],
            goal[1],
            color=color,
            s=80,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    n_success = sum(int(r["success"]) for r in rollouts)
    ax.set_title(f"{title}\n{dataset_id} | success={n_success}/{len(rollouts)}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, zorder=1)

    handles = [
        plt.Line2D([0], [0], color=SUCCESS_COLOR, lw=2, label="success"),
        plt.Line2D([0], [0], color=FAILURE_COLOR, lw=2, label="failure"),
    ]
    ax.legend(handles=handles, loc="upper right")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig
