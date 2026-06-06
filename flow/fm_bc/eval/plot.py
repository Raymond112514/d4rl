"""Plot BC policy rollouts in the point maze."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import minari
import numpy as np

from fm_bc.eval.rollout import RolloutResult


def get_maze_map(dataset: minari.MinariDataset) -> list[list[int]]:
    maze_map = dataset.env_spec.kwargs.get("maze_map")
    if maze_map is None:
        raise ValueError("Dataset env_spec does not contain a maze_map.")
    return maze_map


def maze_extent(
    maze_map: list[list[int]],
    maze_size_scaling: float = 1.0,
) -> tuple[float, float, float, float]:
    map_length = len(maze_map)
    map_width = len(maze_map[0])
    x_center = map_width / 2 * maze_size_scaling
    y_center = map_length / 2 * maze_size_scaling
    return (
        -x_center,
        map_width * maze_size_scaling - x_center,
        y_center - map_length * maze_size_scaling,
        y_center,
    )


def draw_maze(ax: plt.Axes, maze_map: list[list[int]], maze_size_scaling: float = 1.0) -> None:
    maze_array = np.asarray(maze_map, dtype=float)
    extent = maze_extent(maze_map, maze_size_scaling)
    ax.imshow(
        maze_array,
        cmap="gray_r",
        origin="upper",
        extent=extent,
        alpha=0.35,
        vmin=0,
        vmax=1,
        zorder=0,
    )


def plot_policy_rollouts(
    rollouts: list[RolloutResult],
    *,
    dataset_id: str = "D4RL/pointmaze/large-dense-v2",
    title: str = "BC Policy Rollouts",
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    dataset = minari.load_dataset(dataset_id, download=False)
    maze_map = get_maze_map(dataset)

    fig, ax = plt.subplots(figsize=(8, 7))
    draw_maze(ax, maze_map)

    for rollout in rollouts:
        color = "#2ca02c" if rollout.success else "#d62728"
        positions = rollout.positions
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
        ax.scatter(
            rollout.goal[0],
            rollout.goal[1],
            color=color,
            s=80,
            marker="*",
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )

    ax.set_title(f"{title}\n{dataset_id}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2, zorder=1)

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
