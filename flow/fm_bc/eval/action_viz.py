"""Visualize action-chunk distributions over the point maze."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import minari
import numpy as np
import torch

from fm_bc.datasets.pointmaze import PointMazeStats, load_pointmaze_episodes
from fm_bc.eval.plot import draw_maze, get_maze_map
from fm_bc.eval.rollout import _unwrap_env, normalize
from fm_bc.models.flow_policy import FlowPolicy, FlowPolicyConfig


@dataclass
class AnchorState:
    position: np.ndarray
    obs: np.ndarray
    goal: np.ndarray | None


def denormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


def chain_action_chunk(
    actions: np.ndarray,
    *,
    start: np.ndarray,
    scale: float,
) -> np.ndarray:
    """Build a tail-to-tail path from an action chunk, starting at `start`."""
    points = [np.asarray(start, dtype=np.float64)]
    cursor = np.asarray(start, dtype=np.float64)
    for action in actions:
        cursor = cursor + scale * np.asarray(action, dtype=np.float64)
        points.append(cursor.copy())
    return np.stack(points, axis=0)


@torch.no_grad()
def sample_action_chunks(
    model: FlowPolicy,
    stats: PointMazeStats,
    cfg: FlowPolicyConfig,
    anchor: AnchorState,
    *,
    n_samples: int,
    sample_steps: int,
    device: torch.device,
) -> np.ndarray:
    obs_norm = normalize(anchor.obs, stats.obs_mean, stats.obs_std)
    obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)
    obs_batch = obs_t.unsqueeze(0).repeat(n_samples, 1)

    goal_batch = None
    if cfg.mode == "goal_conditioned":
        if anchor.goal is None:
            raise ValueError("goal_conditioned policy requires a goal for each anchor")
        goal_norm = normalize(anchor.goal, stats.goal_mean, stats.goal_std)
        goal_t = torch.as_tensor(goal_norm, dtype=torch.float32, device=device)
        goal_batch = goal_t.unsqueeze(0).repeat(n_samples, 1)

    chunks_norm = model.sample(obs_batch, goal=goal_batch, n_steps=sample_steps)
    chunks = denormalize(
        chunks_norm.cpu().numpy(),
        stats.action_mean,
        stats.action_std,
    )
    return chunks


@torch.no_grad()
def sample_action_chunk_trajectory(
    model: FlowPolicy,
    stats: PointMazeStats,
    cfg: FlowPolicyConfig,
    anchor: AnchorState,
    *,
    sample_steps: int,
    device: torch.device,
    seed: int | None = None,
) -> np.ndarray:
    """Return one denoising trajectory in action space, including initial noise.

    Shape: (n_steps + 1, ac_chunk, action_dim).
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    obs_norm = normalize(anchor.obs, stats.obs_mean, stats.obs_std)
    obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

    goal_batch = None
    if cfg.mode == "goal_conditioned":
        if anchor.goal is None:
            raise ValueError("goal_conditioned policy requires a goal for each anchor")
        goal_norm = normalize(anchor.goal, stats.goal_mean, stats.goal_std)
        goal_t = torch.as_tensor(goal_norm, dtype=torch.float32, device=device)
        goal_batch = goal_t.unsqueeze(0)

    trajectory_norm = model.sample_trajectory(obs_t, goal=goal_batch, n_steps=sample_steps)
    trajectory_norm = trajectory_norm.squeeze(1).cpu().numpy()
    return denormalize(trajectory_norm, stats.action_mean, stats.action_std)


@torch.no_grad()
def sample_action_chunk_trajectories(
    model: FlowPolicy,
    stats: PointMazeStats,
    cfg: FlowPolicyConfig,
    anchor: AnchorState,
    *,
    n_samples: int,
    sample_steps: int,
    device: torch.device,
    seed: int | None = None,
) -> np.ndarray:
    """Return parallel denoising trajectories for one anchor, including initial noise.

    Shape: (n_steps + 1, n_samples, ac_chunk, action_dim).
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    obs_norm = normalize(anchor.obs, stats.obs_mean, stats.obs_std)
    obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)
    obs_batch = obs_t.unsqueeze(0).repeat(n_samples, 1)

    goal_batch = None
    if cfg.mode == "goal_conditioned":
        if anchor.goal is None:
            raise ValueError("goal_conditioned policy requires a goal for each anchor")
        goal_norm = normalize(anchor.goal, stats.goal_mean, stats.goal_std)
        goal_t = torch.as_tensor(goal_norm, dtype=torch.float32, device=device)
        goal_batch = goal_t.unsqueeze(0).repeat(n_samples, 1)

    trajectory_norm = model.sample_trajectory(obs_batch, goal=goal_batch, n_steps=sample_steps)
    trajectory_norm = trajectory_norm.cpu().numpy()
    return denormalize(trajectory_norm, stats.action_mean, stats.action_std)


def get_free_cell_positions(env) -> np.ndarray:
    maze = _unwrap_env(env).maze
    positions: list[np.ndarray] = []
    for row in range(maze.map_length):
        for col in range(maze.map_width):
            if maze.maze_map[row][col] == 1:
                continue
            xy = maze.cell_rowcol_to_xy(np.asarray([row, col], dtype=np.int64))
            positions.append(np.asarray(xy, dtype=np.float64))
    return np.stack(positions, axis=0)


def build_grid_anchors(
    env,
    *,
    stride: int,
    goal: np.ndarray | None,
) -> list[AnchorState]:
    positions = get_free_cell_positions(env)
    anchors: list[AnchorState] = []
    for idx in range(0, len(positions), stride):
        pos = positions[idx]
        obs = np.array([pos[0], pos[1], 0.0, 0.0], dtype=np.float64)
        anchors.append(AnchorState(position=pos, obs=obs, goal=goal))
    return anchors


def build_dataset_anchors(
    dataset_id: str,
    *,
    n_anchors: int,
    seed: int,
    goal: np.ndarray | None,
    use_episode_goals: bool,
) -> list[AnchorState]:
    episodes = load_pointmaze_episodes(dataset_id=dataset_id, download=False)
    achieved = np.concatenate(
        [ep.obs[:, :2] for ep in episodes],
        axis=0,
    )
    obs_all = np.concatenate([ep.obs for ep in episodes], axis=0)
    goals_all = np.concatenate([ep.goals for ep in episodes], axis=0)

    rng = np.random.default_rng(seed)
    replace = n_anchors > len(obs_all)
    indices = rng.choice(len(obs_all), size=n_anchors, replace=replace)

    anchors: list[AnchorState] = []
    for idx in indices:
        anchor_goal = goals_all[idx] if use_episode_goals else goal
        anchors.append(
            AnchorState(
                position=achieved[idx].astype(np.float64),
                obs=obs_all[idx].astype(np.float64),
                goal=None if anchor_goal is None else anchor_goal.astype(np.float64),
            )
        )
    return anchors


def _subsample_anchors(anchors: list[AnchorState], n_show: int, seed: int) -> list[AnchorState]:
    if n_show >= len(anchors):
        return anchors
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(anchors), size=n_show, replace=False)
    return [anchors[i] for i in sorted(indices.tolist())]


def plot_action_chunk_distribution(
    model: FlowPolicy,
    cfg: FlowPolicyConfig,
    stats: PointMazeStats,
    anchors: list[AnchorState],
    *,
    flow_anchors: list[AnchorState],
    dataset_id: str,
    goal: np.ndarray | None,
    n_samples: int,
    n_fan_anchors: int,
    sample_steps: int,
    vector_scale: float,
    device: torch.device,
    seed: int,
    title: str,
    save_path: str | Path | None = None,
    show: bool = True,
) -> plt.Figure:
    dataset = minari.load_dataset(dataset_id, download=False)
    maze_map = get_maze_map(dataset)

    fan_anchors = _subsample_anchors(anchors, n_fan_anchors, seed=seed)

    mean_vectors: list[np.ndarray] = []
    mean_positions: list[np.ndarray] = []

    for anchor in flow_anchors:
        chunks = sample_action_chunks(
            model,
            stats,
            cfg,
            anchor,
            n_samples=n_samples,
            sample_steps=sample_steps,
            device=device,
        )
        endpoints = np.array(
            [
                chain_action_chunk(chunk, start=anchor.position, scale=vector_scale)[-1]
                - anchor.position
                for chunk in chunks
            ]
        )
        mean_vectors.append(endpoints.mean(axis=0))
        mean_positions.append(anchor.position)

    mean_vectors = np.stack(mean_vectors, axis=0)
    mean_positions = np.stack(mean_positions, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, panel_title in zip(
        axes,
        ("Sampled action chunks", "Mean predicted flow"),
    ):
        draw_maze(ax, maze_map)
        ax.set_title(panel_title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, zorder=1)

    if goal is not None:
        for ax in axes:
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

    fan_ax = axes[0]
    cmap = plt.cm.viridis
    for anchor_idx, anchor in enumerate(fan_anchors):
        chunks = sample_action_chunks(
            model,
            stats,
            cfg,
            anchor,
            n_samples=n_samples,
            sample_steps=sample_steps,
            device=device,
        )
        color = cmap(anchor_idx / max(len(fan_anchors) - 1, 1))
        for sample_idx, chunk in enumerate(chunks):
            path = chain_action_chunk(chunk, start=anchor.position, scale=vector_scale)
            fan_ax.plot(
                path[:, 0],
                path[:, 1],
                color=color,
                alpha=0.35,
                linewidth=1.2,
                zorder=2,
            )
            deltas = path[1:] - path[:-1]
            fan_ax.quiver(
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
        fan_ax.scatter(
            anchor.position[0],
            anchor.position[1],
            s=28,
            color=color,
            edgecolors="black",
            linewidths=0.4,
            zorder=4,
        )

    flow_ax = axes[1]
    magnitudes = np.linalg.norm(mean_vectors, axis=1)
    nonzero = magnitudes > 1e-8
    if np.any(nonzero):
        flow_ax.quiver(
            mean_positions[nonzero, 0],
            mean_positions[nonzero, 1],
            mean_vectors[nonzero, 0],
            mean_vectors[nonzero, 1],
            magnitudes[nonzero],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            cmap="coolwarm",
            width=0.004,
            alpha=0.9,
            zorder=4,
        )

    subtitle = (
        f"{dataset_id} | chunk={cfg.ac_chunk} | samples/anchor={n_samples} "
        f"| flow points={len(flow_anchors)} | scale={vector_scale:g}"
    )
    if goal is not None:
        subtitle += f" | goal=({goal[0]:.2f}, {goal[1]:.2f})"
    fig.suptitle(f"{title}\n{subtitle}", fontsize=13, y=1.02)
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
