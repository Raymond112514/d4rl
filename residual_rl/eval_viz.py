"""Evaluation visualization for residual RL rollouts."""

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


def _format_action_chunk(action_chunk: np.ndarray) -> str:
    actions = np.asarray(action_chunk, dtype=np.float64)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    else:
        actions = actions.reshape(-1, actions.shape[-1])
    if actions.shape[0] == 1:
        return f"({actions[0, 0]:+.4f}, {actions[0, 1]:+.4f})"
    rows = ", ".join(f"({row[0]:+.4f}, {row[1]:+.4f})" for row in actions)
    return f"[{rows}]"


def _single_env_slot(env: Any) -> Any:
    if hasattr(env, "_slot"):
        return env._slot
    if hasattr(env, "_slots"):
        return env._slots[0]
    return None


def _step_end_position(info: dict[str, Any], start_pos: np.ndarray, combined: np.ndarray) -> np.ndarray:
    if "terminal_agent_pos" in info:
        return np.asarray(info["terminal_agent_pos"], dtype=np.float64).copy()
    if "achieved_goal" in info:
        return np.asarray(info["achieved_goal"], dtype=np.float64)[:2].copy()
    return start_pos + np.sum(combined, axis=0)


def collect_eval_rollout(
    env: Any,
    agent: Any,
    max_steps: int,
    *,
    rollout_idx: int = 1,
    deterministic: bool = True,
    print_action_stride: int = 0,
) -> dict[str, Any]:
    """Collect one single-env eval rollout."""
    obs, _ = env.reset()
    positions = [env.get_agent_position()]
    sac_steps: list[dict[str, Any]] = []
    success = False
    total_reward = 0.0
    slot = _single_env_slot(env)

    for step_idx in range(max_steps):
        start_pos = env.get_agent_position()
        flow_norm = slot.last_flow_action_norm
        action, _ = agent.predict(obs, deterministic=deterministic)
        residual = np.asarray(action, dtype=np.float32).reshape(env.residual_dim)
        decomp = env.decompose_action(flow_norm, residual)

        if print_action_stride > 0 and step_idx % print_action_stride == 0:
            print(
                f"  rollout {rollout_idx:02d} step {step_idx:03d}: "
                f"flow={_format_action_chunk(decomp['flow'])} "
                f"scaled_residual={_format_action_chunk(decomp['residual'])}"
            )

        obs, reward, terminated, truncated, info = env.step(residual)
        total_reward += float(reward)
        if info.get("success", reward >= 0.0):
            success = True
        end_pos = _step_end_position(info, start_pos, decomp["combined"])
        positions.append(end_pos.copy())

        sac_steps.append(
            {
                "step_idx": step_idx,
                "reward": float(reward),
                "chunk_env_return": float(info.get("chunk_env_return", 0.0)),
                "success": success,
                "start_pos": start_pos,
                "end_pos": end_pos,
                "flow": decomp["flow"],
                "residual": decomp["residual"],
                "combined": decomp["combined"],
            }
        )

        if success or terminated or truncated:
            break

    goal = np.asarray(slot.initial_goal, dtype=np.float64)[:2]
    return {
        "positions": np.stack(positions, axis=0),
        "goal": goal,
        "success": success,
        "total_reward": total_reward,
        "sac_steps": sac_steps,
    }


def collect_eval_rollouts(
    env: Any,
    agent: Any,
    num_rollouts: int,
    max_steps: int,
    *,
    deterministic: bool = True,
    print_action_stride: int = 0,
) -> list[dict[str, Any]]:
    """Collect eval rollouts from a single-env or vectorized wrapper."""
    n_envs = getattr(env, "num_envs", getattr(env, "n_envs", 1))
    is_vec = hasattr(env, "step_async")
    if not is_vec and n_envs == 1:
        return [
            collect_eval_rollout(
                env,
                agent,
                max_steps,
                rollout_idx=rollout_idx + 1,
                deterministic=deterministic,
                print_action_stride=print_action_stride,
            )
            for rollout_idx in range(num_rollouts)
        ]

    n_rounds = int(np.ceil(num_rollouts / n_envs))
    rollouts: list[dict[str, Any]] = []

    for round_idx in range(n_rounds):
        obs = env.reset()
        active = np.ones(n_envs, dtype=bool)
        positions: list[list[np.ndarray]] = [[env.get_agent_position(i)] for i in range(n_envs)]
        sac_steps: list[list[dict[str, Any]]] = [[] for _ in range(n_envs)]
        success = np.zeros(n_envs, dtype=bool)
        total_reward = np.zeros(n_envs, dtype=np.float64)
        goals = [
            np.asarray(env._slots[i].initial_goal, dtype=np.float64)[:2]
            for i in range(n_envs)
        ]

        for step_idx in range(max_steps):
            if not active.any():
                break

            actions, _ = agent.predict(obs, deterministic=deterministic)
            actions = np.asarray(actions, dtype=np.float32).reshape(n_envs, env.residual_dim)

            step_records: list[dict[str, Any] | None] = [None] * n_envs
            for env_idx in range(n_envs):
                if not active[env_idx]:
                    continue
                slot = env._slots[env_idx]
                start_pos = env.get_agent_position(env_idx)
                decomp = env.decompose_action(slot.last_flow_action_norm, actions[env_idx])
                step_records[env_idx] = {
                    "step_idx": step_idx,
                    "start_pos": start_pos,
                    "flow": decomp["flow"],
                    "residual": decomp["residual"],
                    "combined": decomp["combined"],
                }

            obs, rewards, dones, infos = env.step(actions)

            for env_idx in range(n_envs):
                if not active[env_idx]:
                    continue

                reward = float(rewards[env_idx])
                info = infos[env_idx]
                total_reward[env_idx] += reward
                record = step_records[env_idx]
                assert record is not None
                end_pos = _step_end_position(info, record["start_pos"], record["combined"])
                positions[env_idx].append(end_pos.copy())

                record.update(
                    {
                        "reward": reward,
                        "chunk_env_return": float(info.get("chunk_env_return", 0.0)),
                        "success": bool(info.get("success", reward >= 0.0)),
                        "end_pos": end_pos,
                    }
                )
                sac_steps[env_idx].append(record)

                if info.get("success", reward >= 0.0):
                    success[env_idx] = True
                    active[env_idx] = False
                elif dones[env_idx]:
                    active[env_idx] = False

        for env_idx in range(n_envs):
            rollout_idx = round_idx * n_envs + env_idx
            if rollout_idx >= num_rollouts:
                break
            rollouts.append(
                {
                    "positions": np.stack(positions[env_idx], axis=0),
                    "goal": goals[env_idx],
                    "success": bool(success[env_idx]),
                    "total_reward": float(total_reward[env_idx]),
                    "sac_steps": sac_steps[env_idx],
                }
            )

    return rollouts


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
    """Draw flow + combined from start_pos; residual starts at flow tip (triangle)."""
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
    close_fig: bool = True,
    arrow_stride: int = 20,
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

        for step in rollout["sac_steps"]:
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
            f"steps={len(rollout['sac_steps'])} reward={rollout['total_reward']:.1f}"
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15, zorder=1)

    for idx in range(n_rollouts, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].axis("off")

    handles = [
        plt.Line2D([0], [0], color=FLOW_COLOR, lw=2.5, label="flow (from agent)"),
        plt.Line2D([0], [0], color=RESIDUAL_COLOR, lw=2.5, label="scaled residual (from flow tip)"),
        plt.Line2D([0], [0], color=COMBINED_COLOR, lw=2.5, label="combined (from agent)"),
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
    elif close_fig:
        plt.close(fig)

    return fig


def plot_eval_paths(
    rollouts: list[dict[str, Any]],
    *,
    dataset_id: str,
    title: str,
    save_path: str | Path | None = None,
    show: bool = False,
    close_fig: bool = True,
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
    ax.set_title(
        f"{title}\n{dataset_id} | success={n_success}/{len(rollouts)}"
    )
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
    elif close_fig:
        plt.close(fig)

    return fig
