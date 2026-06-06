"""Policy rollout utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch

from fm_bc.datasets.pointmaze import PointMazeStats
from fm_bc.models.flow_policy import FlowPolicy, FlowPolicyConfig

# PointMaze uses a continuing task: reaching a goal samples a new target.
GOAL_TOLERANCE = 0.45


@dataclass
class EvalResetConfig:
    fixed: bool = False
    seed: int = 0
    goal_cell: Optional[tuple[int, int]] = None
    reset_cell: Optional[tuple[int, int]] = None


@dataclass
class RolloutResult:
    positions: np.ndarray
    goal: np.ndarray
    return_: float
    success: bool
    length: int


def normalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x - mean) / std


def denormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std + mean


def _unwrap_env(env):
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


def resolve_eval_reset_cells(
    env,
    *,
    seed: int,
    goal_cell: Optional[tuple[int, int]] = None,
    reset_cell: Optional[tuple[int, int]] = None,
) -> tuple[tuple[int, int], tuple[int, int]]:
    maze = _unwrap_env(env).maze
    rng = np.random.default_rng(seed)

    if goal_cell is None:
        goal_idx = int(rng.integers(len(maze.unique_goal_locations)))
        goal_xy = maze.unique_goal_locations[goal_idx]
        goal_cell = tuple(int(v) for v in maze.cell_xy_to_rowcol(goal_xy))

    if reset_cell is None:
        reset_idx = int(rng.integers(len(maze.unique_reset_locations)))
        reset_xy = maze.unique_reset_locations[reset_idx]
        reset_cell = tuple(int(v) for v in maze.cell_xy_to_rowcol(reset_xy))

    return goal_cell, reset_cell


def cell_to_xy(env, cell: tuple[int, int]) -> np.ndarray:
    maze = _unwrap_env(env).maze
    return np.asarray(
        maze.cell_rowcol_to_xy(np.asarray(cell, dtype=np.int64)),
        dtype=np.float64,
    )


def get_eval_cell_candidates(env) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    maze = _unwrap_env(env).maze
    reset_cells = [
        tuple(int(v) for v in maze.cell_xy_to_rowcol(xy))
        for xy in maze.unique_reset_locations
    ]
    goal_cells = [
        tuple(int(v) for v in maze.cell_xy_to_rowcol(xy))
        for xy in maze.unique_goal_locations
    ]
    return reset_cells, goal_cells


def _maze_free_cells(env) -> set[tuple[int, int]]:
    maze = _unwrap_env(env).maze
    free_cells: set[tuple[int, int]] = set()
    for row in range(maze.map_length):
        for col in range(maze.map_width):
            if maze.maze_map[row][col] == 1:
                continue
            free_cells.add((row, col))
    return free_cells


def maze_shortest_path_length(
    env,
    start_cell: tuple[int, int],
    goal_cell: tuple[int, int],
) -> Optional[int]:
    if start_cell == goal_cell:
        return 0

    free_cells = _maze_free_cells(env)
    if start_cell not in free_cells or goal_cell not in free_cells:
        return None

    frontier = [start_cell]
    visited = {start_cell}
    distance = 0
    while frontier:
        distance += 1
        next_frontier: list[tuple[int, int]] = []
        for row, col in frontier:
            for drow, dcol in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                neighbor = (row + drow, col + dcol)
                if neighbor not in free_cells or neighbor in visited:
                    continue
                if neighbor == goal_cell:
                    return distance
                visited.add(neighbor)
                next_frontier.append(neighbor)
        frontier = next_frontier
    return None


HardestEvalMetric = Literal["path", "euclidean"]


def find_hardest_eval_cells(
    env,
    *,
    metric: HardestEvalMetric = "path",
) -> tuple[tuple[int, int], tuple[int, int], dict[str, float]]:
    """Pick the most challenging fixed start/goal cells for point-maze eval."""
    reset_cells, goal_cells = get_eval_cell_candidates(env)
    best_reset: tuple[int, int] | None = None
    best_goal: tuple[int, int] | None = None
    best_score = -1.0
    best_path = -1.0
    best_euclidean = -1.0

    for reset_cell in reset_cells:
        reset_xy = cell_to_xy(env, reset_cell)
        for goal_cell in goal_cells:
            goal_xy = cell_to_xy(env, goal_cell)
            euclidean = float(np.linalg.norm(goal_xy - reset_xy))
            path_length = maze_shortest_path_length(env, reset_cell, goal_cell)
            if path_length is None:
                continue

            if metric == "path":
                score = float(path_length)
            else:
                score = euclidean

            if score > best_score:
                best_score = score
                best_reset = reset_cell
                best_goal = goal_cell
                best_path = float(path_length)
                best_euclidean = euclidean

    if best_reset is None or best_goal is None:
        raise ValueError("Could not find a valid reset/goal cell pair for eval.")

    return best_reset, best_goal, {
        "score": best_score,
        "path_length": best_path,
        "euclidean": best_euclidean,
        "metric": metric,
    }


def format_eval_cell(env, cell: tuple[int, int]) -> str:
    xy = cell_to_xy(env, cell)
    return f"{cell} xy=[{xy[0]:.3f}, {xy[1]:.3f}]"


def configure_fixed_eval_env(env, *, fixed: bool) -> Optional[bool]:
    if not fixed:
        return None
    base = _unwrap_env(env)
    previous_reset_target = base.reset_target
    base.reset_target = False
    return previous_reset_target


def configure_episodic_eval_env(env) -> tuple[bool, bool]:
    """Use episodic goal-reaching during eval (stop and reset on success)."""
    base = _unwrap_env(env)
    previous = (base.reset_target, base.continuing_task)
    base.reset_target = False
    base.continuing_task = False
    return previous


def restore_episodic_eval_env(env, previous: tuple[bool, bool]) -> None:
    base = _unwrap_env(env)
    base.reset_target, base.continuing_task = previous


def restore_fixed_eval_env(env, previous_reset_target: Optional[bool]) -> None:
    if previous_reset_target is not None:
        _unwrap_env(env).reset_target = previous_reset_target


@torch.no_grad()
def rollout_episode(
    model: FlowPolicy,
    env,
    stats: PointMazeStats,
    cfg: FlowPolicyConfig,
    *,
    seed: int,
    sample_steps: int,
    device: torch.device,
    max_steps: int = 1000,
    eval_reset: Optional[EvalResetConfig] = None,
    episodic_eval: bool = False,
) -> RolloutResult:
    reset_seed = seed
    reset_options: Optional[dict] = None
    fixed_goal_conditioning = episodic_eval or (
        eval_reset is not None and eval_reset.fixed
    )

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
    action_queue: list[np.ndarray] = []
    ep_return = 0.0
    done = False
    steps = 0

    initial_goal = np.asarray(obs_dict["desired_goal"], dtype=np.float64).copy()
    reached_initial_goal = np.linalg.norm(
        obs_dict["achieved_goal"] - initial_goal
    ) <= GOAL_TOLERANCE

    while not done and steps < max_steps:
        if not action_queue:
            obs = normalize(obs_dict["observation"], stats.obs_mean, stats.obs_std)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

            goal_t = None
            if cfg.mode == "goal_conditioned":
                goal = initial_goal if fixed_goal_conditioning else obs_dict["desired_goal"]
                goal_norm = normalize(goal, stats.goal_mean, stats.goal_std)
                goal_t = torch.as_tensor(goal_norm, dtype=torch.float32, device=device).unsqueeze(0)

            action_chunk_norm = model.sample(obs_t, goal=goal_t, n_steps=sample_steps)
            action_chunk = denormalize(
                action_chunk_norm.cpu().numpy()[0],
                stats.action_mean,
                stats.action_std,
            )
            action_queue = [
                np.clip(action, env.action_space.low, env.action_space.high)
                for action in action_chunk
            ]

        action = action_queue.pop(0)
        obs_dict, reward, terminated, truncated, info = env.step(action)
        ep_return += float(reward)
        positions.append(np.asarray(obs_dict["achieved_goal"], dtype=np.float64).copy())
        if not reached_initial_goal:
            reached_initial_goal = np.linalg.norm(
                obs_dict["achieved_goal"] - initial_goal
            ) <= GOAL_TOLERANCE
        done = bool(terminated or truncated) or reached_initial_goal
        steps += 1

    return RolloutResult(
        positions=np.stack(positions, axis=0),
        goal=initial_goal,
        return_=ep_return,
        success=reached_initial_goal,
        length=steps,
    )


@torch.no_grad()
def rollout_policy(
    model: FlowPolicy,
    env,
    stats: PointMazeStats,
    cfg: FlowPolicyConfig,
    *,
    n_episodes: int,
    seed: int,
    sample_steps: int,
    device: torch.device,
    max_steps: int = 1000,
    eval_reset: Optional[EvalResetConfig] = None,
    episodic_eval: bool = False,
) -> tuple[list[RolloutResult], dict[str, float]]:
    previous_episodic_settings = None
    previous_reset_target = None
    if episodic_eval:
        previous_episodic_settings = configure_episodic_eval_env(env)
    else:
        previous_reset_target = configure_fixed_eval_env(
            env,
            fixed=eval_reset is not None and eval_reset.fixed,
        )
    results: list[RolloutResult] = []
    try:
        for ep_idx in range(n_episodes):
            policy_seed = seed + ep_idx
            results.append(
                rollout_episode(
                    model,
                    env,
                    stats,
                    cfg,
                    seed=policy_seed,
                    sample_steps=sample_steps,
                    device=device,
                    max_steps=max_steps,
                    eval_reset=eval_reset,
                    episodic_eval=episodic_eval,
                )
            )
    finally:
        if episodic_eval:
            restore_episodic_eval_env(env, previous_episodic_settings)
        else:
            restore_fixed_eval_env(env, previous_reset_target)

    success_rate = sum(r.success for r in results) / max(n_episodes, 1)
    mean_return = float(np.mean([r.return_ for r in results]))
    metrics = {
        "eval/success_rate": success_rate,
        "eval/mean_return": mean_return,
    }
    return results, metrics
