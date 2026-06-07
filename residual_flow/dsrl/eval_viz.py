"""Evaluation visualization for residual flow DSRL rollouts."""

from __future__ import annotations

import importlib.util
import os

RESIDUAL_RL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "residual_rl")
_RL_EVAL_VIZ_PATH = os.path.join(RESIDUAL_RL_ROOT, "eval_viz.py")
_spec = importlib.util.spec_from_file_location("residual_rl_eval_viz", _RL_EVAL_VIZ_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"Could not load residual RL eval_viz from {_RL_EVAL_VIZ_PATH}")
_rr_eval_viz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rr_eval_viz)

FLOW_COLOR = _rr_eval_viz.FLOW_COLOR
RESIDUAL_COLOR = _rr_eval_viz.RESIDUAL_COLOR
COMBINED_COLOR = _rr_eval_viz.COMBINED_COLOR
TRAJECTORY_COLOR = _rr_eval_viz.TRAJECTORY_COLOR
SUCCESS_COLOR = _rr_eval_viz.SUCCESS_COLOR
FAILURE_COLOR = _rr_eval_viz.FAILURE_COLOR
_format_action_chunk = _rr_eval_viz._format_action_chunk
_single_env_slot = _rr_eval_viz._single_env_slot
_step_end_position = _rr_eval_viz._step_end_position
_draw_action_arrows = _rr_eval_viz._draw_action_arrows
plot_eval_paths = _rr_eval_viz.plot_eval_paths
plot_eval_rollouts = _rr_eval_viz.plot_eval_rollouts

from typing import Any

import numpy as np


def collect_eval_rollout(
    env: Any,
    agent: Any,
    max_steps: int,
    *,
    rollout_idx: int = 1,
    deterministic: bool = False,
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
        noise = np.asarray(action, dtype=np.float32).reshape(env.residual_dim)
        decomp = env.decompose_action(flow_norm, noise, obs_dict=slot.obs_dict)

        if print_action_stride > 0 and step_idx % print_action_stride == 0:
            print(
                f"  rollout {rollout_idx:02d} step {step_idx:03d}: "
                f"flow={_format_action_chunk(decomp['flow'])} "
                f"scaled_residual={_format_action_chunk(decomp['residual'])}"
            )

        obs, reward, terminated, truncated, info = env.step(noise)
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
    deterministic: bool = False,
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
                decomp = env.decompose_action(
                    slot.last_flow_action_norm,
                    actions[env_idx],
                    obs_dict=slot.obs_dict,
                    env_idx=env_idx,
                )
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


__all__ = [
    "FLOW_COLOR",
    "RESIDUAL_COLOR",
    "COMBINED_COLOR",
    "TRAJECTORY_COLOR",
    "SUCCESS_COLOR",
    "FAILURE_COLOR",
    "collect_eval_rollout",
    "collect_eval_rollouts",
    "plot_eval_paths",
    "plot_eval_rollouts",
]
