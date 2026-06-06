"""PointMaze environment wrapper for residual RL on a frozen flow policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import minari
import numpy as np
import torch
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn

from fm_bc.eval.rollout import (
    GOAL_TOLERANCE,
    configure_episodic_eval_env,
    denormalize,
    normalize,
)
from fm_bc.utils.checkpoint import load_checkpoint
from fm_bc.utils.device import resolve_device


@dataclass
class _EnvSlot:
    env: gym.Env
    obs_dict: dict[str, Any] | None = None
    initial_goal: np.ndarray | None = None
    last_flow_action_norm: np.ndarray | None = None
    episode_steps: int = 0


class _FlowPolicyBundle:
    """Shared frozen flow policy and normalization stats."""

    def __init__(
        self,
        *,
        dataset_id: str,
        flow_checkpoint: str,
        sample_steps: int,
        residual_scale: float,
        action_mag: float,
        use_cpu: bool,
    ) -> None:
        self.dataset_id = dataset_id
        self.flow_checkpoint = flow_checkpoint
        self.sample_steps = sample_steps
        self.residual_scale = residual_scale
        self.action_mag = action_mag

        self.device = resolve_device(use_cpu=use_cpu)
        self.flow_policy, self.cfg, self.stats, _payload = load_checkpoint(
            flow_checkpoint,
            device=self.device,
        )
        self.flow_policy.eval()
        for param in self.flow_policy.parameters():
            param.requires_grad = False

        self.ac_chunk = self.cfg.ac_chunk
        self.action_dim = self.cfg.action_dim
        self.residual_dim = self.ac_chunk * self.action_dim

        obs_dim = self.cfg.obs_dim + self.residual_dim
        if self.cfg.mode == "goal_conditioned":
            obs_dim += self.cfg.goal_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-action_mag * np.ones(self.residual_dim, dtype=np.float32),
            high=action_mag * np.ones(self.residual_dim, dtype=np.float32),
            dtype=np.float32,
        )

    def make_base_env(self) -> gym.Env:
        dataset = minari.load_dataset(self.dataset_id, download=False)
        env = dataset.recover_environment()
        configure_episodic_eval_env(env)
        return env

    @torch.no_grad()
    def sample_flow_actions(
        self,
        obs_dicts: list[dict[str, Any]],
        initial_goals: list[np.ndarray],
    ) -> np.ndarray:
        obs_norm = np.stack(
            [
                normalize(
                    np.asarray(obs_dict["observation"], dtype=np.float64),
                    self.stats.obs_mean,
                    self.stats.obs_std,
                )
                for obs_dict in obs_dicts
            ],
            axis=0,
        )
        obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=self.device)

        goal_t = None
        if self.cfg.mode == "goal_conditioned":
            goals = np.stack(initial_goals, axis=0)
            goal_norm = normalize(goals, self.stats.goal_mean, self.stats.goal_std)
            goal_t = torch.as_tensor(goal_norm, dtype=torch.float32, device=self.device)

        chunks = self.flow_policy.sample(
            obs_t,
            goal=goal_t,
            n_steps=self.sample_steps,
        )
        return chunks.cpu().numpy()

    def decompose_action(
        self,
        flow_action_norm: np.ndarray,
        residual: np.ndarray,
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> dict[str, np.ndarray]:
        physical_chunk = denormalize(
            np.asarray(flow_action_norm, dtype=np.float64).reshape(self.ac_chunk, self.action_dim),
            self.stats.action_mean,
            self.stats.action_std,
        )
        residual_chunk = np.asarray(residual, dtype=np.float64).reshape(self.ac_chunk, self.action_dim)
        scaled_residual = self.residual_scale * residual_chunk
        combined_chunk = physical_chunk + scaled_residual
        combined_chunk = np.clip(combined_chunk, action_low, action_high)
        return {
            "flow": physical_chunk,
            "residual": scaled_residual,
            "combined": combined_chunk,
        }

    def build_sac_obs(
        self,
        obs_dict: dict[str, Any],
        initial_goal: np.ndarray | None,
        flow_action_norm: np.ndarray,
    ) -> np.ndarray:
        obs_norm = normalize(
            np.asarray(obs_dict["observation"], dtype=np.float64),
            self.stats.obs_mean,
            self.stats.obs_std,
        ).astype(np.float32)
        flow_flat = flow_action_norm.reshape(-1).astype(np.float32)
        parts = [obs_norm, flow_flat]
        if self.cfg.mode == "goal_conditioned":
            goal = initial_goal
            if goal is None:
                goal = np.asarray(obs_dict["desired_goal"], dtype=np.float64)
            goal_norm = normalize(goal, self.stats.goal_mean, self.stats.goal_std).astype(np.float32)
            parts.insert(1, goal_norm)
        return np.concatenate(parts, axis=0)


def _execute_chunk_on_slot(
    slot: _EnvSlot,
    bundle: _FlowPolicyBundle,
    flow_action_norm: np.ndarray,
    residual: np.ndarray,
    *,
    max_episode_steps: int,
) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    decomp = bundle.decompose_action(
        flow_action_norm,
        residual,
        slot.env.action_space.low,
        slot.env.action_space.high,
    )
    final_chunk = decomp["combined"]

    initial_goal = np.asarray(slot.obs_dict["desired_goal"], dtype=np.float64).copy()
    reached_goal = (
        np.linalg.norm(slot.obs_dict["achieved_goal"] - initial_goal) <= GOAL_TOLERANCE
    )
    chunk_return = 0.0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}

    for action in final_chunk:
        obs_dict, reward, terminated, truncated, info = slot.env.step(action)
        slot.obs_dict = obs_dict
        slot.episode_steps += 1
        chunk_return += float(reward)
        reached_goal = (
            np.linalg.norm(obs_dict["achieved_goal"] - initial_goal) <= GOAL_TOLERANCE
        )
        if reached_goal or terminated or truncated:
            break
        if slot.episode_steps >= max_episode_steps:
            truncated = True
            break

    success = reached_goal
    sparse_reward = 0.0 if success else -1.0
    if success:
        terminated = True

    info = dict(info)
    info["success"] = success
    info["chunk_env_return"] = chunk_return
    info["terminal_agent_pos"] = np.asarray(obs_dict["achieved_goal"], dtype=np.float64)[:2].copy()
    return slot.obs_dict, sparse_reward, terminated, truncated, info


def _reset_slot(
    slot: _EnvSlot,
    bundle: _FlowPolicyBundle,
    *,
    reset_options: dict[str, Any] | None,
    seed: int | None,
) -> np.ndarray:
    if reset_options is None:
        obs_dict, _info = slot.env.reset(seed=seed)
    else:
        obs_dict, _info = slot.env.reset(seed=seed, options=reset_options)

    slot.obs_dict = obs_dict
    slot.initial_goal = np.asarray(obs_dict["desired_goal"], dtype=np.float64).copy()
    slot.episode_steps = 0
    slot.last_flow_action_norm = bundle.sample_flow_actions([obs_dict], [slot.initial_goal])[0]
    return bundle.build_sac_obs(obs_dict, slot.initial_goal, slot.last_flow_action_norm)


class PointMazeResidualWrapper(gym.Env):
    """Single-env wrapper around a frozen flow policy and learnable residuals."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        dataset_id: str = "D4RL/pointmaze/large-dense-v2",
        flow_checkpoint: str,
        sample_steps: int = 20,
        residual_scale: float = 0.1,
        action_mag: float = 2.0,
        max_episode_steps: int = 1000,
        use_cpu: bool = False,
        reset_options: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.reset_options = reset_options
        self._bundle = _FlowPolicyBundle(
            dataset_id=dataset_id,
            flow_checkpoint=flow_checkpoint,
            sample_steps=sample_steps,
            residual_scale=residual_scale,
            action_mag=action_mag,
            use_cpu=use_cpu,
        )
        self.observation_space = self._bundle.observation_space
        self.action_space = self._bundle.action_space
        self.cfg = self._bundle.cfg
        self.stats = self._bundle.stats
        self.ac_chunk = self._bundle.ac_chunk
        self.action_dim = self._bundle.action_dim
        self.residual_dim = self._bundle.residual_dim
        self.residual_scale = self._bundle.residual_scale

        self._slot = _EnvSlot(env=self._bundle.make_base_env())

    @property
    def num_envs(self) -> int:
        return 1

    @property
    def env(self) -> gym.Env:
        return self._slot.env

    def decompose_action(
        self,
        flow_action_norm: np.ndarray,
        residual: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return self._bundle.decompose_action(
            flow_action_norm,
            residual,
            self._slot.env.action_space.low,
            self._slot.env.action_space.high,
        )

    def get_agent_position(self, env_idx: int = 0) -> np.ndarray:
        del env_idx
        if self._slot.obs_dict is None:
            raise RuntimeError("Environment must be reset before reading agent position.")
        return np.asarray(self._slot.obs_dict["achieved_goal"], dtype=np.float64)[:2].copy()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        reset_options = options if options is not None else self.reset_options
        sac_obs = _reset_slot(self._slot, self._bundle, reset_options=reset_options, seed=seed)
        return sac_obs, {}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._slot.obs_dict is None or self._slot.last_flow_action_norm is None:
            raise RuntimeError("Environment must be reset before calling step().")

        obs_dict, reward, terminated, truncated, info = _execute_chunk_on_slot(
            self._slot,
            self._bundle,
            self._slot.last_flow_action_norm,
            action,
            max_episode_steps=self.max_episode_steps,
        )

        done = terminated or truncated
        if not done:
            self._slot.last_flow_action_norm = self._bundle.sample_flow_actions(
                [obs_dict],
                [self._slot.initial_goal],
            )[0]
        sac_obs = self._bundle.build_sac_obs(
            obs_dict,
            self._slot.initial_goal,
            self._slot.last_flow_action_norm,
        )

        if done:
            info["episode"] = {"r": reward, "l": self._slot.episode_steps}
            terminal_obs = sac_obs
            sac_obs = _reset_slot(
                self._slot,
                self._bundle,
                reset_options=self.reset_options,
                seed=None,
            )
            info["terminal_observation"] = terminal_obs

        return sac_obs, reward, terminated, truncated, info

    def step_rand(
        self,
        model: Any = None,
        *,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any], np.ndarray]:
        if self._slot.obs_dict is None or self._slot.last_flow_action_norm is None:
            sac_obs, _ = self.reset()
        else:
            sac_obs = self._bundle.build_sac_obs(
                self._slot.obs_dict,
                self._slot.initial_goal,
                self._slot.last_flow_action_norm,
            )

        if model is not None:
            residual, _ = model.predict(sac_obs, deterministic=deterministic)
            residual = np.asarray(residual, dtype=np.float32).reshape(self.residual_dim)
        else:
            residual = np.zeros(self.residual_dim, dtype=np.float32)

        next_sac_obs, reward, terminated, truncated, info = self.step(residual)
        return next_sac_obs, reward, terminated, truncated, info, residual

    def close(self) -> None:
        self._slot.env.close()


class PointMazeResidualVecEnv(VecEnv):
    """Vectorized PointMaze wrapper with batched flow inference."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_envs: int,
        *,
        dataset_id: str = "D4RL/pointmaze/large-dense-v2",
        flow_checkpoint: str,
        sample_steps: int = 20,
        residual_scale: float = 0.1,
        action_mag: float = 2.0,
        max_episode_steps: int = 1000,
        use_cpu: bool = False,
        reset_options: Optional[dict[str, np.ndarray]] = None,
        base_seed: int = 0,
    ) -> None:
        self.max_episode_steps = max_episode_steps
        self.reset_options = reset_options
        self.base_seed = base_seed

        self._bundle = _FlowPolicyBundle(
            dataset_id=dataset_id,
            flow_checkpoint=flow_checkpoint,
            sample_steps=sample_steps,
            residual_scale=residual_scale,
            action_mag=action_mag,
            use_cpu=use_cpu,
        )
        super().__init__(
            n_envs,
            self._bundle.observation_space,
            self._bundle.action_space,
        )
        self.observation_space = self._bundle.observation_space
        self.action_space = self._bundle.action_space
        self.cfg = self._bundle.cfg
        self.stats = self._bundle.stats
        self.ac_chunk = self._bundle.ac_chunk
        self.action_dim = self._bundle.action_dim
        self.residual_dim = self._bundle.residual_dim
        self.residual_scale = self._bundle.residual_scale

        self._slots = [_EnvSlot(env=self._bundle.make_base_env()) for _ in range(n_envs)]
        self._actions: np.ndarray | None = None

    @property
    def n_envs(self) -> int:
        return self.num_envs

    @property
    def env(self) -> gym.Env:
        return self._slots[0].env

    def decompose_action(
        self,
        flow_action_norm: np.ndarray,
        residual: np.ndarray,
    ) -> dict[str, np.ndarray]:
        return self._bundle.decompose_action(
            flow_action_norm,
            residual,
            self._slots[0].env.action_space.low,
            self._slots[0].env.action_space.high,
        )

    def get_agent_position(self, env_idx: int) -> np.ndarray:
        if self._slots[env_idx].obs_dict is None:
            raise RuntimeError("Environment must be reset before reading agent position.")
        return np.asarray(self._slots[env_idx].obs_dict["achieved_goal"], dtype=np.float64)[:2].copy()

    def reset(self) -> np.ndarray:
        obs = np.zeros((self.n_envs, self.observation_space.shape[0]), dtype=np.float32)
        for env_idx, slot in enumerate(self._slots):
            seed = self.base_seed + env_idx
            obs[env_idx] = _reset_slot(
                slot,
                self._bundle,
                reset_options=self.reset_options,
                seed=seed,
            )
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, dtype=np.float32).reshape(self.n_envs, self.residual_dim)

    def step_wait(self) -> VecEnvStepReturn:
        if self._actions is None:
            raise RuntimeError("step_async must be called before step_wait().")

        obs = np.zeros((self.n_envs, self.observation_space.shape[0]), dtype=np.float32)
        rewards = np.zeros(self.n_envs, dtype=np.float32)
        dones = np.zeros(self.n_envs, dtype=bool)
        infos: list[dict[str, Any]] = [{} for _ in range(self.n_envs)]

        need_flow_indices: list[int] = []
        need_flow_obs: list[dict[str, Any]] = []
        need_flow_goals: list[np.ndarray] = []

        for env_idx, slot in enumerate(self._slots):
            residual = self._actions[env_idx]
            obs_dict, reward, terminated, truncated, info = _execute_chunk_on_slot(
                slot,
                self._bundle,
                slot.last_flow_action_norm,
                residual,
                max_episode_steps=self.max_episode_steps,
            )
            done = terminated or truncated
            rewards[env_idx] = reward
            dones[env_idx] = done
            infos[env_idx] = info

            if not done:
                need_flow_indices.append(env_idx)
                need_flow_obs.append(obs_dict)
                need_flow_goals.append(slot.initial_goal)
                obs[env_idx] = 0.0
            else:
                info["episode"] = {"r": reward, "l": slot.episode_steps}
                terminal_obs = self._bundle.build_sac_obs(
                    obs_dict,
                    slot.initial_goal,
                    slot.last_flow_action_norm,
                )
                info["terminal_observation"] = terminal_obs
                obs[env_idx] = _reset_slot(
                    slot,
                    self._bundle,
                    reset_options=self.reset_options,
                    seed=self.base_seed + env_idx,
                )

        if need_flow_indices:
            flow_actions = self._bundle.sample_flow_actions(need_flow_obs, need_flow_goals)
            for batch_idx, env_idx in enumerate(need_flow_indices):
                slot = self._slots[env_idx]
                slot.last_flow_action_norm = flow_actions[batch_idx]
                obs[env_idx] = self._bundle.build_sac_obs(
                    slot.obs_dict,
                    slot.initial_goal,
                    slot.last_flow_action_norm,
                )

        self._actions = None
        return obs, rewards, dones, infos

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        self.step_async(actions)
        return self.step_wait()

    def step_rand(
        self,
        obs: np.ndarray,
        model: Any = None,
        *,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], np.ndarray]:
        if model is not None:
            residuals, _ = model.predict(obs, deterministic=deterministic)
            residuals = np.asarray(residuals, dtype=np.float32).reshape(self.n_envs, self.residual_dim)
        else:
            residuals = np.zeros((self.n_envs, self.residual_dim), dtype=np.float32)

        next_obs, rewards, dones, infos = self.step(residuals)
        return next_obs, rewards, dones, infos, residuals

    def close(self) -> None:
        for slot in self._slots:
            slot.env.close()

    def env_is_wrapped(self, wrapper_class: type[gym.Wrapper], indices=None) -> list[bool]:
        del wrapper_class, indices
        return [False] * self.n_envs

    def get_attr(self, attr_name: str, indices=None):
        del indices
        return [getattr(slot.env, attr_name) for slot in self._slots]

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        target = self._slots if indices is None else [self._slots[i] for i in indices]
        for slot in target:
            setattr(slot.env, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        target = self._slots if indices is None else [self._slots[i] for i in indices]
        return [getattr(slot.env, method_name)(*method_args, **method_kwargs) for slot in target]
