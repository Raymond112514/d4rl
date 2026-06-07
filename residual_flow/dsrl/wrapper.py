"""PointMaze environment wrapper for DSRL on a frozen residual flow policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import minari
import numpy as np
import torch
from gymnasium import spaces

from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn

from eval import combine_actions, load_residual_checkpoint
from fm_bc.eval.rollout import (
    GOAL_TOLERANCE,
    configure_episodic_eval_env,
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
    last_noise_action: np.ndarray | None = None
    episode_steps: int = 0


class PointMazeResidualFlowBundle:
    """Shared frozen base flow policy, frozen residual flow policy, and stats."""

    def __init__(
        self,
        *,
        dataset_id: str,
        flow_checkpoint: str,
        residual_checkpoint: str,
        flow_sample_steps: int,
        residual_sample_steps: int,
        residual_scale: float,
        action_mag: float,
        use_cpu: bool,
    ) -> None:
        self.dataset_id = dataset_id
        self.flow_checkpoint = flow_checkpoint
        self.residual_checkpoint = residual_checkpoint
        self.flow_sample_steps = flow_sample_steps
        self.residual_sample_steps = residual_sample_steps
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

        self.residual_flow, self.residual_cfg, self.residual_stats, _res_payload = (
            load_residual_checkpoint(residual_checkpoint, device=self.device)
        )
        self.residual_flow.eval()
        for param in self.residual_flow.parameters():
            param.requires_grad = False

        self.ac_chunk = self.residual_cfg.ac_chunk
        self.action_dim = self.residual_cfg.action_dim
        self.noise_dim = self.residual_cfg.flow_dim

        obs_dim = self.cfg.obs_dim + self.noise_dim
        if self.cfg.mode == "goal_conditioned":
            obs_dim += self.cfg.goal_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-action_mag * np.ones(self.noise_dim, dtype=np.float32),
            high=action_mag * np.ones(self.noise_dim, dtype=np.float32),
            dtype=np.float32,
        )

    def make_base_env(self) -> gym.Env:
        dataset = minari.load_dataset(self.dataset_id, download=False)
        env = dataset.recover_environment()
        configure_episodic_eval_env(env)
        return env

    def _obs_norm_from_dict(self, obs_dict: dict[str, Any]) -> np.ndarray:
        return normalize(
            np.asarray(obs_dict["observation"], dtype=np.float64),
            self.stats.obs_mean,
            self.stats.obs_std,
        ).astype(np.float32)

    @torch.no_grad()
    def sample_flow_actions(
        self,
        obs_dicts: list[dict[str, Any]],
        initial_goals: list[np.ndarray],
    ) -> np.ndarray:
        obs_norm = np.stack(
            [self._obs_norm_from_dict(obs_dict) for obs_dict in obs_dicts],
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
            n_steps=self.flow_sample_steps,
        )
        return chunks.cpu().numpy()

    @torch.no_grad()
    def denoise_noise_to_residual_norm(
        self,
        noise: np.ndarray,
        *,
        obs_dict: dict[str, Any],
        flow_action_norm: np.ndarray,
    ) -> np.ndarray:
        obs_norm = self._obs_norm_from_dict(obs_dict)
        obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=self.device).unsqueeze(0)
        noise_t = torch.as_tensor(
            np.asarray(noise, dtype=np.float32).reshape(-1),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        anchor_t = None
        if self.residual_cfg.uses_action_conditioning:
            anchor_flat = np.asarray(flow_action_norm, dtype=np.float32).reshape(-1)
            anchor_t = torch.as_tensor(anchor_flat, dtype=torch.float32, device=self.device).unsqueeze(0)

        residual = self.residual_flow.sample_from_noise(
            noise_t,
            obs_t,
            anchor_action=anchor_t,
            n_steps=self.residual_sample_steps,
        )
        return residual.cpu().numpy()[0].reshape(-1).astype(np.float32)

    @torch.no_grad()
    def denoise_noise_batch(
        self,
        noises: np.ndarray,
        obs_dicts: list[dict[str, Any]],
        flow_action_norms: list[np.ndarray],
    ) -> np.ndarray:
        obs_norm = np.stack(
            [self._obs_norm_from_dict(obs_dict) for obs_dict in obs_dicts],
            axis=0,
        )
        obs_t = torch.as_tensor(obs_norm, dtype=torch.float32, device=self.device)
        noise_t = torch.as_tensor(
            np.asarray(noises, dtype=np.float32).reshape(len(noises), -1),
            dtype=torch.float32,
            device=self.device,
        )

        anchor_t = None
        if self.residual_cfg.uses_action_conditioning:
            anchors = np.stack(
                [
                    np.asarray(flow_norm, dtype=np.float32).reshape(-1)
                    for flow_norm in flow_action_norms
                ],
                axis=0,
            )
            anchor_t = torch.as_tensor(anchors, dtype=torch.float32, device=self.device)

        residuals = self.residual_flow.sample_from_noise(
            noise_t,
            obs_t,
            anchor_action=anchor_t,
            n_steps=self.residual_sample_steps,
        )
        return residuals.cpu().numpy().reshape(len(noises), -1).astype(np.float32)

    def decompose_action(
        self,
        flow_action_norm: np.ndarray,
        noise: np.ndarray,
        *,
        obs_dict: dict[str, Any],
        action_low: np.ndarray,
        action_high: np.ndarray,
    ) -> dict[str, np.ndarray]:
        residual_norm = self.denoise_noise_to_residual_norm(
            noise,
            obs_dict=obs_dict,
            flow_action_norm=flow_action_norm,
        )
        return combine_actions(
            flow_action_norm,
            residual_norm,
            stats=self.stats,
            residual_cfg=self.residual_cfg,
            residual_scale=self.residual_scale,
            action_low=action_low,
            action_high=action_high,
        )

    def build_sac_obs(
        self,
        obs_dict: dict[str, Any],
        initial_goal: np.ndarray | None,
        flow_action_norm: np.ndarray,
    ) -> np.ndarray:
        obs_norm = self._obs_norm_from_dict(obs_dict)
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
    bundle: PointMazeResidualFlowBundle,
    flow_action_norm: np.ndarray,
    noise: np.ndarray,
    *,
    max_episode_steps: int,
) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    if slot.obs_dict is None:
        raise RuntimeError("Environment slot must be reset before stepping.")

    slot.last_noise_action = np.asarray(noise, dtype=np.float32).reshape(-1).copy()
    decomp = bundle.decompose_action(
        flow_action_norm,
        noise,
        obs_dict=slot.obs_dict,
        action_low=slot.env.action_space.low,
        action_high=slot.env.action_space.high,
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
    bundle: PointMazeResidualFlowBundle,
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
    slot.last_noise_action = None
    slot.last_flow_action_norm = bundle.sample_flow_actions([obs_dict], [slot.initial_goal])[0]
    return bundle.build_sac_obs(obs_dict, slot.initial_goal, slot.last_flow_action_norm)


class PointMazeResidualFlowWrapper(gym.Env):
    """Single-env wrapper where SAC selects noise for a frozen residual flow policy."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        dataset_id: str = "D4RL/pointmaze/large-dense-v2",
        flow_checkpoint: str,
        residual_checkpoint: str,
        flow_sample_steps: int = 20,
        residual_sample_steps: int = 20,
        residual_scale: float = 1.0,
        action_mag: float = 2.0,
        max_episode_steps: int = 1000,
        use_cpu: bool = False,
        reset_options: Optional[dict[str, np.ndarray]] = None,
    ) -> None:
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.reset_options = reset_options
        self._bundle = PointMazeResidualFlowBundle(
            dataset_id=dataset_id,
            flow_checkpoint=flow_checkpoint,
            residual_checkpoint=residual_checkpoint,
            flow_sample_steps=flow_sample_steps,
            residual_sample_steps=residual_sample_steps,
            residual_scale=residual_scale,
            action_mag=action_mag,
            use_cpu=use_cpu,
        )
        self.observation_space = self._bundle.observation_space
        self.action_space = self._bundle.action_space
        self.cfg = self._bundle.cfg
        self.residual_cfg = self._bundle.residual_cfg
        self.stats = self._bundle.stats
        self.ac_chunk = self._bundle.ac_chunk
        self.action_dim = self._bundle.action_dim
        self.residual_dim = self._bundle.noise_dim
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
        noise: np.ndarray,
        *,
        obs_dict: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        if obs_dict is None:
            if self._slot.obs_dict is None:
                raise RuntimeError("Environment must be reset before decomposing actions.")
            obs_dict = self._slot.obs_dict
        return self._bundle.decompose_action(
            flow_action_norm,
            noise,
            obs_dict=obs_dict,
            action_low=self._slot.env.action_space.low,
            action_high=self._slot.env.action_space.high,
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
            noise, _ = model.predict(sac_obs, deterministic=deterministic)
            noise = np.asarray(noise, dtype=np.float32).reshape(self.residual_dim)
        else:
            noise = np.zeros(self.residual_dim, dtype=np.float32)

        next_sac_obs, reward, terminated, truncated, info = self.step(noise)
        return next_sac_obs, reward, terminated, truncated, info, noise

    def close(self) -> None:
        self._slot.env.close()


class PointMazeResidualFlowVecEnv(VecEnv):
    """Vectorized PointMaze wrapper with batched flow and residual-flow inference."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_envs: int,
        *,
        dataset_id: str = "D4RL/pointmaze/large-dense-v2",
        flow_checkpoint: str,
        residual_checkpoint: str,
        flow_sample_steps: int = 20,
        residual_sample_steps: int = 20,
        residual_scale: float = 1.0,
        action_mag: float = 2.0,
        max_episode_steps: int = 1000,
        use_cpu: bool = False,
        reset_options: Optional[dict[str, np.ndarray]] = None,
        base_seed: int = 0,
    ) -> None:
        self.max_episode_steps = max_episode_steps
        self.reset_options = reset_options
        self.base_seed = base_seed

        self._bundle = PointMazeResidualFlowBundle(
            dataset_id=dataset_id,
            flow_checkpoint=flow_checkpoint,
            residual_checkpoint=residual_checkpoint,
            flow_sample_steps=flow_sample_steps,
            residual_sample_steps=residual_sample_steps,
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
        self.residual_cfg = self._bundle.residual_cfg
        self.stats = self._bundle.stats
        self.ac_chunk = self._bundle.ac_chunk
        self.action_dim = self._bundle.action_dim
        self.residual_dim = self._bundle.noise_dim
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
        noise: np.ndarray,
        *,
        obs_dict: dict[str, Any] | None = None,
        env_idx: int = 0,
    ) -> dict[str, np.ndarray]:
        slot = self._slots[env_idx]
        if obs_dict is None:
            if slot.obs_dict is None:
                raise RuntimeError("Environment must be reset before decomposing actions.")
            obs_dict = slot.obs_dict
        return self._bundle.decompose_action(
            flow_action_norm,
            noise,
            obs_dict=obs_dict,
            action_low=slot.env.action_space.low,
            action_high=slot.env.action_space.high,
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
            noise = self._actions[env_idx]
            obs_dict, reward, terminated, truncated, info = _execute_chunk_on_slot(
                slot,
                self._bundle,
                slot.last_flow_action_norm,
                noise,
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
            noises, _ = model.predict(obs, deterministic=deterministic)
            noises = np.asarray(noises, dtype=np.float32).reshape(self.n_envs, self.residual_dim)
        else:
            noises = np.zeros((self.n_envs, self.residual_dim), dtype=np.float32)

        next_obs, rewards, dones, infos = self.step(noises)
        return next_obs, rewards, dones, infos, noises

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
