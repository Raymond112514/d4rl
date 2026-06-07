"""Minari point-maze dataset with action-chunk sampling and padding masks."""

from __future__ import annotations

from dataclasses import dataclass

import minari
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class PointMazeStats:
    obs_mean: np.ndarray
    obs_std: np.ndarray
    goal_mean: np.ndarray
    goal_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def to_dict(self) -> dict:
        return {
            "obs_mean": self.obs_mean.tolist(),
            "obs_std": self.obs_std.tolist(),
            "goal_mean": self.goal_mean.tolist(),
            "goal_std": self.goal_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PointMazeStats":
        return cls(
            obs_mean=np.asarray(data["obs_mean"], dtype=np.float32),
            obs_std=np.asarray(data["obs_std"], dtype=np.float32),
            goal_mean=np.asarray(data["goal_mean"], dtype=np.float32),
            goal_std=np.asarray(data["goal_std"], dtype=np.float32),
            action_mean=np.asarray(data["action_mean"], dtype=np.float32),
            action_std=np.asarray(data["action_std"], dtype=np.float32),
        )


@dataclass
class EpisodeBuffer:
    obs: np.ndarray
    actions: np.ndarray
    goals: np.ndarray


def episode_main_goal(desired_goals: np.ndarray) -> np.ndarray:
    if len(np.unique(desired_goals, axis=0)) == 1:
        return desired_goals[0]
    return desired_goals[1]


def load_pointmaze_episodes(
    dataset_id: str = "D4RL/pointmaze/large-dense-v2",
    download: bool = True,
) -> list[EpisodeBuffer]:
    dataset = minari.load_dataset(dataset_id, download=download)
    episodes: list[EpisodeBuffer] = []

    for episode in dataset:
        observations = np.asarray(episode.observations["observation"], dtype=np.float32)
        actions = np.asarray(episode.actions, dtype=np.float32)
        desired = np.asarray(episode.observations["desired_goal"], dtype=np.float32)
        main_goal = episode_main_goal(desired)
        goals = np.repeat(main_goal[None, :], len(actions), axis=0)

        episodes.append(
            EpisodeBuffer(
                obs=observations[: len(actions)],
                actions=actions,
                goals=goals,
            )
        )
    return episodes


def compute_stats(episodes: list[EpisodeBuffer], eps: float = 1e-6) -> PointMazeStats:
    obs = np.concatenate([ep.obs for ep in episodes], axis=0)
    actions = np.concatenate([ep.actions for ep in episodes], axis=0)
    goals = np.concatenate([ep.goals for ep in episodes], axis=0)
    return PointMazeStats(
        obs_mean=obs.mean(axis=0),
        obs_std=obs.std(axis=0) + eps,
        goal_mean=goals.mean(axis=0),
        goal_std=goals.std(axis=0) + eps,
        action_mean=actions.mean(axis=0),
        action_std=actions.std(axis=0) + eps,
    )


def build_chunk_indices(episodes: list[EpisodeBuffer]) -> list[tuple[int, int]]:
    indices: list[tuple[int, int]] = []
    for ep_idx, episode in enumerate(episodes):
        for start in range(len(episode.actions)):
            indices.append((ep_idx, start))
    return indices


def subsample_chunk_indices(
    indices: list[tuple[int, int]],
    data_fraction: float,
    *,
    seed: int = 0,
) -> list[tuple[int, int]]:
    if not 0.0 < data_fraction <= 1.0:
        raise ValueError(f"data_fraction must be in (0, 1], got {data_fraction}")
    if data_fraction >= 1.0:
        return indices

    n_total = len(indices)
    n_keep = max(1, int(n_total * data_fraction))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n_total, size=n_keep, replace=False)
    return [indices[int(i)] for i in chosen]


def compute_stats_from_indices(
    episodes: list[EpisodeBuffer],
    indices: list[tuple[int, int]],
    eps: float = 1e-6,
) -> PointMazeStats:
    obs = np.stack([episodes[ep_idx].obs[start] for ep_idx, start in indices], axis=0)
    actions = np.stack(
        [episodes[ep_idx].actions[start] for ep_idx, start in indices],
        axis=0,
    )
    goals = np.stack([episodes[ep_idx].goals[start] for ep_idx, start in indices], axis=0)
    return PointMazeStats(
        obs_mean=obs.mean(axis=0),
        obs_std=obs.std(axis=0) + eps,
        goal_mean=goals.mean(axis=0),
        goal_std=goals.std(axis=0) + eps,
        action_mean=actions.mean(axis=0),
        action_std=actions.std(axis=0) + eps,
    )


class PointMazeChunkDataset(Dataset):
    def __init__(
        self,
        episodes: list[EpisodeBuffer],
        stats: PointMazeStats,
        ac_chunk: int = 10,
    ):
        self.episodes = episodes
        self.stats = stats
        self.ac_chunk = ac_chunk
        self.indices = build_chunk_indices(episodes)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ep_idx, start = self.indices[idx]
        episode = self.episodes[ep_idx]

        obs = (episode.obs[start] - self.stats.obs_mean) / self.stats.obs_std
        goal = (episode.goals[start] - self.stats.goal_mean) / self.stats.goal_std

        chunked_actions: list[np.ndarray] = []
        mask: list[float] = []
        for offset in range(self.ac_chunk):
            step = start + offset
            if step < len(episode.actions):
                action = episode.actions[step]
                mask.append(1.0)
            else:
                action = episode.actions[-1]
                mask.append(0.0)
            action = (action - self.stats.action_mean) / self.stats.action_std
            chunked_actions.append(action)

        actions = np.stack(chunked_actions, axis=0).astype(np.float32)
        mask_arr = np.asarray(mask, dtype=np.float32)

        return {
            "obs": torch.as_tensor(obs, dtype=torch.float32),
            "goal": torch.as_tensor(goal, dtype=torch.float32),
            "action": torch.as_tensor(actions, dtype=torch.float32),
            "mask": torch.as_tensor(mask_arr, dtype=torch.float32),
        }


def build_pointmaze_dataset(
    dataset_id: str = "D4RL/pointmaze/large-dense-v2",
    ac_chunk: int = 10,
    download: bool = True,
    data_fraction: float = 1.0,
    seed: int = 0,
) -> tuple[PointMazeChunkDataset, PointMazeStats, int]:
    episodes = load_pointmaze_episodes(dataset_id=dataset_id, download=download)
    all_indices = build_chunk_indices(episodes)
    indices = subsample_chunk_indices(all_indices, data_fraction, seed=seed)
    stats = compute_stats_from_indices(episodes, indices)
    dataset = PointMazeChunkDataset(episodes, stats, ac_chunk=ac_chunk)
    dataset.indices = indices
    return dataset, stats, len(dataset)
