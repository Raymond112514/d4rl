"""Rectified flow matching policy for chunked behavior cloning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

ConditioningMode = Literal["unconditional", "goal_conditioned"]


def sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -torch.log(torch.tensor(10000.0, device=t.device))
        * torch.arange(half, device=t.device).float()
        / max(half - 1, 1)
    )
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(emb.shape[0], 1, device=t.device)], dim=-1)
    return emb


@dataclass
class FlowPolicyConfig:
    action_dim: int = 2
    ac_chunk: int = 10
    obs_dim: int = 4
    goal_dim: int = 2
    mode: ConditioningMode = "unconditional"
    hidden_dim: int = 256
    n_layers: int = 4
    time_emb_dim: int = 64

    @property
    def cond_dim(self) -> int:
        if self.mode == "unconditional":
            return self.obs_dim
        if self.mode == "goal_conditioned":
            return self.obs_dim + self.goal_dim
        raise ValueError(f"Unknown mode: {self.mode}")

    @property
    def flow_dim(self) -> int:
        return self.action_dim * self.ac_chunk


class FlowPolicy(nn.Module):
    """Velocity field v(x_t, t, cond) over flattened action chunks."""

    def __init__(self, cfg: FlowPolicyConfig):
        super().__init__()
        self.cfg = cfg
        in_dim = cfg.flow_dim + cfg.time_emb_dim + cfg.cond_dim

        layers: list[nn.Module] = []
        width = in_dim
        for _ in range(cfg.n_layers):
            layers.extend([nn.Linear(width, cfg.hidden_dim), nn.SiLU()])
            width = cfg.hidden_dim
        layers.append(nn.Linear(width, cfg.flow_dim))
        self.net = nn.Sequential(*layers)

    def _build_condition(
        self,
        obs: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.cfg.mode == "unconditional":
            return obs
        if self.cfg.mode == "goal_conditioned":
            if goal is None:
                raise ValueError("goal_conditioned mode requires goal")
            return torch.cat([obs, goal], dim=-1)
        raise ValueError(f"Unknown mode: {self.cfg.mode}")

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        obs: torch.Tensor,
        goal: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.cfg.time_emb_dim)
        cond = self._build_condition(obs, goal)
        h = torch.cat([x_t, t_emb, cond], dim=-1)
        return self.net(h)

    @torch.no_grad()
    def sample(
        self,
        obs: torch.Tensor,
        *,
        goal: Optional[torch.Tensor] = None,
        n_steps: int = 20,
    ) -> torch.Tensor:
        return self.sample_trajectory(obs, goal=goal, n_steps=n_steps)[-1]

    @torch.no_grad()
    def sample_trajectory(
        self,
        obs: torch.Tensor,
        *,
        goal: Optional[torch.Tensor] = None,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """Return action chunks at each denoising step, including initial noise.

        Shape: (n_steps + 1, batch, ac_chunk, action_dim).
        Index 0 is pure noise; index n_steps is the final sample.
        """
        batch_size = obs.shape[0]
        device = obs.device
        chunk_shape = (batch_size, self.cfg.ac_chunk, self.cfg.action_dim)
        trajectory: list[torch.Tensor] = []

        x = torch.randn(batch_size, self.cfg.flow_dim, device=device)
        trajectory.append(x.reshape(chunk_shape))

        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.full((batch_size,), step / n_steps, device=device)
            v = self.forward(x, t, obs=obs, goal=goal)
            x = x + dt * v
            trajectory.append(x.reshape(chunk_shape))

        return torch.stack(trajectory, dim=0)


def expand_action_mask(mask: torch.Tensor, action_dim: int) -> torch.Tensor:
    """Expand per-step mask (B, H) to per-dimension mask (B, H * A)."""
    return mask[:, :, None].expand(-1, -1, action_dim).reshape(mask.shape[0], -1)


def flow_matching_loss(
    model: FlowPolicy,
    actions: torch.Tensor,
    mask: torch.Tensor,
    *,
    obs: torch.Tensor,
    goal: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Masked rectified flow matching loss on action chunks."""
    batch_size, ac_chunk, action_dim = actions.shape
    actions_flat = actions.reshape(batch_size, ac_chunk * action_dim)
    mask_flat = expand_action_mask(mask, action_dim)

    x0 = torch.randn_like(actions_flat)
    device = actions.device
    t = torch.rand(batch_size, device=device)
    t_expand = t[:, None]
    x_t = (1.0 - t_expand) * x0 + t_expand * actions_flat
    target_v = actions_flat - x0
    pred_v = model(x_t, t, obs=obs, goal=goal)

    sq_err = (pred_v - target_v) ** 2
    return (sq_err * mask_flat).sum() / mask_flat.sum().clamp(min=1.0)
