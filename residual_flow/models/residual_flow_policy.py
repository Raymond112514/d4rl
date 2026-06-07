"""Rectified flow matching policy for action-difference residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn

from fm_bc.models.flow_policy import sinusoidal_time_embedding

ResidualCondMode = Literal[
    "unconditioned",
    "obs_conditioned",
    "action_conditioned",
    "obs_action_conditioned",
]


@dataclass
class ResidualFlowConfig:
    action_dim: int = 2
    ac_chunk: int = 10
    obs_dim: int = 4
    goal_dim: int = 2
    mode: ResidualCondMode = "unconditioned"
    hidden_dim: int = 256
    n_layers: int = 4
    time_emb_dim: int = 64

    @property
    def cond_dim(self) -> int:
        if self.mode == "unconditioned":
            return 0
        if self.mode == "obs_conditioned":
            return self.obs_dim
        if self.mode == "action_conditioned":
            return self.flow_dim
        if self.mode == "obs_action_conditioned":
            return self.obs_dim + self.flow_dim
        raise ValueError(f"Unknown mode: {self.mode}")

    @property
    def flow_dim(self) -> int:
        return self.action_dim * self.ac_chunk

    @property
    def uses_obs_conditioning(self) -> bool:
        return self.mode in ("obs_conditioned", "obs_action_conditioned")

    @property
    def uses_action_conditioning(self) -> bool:
        return self.mode in ("action_conditioned", "obs_action_conditioned")


class ResidualFlowPolicy(nn.Module):
    """Velocity field v(x_t, t, cond) over flattened action-difference vectors."""

    def __init__(self, cfg: ResidualFlowConfig):
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
        anchor_action: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.cfg.mode == "unconditioned":
            return None
        if self.cfg.mode == "obs_conditioned":
            return obs
        if self.cfg.mode == "action_conditioned":
            if anchor_action is None:
                raise ValueError("action_conditioned mode requires anchor_action")
            return anchor_action
        if self.cfg.mode == "obs_action_conditioned":
            if anchor_action is None:
                raise ValueError("obs_action_conditioned mode requires anchor_action")
            return torch.cat([obs, anchor_action], dim=-1)
        raise ValueError(f"Unknown mode: {self.cfg.mode}")

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        *,
        obs: torch.Tensor,
        anchor_action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(t, self.cfg.time_emb_dim)
        cond = self._build_condition(obs, anchor_action)
        if cond is None:
            h = torch.cat([x_t, t_emb], dim=-1)
        else:
            h = torch.cat([x_t, t_emb, cond], dim=-1)
        return self.net(h)

    @torch.no_grad()
    def sample_from_noise(
        self,
        noise: torch.Tensor,
        obs: torch.Tensor,
        *,
        anchor_action: Optional[torch.Tensor] = None,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """Denoise explicit starting noise into an action-difference vector."""
        x = noise.clone()
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.full((x.shape[0],), step / n_steps, device=x.device)
            v = self.forward(x, t, obs=obs, anchor_action=anchor_action)
            x = x + dt * v
        return x.reshape(x.shape[0], self.cfg.ac_chunk, self.cfg.action_dim)

    @torch.no_grad()
    def sample(
        self,
        obs: torch.Tensor,
        *,
        anchor_action: Optional[torch.Tensor] = None,
        n_steps: int = 20,
    ) -> torch.Tensor:
        return self.sample_trajectory(
            obs,
            anchor_action=anchor_action,
            n_steps=n_steps,
        )[-1]

    @torch.no_grad()
    def sample_trajectory(
        self,
        obs: torch.Tensor,
        *,
        anchor_action: Optional[torch.Tensor] = None,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """Return action-difference vectors at each denoising step.

        Shape: (n_steps + 1, batch, ac_chunk, action_dim).
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
            v = self.forward(x, t, obs=obs, anchor_action=anchor_action)
            x = x + dt * v
            trajectory.append(x.reshape(chunk_shape))

        return torch.stack(trajectory, dim=0)


def residual_flow_matching_loss(
    model: ResidualFlowPolicy,
    diff_target: torch.Tensor,
    *,
    obs: torch.Tensor,
    anchor_action: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rectified flow matching loss on action-difference targets."""
    batch_size = diff_target.shape[0]
    x0 = torch.randn_like(diff_target)
    device = diff_target.device
    t = torch.rand(batch_size, device=device)
    t_expand = t[:, None]
    x_t = (1.0 - t_expand) * x0 + t_expand * diff_target
    target_v = diff_target - x0
    pred_v = model(x_t, t, obs=obs, anchor_action=anchor_action)
    return ((pred_v - target_v) ** 2).mean()
