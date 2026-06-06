"""Device selection helpers."""

from __future__ import annotations

import torch


def resolve_device(use_cpu: bool = False) -> torch.device:
    if use_cpu:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build or pass --cpu."
        )
    return torch.device("cuda")
