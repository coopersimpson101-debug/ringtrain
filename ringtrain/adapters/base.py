"""The adapter protocol every workload implements."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch
from torch import nn


@runtime_checkable
class Adapter(Protocol):
    name: str
    num_samples: int
    global_batch: int

    def build_model(self, seed: int) -> nn.Module:
        """Deterministic for a given seed (rank 0's copy is broadcast anyway)."""

    def build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        ...

    def batch(self, indices: np.ndarray) -> Any:
        """The samples at these global dataset indices."""

    def loss_sum(self, model: nn.Module, batch: Any) -> tuple[torch.Tensor, int]:
        """Return the SUM of per-sample losses and the number of samples.  The trainer divides by
        the *global* batch size, so gradients are identical whatever the world size."""
