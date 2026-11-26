"""World-size-independent data order.

The global sample stream is fixed by ``(seed, step)`` alone: step ``s`` consumes positions
``[s B, (s+1) B)`` of an infinite sequence made of per-epoch permutations.  Rank ``r`` of ``N``
takes a contiguous slice of that global batch.  The *data cursor* in a checkpoint is therefore
just the step number, and a run that loses a worker and continues on ``N-1`` sees exactly the same
global batches it would have seen on ``N`` -- the elastic trainer keeps the global batch size
fixed and sums (not averages) per-sample losses, so the mathematical update is unchanged.
"""
from __future__ import annotations

import numpy as np

from .collectives import chunk_bounds


def epoch_permutation(n: int, seed: int, epoch: int) -> np.ndarray:
    return np.random.default_rng([seed, epoch]).permutation(n)


def global_batch_indices(n: int, batch: int, step: int, seed: int) -> np.ndarray:
    pos = np.arange(step * batch, (step + 1) * batch, dtype=np.int64)
    epochs = pos // n
    out = np.empty(batch, dtype=np.int64)
    for e in np.unique(epochs):
        sel = epochs == e
        out[sel] = epoch_permutation(n, seed, int(e))[pos[sel] % n]
    return out


def shard(indices: np.ndarray, rank: int, world: int) -> np.ndarray:
    lo, hi = chunk_bounds(len(indices), world)[rank]
    return indices[lo:hi]
