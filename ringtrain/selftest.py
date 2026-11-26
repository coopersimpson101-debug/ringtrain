"""Entry points used by the multi-process tests (they must be importable by spawned workers)."""
from __future__ import annotations

import hashlib

import numpy as np


def _data(rank: int, n: int) -> np.ndarray:
    return np.random.default_rng(1000 + rank).integers(-(1 << 12), 1 << 12, size=n).astype(np.float32)


def allreduce_check(pg, args: dict) -> dict:
    n = int(args.get("n", 10_000))
    a = _data(pg.rank, n)
    pg.allreduce(a)
    expect = sum(_data(r, n) for r in range(pg.world))
    blob = pg.broadcast_bytes(b"x" * 100_000 if pg.rank == 0 else None)
    pg.barrier()
    return {"ok": bool(np.array_equal(a, expect)) and blob == b"x" * 100_000,
            "digest": hashlib.sha256(a.tobytes()).hexdigest()}
