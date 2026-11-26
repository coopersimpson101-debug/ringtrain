"""Gradient reducers: exact ring all-reduce, top-k sparsification with error feedback, PowerSGD.

All reducers take the list of gradient tensors of one DDP bucket and return the *summed* gradient
over ranks (DDP divides by the world size afterwards if it averages).

Top-k with error feedback (Stich et al. 2018, "Sparsified SGD with memory"; Lin et al. 2018, DGC)
------------------------------------------------------------------------------------------------
Each rank keeps a residual ``e``.  Per step: ``a = g + e``; keep the ``k`` entries of largest
``|a|`` (indices ``I``); ``e <- a`` with ``a[I]`` zeroed.  Ranks all-gather their ``(I, a[I])``
pairs (``k`` int32 + ``k`` float32 each) and scatter-add them into a dense sum.  Nothing is ever
dropped, only delayed, which is what makes biased compressors converge.  Traffic per rank:
``8 k (N-1)`` bytes vs ``8 n (N-1)/N`` for ring all-reduce of float32.

PowerSGD (Vogels, Karimireddy & Jaggi, NeurIPS 2019)
----------------------------------------------------
For each gradient matrix ``M_i`` (``n x m``, conv/linear weights reshaped) with error feedback,
and a rank-``q`` right factor ``Q`` that is identical on every rank (warm-started from the
previous step):

    P   = sum_i (M_i Q)               all-reduce of an n x q matrix
    P^  = orthonormalise(P)           Gram-Schmidt, identical on every rank
    Q'  = sum_i (M_i^T P^)            all-reduce of an m x q matrix
    sum_i M_i ~= P^ Q'^T              rank-q approximation of the summed gradient
    e_i = M_i - (P^ Q'^T) / N         local error feedback

Traffic per matrix: ``(n + m) q`` floats instead of ``n m``.  Vectors (biases, norms) and
matrices too small to benefit are all-reduced exactly.
"""
from __future__ import annotations

import numpy as np
import torch

from .collectives import ProcessGroup


def _flat(tensors: list[torch.Tensor]) -> np.ndarray:
    return torch.cat([t.reshape(-1) for t in tensors]).detach().to(torch.float32).numpy().copy()


def _unflat(flat: np.ndarray, like: list[torch.Tensor]) -> list[torch.Tensor]:
    out, o = [], 0
    for t in like:
        n = t.numel()
        out.append(torch.from_numpy(flat[o:o + n].copy()).view_as(t).to(t.dtype))
        o += n
    return out


class Reducer:
    name = "allreduce"

    def reduce(self, pg: ProcessGroup, key: int, grads: list[torch.Tensor], channel: int = 1) -> list[torch.Tensor]:
        if pg.world == 1:
            return grads  # nothing to exchange (the grads are already private clones)
        flat = _flat(grads)
        pg.allreduce(flat, channel=channel)
        return _unflat(flat, grads)

    def reset(self) -> None:
        """Drop per-rank state (error feedback) -- called when the group membership changes."""


class TopKReducer(Reducer):
    name = "topk"

    def __init__(self, ratio: float = 0.01, min_k: int = 1):
        self.ratio, self.min_k = ratio, min_k
        self.residual: dict[int, torch.Tensor] = {}

    def reset(self) -> None:
        self.residual.clear()

    def reduce(self, pg, key, grads, channel=1):
        flat = torch.cat([g.reshape(-1).to(torch.float32) for g in grads])
        n = flat.numel()
        e = self.residual.get(key)
        acc = flat + e if e is not None else flat.clone()
        k = max(self.min_k, min(n, int(round(self.ratio * n))))
        idx = torch.topk(acc.abs(), k, sorted=False).indices
        vals = acc[idx]
        resid = acc.clone()
        resid[idx] = 0.0
        self.residual[key] = resid
        payload = np.concatenate([idx.to(torch.int32).numpy().view(np.uint8),
                                  vals.numpy().astype(np.float32).view(np.uint8)])
        dense = np.zeros(n, dtype=np.float32)
        # sum in rank order so every rank computes bit-identical results
        for blob in pg.all_gather_bytes(payload.tobytes(), channel=channel):
            arr = np.frombuffer(blob, dtype=np.uint8)
            ii = arr[: 4 * k].view(np.int32)
            vv = arr[4 * k:].view(np.float32)
            np.add.at(dense, ii, vv)
        return _unflat(dense, grads)


def orthonormalise_(p: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """In-place modified Gram-Schmidt on the columns of ``p`` (n x q)."""
    for i in range(p.shape[1]):
        col = p[:, i]
        for j in range(i):
            col -= torch.dot(p[:, j], col) * p[:, j]
        col /= col.norm() + eps
    return p


class PowerSGDReducer(Reducer):
    name = "powersgd"

    def __init__(self, rank: int = 2, min_compression: float = 2.0, seed: int = 0, error_feedback: bool = True):
        self.q, self.min_compression, self.seed = rank, min_compression, seed
        self.error_feedback = error_feedback
        self.Q: dict[tuple[int, int], torch.Tensor] = {}
        self.err: dict[tuple[int, int], torch.Tensor] = {}

    def reset(self) -> None:
        self.err.clear()  # Q stays: it is identical on every rank and only a warm start

    def _compressible(self, g: torch.Tensor) -> bool:
        if g.dim() < 2:
            return False
        n, m = g.shape[0], g.numel() // g.shape[0]
        q = min(self.q, n, m)
        return n * m >= self.min_compression * (n + m) * q

    def reduce(self, pg, key, grads, channel=1):
        world = pg.world
        out: list[torch.Tensor | None] = [None] * len(grads)
        dense = [i for i, g in enumerate(grads) if not self._compressible(g)]
        mats = [i for i, g in enumerate(grads) if self._compressible(g)]
        if dense:
            flat = _flat([grads[i] for i in dense])
            pg.allreduce(flat, channel=channel)
            for i, t in zip(dense, _unflat(flat, [grads[i] for i in dense])):
                out[i] = t
        if not mats:
            return out  # type: ignore[return-value]
        Ms, Ps, Qs = {}, {}, {}
        for i in mats:
            g = grads[i].to(torch.float32)
            n, m = g.shape[0], g.numel() // g.shape[0]
            q = min(self.q, n, m)
            M = g.reshape(n, m).clone()
            if self.error_feedback and (key, i) in self.err:
                M += self.err[(key, i)]
            Q = self.Q.get((key, i))
            if Q is None:
                gen = torch.Generator().manual_seed(self.seed * 1_000_003 + key * 1009 + i)
                Q = orthonormalise_(torch.randn(m, q, generator=gen))
            Ms[i], Qs[i] = M, Q
            Ps[i] = M @ Q
        # one all-reduce for every P in the bucket, one for every Q
        flatP = np.concatenate([Ps[i].numpy().reshape(-1) for i in mats]).astype(np.float32)
        pg.allreduce(flatP, channel=channel)
        o = 0
        for i in mats:
            n, q = Ps[i].shape
            Ps[i] = orthonormalise_(torch.from_numpy(flatP[o:o + n * q].copy()).view(n, q))
            o += n * q
            Qs[i] = Ms[i].t() @ Ps[i]
        flatQ = np.concatenate([Qs[i].numpy().reshape(-1) for i in mats]).astype(np.float32)
        pg.allreduce(flatQ, channel=channel)
        o = 0
        for i in mats:
            m, q = Qs[i].shape
            Qsum = torch.from_numpy(flatQ[o:o + m * q].copy()).view(m, q)
            o += m * q
            approx_sum = Ps[i] @ Qsum.t()
            if self.error_feedback:
                self.err[(key, i)] = Ms[i] - approx_sum / world
            self.Q[(key, i)] = Qsum.clone()  # warm start (identical on every rank)
            out[i] = approx_sum.view_as(grads[i]).to(grads[i].dtype)
        return out  # type: ignore[return-value]


def make_reducer(spec: str | None) -> Reducer:
    """``none`` | ``topk:0.01`` | ``powersgd:2``"""
    if not spec or spec == "none":
        return Reducer()
    name, _, arg = spec.partition(":")
    if name == "topk":
        return TopKReducer(float(arg or 0.01))
    if name == "powersgd":
        return PowerSGDReducer(int(arg or 2))
    raise ValueError(f"unknown compression {spec!r}")
