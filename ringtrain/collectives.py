"""Collectives built on :class:`~ringtrain.transport.Transport`: ring all-reduce, ring
reduce-scatter / all-gather, pipelined ring broadcast, dissemination barrier, point-to-point
tensors, plus naive reference implementations used by the tests and the benchmarks.

Ring all-reduce (Patarasuk & Yuan 2009; popularised for deep learning by Baidu / Horovod)
-----------------------------------------------------------------------------------------
Split the length-``n`` buffer into ``N`` contiguous chunks.  Rank ``r`` sends to ``r+1`` and
receives from ``r-1``.

* reduce-scatter, steps ``s = 0 .. N-2``: send chunk ``(r - s) mod N``, receive chunk
  ``(r - s - 1) mod N`` and add it into the local copy.  After ``N-1`` steps rank ``r`` holds the
  complete sum of chunk ``(r + 1) mod N``.
* all-gather, steps ``s = 0 .. N-2``: send chunk ``(r - s + 1) mod N`` (the one just completed),
  receive chunk ``(r - s) mod N`` and overwrite.

Each rank sends ``2 (N-1) n / N`` elements in total, independent of ``N`` as ``N`` grows -- the
bandwidth lower bound for all-reduce -- whereas reduce-to-root + broadcast makes the root send and
receive ``(N-1) n``.  Every chunk is further cut into segments of at most ``segment_bytes`` so
that the receiver adds segment ``k`` while segment ``k+1`` is still on the wire, and sends are
asynchronous so the outgoing and incoming chunk of one step overlap.

Every operation draws a tag from a per-*channel* sequence counter.  Ranks issue the operations of
one channel in the same order, so tags match without any negotiation; separate channels (the
training thread vs DDP's communication thread) can be in flight at the same time.
"""
from __future__ import annotations

import json
import threading

import numpy as np

from .transport import Transport

OPS = {
    "sum": np.add,
    "max": np.maximum,
    "min": np.minimum,
}


def chunk_bounds(n: int, parts: int) -> list[tuple[int, int]]:
    """Contiguous [lo, hi) ranges of almost equal size (the first ``n % parts`` get one extra)."""
    base, extra = divmod(n, parts)
    out, lo = [], 0
    for i in range(parts):
        hi = lo + base + (1 if i < extra else 0)
        out.append((lo, hi))
        lo = hi
    return out


class ProcessGroup:
    """The collectives of one generation.  ``rank``/``world`` are fixed for its lifetime."""

    def __init__(self, transport: Transport, segment_bytes: int = 1 << 20):
        self.t = transport
        self.rank, self.world = transport.rank, transport.world
        self.segment_bytes = segment_bytes
        self._seq: dict[int, int] = {}
        self._lock = threading.Lock()

    # -------------------------------------------------------------- plumbing
    def _next_tag(self, channel: int) -> int:
        with self._lock:
            seq = self._seq.get(channel, 0)
            self._seq[channel] = seq + 1
        # [channel:8][seq:40][sub:16]
        return ((channel & 0xFF) << 56) | ((seq & ((1 << 40) - 1)) << 16)

    @property
    def bytes_sent(self) -> int:
        return self.t.bytes_sent

    def _segments(self, lo: int, hi: int, itemsize: int) -> list[tuple[int, int]]:
        step = max(1, self.segment_bytes // itemsize)
        return [(a, min(a + step, hi)) for a in range(lo, hi, step)] or [(lo, lo)]

    # ------------------------------------------------------------ all-reduce
    def allreduce(self, arr: np.ndarray, op: str = "sum", channel: int = 0) -> np.ndarray:
        """In-place ring all-reduce of a contiguous 1-D numpy array.  Returns ``arr``."""
        if arr.ndim != 1 or not arr.flags.c_contiguous:
            raise ValueError("allreduce expects a contiguous 1-D array (use .reshape(-1))")
        if self.world == 1 or arr.size == 0:
            return arr
        tag = self._next_tag(channel)
        self.reduce_scatter_(arr, op, tag)
        self.all_gather_(arr, tag | 0x8000)
        return arr

    def reduce_scatter_(self, arr: np.ndarray, op: str = "sum", tag: int | None = None) -> tuple[int, int]:
        """Ring reduce-scatter in place.  Returns the ``[lo, hi)`` chunk this rank now owns fully
        reduced (chunk ``(rank + 1) mod N``)."""
        N, r = self.world, self.rank
        tag = self._next_tag(0) if tag is None else tag
        bounds = chunk_bounds(arr.size, N)
        if N == 1:
            return bounds[0]
        fn = OPS[op]
        right, left = (r + 1) % N, (r - 1) % N
        isz = arr.itemsize
        for s in range(N - 1):
            send_c, recv_c = (r - s) % N, (r - s - 1) % N
            sub = tag | ((s & 0x7F) << 8)
            handles = [self.t.isend(right, sub | (k & 0xFF), memoryview(arr[a:b]))
                       for k, (a, b) in enumerate(self._segments(*bounds[send_c], isz))]
            for k, (a, b) in enumerate(self._segments(*bounds[recv_c], isz)):
                data = np.frombuffer(self.t.recv(left, sub | (k & 0xFF)), dtype=arr.dtype)
                fn(arr[a:b], data, out=arr[a:b])
            for h in handles:
                h.wait()
        return bounds[(r + 1) % N]

    def all_gather_(self, arr: np.ndarray, tag: int | None = None) -> np.ndarray:
        """Ring all-gather in place, assuming rank ``r`` holds the final value of chunk
        ``(r + 1) mod N`` (the layout :meth:`reduce_scatter_` leaves behind)."""
        N, r = self.world, self.rank
        tag = self._next_tag(0) if tag is None else tag
        if N == 1:
            return arr
        bounds = chunk_bounds(arr.size, N)
        right, left = (r + 1) % N, (r - 1) % N
        isz = arr.itemsize
        for s in range(N - 1):
            send_c, recv_c = (r - s + 1) % N, (r - s) % N
            sub = tag | ((s & 0x7F) << 8)
            handles = [self.t.isend(right, sub | (k & 0xFF), memoryview(arr[a:b]))
                       for k, (a, b) in enumerate(self._segments(*bounds[send_c], isz))]
            for k, (a, b) in enumerate(self._segments(*bounds[recv_c], isz)):
                arr[a:b] = np.frombuffer(self.t.recv(left, sub | (k & 0xFF)), dtype=arr.dtype)
            for h in handles:
                h.wait()
        return arr

    # ------------------------------------------------------------ all-gather
    def all_gather_bytes(self, blob: bytes, channel: int = 0) -> list[bytes]:
        """Every rank contributes one (possibly different-sized) blob; all ranks get all blobs,
        passed around the ring in ``N-1`` steps."""
        N, r = self.world, self.rank
        out: list[bytes | None] = [None] * N
        out[r] = bytes(blob)
        if N == 1:
            return out  # type: ignore[return-value]
        tag = self._next_tag(channel)
        right, left = (r + 1) % N, (r - 1) % N
        for s in range(N - 1):
            send_i, recv_i = (r - s) % N, (r - s - 1) % N
            h = self.t.isend(right, tag | s, out[send_i])
            out[recv_i] = bytes(self.t.recv(left, tag | s))
            h.wait()
        return out  # type: ignore[return-value]

    def all_gather_array(self, arr: np.ndarray, channel: int = 0) -> list[np.ndarray]:
        blobs = self.all_gather_bytes(np.ascontiguousarray(arr).tobytes(), channel)
        return [np.frombuffer(b, dtype=arr.dtype) for b in blobs]

    def all_gather_json(self, obj, channel: int = 0) -> list:
        return [json.loads(b) for b in self.all_gather_bytes(json.dumps(obj).encode(), channel)]

    # ------------------------------------------------------------- broadcast
    def broadcast_bytes(self, blob: bytes | None, root: int = 0, channel: int = 0) -> bytes:
        """Pipelined ring broadcast: the root streams segments to ``root+1``, every other rank
        forwards each segment to its right neighbour as soon as it arrives (``N-1`` hops overlap,
        so a large blob costs about one transfer time, not ``N-1``)."""
        N, r = self.world, self.rank
        if N == 1:
            return bytes(blob or b"")
        tag = self._next_tag(channel)
        right, left = (r + 1) % N, (r - 1) % N
        last = (root - 1) % N  # the rank that does not forward
        seg = self.segment_bytes
        if r == root:
            data = memoryview(bytes(blob or b""))
            nseg = max(1, -(-len(data) // seg))
            hs = [self.t.isend(right, tag, np.array([len(data), nseg], dtype=np.int64))]
            hs += [self.t.isend(right, tag | (1 + k), data[k * seg:(k + 1) * seg]) for k in range(nseg)]
            for h in hs:
                h.wait()
            return bytes(data)
        hdr = self.t.recv(left, tag)
        size, nseg = (int(x) for x in np.frombuffer(hdr, dtype=np.int64))
        hs = [self.t.isend(right, tag, hdr)] if r != last else []
        parts = []
        for k in range(nseg):
            p = self.t.recv(left, tag | (1 + k))
            if r != last:
                hs.append(self.t.isend(right, tag | (1 + k), p))
            parts.append(p)
        for h in hs:
            h.wait()
        out = b"".join(parts)
        assert len(out) == size
        return out

    def broadcast(self, arr: np.ndarray, root: int = 0, channel: int = 0) -> np.ndarray:
        """In-place broadcast of a numpy array (same shape/dtype on every rank)."""
        data = self.broadcast_bytes(arr.tobytes() if self.rank == root else None, root, channel)
        if self.rank != root:
            arr.reshape(-1)[:] = np.frombuffer(data, dtype=arr.dtype)
        return arr

    # --------------------------------------------------------------- barrier
    def barrier(self, channel: int = 0) -> None:
        """Dissemination barrier (Hensgen, Finkel & Manber 1988): in round ``k`` rank ``r``
        signals ``r + 2^k`` and waits for ``r - 2^k``; ``ceil(log2 N)`` rounds."""
        N, r = self.world, self.rank
        if N == 1:
            return
        tag = self._next_tag(channel)
        k, d = 0, 1
        while d < N:
            h = self.t.isend((r + d) % N, tag | k, b"\x01")
            self.t.recv((r - d) % N, tag | k)
            h.wait()
            k, d = k + 1, d * 2

    # --------------------------------------------------------- point-to-point
    def send_array(self, dst: int, arr: np.ndarray, tag: int):
        """Send an array with its shape and dtype (used by pipeline parallelism)."""
        arr = np.ascontiguousarray(arr)
        meta = json.dumps({"s": list(arr.shape), "d": arr.dtype.str}).encode()
        payload = len(meta).to_bytes(4, "little") + meta + arr.tobytes()
        return self.t.isend(dst, tag, payload)

    def recv_array(self, src: int, tag: int) -> np.ndarray:
        buf = self.t.recv(src, tag)
        n = int.from_bytes(buf[:4], "little")
        meta = json.loads(bytes(buf[4:4 + n]))
        return np.frombuffer(memoryview(buf)[4 + n:], dtype=np.dtype(meta["d"])).reshape(meta["s"])

    def close(self) -> None:
        self.t.close()


# ------------------------------------------------------------------ baselines
def naive_allreduce(pg: ProcessGroup, arr: np.ndarray, channel: int = 0) -> np.ndarray:
    """Reduce-to-root then broadcast (star).  The root moves ``(N-1) n`` elements each way; this
    is the baseline the ring is compared against in ``scripts/bench_allreduce.py``.  The sum is
    taken in rank order 0, 1, ..., N-1, i.e. exactly ``sum(x_i)`` evaluated left to right."""
    N, r = pg.world, pg.rank
    if N == 1:
        return arr
    tag = pg._next_tag(channel)
    if r == 0:
        acc = arr.copy()
        for src in range(1, N):
            acc += np.frombuffer(pg.t.recv(src, tag), dtype=arr.dtype)
        arr[:] = acc
        hs = [pg.t.isend(dst, tag | 1, memoryview(arr)) for dst in range(1, N)]
        for h in hs:
            h.wait()
    else:
        pg.t.send(0, tag, memoryview(arr))
        arr[:] = np.frombuffer(pg.t.recv(0, tag | 1), dtype=arr.dtype)
    return arr
