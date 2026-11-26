"""Data parallelism: DDP-style gradient bucketing with communication/computation overlap.

How it works (the same idea as PyTorch DDP, Li et al. VLDB 2020, written from scratch):

1. Parameters are grouped into buckets of about ``bucket_mb`` megabytes in *reverse* registration
   order, because backward produces gradients roughly from the last layer to the first.
2. A ``post_accumulate_grad`` hook on every parameter marks it ready.  When every parameter of a
   bucket is ready, the bucket is handed to a background communication thread -- strictly in
   bucket order, so every rank issues the same collectives in the same order.
3. The comm thread all-reduces (or compresses, see :mod:`ringtrain.compression`) the bucket while
   autograd is still computing gradients of earlier layers.  Socket I/O and numpy releases the
   GIL, so the two really run at the same time.
4. :meth:`DistributedDataParallel.finish` (call it after ``backward``, before ``optimizer.step``)
   flushes buckets that never became ready (unused parameters count as zero), waits for the comm
   thread and writes the reduced gradients back into ``p.grad``.

With ``overlap=False`` all buckets are reduced after backward instead -- the baseline used to
measure what overlap buys.
"""
from __future__ import annotations

import queue
import threading
import time

import torch
from torch import nn

from .collectives import ProcessGroup
from .compression import Reducer

DDP_CHANNEL = 1


def broadcast_module(pg: ProcessGroup, module: nn.Module, root: int = 0) -> None:
    """Make every rank's parameters and buffers bit-identical to the root's."""
    tensors = [p.data for p in module.parameters()] + [b for b in module.buffers()]
    if pg.world == 1 or not tensors:
        return
    flat = torch.cat([t.reshape(-1).to(torch.float64) for t in tensors]).numpy()
    pg.broadcast(flat, root)
    o = 0
    for t in tensors:
        n = t.numel()
        t.copy_(torch.from_numpy(flat[o:o + n].copy()).view_as(t).to(t.dtype))
        o += n


class _Bucket:
    def __init__(self, index: int, params: list[nn.Parameter]):
        self.index, self.params = index, params
        self.ready: set[int] = set()
        self.launched = False
        self.result: list[torch.Tensor] | None = None


class DistributedDataParallel(nn.Module):
    def __init__(self, module: nn.Module, pg: ProcessGroup, reducer: Reducer | None = None,
                 bucket_mb: float = 1.0, average: bool = True, overlap: bool = True,
                 broadcast_init: bool = True):
        super().__init__()
        self.module, self.pg = module, pg
        self.reducer = reducer or Reducer()
        self.average, self.overlap = average, overlap
        if broadcast_init:
            broadcast_module(pg, module)
        params = [p for p in module.parameters() if p.requires_grad]
        self.buckets: list[_Bucket] = []
        cur, size, cap = [], 0, int(bucket_mb * (1 << 20))
        for p in reversed(params):
            cur.append(p)
            size += p.numel() * 4
            if size >= cap:
                self.buckets.append(_Bucket(len(self.buckets), cur))
                cur, size = [], 0
        if cur:
            self.buckets.append(_Bucket(len(self.buckets), cur))
        self._where = {id(p): (b.index, j) for b in self.buckets for j, p in enumerate(b.params)}
        self._lock = threading.Lock()
        self._next_launch = 0
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._error: BaseException | None = None
        self._done = threading.Condition()
        self._completed = 0
        self.comm_seconds = 0.0
        self.stats = {"steps": 0, "wait_s": 0.0}
        self._hooks = [p.register_post_accumulate_grad_hook(self._on_grad) for p in params]
        self._thread = threading.Thread(target=self._comm_loop, name="rt-ddp-comm", daemon=True)
        self._thread.start()

    def forward(self, *args, **kw):
        return self.module(*args, **kw)

    # ----------------------------------------------------------- grad hooks
    def _on_grad(self, p: torch.Tensor) -> None:
        if not self.overlap:
            return
        bi, j = self._where[id(p)]
        with self._lock:
            self.buckets[bi].ready.add(j)
            self._launch_ready()

    def _launch_ready(self) -> None:
        # launch strictly in bucket order so every rank issues identical collectives
        while self._next_launch < len(self.buckets):
            b = self.buckets[self._next_launch]
            if len(b.ready) < len(b.params):
                return
            self._launch(b)

    def _launch(self, b: _Bucket) -> None:
        grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in b.params]
        b.launched = True
        self._next_launch += 1
        self._q.put((b, grads))

    def _comm_loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                return
            b, grads = item
            t0 = time.perf_counter()
            try:
                if self._error is None:
                    b.result = self.reducer.reduce(self.pg, b.index, grads, channel=DDP_CHANNEL)
            except BaseException as e:  # noqa: BLE001 - surfaced in finish()
                self._error = e
            self.comm_seconds += time.perf_counter() - t0
            with self._done:
                self._completed += 1
                self._done.notify_all()

    # --------------------------------------------------------------- finish
    def finish(self) -> None:
        """Wait for all buckets and write reduced gradients into ``p.grad``."""
        t0 = time.perf_counter()
        with self._lock:
            while self._next_launch < len(self.buckets):
                self._launch(self.buckets[self._next_launch])
        with self._done:
            while self._completed < len(self.buckets):
                self._done.wait(0.05)
                if self._error is None:
                    try:
                        self.pg.t.check()
                    except BaseException as e:  # noqa: BLE001
                        self._error = e
                if self._error is not None and self._completed < len(self.buckets):
                    # the comm thread may be blocked in a recv that will raise shortly
                    self.pg.t.abort(str(self._error))
        err, self._error = self._error, None
        scale = 1.0 / self.pg.world if self.average else 1.0
        for b in self.buckets:
            if err is None:
                for p, g in zip(b.params, b.result):
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)
                    p.grad.copy_(g)
                    if scale != 1.0:
                        p.grad.mul_(scale)
            b.ready.clear()
            b.launched = False
            b.result = None
        self._next_launch = 0
        self._completed = 0
        self.stats["steps"] += 1
        self.stats["wait_s"] += time.perf_counter() - t0
        if err is not None:
            raise err

    def close(self) -> None:
        for h in self._hooks:
            h.remove()
        self._q.put(None)
