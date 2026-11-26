"""Pipeline parallelism: one model stage per rank, GPipe and 1F1B micro-batch schedules.

A mini-batch is cut into ``M`` micro-batches.  Stage ``s`` receives activations from ``s-1``,
runs its layers, and sends its output to ``s+1``; in backward it receives the gradient of its
output from ``s+1`` and sends the gradient of its input to ``s-1``.  The last stage computes the
loss, scaled by ``1/M`` so the accumulated gradient equals that of the whole mini-batch.

Schedules (``S`` stages, ``M`` micro-batches)
--------------------------------------------
* **GPipe** (Huang et al. 2019): all ``M`` forwards, then all ``M`` backwards.  Bubble fraction
  ``(S-1)/(M+S-1)``; every stage stores activations for all ``M`` micro-batches.
* **1F1B** (PipeDream-Flush, Narayanan et al. 2019/2021): stage ``s`` runs ``S-s-1`` warm-up
  forwards, then alternates one forward / one backward, then drains.  Same bubble as GPipe for a
  flush schedule, but at most ``S - s`` micro-batches are in flight on stage ``s``, so activation
  memory is bounded by the depth of the pipeline rather than by ``M``.

Both are *synchronous* (one optimizer step per mini-batch, no weight staleness), so the result
matches single-process training on the same micro-batches up to floating-point summation order.
"""
from __future__ import annotations

import time

import torch
from torch import nn

from .collectives import ProcessGroup

PIPE_CHANNEL = 2
_FWD, _BWD = 1, 2


def gpipe_schedule(num_stages: int, stage: int, num_micro: int) -> list[tuple[str, int]]:
    return [("F", m) for m in range(num_micro)] + [("B", m) for m in range(num_micro)]


def one_f_one_b_schedule(num_stages: int, stage: int, num_micro: int) -> list[tuple[str, int]]:
    warm = min(num_stages - stage - 1, num_micro)
    acts: list[tuple[str, int]] = [("F", m) for m in range(warm)]
    f, b = warm, 0
    while f < num_micro:
        acts.append(("F", f))
        f += 1
        acts.append(("B", b))
        b += 1
    while b < num_micro:
        acts.append(("B", b))
        b += 1
    return acts


SCHEDULES = {"gpipe": gpipe_schedule, "1f1b": one_f_one_b_schedule}


def split_sequential(model: nn.Sequential, num_stages: int, weights: list[float] | None = None) -> list[nn.Sequential]:
    """Split an ``nn.Sequential`` into ``num_stages`` contiguous pieces with the smallest possible
    maximum stage weight (parameter count by default, or the given per-layer ``weights``).

    Parameter-free layers (activations) stay attached to the layer before them, so no stage ends up
    without parameters when it can be avoided.  The partition is exact: a small dynamic program
    over contiguous splits (the classic linear-partition problem)."""
    layers = list(model.children())
    w = weights or [float(sum(p.numel() for p in layer.parameters())) for layer in layers]
    units: list[list[int]] = []  # each unit = a weighted layer + the weight-free layers after it
    for i, wi in enumerate(w):
        if units and wi == 0:
            units[-1].append(i)
        else:
            units.append([i])
    if len(units) < num_stages:
        units = [[i] for i in range(len(layers))]
    if len(units) < num_stages:
        raise ValueError(f"cannot split {len(layers)} layers into {num_stages} stages")
    uw = [sum(w[i] for i in u) for u in units]
    n, inf = len(units), float("inf")
    pre = [0.0]
    for x in uw:
        pre.append(pre[-1] + x)
    # best[k][j]: minimal max-stage weight putting the first j units into k stages
    best = [[inf] * (n + 1) for _ in range(num_stages + 1)]
    cut = [[0] * (n + 1) for _ in range(num_stages + 1)]
    best[0][0] = 0.0
    for k in range(1, num_stages + 1):
        for j in range(k, n + 1):
            for i in range(k - 1, j):
                v = max(best[k - 1][i], pre[j] - pre[i])
                if v < best[k][j]:
                    best[k][j], cut[k][j] = v, i
    bounds, j = [n], n
    for k in range(num_stages, 0, -1):
        j = cut[k][j]
        bounds.append(j)
    bounds.reverse()
    return [nn.Sequential(*[layers[i] for u in units[bounds[k]:bounds[k + 1]] for i in u])
            for k in range(num_stages)]


class PipelineStage:
    def __init__(self, module: nn.Module, pg: ProcessGroup, loss_fn=None, schedule: str = "1f1b"):
        self.module, self.pg = module, pg
        self.stage, self.num_stages = pg.rank, pg.world
        self.loss_fn = loss_fn
        self.schedule = schedule
        self.step_id = 0
        self.timeline: list[dict] = []
        self.max_inflight = 0

    @property
    def is_first(self) -> bool:
        return self.stage == 0

    @property
    def is_last(self) -> bool:
        return self.stage == self.num_stages - 1

    def _tag(self, kind: int, mb: int) -> int:
        return (PIPE_CHANNEL << 56) | ((self.step_id & 0xFFFFFFFF) << 24) | (kind << 20) | (mb & 0xFFFFF)

    def train_step(self, inputs: torch.Tensor | None = None, targets: torch.Tensor | None = None,
                   num_micro: int = 4, record: bool = False) -> float | None:
        """Run forward+backward for one mini-batch.  Stage 0 passes ``inputs``, the last stage
        passes ``targets``.  Gradients accumulate into the stage's parameters (call
        ``optimizer.step()`` afterwards).  Returns the mini-batch loss on the last stage."""
        acts = SCHEDULES[self.schedule](self.num_stages, self.stage, num_micro)
        x_mb = list(torch.tensor_split(inputs, num_micro)) if self.is_first else None
        y_mb = list(torch.tensor_split(targets, num_micro)) if self.is_last else None
        saved: dict[int, tuple[torch.Tensor | None, torch.Tensor]] = {}
        pending = []
        total_loss = 0.0
        t_origin = time.perf_counter()
        for kind, m in acts:
            t0 = time.perf_counter()
            if kind == "F":
                if self.is_first:
                    x = x_mb[m]
                    inp = None
                else:
                    x = torch.from_numpy(self.pg.recv_array(self.stage - 1, self._tag(_FWD, m)).copy())
                    x.requires_grad_(True)
                    inp = x
                tc = time.perf_counter()  # input has arrived: compute starts here
                y = self.module(x)
                if self.is_last:
                    loss = self.loss_fn(y, y_mb[m]) / num_micro
                    total_loss += float(loss.detach())
                    saved[m] = (inp, loss)
                else:
                    pending.append(self.pg.send_array(self.stage + 1, y.detach().numpy(), self._tag(_FWD, m)))
                    saved[m] = (inp, y)
                self.max_inflight = max(self.max_inflight, len(saved))
            else:
                inp, out = saved.pop(m)
                if self.is_last:
                    tc = time.perf_counter()
                    out.backward()
                else:
                    g = torch.from_numpy(self.pg.recv_array(self.stage + 1, self._tag(_BWD, m)).copy())
                    tc = time.perf_counter()
                    torch.autograd.backward(out, g)
                if not self.is_first:
                    pending.append(self.pg.send_array(self.stage - 1, inp.grad.numpy(), self._tag(_BWD, m)))
            if record:
                self.timeline.append({"step": self.step_id, "stage": self.stage, "kind": kind, "mb": m,
                                      "wait0": t0 - t_origin, "t0": tc - t_origin,
                                      "t1": time.perf_counter() - t_origin,
                                      "inflight": len(saved)})
        for h in pending:
            h.wait()
        self.step_id += 1
        return total_loss if self.is_last else None


def reference_step(stages: list[nn.Module], inputs: torch.Tensor, targets: torch.Tensor, loss_fn,
                   num_micro: int) -> float:
    """Single-process equivalent of one pipeline step (same micro-batching and loss scaling)."""
    total = 0.0
    for x, y in zip(torch.tensor_split(inputs, num_micro), torch.tensor_split(targets, num_micro)):
        h = x
        for s in stages:
            h = s(h)
        loss = loss_fn(h, y) / num_micro
        loss.backward()
        total += float(loss.detach())
    return total


def bubble_fraction(num_stages: int, num_micro: int) -> float:
    return (num_stages - 1) / (num_micro + num_stages - 1)


__all__ = ["PipelineStage", "gpipe_schedule", "one_f_one_b_schedule", "split_sequential",
           "reference_step", "bubble_fraction"]
