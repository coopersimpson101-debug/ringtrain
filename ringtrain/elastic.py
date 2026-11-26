"""The elastic, fault-tolerant data-parallel trainer.

One worker process runs :class:`ElasticTrainer`.  Its life is a loop over *generations*::

    join the coordinator ──> ASSIGN(gen, rank, world, peers)
          ^                         │
          │                         v
          │       connect full mesh; rank 0 fetches LATEST checkpoint from the store
          │       and broadcasts it; everyone restores model / optimiser / RNG / step
          │                         │
          │                         v
          │       train: shard of the global batch ─> backward ─> DDP ring all-reduce ─> step
          │              every K steps rank 0 checkpoints asynchronously to raftstore
          │                         │
          ├──── PeerFailure / ABORT (a worker died): report, drop everything since the last
          │     checkpoint, wait for the next generation of survivors
          └──── a newcomer is waiting (scale-up): all ranks agree at a step boundary through the
                per-step metrics all-reduce, rank 0 checkpoints synchronously, barrier, re-form

Why roll back instead of continuing from the survivors' memory?  A rank can die half-way through
the last all-gather, when some survivors have already applied step ``s`` and others have not, so
in-memory state is not guaranteed consistent.  The checkpoint is.  The cost is at most
``ckpt_every`` steps of recomputation, which is what the fault demo measures.

The global batch size is fixed for the run and losses are summed and divided by it, so after a
membership change the survivors compute the *same* update as the full group would have (up to
floating-point summation order); only throughput changes.
"""
from __future__ import annotations

import json
import os
import random
import time

import numpy as np
import torch

from .checkpoint import CheckpointStore, loads_state
from .collectives import ProcessGroup
from .compression import make_reducer
from .data import global_batch_indices, shard
from .ddp import DistributedDataParallel
from .rendezvous import RendezvousClient
from .transport import Endpoint, PeerFailure, Transport


def _rng_state() -> dict:
    s = np.random.get_state()
    return {"torch": torch.get_rng_state(), "python": random.getstate(),
            "numpy": {"keys": torch.from_numpy(s[1].astype(np.int64)), "pos": int(s[2]),
                      "has_gauss": int(s[3]), "gauss": float(s[4])}}


def _set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    random.setstate(st["python"])
    n = st["numpy"]
    np.random.set_state(("MT19937", n["keys"].numpy().astype(np.uint32), n["pos"], n["has_gauss"], n["gauss"]))


class ElasticTrainer:
    def __init__(self, adapter, store: CheckpointStore, coord_addr: str, worker_id: str, total_steps: int,
                 ckpt_every: int = 25, seed: int = 0, compression: str | None = None, bucket_mb: float = 1.0,
                 log_path: str | None = None, host: str = "127.0.0.1", port: int = 0,
                 advertise: str | None = None, hb_timeout: float = 8.0, min_step_s: float = 0.0):
        self.adapter, self.store = adapter, store
        self.coord_addr, self.id = coord_addr, worker_id
        self.total_steps, self.ckpt_every, self.seed = total_steps, ckpt_every, seed
        self.compression, self.bucket_mb = compression, bucket_mb
        self.log_path = log_path
        self.host, self.port, self.advertise = host, port, advertise
        self.hb_timeout = hb_timeout
        self.min_step_s = min_step_s  # pad steps (emulate slower hardware in demos/tests)
        self._t: Transport | None = None
        self.summary = {"generations": 0, "failures": 0, "steps_run": 0, "restores": []}

    # ----------------------------------------------------------------- utils
    def log(self, event: str, **kw) -> None:
        if not self.log_path:
            return
        rec = {"t": time.time(), "worker": self.id, "pid": os.getpid(), "event": event, **kw}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def _on_abort(self, gen: int, reason: str) -> None:
        t = self._t
        if t is not None and t.gen == gen:
            t.abort(reason)

    # ------------------------------------------------------------------ main
    def run(self) -> dict:
        ep = Endpoint(self.host, self.port, self.advertise)
        rdv = RendezvousClient(self.coord_addr, self.id, ep.addr, on_abort=self._on_abort)
        self.log("start", addr=ep.addr)
        last_gen = -1
        try:
            while True:
                a = rdv.wait_assignment(last_gen)
                if a is None:
                    self.log("run_finished_elsewhere")
                    break
                last_gen = a.gen
                self.summary["generations"] += 1
                self.log("assigned", gen=a.gen, rank=int(a["rank"]), world=int(a["world"]), ids=a["ids"])
                pg = None
                try:
                    t = Transport(ep, a.gen, int(a["rank"]), list(a["peers"]), hb_timeout=self.hb_timeout)
                    self._t = t
                    if rdv.is_aborted(a.gen):
                        t.abort("aborted before start")
                    pg = ProcessGroup(t)
                    outcome = self._train_generation(pg, rdv, a.gen)
                    if outcome == "done":
                        rdv.done(a.gen)
                        self.log("done", gen=a.gen)
                        break
                    self.log("reform", gen=a.gen)
                except PeerFailure as e:
                    self.summary["failures"] += 1
                    self.log("failure", gen=a.gen, peer=e.peer, reason=e.reason, kind=type(e).__name__)
                    rdv.report_failure(a.gen, e.reason, e.peer)
                finally:
                    self._t = None
                    if pg is not None:
                        pg.close()
        finally:
            rdv.close()
            ep.close()
        return self.summary

    def _restore(self, pg: ProcessGroup):
        """Rank 0 reads LATEST from the store and broadcasts it; returns (state | None, ptr)."""
        t0 = time.time()
        blob, ptr = b"", None
        if pg.rank == 0:
            try:
                self.store.wait()  # a previous generation's async upload must land first
            except Exception as e:  # noqa: BLE001
                self.log("ckpt_upload_failed", error=str(e))
            got = self.store.load_latest_blob()
            if got is not None:
                ptr, blob = got
        hdr = pg.broadcast_bytes(json.dumps(ptr).encode() if pg.rank == 0 else None)
        ptr = json.loads(hdr)
        blob = pg.broadcast_bytes(blob if pg.rank == 0 else None)
        state = loads_state(blob) if blob else None
        return state, ptr, time.time() - t0, len(blob)

    def _train_generation(self, pg: ProcessGroup, rdv: RendezvousClient, gen: int) -> str:
        ad = self.adapter
        state, ptr, restore_s, nbytes = self._restore(pg)
        model = ad.build_model(self.seed)
        opt = ad.build_optimizer(model)
        step = 0
        if state is not None:
            model.load_state_dict(state["model"])
            opt.load_state_dict(state["optim"])
            _set_rng_state(state["rng"])
            step = int(state["step"])
        self.summary["restores"].append({"gen": gen, "step": step, "seconds": restore_s})
        self.log("restored", gen=gen, step=step, from_ckpt=ptr is not None, seconds=restore_s,
                 bytes=nbytes, world=pg.world, rank=pg.rank)
        reducer = make_reducer(self.compression)  # fresh error-feedback state for the new group
        ddp = DistributedDataParallel(model, pg, reducer, bucket_mb=self.bucket_mb, average=False,
                                      broadcast_init=state is None)
        B = ad.global_batch
        try:
            while step < self.total_steps:
                t0 = time.perf_counter()
                local = shard(global_batch_indices(ad.num_samples, B, step, self.seed), pg.rank, pg.world)
                opt.zero_grad(set_to_none=True)
                loss_sum, count = ad.loss_sum(model, ad.batch(local))
                (loss_sum / B).backward()
                ddp.finish()
                opt.step()
                step += 1
                self.summary["steps_run"] += 1
                # one tiny all-reduce per step: global loss, sample count, and the scale-up vote
                if self.min_step_s:
                    time.sleep(max(0.0, self.min_step_s - (time.perf_counter() - t0)))
                m = np.array([float(loss_sum.detach()), float(count), 1.0 if rdv.pending else 0.0])
                pg.allreduce(m)
                dt = time.perf_counter() - t0
                self.log("step", gen=gen, rank=pg.rank, world=pg.world, step=step,
                         loss=m[0] / max(m[1], 1.0), samples=int(m[1]), step_s=dt)
                ckpt_due = step % self.ckpt_every == 0 or step == self.total_steps
                if pg.rank == 0 and ckpt_due:
                    self.store.save_async(step, self._state(model, opt, step),
                                          {"world": pg.world, "gen": gen})
                    self.log("checkpoint", gen=gen, step=step, mode="async")
                if m[2] > 0 and step < self.total_steps:
                    # graceful scale-up: everyone voted at the same step, so stop here together
                    if pg.rank == 0:
                        if not ckpt_due:
                            self.store.save_async(step, self._state(model, opt, step),
                                                  {"world": pg.world, "gen": gen})
                        self.store.wait()
                        self.log("checkpoint", gen=gen, step=step, mode="sync-before-reform")
                    pg.barrier()
                    if pg.rank == 0:
                        rdv.request_reform(gen)
                    return "reform"
            if pg.rank == 0:
                self.store.wait()
            pg.barrier()
            return "done"
        finally:
            ddp.close()

    def _state(self, model, opt, step: int) -> dict:
        return {"model": {k: v.detach().clone() for k, v in model.state_dict().items()},
                "optim": opt.state_dict(), "rng": _rng_state(), "step": step,
                "data_cursor": {"step": step, "samples_consumed": step * self.adapter.global_batch,
                                "seed": self.seed}}
