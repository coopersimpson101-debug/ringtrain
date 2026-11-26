"""Benchmark entry points, run with :func:`ringtrain.launch.launch_processes`.

They return plain dicts; ``scripts/bench_scaling.py`` and ``scripts/bench_allreduce.py`` turn them
into ``results/*.json`` and plots.
"""
from __future__ import annotations

import time

import numpy as np

from .collectives import ProcessGroup, naive_allreduce
from .data import global_batch_indices, shard


def allreduce_bench(pg: ProcessGroup, args: dict) -> dict:
    """Time ring vs naive all-reduce over a list of float32 sizes."""
    out = []
    if "segment_bytes" in args:
        pg.segment_bytes = int(args["segment_bytes"])
    for n in args.get("sizes", [1 << 10, 1 << 16, 1 << 20, 1 << 22]):
        for algo in args.get("algos", ["ring", "naive"]):
            a = np.ones(n, dtype=np.float32)
            reps = max(3, min(50, int(args.get("budget_bytes", 2e8) // max(4 * n, 1))))
            for _ in range(2):  # warm-up
                (pg.allreduce(a) if algo == "ring" else naive_allreduce(pg, a))
            pg.barrier()
            t0 = time.perf_counter()
            for _ in range(reps):
                (pg.allreduce(a) if algo == "ring" else naive_allreduce(pg, a))
            dt = (time.perf_counter() - t0) / reps
            # every rank must agree on the time (take the slowest)
            m = np.array([dt])
            pg.allreduce(m, op="max")
            dt = float(m[0])
            nbytes = 4 * n
            out.append({"algo": algo, "elements": n, "bytes": nbytes, "seconds": dt,
                        "algbw_GBps": nbytes / dt / 1e9,
                        "busbw_GBps": nbytes / dt / 1e9 * (2 * (pg.world - 1) / pg.world)})
    return {"world": pg.world, "rank": pg.rank, "results": out}


def train_bench(pg: ProcessGroup, args: dict) -> dict:
    """Data-parallel training throughput of an adapter for a fixed number of steps.

    ``mode=weak``: per-worker batch fixed (global batch = per_worker * N).
    ``mode=strong``: global batch fixed.
    """
    import torch

    from .adapters import make_adapter
    from .compression import make_reducer
    from .ddp import DistributedDataParallel
    kw = dict(args.get("adapter_args", {}))
    per = int(args.get("per_worker_batch", 128))
    if args.get("mode", "weak") == "weak":
        kw["global_batch"] = per * pg.world
    ad = make_adapter(args.get("adapter", "mlp"), **kw)
    model = ad.build_model(0)
    opt = ad.build_optimizer(model)
    ddp = DistributedDataParallel(model, pg, make_reducer(args.get("compression")),
                                  bucket_mb=float(args.get("bucket_mb", 1.0)), average=False,
                                  overlap=bool(args.get("overlap", True)))
    B = ad.global_batch
    steps, warm = int(args.get("steps", 30)), int(args.get("warmup", 5))
    compute_s = 0.0
    bytes0 = 0
    losses = []
    for s in range(steps + warm):
        if s == warm:
            pg.barrier()
            t0 = time.perf_counter()
            bytes0 = pg.bytes_sent
            compute_s = 0.0
            ddp.stats["wait_s"] = 0.0
        local = shard(global_batch_indices(ad.num_samples, B, s, 0), pg.rank, pg.world)
        tc = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss, cnt = ad.loss_sum(model, ad.batch(local))
        (loss / B).backward()
        compute_s += time.perf_counter() - tc
        ddp.finish()
        opt.step()
        losses.append(float(loss.detach()) / max(cnt, 1))
    pg.barrier()
    wall = time.perf_counter() - t0
    m = np.array([wall, compute_s, ddp.stats["wait_s"]])
    pg.allreduce(m, op="max")
    ddp.close()
    nparams = sum(p.numel() for p in model.parameters())
    return {"world": pg.world, "rank": pg.rank, "global_batch": B, "steps": steps, "wall_s": float(m[0]),
            "samples_per_s": B * steps / float(m[0]), "compute_s_per_step": float(m[1]) / steps,
            "comm_wait_s_per_step": float(m[2]) / steps, "params": nparams,
            "bytes_sent_per_step": (pg.bytes_sent - bytes0) / steps, "last_loss": losses[-1],
            "torch_threads": torch.get_num_threads()}


def pipeline_bench(pg: ProcessGroup, args: dict) -> dict:
    """Train the char-LM split into ``world`` pipeline stages with one schedule; return per-step
    times, the event timeline of the last step, peak in-flight micro-batches and losses."""
    import torch

    from .adapters.charlm import CharLMAdapter, lm_loss
    from .pipeline import PipelineStage, split_sequential
    ad = CharLMAdapter(ctx=int(args.get("ctx", 64)), d=int(args.get("d", 96)), layers=int(args.get("layers", 6)),
                       global_batch=int(args.get("batch", 32)), lr=float(args.get("lr", 3e-3)))
    full = ad.build_model(0)
    stage = split_sequential(full, pg.world)[pg.rank]
    opt = torch.optim.AdamW(stage.parameters(), lr=ad.lr)
    st = PipelineStage(stage, pg, loss_fn=lm_loss, schedule=args.get("schedule", "1f1b"))
    steps, M = int(args.get("steps", 10)), int(args.get("micro", 8))
    times, losses = [], []
    for s in range(steps):
        x, y = ad.batch(global_batch_indices(ad.num_samples, ad.global_batch, s, 0))
        pg.barrier()
        t0 = time.perf_counter()
        opt.zero_grad()
        loss = st.train_step(x if st.is_first else None, y if st.is_last else None, num_micro=M,
                             record=(s == steps - 1))
        opt.step()
        times.append(time.perf_counter() - t0)
        losses.append(loss)
    return {"rank": pg.rank, "schedule": st.schedule, "micro": M, "step_s": times, "losses": losses,
            "timeline": st.timeline, "max_inflight": st.max_inflight,
            "stage_params": sum(p.numel() for p in stage.parameters())}


def convergence_run(pg: ProcessGroup, args: dict) -> dict:
    """Train with a given gradient reducer; return the global loss per step and bytes sent."""
    import torch

    from .adapters import make_adapter
    from .compression import make_reducer
    from .ddp import DistributedDataParallel
    ad = make_adapter(args.get("adapter", "mlp"), **args.get("adapter_args", {}))
    model = ad.build_model(0)
    opt = ad.build_optimizer(model)
    ddp = DistributedDataParallel(model, pg, make_reducer(args.get("compression")), bucket_mb=1.0, average=False)
    B, steps = ad.global_batch, int(args.get("steps", 200))
    losses, b0, t0 = [], pg.bytes_sent, time.perf_counter()
    for s in range(steps):
        local = shard(global_batch_indices(ad.num_samples, B, s, 0), pg.rank, pg.world)
        opt.zero_grad(set_to_none=True)
        loss, _ = ad.loss_sum(model, ad.batch(local))
        (loss / B).backward()
        ddp.finish()
        opt.step()
        m = np.array([float(loss.detach())])
        pg.allreduce(m)
        losses.append(float(m[0]) / B)
    wall = time.perf_counter() - t0
    ddp.close()
    # the per-step loss all-reduce is 8 bytes + headers; subtract nothing, it is tiny and identical
    acc = ad.accuracy(model) if hasattr(ad, "accuracy") else None
    return {"rank": pg.rank, "losses": losses, "bytes_per_step": (pg.bytes_sent - b0) / steps,
            "accuracy": acc, "wall_s": wall, "params": sum(p.numel() for p in model.parameters()),
            "torch_threads": torch.get_num_threads()}
