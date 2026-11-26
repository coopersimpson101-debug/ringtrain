#!/usr/bin/env python3
"""Scaling benchmarks on ONE machine: 1-4 worker processes over loopback TCP.

    python scripts/bench_scaling.py            # ~5-10 min; writes results/scaling.json, results/allreduce.json
    python scripts/bench_scaling.py --quick    # smaller, for CI

This is *not* a cluster measurement: all workers share one CPU, one memory bus and the kernel's
loopback device, so it shows the overheads of the implementation (protocol, copies, Python
threads), not network scaling.  Multi-machine numbers are a TODO for the author (deploy/).
"""
from __future__ import annotations

import argparse
import statistics
import time

from _common import CRASHES, launch_retry, machine, save


def median_run(n, entry, args, reps):
    runs = [launch_retry(n, entry, args)[0] for _ in range(reps)]
    return runs, statistics.median([r["samples_per_s"] for r in runs]) if "samples_per_s" in runs[0] else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-workers", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    reps = 1 if a.quick else a.reps
    adapter_args = {"hidden": 1024, "depth": 3, "num_samples": 32768} if not a.quick else \
        {"hidden": 256, "depth": 2, "num_samples": 8192}
    steps = 30 if not a.quick else 8
    t0 = time.time()
    out = {"machine": machine(), "adapter": "mlp", "adapter_args": adapter_args, "steps": steps, "reps": reps,
           "torch_threads_per_worker": 1, "weak": [], "strong": [], "variants": []}
    for n in range(1, a.max_workers + 1):
        for mode in ("weak", "strong"):
            args = {"adapter_args": dict(adapter_args, global_batch=512), "mode": mode, "per_worker_batch": 128,
                    "steps": steps, "warmup": 3}
            runs, med = median_run(n, "ringtrain.bench:train_bench", args, reps)
            best = sorted(runs, key=lambda r: r["samples_per_s"])[len(runs) // 2]
            best["samples_per_s_median"] = med
            best["samples_per_s_all"] = [r["samples_per_s"] for r in runs]
            out[mode].append(best)
            print(f"{mode:6s} N={n}: {med:8.0f} samples/s  compute {best['compute_s_per_step'] * 1e3:6.1f} ms/step"
                  f"  comm-wait {best['comm_wait_s_per_step'] * 1e3:6.1f} ms/step", flush=True)
    n = a.max_workers
    for name, extra in [("no-overlap", {"overlap": False}), ("topk:0.01", {"compression": "topk:0.01"}),
                        ("powersgd:4", {"compression": "powersgd:4"})]:
        args = {"adapter_args": dict(adapter_args, global_batch=512), "mode": "weak", "per_worker_batch": 128,
                "steps": steps, "warmup": 3, **extra}
        runs, med = median_run(n, "ringtrain.bench:train_bench", args, reps)
        r = runs[0]
        r.update({"variant": name, "samples_per_s_median": med, "samples_per_s_all": [x["samples_per_s"] for x in runs]})
        out["variants"].append(r)
        print(f"variant {name:12s} N={n}: {med:8.0f} samples/s, {r['bytes_sent_per_step'] / 1e6:7.2f} MB sent/step/rank",
              flush=True)
    out["seconds"] = time.time() - t0
    out["crashed_attempts"] = CRASHES
    save("scaling.json", out)

    sizes = [1 << 10, 1 << 14, 1 << 18, 1 << 20, 1 << 22] if not a.quick else [1 << 10, 1 << 16]
    ar = {"machine": machine(), "runs": []}
    for n in range(2, a.max_workers + 1):
        res = launch_retry(n, "ringtrain.bench:allreduce_bench", {"sizes": sizes})[0]
        ar["runs"].append(res)
        for x in res["results"]:
            print(f"allreduce N={n} {x['algo']:5s} {x['bytes']:>10d} B  {x['seconds'] * 1e3:8.2f} ms  "
                  f"busbw {x['busbw_GBps']:.3f} GB/s", flush=True)
    save("allreduce.json", ar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
