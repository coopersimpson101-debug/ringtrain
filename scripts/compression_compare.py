#!/usr/bin/env python3
"""Convergence vs traffic for exact all-reduce, top-k + error feedback and PowerSGD.

    python scripts/compression_compare.py          # 4 worker processes, ~3-6 min; results/compression.json

Every variant trains the same teacher-student MLP from the same initialisation on the same global
batches; only the gradient reducer differs.  Bytes are counted by the transport (frame headers
included), per rank per step.
"""
from __future__ import annotations

import argparse
import time

from _common import CRASHES, launch_retry, machine, save


VARIANTS = ["none", "topk:0.1", "topk:0.01", "powersgd:4", "powersgd:1"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    steps = 40 if a.quick else a.steps
    adapter_args = {"hidden": 512, "depth": 3, "num_samples": 32768, "global_batch": 256, "lr": 0.05}
    out = {"machine": machine(), "workers": a.workers, "steps": steps, "adapter_args": adapter_args, "variants": []}
    for v in VARIANTS:
        t0 = time.time()
        r = launch_retry(a.workers, "ringtrain.bench:convergence_run",
                             {"adapter_args": adapter_args, "compression": v, "steps": steps}, timeout=1800)[0]
        r["compression"] = v
        out["variants"].append(r)
        print(f"{v:11s} final loss {sum(r['losses'][-20:]) / 20:.4f}  acc {r['accuracy']:.3f}  "
              f"{r['bytes_per_step'] / 1e6:7.3f} MB/step/rank  ({time.time() - t0:.0f}s)", flush=True)
    out["crashed_attempts"] = CRASHES
    save("compression.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
