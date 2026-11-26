#!/usr/bin/env python3
"""Pipeline parallelism on 4 worker processes: GPipe vs 1F1B on the char-level transformer.

    python scripts/pipeline_timeline.py          # results/pipeline.json (real per-stage event timelines)

The model (embedding, 6 transformer blocks, head) is split into 4 stages, one per process; the
mini-batch is cut into 8 micro-batches.  Recorded: step times, peak in-flight micro-batches per
stage, and the forward/backward timeline of the last step (what the Gantt chart in the README
shows).  Losses are checked against the single-process run on the same micro-batches.
"""
from __future__ import annotations

import argparse
import copy

from _common import CRASHES, launch_retry, machine, save



def reference(args: dict, steps: int) -> list[float]:
    import torch

    from ringtrain.adapters.charlm import CharLMAdapter, lm_loss
    from ringtrain.data import global_batch_indices
    from ringtrain.pipeline import reference_step, split_sequential
    torch.set_num_threads(1)
    ad = CharLMAdapter(ctx=args["ctx"], d=args["d"], layers=args["layers"], global_batch=args["batch"], lr=args["lr"])
    stages = split_sequential(copy.deepcopy(ad.build_model(0)), 4)
    opts = [torch.optim.AdamW(s.parameters(), lr=ad.lr) for s in stages]
    out = []
    for s in range(steps):
        x, y = ad.batch(global_batch_indices(ad.num_samples, ad.global_batch, s, 0))
        for o in opts:
            o.zero_grad()
        out.append(reference_step(stages, x, y, lm_loss, args["micro"]))
        for o in opts:
            o.step()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--micro", type=int, default=8)
    a = ap.parse_args()
    base = {"ctx": 64, "d": 128, "layers": 6, "batch": 32, "lr": 3e-3, "micro": a.micro, "steps": a.steps}
    out = {"machine": machine(), "config": base, "stages": 4, "runs": {}}
    ref = reference(base, a.steps)
    out["reference_losses"] = ref
    for sched in ("gpipe", "1f1b"):
        res = launch_retry(4, "ringtrain.bench:pipeline_bench", dict(base, schedule=sched), timeout=1800)
        losses = res[-1]["losses"]
        diff = max(abs(x - y) for x, y in zip(losses, ref))
        out["runs"][sched] = {"stages": res, "max_abs_loss_diff_vs_reference": diff}
        times = sorted(res[0]["step_s"][2:])
        print(f"{sched:5s}: median step {times[len(times) // 2] * 1e3:7.1f} ms; peak in-flight per stage "
              f"{[r['max_inflight'] for r in res]}; max |loss - reference| = {diff:.2e}", flush=True)
    out["crashed_attempts"] = CRASHES
    save("pipeline.json", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
