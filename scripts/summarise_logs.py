#!/usr/bin/env python3
"""Summarise ringtrain worker logs (the ``--log`` JSONL files): per generation, the world size,
steps, samples/s, and the failure / restore timeline.

    python scripts/summarise_logs.py results/fault_demo/run/w*.jsonl
"""
from __future__ import annotations

import json
import sys


def main(paths: list[str]) -> int:
    recs = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except ValueError:
                    pass
    if not recs:
        print("no records")
        return 1
    t0 = min(r["t"] for r in recs)
    steps: dict[tuple[int, int], dict] = {}
    for r in recs:
        if r["event"] == "step":
            steps.setdefault((r["gen"], r["step"]), r)
    print(f"{'gen':>4} {'world':>5} {'steps':>11} {'t_start':>8} {'t_end':>8} {'samples/s':>10} {'median step ms':>15}")
    for g in sorted({g for g, _ in steps}):
        rows = sorted((r for (gg, _), r in steps.items() if gg == g), key=lambda r: r["t"])
        span = rows[-1]["t"] - rows[0]["t"]
        sps = sum(r["samples"] for r in rows[1:]) / span if span > 0 else float("nan")
        med = sorted(r["step_s"] for r in rows)[len(rows) // 2] * 1e3
        print(f"{g:>4} {rows[0]['world']:>5} {rows[0]['step']:>5}-{rows[-1]['step']:<5} {rows[0]['t'] - t0:8.2f} "
              f"{rows[-1]['t'] - t0:8.2f} {sps:10.0f} {med:15.1f}")
    print("\nevents:")
    seen = set()
    for r in sorted(recs, key=lambda r: r["t"]):
        if r["event"] in ("failure", "restored", "reform", "assigned", "done"):
            key = (r["event"], r.get("gen"))
            if r["event"] != "failure" and key in seen:
                continue
            seen.add(key)
            extra = {k: r[k] for k in ("gen", "world", "step", "reason", "seconds") if k in r}
            print(f"  {r['t'] - t0:8.2f}s  {r['worker']:>4}  {r['event']:9s} {extra}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
