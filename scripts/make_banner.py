#!/usr/bin/env python3
"""Generate docs/assets/banner.svg.  The motif on the right is REAL output: the loss curve of the
fault-injection run in results/fault_demo (one polyline per generation, the kill marked), drawn
around a 4-node ring.

    python scripts/make_banner.py
    node render_png.mjs docs/assets/banner.svg docs/assets/banner.png   # optional, via @resvg/resvg-js
"""
from __future__ import annotations

import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCENT = "#38BDF8"


def runs():
    d = os.path.join(ROOT, "results", "fault_demo", "run")
    steps = {}
    for p in sorted(glob.glob(os.path.join(d, "w*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("event") == "step":
                    steps.setdefault((r["gen"], r["step"]), (r["loss"], r["world"]))
    summ = json.load(open(os.path.join(ROOT, "results", "fault_demo", "summary.json"), encoding="utf-8"))
    return steps, summ


def main() -> None:
    steps, summ = runs()
    W, H = 1280, 400
    x0, x1, y0, y1 = 700, 1230, 70, 330
    smax = max(s for _, s in steps)
    lmin = math.log(min(v[0] for v in steps.values()))
    lmax = math.log(max(v[0] for v in steps.values()))

    def X(s):
        return x0 + (x1 - x0) * s / smax

    def Y(loss):
        return y1 - (y1 - y0) * (math.log(loss) - lmin) / (lmax - lmin)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">',
             '<defs><pattern id="g" width="32" height="32" patternUnits="userSpaceOnUse">'
             '<path d="M32 0H0V32" fill="none" stroke="#161b22" stroke-width="1"/></pattern></defs>',
             f'<rect width="{W}" height="{H}" fill="#0d1117"/><rect width="{W}" height="{H}" fill="url(#g)"/>']
    # ring of 4 workers behind the curve
    cx, cy, R = 965, 200, 150
    pts = [(cx + R * math.cos(math.pi / 2 * k - math.pi / 4), cy + R * math.sin(math.pi / 2 * k - math.pi / 4)) for k in range(4)]
    parts.append(f'<circle cx="{cx}" cy="{cy}" r="{R}" fill="none" stroke="{ACCENT}" stroke-opacity="0.18" '
                 'stroke-width="2" stroke-dasharray="6 8"/>')
    for k, (px, py) in enumerate(pts):
        dead = k == 2
        parts.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="13" fill="#0d1117" stroke="'
                     f'{"#f85149" if dead else ACCENT}" stroke-opacity="0.55" stroke-width="2"/>')
    # loss curve, one polyline per generation
    gens = sorted({g for g, _ in steps})
    for g in gens:
        rows = sorted((s, v) for (gg, s), v in steps.items() if gg == g)
        rows = rows[:: max(1, len(rows) // 160)]
        d = " ".join(f"{X(s):.1f},{Y(v[0]):.1f}" for s, v in rows)
        parts.append(f'<polyline points="{d}" fill="none" stroke="{ACCENT}" stroke-width="2.4" '
                     'stroke-linejoin="round" stroke-opacity="0.95"/>')
    kill = next((e for e in summ["injected"] if e["event"] == "kill_worker"), None)
    if kill:
        kx = X(kill["seen_step"])
        parts.append(f'<line x1="{kx:.1f}" y1="{y0 - 10}" x2="{kx:.1f}" y2="{y1 + 10}" stroke="#f85149" '
                     'stroke-width="2" stroke-dasharray="5 5"/>')
        parts.append(f'<text x="{kx + 8:.1f}" y="{y0 + 4}" fill="#f85149" font-family="ui-monospace,SFMono-Regular,'
                     f'Menlo,Consolas,monospace" font-size="15">worker killed</text>')
    mono = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
    parts += [
        f'<text x="64" y="178" fill="#e6edf3" font-family="{mono}" font-size="88" font-weight="bold">ringtrain</text>',
        f'<text x="68" y="228" fill="#8b949e" font-family="{mono}" font-size="22">ring all-reduce over TCP · DDP + pipeline</text>',
        f'<text x="68" y="260" fill="#8b949e" font-family="{mono}" font-size="22">compression · checkpoints in raftstore</text>',
        f'<text x="68" y="314" fill="{ACCENT}" font-family="{mono}" font-size="17">training that survives a node dying</text>',
        f'<text x="{x1}" y="{H - 24}" fill="#8b949e" font-family="{mono}" font-size="12" text-anchor="end">'
        'real loss curve from scripts/fault_demo.py</text>',
        "</svg>"]
    out = os.path.join(ROOT, "docs", "assets", "banner.svg")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")
    print("wrote docs/assets/banner.svg")


if __name__ == "__main__":
    main()
