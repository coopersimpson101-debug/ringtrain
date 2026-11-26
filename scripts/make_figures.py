#!/usr/bin/env python3
"""Render every README figure from the JSON/JSONL files in results/ (nothing is typed in).

    python scripts/make_figures.py
"""
from __future__ import annotations

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "assets")

# validated categorical order (blue, orange, aqua, yellow, magenta); text in ink colours
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlecolor": INK, "legend.frameon": False, "lines.linewidth": 2.0, "svg.fonttype": "none",
    "font.family": "DejaVu Sans",
})


def read_jsonl(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                pass
    return out


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=110 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("wrote docs/assets/" + name + ".{svg,png}")


def load_run(d):
    """Merge per-worker logs: the per-step loss is the all-reduced global loss, identical on every
    rank, so take one record per (generation, step)."""
    steps, events = {}, []
    for p in sorted(glob.glob(os.path.join(d, "w*.jsonl"))):
        for r in read_jsonl(p):
            if r["event"] == "step":
                steps.setdefault((r["gen"], r["step"]), r)
            else:
                events.append(r)
    coord = read_jsonl(os.path.join(d, "coordinator.jsonl")) if os.path.exists(os.path.join(d, "coordinator.jsonl")) else []
    return steps, events, coord


# ------------------------------------------------------------------ fault demo
def fig_fault():
    base = os.path.join(RES, "fault_demo")
    if not os.path.exists(os.path.join(base, "summary.json")):
        return
    summ = json.load(open(os.path.join(base, "summary.json"), encoding="utf-8"))
    steps, events, coord = load_run(os.path.join(base, "run"))
    ref_steps, _, _ = load_run(os.path.join(base, "reference")) if os.path.isdir(os.path.join(base, "reference")) else ({}, [], [])
    t0 = min(r["t"] for r in steps.values())
    gens = sorted({g for g, _ in steps})
    inj = {e["event"]: e for e in summ["injected"]}

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 6.6), gridspec_kw={"height_ratios": [3, 2]})
    if ref_steps:
        rs = sorted(ref_steps.values(), key=lambda r: r["step"])
        ax.plot([r["step"] for r in rs], [r["loss"] for r in rs], color="#b8b7b1", lw=3.5,
                label="reference: 4 workers, no faults", zorder=1)
    for i, g in enumerate(gens):
        rows = sorted((r for (gg, _), r in steps.items() if gg == g), key=lambda r: r["step"])
        w = rows[0]["world"]
        ax.plot([r["step"] for r in rows], [r["loss"] for r in rows], color=C[[4, 3, 2, 1].index(w) if w in (4, 3, 2, 1) else 0],
                lw=1.6, zorder=2)
        ax2.plot([r["t"] - t0 for r in rows], [r["step"] for r in rows], color=C[[4, 3, 2, 1].index(w)], lw=2.2)
    # legend proxies by world size
    for w in sorted({r["world"] for r in steps.values()}, reverse=True):
        ax.plot([], [], color=C[[4, 3, 2, 1].index(w)], label=f"faulty run, {w} workers")
    kill = inj.get("kill_worker")
    restores = sorted((e for e in events if e["event"] == "restored" and e.get("from_ckpt")), key=lambda e: e["t"])
    if kill:
        ks = kill["seen_step"]
        ax.axvline(ks, color="#e34948", lw=1.4, ls="--")
        ax.annotate(f"SIGKILL worker {kill['id']}\n(step {ks})", (ks, ax.get_ylim()[1]), xytext=(6, -4),
                    textcoords="offset points", va="top", color=INK, fontsize=9)
        ax2.axvline(kill["t"] - t0, color="#e34948", lw=1.4, ls="--")
        ax2.annotate("worker killed", (kill["t"] - t0, 0), xytext=(4, 4), textcoords="offset points", color=INK, fontsize=9)
        rb = [e for e in restores if e["t"] > kill["t"]]
        if rb:
            ax.annotate(f"rolled back to checkpoint {rb[0]['step']}", (rb[0]["step"], ax.get_ylim()[1]),
                        xytext=(6, -34), textcoords="offset points", va="top", color=INK2, fontsize=8.5)
    sk = inj.get("kill_store_leader")
    if sk:
        ax.axvline(sk["seen_step"], color=C[3], lw=1.2, ls=":")
        ax.annotate(f"raftstore leader\n'{sk['id']}' killed", (sk["seen_step"], ax.get_ylim()[1]), xytext=(6, -4),
                    textcoords="offset points", va="top", color=INK, fontsize=9)
        ax2.axvline(sk["t"] - t0, color=C[3], lw=1.2, ls=":")
    rj = inj.get("restart_worker")
    if rj:
        ax2.axvline(rj["t"] - t0, color=C[2], lw=1.2, ls="-.")
        ax2.annotate(f"{rj['id']} restarted", (rj["t"] - t0, 0), xytext=(4, 4), textcoords="offset points",
                     color=INK, fontsize=9)
        later = [e for e in coord if e["event"] == "form" and e["t"] > rj["t"]]
        if later:
            rows = sorted((r for (g, _), r in steps.items() if g == later[0]["gen"]), key=lambda r: r["t"])
            if rows:
                ax.axvline(rows[0]["step"], color=C[2], lw=1.2, ls="-.")
                ax.annotate(f"{rj['id']} re-joined\n(step {rows[0]['step']})", (rows[0]["step"], ax.get_ylim()[1]),
                            xytext=(-6, -4), textcoords="offset points", va="top", ha="right", color=INK, fontsize=9)
                ax2.annotate(f"re-joined, gen {later[0]['gen']}", (rows[0]["t"] - t0, rows[0]["step"]), xytext=(-6, 4),
                             textcoords="offset points", ha="right", color=INK, fontsize=9)
    ax.set_yscale("log")
    ax.set_xlabel("global step")
    ax.set_ylabel("training loss (global batch 512)")
    ax.set_title("Training continues after a worker is killed mid-run")
    ax.legend(loc="lower left", fontsize=9)
    ax2.set_xlabel("wall-clock seconds")
    ax2.set_ylabel("global step")
    ax2.set_title("Timeline (colour = workers in the group)")
    detect = ""
    if kill:
        forms = [e for e in coord if e["event"] == "form" and e["t"] > kill["t"]]
        resumed = [r["t"] for r in steps.values() if r["t"] > kill["t"] and forms and r["gen"] == forms[0]["gen"]]
        if resumed:
            detect = f"Training resumed {min(resumed) - kill['t']:.1f} s after the kill. "
        first = next((e for e in coord if e["event"] == "form"), None)
        if first and first["ids"][0] == kill["id"]:
            detect += f"{kill['id']} was rank 0, the checkpoint writer: its in-flight upload died with it."
    m = summ.get("machine", {})
    fig.text(0.01, -0.035, f"Measured {summ.get('date', '')} on one {m.get('os', '')} desktop ({m.get('cpu', '?')}), "
             f"{m.get('host_load', '')}: 4 worker processes + 3 raftstore processes over loopback TCP, "
             "not a cluster.\n" + detect, fontsize=8, color=INK2, ha="left")
    fig.tight_layout()
    save(fig, "fault_demo")


# --------------------------------------------------------------------- scaling
def fig_scaling():
    p = os.path.join(RES, "scaling.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p, encoding="utf-8"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, mode in enumerate(("weak", "strong")):
        rows = d[mode]
        n = [r["world"] for r in rows]
        sps = [r["samples_per_s_median"] for r in rows]
        a1.plot(n, sps, marker="o", ms=7, color=C[i], label=f"{mode} scaling (measured)")
        if mode == "weak":
            a1.plot(n, [sps[0] * k for k in n], color=C[i], lw=1, ls="--", label="ideal (linear)")
    a1.set_xticks([1, 2, 3, 4])
    a1.set_xlabel("worker processes on one host")
    a1.set_ylabel("training samples / s")
    a1.set_title("Throughput, 1-4 processes, same host, loopback")
    a1.legend(fontsize=8.5)
    rows = d["weak"]
    n = [r["world"] for r in rows]
    comp = [1e3 * r["compute_s_per_step"] for r in rows]
    comm = [1e3 * r["comm_wait_s_per_step"] for r in rows]
    a2.bar(n, comp, color=C[0], width=0.6, label="forward + backward")
    a2.bar(n, comm, bottom=comp, color=C[1], width=0.6, label="waiting for all-reduce (not hidden by overlap)",
           edgecolor=SURF, linewidth=2)
    a2.set_xticks([1, 2, 3, 4])
    a2.set_xlabel("worker processes (weak scaling, 128 samples each)")
    a2.set_ylabel("ms per step")
    a2.set_title("Where a step's time goes")
    a2.legend(fontsize=8.5, loc="upper left")
    m = d["machine"]
    fig.text(0.01, -0.07, f"Measured {m['date']}, {m['os']}, {m['logical_cpus']} logical CPUs, 1 torch thread per "
             f"worker; {rows[0]['params']:,}-parameter MLP; median of {d['reps']} runs.\nSingle host - NOT a cluster. "
             f"Host load: {m.get('host_load', 'unknown')}, so run-to-run noise is large.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "scaling")


def fig_allreduce():
    p = os.path.join(RES, "allreduce.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p, encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, run in enumerate(d["runs"]):
        for algo, ls in (("ring", "-"), ("naive", ":")):
            rs = [x for x in run["results"] if x["algo"] == algo]
            ax.plot([x["bytes"] for x in rs], [x["algbw_GBps"] for x in rs], ls=ls, marker="o", ms=5, color=C[i],
                    label=f"N={run['world']} {'ring' if algo == 'ring' else 'reduce-to-root + broadcast'}")
    ax.set_xscale("log")
    ax.set_xlabel("message size (bytes, float32)")
    ax.set_ylabel("algorithm bandwidth (GB/s)")
    ax.set_title("All-reduce over loopback TCP (one host)")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    fig.text(0.01, -0.1, f"Measured {d['machine']['date']}, one host, worker processes. "
             f"Host load: {d['machine'].get('host_load', 'unknown')}.", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "allreduce")


def fig_compression():
    p = os.path.join(RES, "compression.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p, encoding="utf-8"))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    import numpy as np
    for i, v in enumerate(d["variants"]):
        L = np.array(v["losses"])
        k = 10
        sm = np.convolve(L, np.ones(k) / k, mode="valid")
        x = np.arange(k, len(L) + 1)
        name = {"none": "exact ring all-reduce"}.get(v["compression"], v["compression"])
        a1.plot(x, sm, color=C[i], lw=1.6, label=name)
        a2.plot(x * v["bytes_per_step"] / 1e6, sm, color=C[i], lw=1.6, label=name)
    for a in (a1, a2):
        a.set_yscale("log")
        a.set_ylabel("loss (10-step mean)")
    a1.set_xlabel("step")
    a2.set_xscale("log")
    a2.set_xlabel("MB sent per worker (cumulative, log)")
    a1.set_title(f"Convergence, {d['workers']} workers")
    a2.set_title("Same runs against traffic")
    a1.legend(fontsize=8.5)
    fig.text(0.01, -0.03, f"Measured {d['machine']['date']} on one host (loopback). Top-k keeps the given fraction "
             "of entries with error feedback; PowerSGD:r is rank-r with error feedback.", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "compression")


def fig_pipeline():
    p = os.path.join(RES, "pipeline.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p, encoding="utf-8"))
    fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
    tmax = 0
    for ax, sched in zip(axes, ("gpipe", "1f1b")):
        stages = d["runs"][sched]["stages"]
        # align stage clocks on each stage's own start (each records relative to its own step start,
        # all stages start the step right after a barrier)
        for r in stages:
            for ev in r["timeline"]:
                col = C[0] if ev["kind"] == "F" else C[1]
                ax.broken_barh([(ev["t0"] * 1e3, (ev["t1"] - ev["t0"]) * 1e3)], (r["rank"] - 0.38, 0.76),
                               facecolor=col, edgecolor=SURF, linewidth=1)
                tmax = max(tmax, ev["t1"] * 1e3)
        peaks = [r["max_inflight"] for r in stages]
        step = sorted(stages[0]["step_s"][2:])
        ax.set_yticks(range(4), [f"stage {i}" for i in range(4)])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.set_title(f"{'GPipe' if sched == 'gpipe' else '1F1B'}: median step {1e3 * step[len(step) // 2]:.0f} ms, "
                     f"peak stored micro-batches per stage {peaks}", loc="left", fontsize=10)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=C[0], label="forward compute"), Patch(color=C[1], label="backward compute")],
                   loc="lower right", bbox_to_anchor=(1.0, 1.08), fontsize=8.5, ncol=2)
    axes[1].set_xlabel("ms since the start of the (last) step, per stage clock; gaps = waiting for a neighbour (bubble)")
    cfg = d["config"]
    fig.text(0.01, -0.07, f"Measured {d['machine']['date']}: char-level transformer ({cfg['layers']} blocks, d={cfg['d']}) "
             f"split over 4 worker processes on one host, {cfg['micro']} micro-batches.\n"
             f"Host load: {d['machine'].get('host_load', 'unknown')}; individual blocks vary with contention.",
             fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "pipeline")


def fig_dog2real():
    p = os.path.join(RES, "dog2real", "train_history.json")
    if not os.path.exists(p):
        return
    d = json.load(open(p, encoding="utf-8"))
    h = d["history"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot([x["iteration"] for x in h], [x["train/mean_step_reward"] for x in h], color=C[0])
    ax.set_xlabel("PPO iteration")
    ax.set_ylabel("mean reward per env step")
    env_steps = sum(x["perf/env_steps"] for x in h)
    ax.set_title(f"dog2real walking task, {d['workers']} data-parallel workers ({env_steps:,} env steps)")
    fig.text(0.01, -0.04, f"Measured {d['machine']['date']} in MuJoCo simulation on one host; "
             f"{d['num_envs_per_worker']} envs per worker; wall time {d['wall_s'] / 60:.1f} min. "
             "Short run: shows integration, not a finished walking policy.", fontsize=8, color=INK2)
    fig.tight_layout()
    save(fig, "dog2real")


if __name__ == "__main__":
    fig_fault()
    fig_scaling()
    fig_allreduce()
    fig_compression()
    fig_pipeline()
    fig_dog2real()
