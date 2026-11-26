#!/usr/bin/env python3
"""Fault-injection demo: kill a training worker mid-run and watch the run carry on.

    python scripts/fault_demo.py                      # default scenario, ~2 minutes
    python scripts/fault_demo.py --no-reference       # skip the uninterrupted baseline run

Everything is real and on this machine: a 3-process raftstore cluster (the checkpoint store),
a ringtrain coordinator process, and 4 ringtrain worker processes talking over loopback TCP.

Timeline of the default scenario:
  1. 4 workers train the teacher-student MLP (global batch 256, checkpoint every 25 steps).
  2. at step >= --kill-step, worker w2 is killed with Popen.kill() -- SIGKILL on POSIX,
     TerminateProcess on Windows, so no cleanup code runs in it.
  3. survivors detect the dead peer, the coordinator forms generation 2 with 3 workers, which
     restore the last checkpoint from raftstore and continue.
  4. at step >= --store-kill-step the raftstore *leader* is killed too; checkpoints keep working
     through the remaining 2-of-3 majority.
  5. at step >= --rejoin-step a fresh w2 process is started; it joins, the group scales back up to
     4 at a step boundary (no rollback), and the run finishes.
An uninterrupted 4-worker run with the same seed is recorded as the reference curve.

Outputs: results/fault_demo/{run,reference}/w*.jsonl, coordinator.jsonl and summary.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from _common import machine  # noqa: E402

T0 = time.time()


def say(msg: str) -> None:
    print(f"[{time.time() - T0:7.2f}s] {msg}", flush=True)


def read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass  # a line being written right now
    return out


def max_step(logdir: str) -> int:
    best = 0
    for p in glob.glob(os.path.join(logdir, "w*.jsonl")):
        for r in read_jsonl(p)[-50:]:
            if r.get("event") == "step":
                best = max(best, r["step"])
    return best


class Scenario:
    def __init__(self, a, logdir: str, store: str, run_name: str):
        self.a, self.logdir, self.store, self.run_name = a, logdir, store, run_name
        os.makedirs(logdir, exist_ok=True)
        for p in glob.glob(os.path.join(logdir, "*.jsonl")):
            os.remove(p)
        self.env = dict(os.environ, OMP_NUM_THREADS="1",
                        PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
        pf = os.path.join(tempfile.mkdtemp(prefix="rt-coord-"), "port")
        self.coord = subprocess.Popen(
            [sys.executable, "-m", "ringtrain", "coordinator", "--port", "0", "--port-file", pf,
             "--min-workers", "1", "--max-workers", str(a.workers), "--initial-workers", str(a.workers), "--settle", "1.0",
             "--events", os.path.join(logdir, "coordinator.jsonl")], env=self.env)
        while not os.path.exists(pf) or not open(pf).read().strip():
            time.sleep(0.05)
        self.coord_addr = f"127.0.0.1:{open(pf).read().strip()}"
        self.workers: dict[str, subprocess.Popen] = {}

    def start_worker(self, wid: str) -> None:
        self.workers[wid] = subprocess.Popen(
            [sys.executable, "-m", "ringtrain", "worker", "--coordinator", self.coord_addr, "--id", wid,
             "--adapter", "mlp", "--adapter-args", self.a.adapter_args, "--store", self.store, "--run", self.run_name, "--steps", str(self.a.steps),
             "--ckpt-every", str(self.a.ckpt_every), "--threads", "1",
             "--log", os.path.join(self.logdir, f"{wid}.jsonl")], env=self.env,
            stdout=subprocess.DEVNULL)

    def kill_worker(self, wid: str) -> None:
        self.workers[wid].kill()  # SIGKILL / TerminateProcess
        self.workers[wid].wait(10)

    def wait_done(self, timeout: float) -> None:
        end = time.time() + timeout
        for wid, p in self.workers.items():
            if p.poll() is None:
                p.wait(max(1.0, end - time.time()))
        self.coord.kill()
        self.coord.wait(10)


def run_reference(a, out: str, store: str) -> None:
    say(f"reference run: {a.workers} workers, no faults, {a.steps} steps")
    sc = Scenario(a, out, store, "reference")
    for i in range(a.workers):
        sc.start_worker(f"w{i}")
    sc.wait_done(a.timeout)
    say(f"reference finished at step {max_step(out)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--ckpt-every", type=int, default=25)
    ap.add_argument("--kill-step", type=int, default=190)
    ap.add_argument("--store-kill-step", type=int, default=300)
    ap.add_argument("--rejoin-step", type=int, default=400)
    ap.add_argument("--victim", default="w2")
    ap.add_argument("--adapter-args", default='{"hidden": 768, "depth": 3, "global_batch": 512, "lr": 0.05}')
    ap.add_argument("--store", default="raftstore", choices=["raftstore", "dir"])
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "fault_demo"))
    ap.add_argument("--timeout", type=float, default=900)
    a = ap.parse_args()

    work = tempfile.mkdtemp(prefix="ringtrain-fault-")
    dc = None
    if a.store == "raftstore":
        from raftstore.devcluster import DevCluster
        dc = DevCluster(os.path.join(work, "raftstore"), names=("nas", "pi5", "server"), tick_ms=50)
        dc.start_all()
        leader = dc.wait_leader(30)
        store = "raftstore:" + ",".join(dc.urls())
        say(f"raftstore: 3 local processes (nas, pi5, server), leader '{leader}'")
    else:
        store = "dir:" + os.path.join(work, "ckpt")
    events: list[dict] = []
    mach = machine()  # sampled before our own processes start
    try:
        if not a.no_reference:
            run_reference(a, os.path.join(a.out, "reference"), store)
        logdir = os.path.join(a.out, "run")
        sc = Scenario(a, logdir, store, "faulty")
        say(f"coordinator at {sc.coord_addr}; starting {a.workers} workers")
        for i in range(a.workers):
            sc.start_worker(f"w{i}")
        phase = "kill"
        while True:
            alive = [w for w, p in sc.workers.items() if p.poll() is None]
            if not alive:
                break
            s = max_step(logdir)
            if phase == "kill" and s >= a.kill_step:
                t_kill = time.time()
                sc.kill_worker(a.victim)
                events.append({"t": t_kill, "event": "kill_worker", "id": a.victim, "seen_step": s})
                say(f"KILLED worker {a.victim} (TerminateProcess/SIGKILL) at global step ~{s}")
                phase = "store"
            elif phase == "store" and s >= a.store_kill_step:
                if dc is not None:
                    lead = dc.wait_leader(10)
                    dc.kill(lead)
                    events.append({"t": time.time(), "event": "kill_store_leader", "id": lead, "seen_step": s})
                    say(f"KILLED raftstore leader '{lead}' at step ~{s}; 2 of 3 store nodes remain")
                phase = "rejoin"
            elif phase == "rejoin" and s >= a.rejoin_step:
                sc.start_worker(a.victim)
                events.append({"t": time.time(), "event": "restart_worker", "id": a.victim, "seen_step": s})
                say(f"restarted {a.victim} as a fresh process at step ~{s}; it will re-join")
                phase = "finish"
            time.sleep(0.05)
        sc.wait_done(a.timeout)
        final = max_step(logdir)
        say(f"run finished at step {final}")
        coord = read_jsonl(os.path.join(logdir, "coordinator.jsonl"))
        for ev in coord:
            if ev["event"] in ("form", "abort"):
                say(f"  coordinator: {ev['event']} {({k: v for k, v in ev.items() if k not in ('t', 'event')})}")
        with open(os.path.join(a.out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump({"injected": events, "final_step": final, "steps": a.steps, "ckpt_every": a.ckpt_every,
                       "workers": a.workers, "store": a.store, "t_start": T0,
                       "adapter_args": json.loads(a.adapter_args), "machine": mach,
                       "date": mach["date"],
                       "host": "single host, loopback TCP (not a cluster)"}, f, indent=1)
        if final < a.steps:
            say("run did NOT finish")
            return 1
        return 0
    finally:
        if dc is not None:
            dc.stop_all()
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
