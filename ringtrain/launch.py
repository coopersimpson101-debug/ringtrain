"""Launch helpers: form a static group in threads (tests) or in separate processes (benchmarks).

    python -m ringtrain.launch --nproc 4 --entry ringtrain.bench:allreduce_bench --args '{"sizes":[1024]}'

Each process joins an in-launcher coordinator, receives its rank, builds a ProcessGroup and calls
``entry(pg, args)``; the returned JSON-serialisable value is written to a result file and
collected by the launcher.  This is the non-elastic path; see :mod:`ringtrain.elastic` for the
fault-tolerant trainer.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from .collectives import ProcessGroup
from .rendezvous import Coordinator, RendezvousClient
from .transport import Endpoint, Transport


def join_static(coord_addr: str, worker_id: str, host: str = "127.0.0.1", **tkw):
    """Join a coordinator once and return ``(pg, endpoint, client)``."""
    ep = Endpoint(host)
    holder: dict = {}

    def on_abort(gen, reason):
        t = holder.get("t")
        if t is not None and t.gen == gen:
            t.abort(reason)
    rdv = RendezvousClient(coord_addr, worker_id, ep.addr, on_abort=on_abort)
    a = rdv.wait_assignment(-1, timeout=120)
    t = Transport(ep, a.gen, int(a["rank"]), list(a["peers"]), **tkw)
    holder["t"] = t
    return ProcessGroup(t), ep, rdv


def run_threads(n: int, fn, timeout: float = 900.0, **tkw) -> list:
    """Run ``fn(pg)`` on ``n`` ranks, each in its own thread of this process, over real loopback
    TCP.  Returns the results ordered by rank; re-raises the first exception."""
    coord = Coordinator(min_workers=n, max_workers=n, settle_s=0.0).start()
    results: dict[int, object] = {}
    errors: list[BaseException] = []

    def worker(i):
        pg = ep = rdv = None
        try:
            pg, ep, rdv = join_static(coord.addr, f"w{i}", **tkw)
            results[pg.rank] = fn(pg)
        except BaseException as e:  # noqa: BLE001
            errors.append(e)
        finally:
            if pg is not None:
                try:
                    if not errors:
                        pg.barrier(channel=7)  # nobody closes while a peer still needs it
                        rdv.done(pg.t.gen)  # so our disconnect is not treated as a failure
                except BaseException:  # noqa: BLE001
                    pass
                pg.close()
            if rdv is not None:
                rdv.close()
            if ep is not None:
                ep.close()
    ths = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(timeout)
    coord.stop()
    if errors:
        raise errors[0]
    if len(results) != n:
        raise TimeoutError(f"only {len(results)}/{n} ranks finished")
    return [results[r] for r in range(n)]


def _resolve(entry: str):
    mod, _, fn = entry.partition(":")
    return getattr(importlib.import_module(mod), fn)


def launch_processes(n: int, entry: str, args: dict | None = None, timeout: float = 600.0,
                     env: dict | None = None, threads_per_proc: int = 1) -> list:
    """Spawn ``n`` worker processes running ``entry(pg, args)`` and return their results by rank."""
    coord = Coordinator(min_workers=n, max_workers=n, settle_s=0.0).start()
    tmp = tempfile.mkdtemp(prefix="ringtrain-launch-")
    run_id = uuid.uuid4().hex[:8]
    e = dict(os.environ if env is None else env)
    e.setdefault("OMP_NUM_THREADS", str(threads_per_proc))
    e["RINGTRAIN_THREADS"] = str(threads_per_proc)
    procs = []
    for i in range(n):
        out = os.path.join(tmp, f"{run_id}-{i}.json")
        procs.append((out, subprocess.Popen(
            [sys.executable, "-m", "ringtrain.launch", "--worker", "--coord", coord.addr, "--id", f"p{i}",
             "--entry", entry, "--args", json.dumps(args or {}), "--out", out], env=e)))
    end = time.time() + timeout
    try:
        for _, p in procs:
            p.wait(max(1.0, end - time.time()))
    finally:
        for _, p in procs:
            if p.poll() is None:
                p.kill()
        coord.stop()
    res = {}
    for out, p in procs:
        if p.returncode != 0:
            raise RuntimeError(f"worker exited with code {p.returncode}")
        with open(out, encoding="utf-8") as f:
            d = json.load(f)
        res[d["rank"]] = d["result"]
    return [res[r] for r in range(n)]


def _worker_main(a) -> int:
    if os.environ.get("RINGTRAIN_DUMP_AFTER"):
        import faulthandler
        faulthandler.dump_traceback_later(float(os.environ["RINGTRAIN_DUMP_AFTER"]), exit=False)
    nthreads = int(os.environ.get("RINGTRAIN_THREADS", "0"))
    entry = _resolve(a.entry)
    if nthreads and "torch" in sys.modules:  # only pay for importing torch if the entry uses it
        sys.modules["torch"].set_num_threads(nthreads)
    pg, ep, rdv = join_static(a.coord, a.id)
    try:
        result = entry(pg, json.loads(a.args))
        pg.barrier(channel=7)
        rdv.done(pg.t.gen)
    finally:
        pg.close()
        rdv.close()
        ep.close()
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"rank": pg.rank, "result": result}, f)
    # The result is on disk and every socket is closed.  Skip interpreter teardown: on a heavily
    # loaded Windows host, workers occasionally died with a native error (0xC000070A) and this
    # rules out teardown as the place it can happen.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="launch a static ringtrain group on this machine")
    ap.add_argument("--nproc", type=int, default=2)
    ap.add_argument("--entry", required=True, help="module:function taking (pg, args)")
    ap.add_argument("--args", default="{}")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--coord", help=argparse.SUPPRESS)
    ap.add_argument("--id", help=argparse.SUPPRESS)
    ap.add_argument("--out", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.worker:
        return _worker_main(a)
    res = launch_processes(a.nproc, a.entry, json.loads(a.args))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
