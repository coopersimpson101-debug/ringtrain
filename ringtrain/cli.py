"""Command line: ``python -m ringtrain {coordinator,worker,info}``."""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys

from . import __version__


def _cmd_coordinator(a) -> int:
    from .rendezvous import Coordinator
    c = Coordinator(a.host, a.port, a.min_workers, a.max_workers, a.settle, a.hb_timeout, a.events,
                    a.initial_workers, a.initial_wait)
    print(f"ringtrain coordinator listening on {c.addr} (min={a.min_workers} max={a.max_workers})", flush=True)
    if a.port_file:
        with open(a.port_file, "w", encoding="utf-8") as f:
            f.write(str(c.port))
    c.serve_forever()
    return 0


def _cmd_worker(a) -> int:
    import torch

    from .adapters import make_adapter
    from .checkpoint import CheckpointStore, make_backend
    from .elastic import ElasticTrainer
    if a.threads:
        torch.set_num_threads(a.threads)
    adapter = make_adapter(a.adapter, **json.loads(a.adapter_args))
    store = CheckpointStore(make_backend(a.store), a.run, keep=a.keep)
    wid = a.id or f"{socket.gethostname()}-{os.getpid()}"
    tr = ElasticTrainer(adapter, store, a.coordinator, wid, a.steps, a.ckpt_every, a.seed, a.compression,
                        a.bucket_mb, a.log, a.host, a.port, a.advertise, a.hb_timeout, a.min_step_time)
    summary = tr.run()
    print(json.dumps({"worker": wid, **summary}), flush=True)
    return 0


def _cmd_info(a) -> int:
    print(json.dumps({"ringtrain": __version__, "python": sys.version.split()[0]}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ringtrain", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("coordinator", help="run the rendezvous / membership service")
    c.add_argument("--host", default="127.0.0.1")
    c.add_argument("--port", type=int, default=29400)
    c.add_argument("--min-workers", type=int, default=1)
    c.add_argument("--max-workers", type=int, default=64)
    c.add_argument("--initial-workers", type=int, help="first group waits for this many (default: --min-workers)")
    c.add_argument("--initial-wait", type=float, default=120.0, help="...but at most this many seconds")
    c.add_argument("--settle", type=float, default=1.0, help="seconds without joins before forming a group")
    c.add_argument("--hb-timeout", type=float, default=8.0)
    c.add_argument("--events", help="append membership events (JSONL) here")
    c.add_argument("--port-file", help=argparse.SUPPRESS)
    c.set_defaults(fn=_cmd_coordinator)

    w = sub.add_parser("worker", help="run one elastic training worker")
    w.add_argument("--coordinator", required=True, help="host:port")
    w.add_argument("--id", help="stable worker id (default hostname-pid)")
    w.add_argument("--adapter", default="mlp", help="mlp | charlm | tinylm | module:factory")
    w.add_argument("--adapter-args", default="{}", help="JSON kwargs for the adapter")
    w.add_argument("--store", required=True, help="raftstore:URL[,URL...] or dir:PATH")
    w.add_argument("--run", default="default", help="run name (checkpoint key prefix)")
    w.add_argument("--steps", type=int, default=1000)
    w.add_argument("--ckpt-every", type=int, default=25)
    w.add_argument("--keep", type=int, default=3)
    w.add_argument("--seed", type=int, default=0)
    w.add_argument("--compression", default="none", help="none | topk:RATIO | powersgd:RANK")
    w.add_argument("--bucket-mb", type=float, default=1.0)
    w.add_argument("--threads", type=int, default=0, help="torch threads (0 = default)")
    w.add_argument("--host", default="127.0.0.1", help="interface to listen on for peers")
    w.add_argument("--port", type=int, default=0)
    w.add_argument("--advertise", help="address peers should dial (default: --host)")
    w.add_argument("--hb-timeout", type=float, default=8.0)
    w.add_argument("--log", help="append per-step JSONL events here")
    w.add_argument("--min-step-time", type=float, default=0.0,
                   help="pad every step to at least this many seconds (emulates slower machines in demos/tests)")
    w.set_defaults(fn=_cmd_worker)

    i = sub.add_parser("info")
    i.set_defaults(fn=_cmd_info)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(message)s")
    return a.fn(a)
