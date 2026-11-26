#!/usr/bin/env python3
"""Train a dog2real walking policy with data-parallel PPO over ringtrain.

    git clone https://github.com/coopersimpson101-debug/dog2real ../dog2real
    pip install "dog2real @ git+https://github.com/coopersimpson101-debug/dog2real" mujoco pyyaml gymnasium
    python scripts/train_dog2real.py --dog2real-root ../dog2real --workers 4 --iterations 60

Each worker process runs its own MuJoCo environments (dog2real's VecQuadrupedEnv, unchanged);
gradients, normaliser statistics and the adaptive learning rate are synchronised over ringtrain.
The final checkpoint goes to the checkpoint store (a local dir by default, raftstore with
``--store raftstore:URLS``) and to ``--out`` in dog2real's own checkpoint format, so
``python scripts/play.py --checkpoint <out>`` in dog2real can load it.
"""
from __future__ import annotations

import argparse
import os
import tempfile
import time

from _common import CRASHES, RESULTS, launch_retry, machine, save



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dog2real-root", required=True, help="a dog2real checkout (for configs/)")
    ap.add_argument("--config", default="train_walk.yaml")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--iterations", type=int, default=60)
    ap.add_argument("--num-envs", type=int, default=16, help="environments per worker")
    ap.add_argument("--compression", default="none")
    ap.add_argument("--store", default=None, help="dir:PATH or raftstore:URLS (default: a temp dir)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "dog2real", "policy_ringtrain.pt"))
    ap.add_argument("--small", action="store_true", help="[64,64] networks (faster)")
    a = ap.parse_args()
    store = a.store or "dir:" + tempfile.mkdtemp(prefix="ringtrain-d2r-")
    args = {"dog2real_root": os.path.abspath(a.dog2real_root), "config": a.config, "iterations": a.iterations,
            "num_envs": a.num_envs, "compression": a.compression, "store": store, "run": "dog2real-walk",
            "out": os.path.abspath(a.out)}
    if a.small:
        args["overrides"] = {"policy": {"actor_hidden": [64, 64], "critic_hidden": [64, 64]}}
    t0 = time.time()
    res = launch_retry(a.workers, "ringtrain.integrations.dog2real_ppo:run_entry", args, timeout=24 * 3600)
    digests = res[0]["param_digests"]
    out = {"machine": machine(), "workers": a.workers, "num_envs_per_worker": a.num_envs,
           "iterations": a.iterations, "config": a.config, "compression": a.compression,
           "small_networks": a.small, "history": res[0]["history"], "param_digests": digests,
           "ranks_identical": len(set(digests)) == 1, "wall_s": time.time() - t0}
    out["crashed_attempts"] = CRASHES
    save(os.path.join("dog2real", "train_history.json"), out)
    h = res[0]["history"]
    print(f"done: {len(h)} iterations, {sum(x['perf/env_steps'] for x in h)} env steps in {out['wall_s']:.0f}s; "
          f"mean step reward {h[0]['train/mean_step_reward']:+.4f} -> {h[-1]['train/mean_step_reward']:+.4f}; "
          f"policies identical on all ranks: {out['ranks_identical']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
