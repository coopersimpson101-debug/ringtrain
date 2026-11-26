"""End-to-end elasticity: real worker processes, one killed with SIGKILL / TerminateProcess,
re-join, and the loss trajectory compared with single-process training."""
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest
import torch

from ringtrain.adapters.mlp import TeacherMLPAdapter
from ringtrain.checkpoint import CheckpointStore, DirBackend, RaftstoreBackend
from ringtrain.data import global_batch_indices
from ringtrain.rendezvous import Coordinator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADAPTER_ARGS = {"num_samples": 4096, "hidden": 64, "depth": 2, "global_batch": 96, "lr": 0.05}
STEPS = 200


def reference_losses():
    torch.set_num_threads(1)
    ad = TeacherMLPAdapter(**ADAPTER_ARGS)
    model, out = ad.build_model(0), []
    opt = ad.build_optimizer(model)
    for s in range(STEPS):
        opt.zero_grad()
        loss, _ = ad.loss_sum(model, ad.batch(global_batch_indices(ad.num_samples, ad.global_batch, s, 0)))
        (loss / ad.global_batch).backward()
        opt.step()
        out.append(float(loss.detach()) / ad.global_batch)
    return out


def read(path):
    out = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def _wait(cond, what, timeout=300):
    end = time.time() + timeout
    while not cond():
        assert time.time() < end, f"timed out waiting for {what}"
        time.sleep(0.05)


def _paused(logdir, wid, gen_pred):
    return any(r["event"] == "paused" and gen_pred(r["gen"]) for r in read(os.path.join(logdir, f"{wid}.jsonl")))


def run_scenario(tmp_path, store_spec, kill_at=40):
    """Deterministic kill / re-join, independent of how long processes take to start:

    1. all 3 workers train and then hold at step ``kill_at`` (``--pause-file`` containing the step);
    2. w1 is killed while they hold; the survivors re-form as generation 2 and hold again;
    3. a fresh w1 is started; only once the coordinator has registered it is the pause lifted,
       so the survivors cannot finish the run before it arrives.
    """
    logdir = str(tmp_path / "logs")
    os.makedirs(logdir)
    pause = str(tmp_path / "pause")
    with open(pause, "w", encoding="utf-8") as f:
        f.write(str(kill_at))
    coord = Coordinator(min_workers=1, max_workers=3, settle_s=0.5, initial_workers=3).start()
    env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))

    def start(wid):
        return subprocess.Popen(
            [sys.executable, "-m", "ringtrain", "worker", "--coordinator", coord.addr, "--id", wid,
             "--adapter", "mlp", "--adapter-args", json.dumps(ADAPTER_ARGS), "--store", store_spec,
             "--run", "t", "--steps", str(STEPS), "--ckpt-every", "10", "--threads", "1",
             "--pause-file", pause, "--log", os.path.join(logdir, f"{wid}.jsonl")],
            env=env, stdout=subprocess.DEVNULL)

    def forms():
        return [e for e in list(coord.history) if e["event"] == "form"]
    procs = {w: start(w) for w in ("w0", "w1", "w2")}
    try:
        _wait(lambda: all(_paused(logdir, w, lambda g: g == 1) for w in ("w0", "w1", "w2")),
              f"all workers holding at step {kill_at}")
        procs["w1"].kill()  # SIGKILL on POSIX, TerminateProcess on Windows
        procs["w1"].wait(10)
        procs.pop("w1")
        _wait(lambda: any(e["world"] == 2 for e in forms()), "the survivors to re-form as 2")
        gen2 = next(e["gen"] for e in forms() if e["world"] == 2)
        _wait(lambda: all(_paused(logdir, w, lambda g: g == gen2) for w in ("w0", "w2")),
              "the survivors to restore and hold")
        t_restart = time.time()
        procs["w1b"] = start("w1")  # a fresh process with the same id
        _wait(lambda: any(e["event"] == "join" and e["id"] == "w1" and e["t"] >= t_restart
                          for e in list(coord.history)), "the restarted worker to register")
        os.remove(pause)  # let training go on; the waiting newcomer triggers a graceful re-form
        _wait(lambda: all(p.poll() is not None for p in procs.values()), "the run to finish", timeout=600)
    finally:
        for p in procs.values():
            if p.poll() is None:
                p.kill()
        coord.stop()
    return logdir, coord.history, {w: p.returncode for w, p in procs.items()}


def check(logdir, history, codes, store):
    assert all(c == 0 for c in codes.values()), codes
    forms = [e for e in history if e["event"] == "form"]
    worlds = [e["world"] for e in forms]
    assert worlds[0] == 3 and 2 in worlds, worlds  # shrank after the kill ...
    assert worlds[-1] == 3, worlds  # ... and grew back after the re-join
    aborts = [e for e in history if e["event"] == "abort"]
    assert any(e["failure"] for e in aborts) and any(not e["failure"] for e in aborts)
    ptr = store.latest_pointer()
    assert ptr["step"] == STEPS
    # loss of every step, taken from the generation that finally committed it, matches the
    # single-process run on the same global batches (only float summation order differs)
    final_loss = {}
    restores = []
    for p in glob.glob(os.path.join(logdir, "*.jsonl")):
        for r in read(p):
            if r["event"] == "step":
                final_loss[(r["gen"], r["step"])] = r["loss"]
            if r["event"] == "restored":
                restores.append(r)
    by_step = {}
    for (gen, step), loss in sorted(final_loss.items()):
        by_step[step] = loss  # later generations overwrite replayed steps
    assert sorted(by_step) == list(range(1, STEPS + 1))
    ref = reference_losses()
    got = [by_step[s] for s in range(1, STEPS + 1)]
    np.testing.assert_allclose(got, ref, rtol=2e-4, atol=2e-5)
    assert any(r["from_ckpt"] and r["step"] > 0 for r in restores)


@pytest.mark.slow
def test_kill_a_worker_and_rejoin_dir_store(tmp_path):
    store_dir = str(tmp_path / "ckpt")
    logdir, hist, codes = run_scenario(tmp_path, f"dir:{store_dir}")
    check(logdir, hist, codes, CheckpointStore(DirBackend(store_dir), "t"))


@pytest.mark.slow
@pytest.mark.cluster
def test_kill_a_worker_with_raftstore_checkpoints(tmp_path, raftstore_cluster):
    urls = raftstore_cluster.urls()
    logdir, hist, codes = run_scenario(tmp_path, "raftstore:" + ",".join(urls))
    check(logdir, hist, codes, CheckpointStore(RaftstoreBackend(urls), "t"))
