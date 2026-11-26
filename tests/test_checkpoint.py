"""Checkpoint stores: local directory and a real 3-process raftstore cluster."""
import json

import pytest
import torch

from ringtrain.checkpoint import (CheckpointStore, DirBackend, RaftstoreBackend, StoreError, dumps_state,
                                  loads_state, make_backend)
from ringtrain.elastic import _rng_state, _set_rng_state


def _state(step):
    torch.manual_seed(step)
    model = torch.nn.Linear(8, 4)
    opt = torch.optim.Adam(model.parameters())
    model(torch.randn(2, 8)).sum().backward()
    opt.step()
    return {"model": model.state_dict(), "optim": opt.state_dict(), "rng": _rng_state(), "step": step,
            "data_cursor": {"step": step, "samples_consumed": step * 64, "seed": 0}}


def test_state_roundtrip_with_weights_only_loader():
    st = _state(3)
    back = loads_state(dumps_state(st))
    assert back["step"] == 3
    for k, v in st["model"].items():
        assert torch.equal(v, back["model"][k])
    assert back["optim"]["state"][0]["step"] == st["optim"]["state"][0]["step"]


def test_rng_state_restores_exactly():
    torch.manual_seed(1)
    import random

    import numpy as np
    np.random.seed(5)
    random.seed(7)
    st = loads_state(dumps_state({"rng": _rng_state()}))["rng"]
    a = (torch.rand(3), np.random.rand(3), random.random())
    _set_rng_state(st)
    b = (torch.rand(3), np.random.rand(3), random.random())
    assert torch.equal(a[0], b[0]) and (a[1] == b[1]).all() and a[2] == b[2]


def test_dir_store_latest_and_pruning(tmp_path):
    cs = CheckpointStore(DirBackend(str(tmp_path)), "run1", keep=2)
    assert cs.load_latest() is None
    for s in (10, 20, 30):
        cs.save(s, _state(s), {"world": 4})
    ptr, st = cs.load_latest()
    assert ptr["step"] == 30 and st["step"] == 30 and ptr["world"] == 4
    assert cs.backend.get("ringtrain/run1/step-000000010") is None  # pruned (keep=2)
    assert cs.backend.get("ringtrain/run1/step-000000020") is not None


def test_async_save_then_wait(tmp_path):
    cs = CheckpointStore(DirBackend(str(tmp_path)), "r")
    st = _state(5)
    cs.save_async(5, st)
    st["model"]["weight"].zero_()  # mutation after save_async must not leak into the checkpoint
    cs.wait()
    _, back = cs.load_latest()
    assert not torch.equal(back["model"]["weight"], st["model"]["weight"])


def test_corrupt_blob_is_detected(tmp_path):
    cs = CheckpointStore(DirBackend(str(tmp_path)), "r")
    cs.save(1, _state(1))
    key = cs.latest_pointer()["key"]
    blob = bytearray(cs.backend.get(key))
    blob[100] ^= 0xFF
    cs.backend.put(key, bytes(blob))
    with pytest.raises(StoreError, match="checksum"):
        cs.load_latest()


def test_make_backend_specs(tmp_path):
    assert isinstance(make_backend(f"dir:{tmp_path}"), DirBackend)
    rb = make_backend("raftstore:http://10.0.0.10:8080,10.0.0.11:8080")
    assert rb.urls == ["http://10.0.0.10:8080", "http://10.0.0.11:8080"]
    with pytest.raises(ValueError):
        make_backend("s3:bucket")


@pytest.mark.cluster
def test_raftstore_store_survives_leader_kill(raftstore_cluster):
    dc = raftstore_cluster
    cs = CheckpointStore(RaftstoreBackend(dc.urls()), "rs-test", keep=2)
    st = _state(7)
    cs.save(7, st, {"world": 3})
    ptr, back = cs.load_latest()
    assert ptr["step"] == 7 and torch.equal(back["model"]["weight"], st["model"]["weight"])
    leader = dc.wait_leader(20)
    dc.kill(leader)  # SIGKILL / TerminateProcess the raftstore leader
    try:
        ptr, back = cs.load_latest()  # still readable from the 2-node majority
        assert ptr["step"] == 7
        cs.save(8, _state(8))  # and still writable
        assert cs.load_latest()[0]["step"] == 8
        assert json.loads(cs.backend.get("ringtrain/rs-test/LATEST"))["step"] == 8
    finally:
        dc.start(leader)
        dc.wait_converged(timeout=30)
