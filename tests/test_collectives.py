"""Collectives vs naive references, on real loopback TCP (threads and processes)."""
import time

import numpy as np
import pytest

from ringtrain.collectives import chunk_bounds, naive_allreduce
from ringtrain.launch import launch_processes, run_threads


def _data(rank, n, dtype, integer=True):
    rng = np.random.default_rng(1000 + rank)
    if integer:
        return rng.integers(-(1 << 12), 1 << 12, size=n).astype(dtype)
    return rng.standard_normal(n).astype(dtype)


def test_chunk_bounds_cover_exactly():
    for n in range(0, 40):
        for p in range(1, 7):
            b = chunk_bounds(n, p)
            assert b[0][0] == 0 and b[-1][1] == n and len(b) == p
            assert all(b[i][1] == b[i + 1][0] for i in range(p - 1))
            sizes = [hi - lo for lo, hi in b]
            assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize("world", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("dtype", [np.float32, np.int64])
@pytest.mark.parametrize("n", [0, 1, 3, 1000, 65537])
def test_ring_allreduce_exactly_equals_naive_sum(world, dtype, n):
    """Integer-valued data: every partial sum is exactly representable, so the ring result must be
    bit-identical to the naive sum, whatever the summation order."""
    def fn(pg):
        a = _data(pg.rank, n, dtype)
        pg.allreduce(a)
        return a
    out = run_threads(world, fn)
    expect = sum(_data(r, n, dtype) for r in range(world)) if n else np.zeros(0, dtype)
    for a in out:
        assert a.dtype == dtype
        assert np.array_equal(a, expect)


@pytest.mark.parametrize("world", [2, 3, 4])
def test_ring_allreduce_floats_identical_on_every_rank(world):
    n = 10_001

    def fn(pg):
        a = _data(pg.rank, n, np.float64, integer=False)
        pg.allreduce(a)
        return a
    out = run_threads(world, fn)
    expect = sum(_data(r, n, np.float64, integer=False) for r in range(world))
    for a in out:
        assert np.array_equal(a, out[0])  # all-gather copies: bitwise agreement across ranks
        np.testing.assert_allclose(a, expect, rtol=1e-12, atol=1e-12)


def test_segmentation_does_not_change_the_result():
    n = 50_000

    def fn(pg):
        pg.segment_bytes = 4096  # force many segments per chunk (and >256 per chunk: tag wrap)
        a = _data(pg.rank, n, np.float32)
        pg.allreduce(a)
        b = _data(pg.rank, n, np.float32)
        pg.segment_bytes = 1 << 30
        pg.allreduce(b)
        return a, b
    for a, b in run_threads(3, fn):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("op,ref", [("max", np.maximum), ("min", np.minimum)])
def test_allreduce_max_min(op, ref):
    def fn(pg):
        a = _data(pg.rank, 777, np.float32)
        pg.allreduce(a, op=op)
        return a
    out = run_threads(4, fn)
    expect = _data(0, 777, np.float32)
    for r in range(1, 4):
        expect = ref(expect, _data(r, 777, np.float32))
    for a in out:
        assert np.array_equal(a, expect)


def test_reduce_scatter_ownership():
    world, n = 4, 103

    def fn(pg):
        a = _data(pg.rank, n, np.int64)
        lo, hi = pg.reduce_scatter_(a)
        return lo, hi, a[lo:hi].copy()
    expect = sum(_data(r, n, np.int64) for r in range(world))
    owned = []
    for r, (lo, hi, seg) in enumerate(run_threads(world, fn)):
        assert (lo, hi) == chunk_bounds(n, world)[(r + 1) % world]
        assert np.array_equal(seg, expect[lo:hi])
        owned.append((lo, hi))
    assert sorted(owned) == chunk_bounds(n, world)


def test_naive_allreduce_matches():
    def fn(pg):
        a = _data(pg.rank, 5000, np.float32)
        naive_allreduce(pg, a)
        return a
    expect = sum(_data(r, 5000, np.float32) for r in range(3))
    for a in run_threads(3, fn):
        assert np.array_equal(a, expect)


@pytest.mark.parametrize("size", [0, 1, 5000, 3 * (1 << 20) + 17])
@pytest.mark.parametrize("root", [0, 2])
def test_broadcast_bytes(size, root):
    payload = np.random.default_rng(size).integers(0, 256, size, dtype=np.uint8).tobytes()

    def fn(pg):
        return pg.broadcast_bytes(payload if pg.rank == root else None, root=root)
    for got in run_threads(4, fn):
        assert got == payload


def test_broadcast_array_in_place():
    def fn(pg):
        a = np.arange(10, dtype=np.float32) * (pg.rank + 1)
        pg.broadcast(a, root=1)
        return a
    for a in run_threads(3, fn):
        assert np.array_equal(a, np.arange(10, dtype=np.float32) * 2)


def test_barrier_holds_everyone_until_the_last_arrives():
    def fn(pg):
        time.sleep(0.3 if pg.rank == 2 else 0.0)
        t_arrive = time.monotonic()
        pg.barrier()
        return t_arrive, time.monotonic()
    out = run_threads(5, fn)
    last_arrival = max(a for a, _ in out)
    assert all(leave >= last_arrival for _, leave in out)


def test_all_gather_variable_sizes():
    def fn(pg):
        return pg.all_gather_bytes(bytes([pg.rank]) * (pg.rank * 1000 + 1))
    for got in run_threads(4, fn):
        assert got == [bytes([r]) * (r * 1000 + 1) for r in range(4)]


def test_all_gather_json_and_array():
    def fn(pg):
        return pg.all_gather_json({"r": pg.rank}), [a.tolist() for a in pg.all_gather_array(np.array([pg.rank, 7]))]
    for js, arrs in run_threads(3, fn):
        assert js == [{"r": 0}, {"r": 1}, {"r": 2}]
        assert arrs == [[0, 7], [1, 7], [2, 7]]


def test_point_to_point_arrays_keep_shape_and_dtype():
    def fn(pg):
        if pg.rank == 0:
            pg.send_array(1, np.arange(24, dtype=np.float64).reshape(2, 3, 4), tag=99).wait()
            return None
        return pg.recv_array(0, tag=99)
    _, got = run_threads(2, fn)
    assert got.shape == (2, 3, 4) and got.dtype == np.float64
    assert np.array_equal(got, np.arange(24, dtype=np.float64).reshape(2, 3, 4))


def test_two_channels_interleave_without_confusion():
    """DDP's comm thread (channel 1) and the training thread (channel 0) share connections."""
    import threading

    def fn(pg):
        res = {}

        def side():
            b = np.full(1000, pg.rank + 1, dtype=np.int64)
            for _ in range(20):
                pg.allreduce(b, channel=1)
            res["b"] = b
        t = threading.Thread(target=side)
        t.start()
        a = np.full(999, pg.rank, dtype=np.int64)
        for _ in range(20):
            pg.allreduce(a, channel=0)
        t.join()
        return a, res["b"]
    for a, b in run_threads(3, fn):
        assert np.all(a == 3 ** 20) and np.all(b == 6 * 3 ** 19)  # (0+1+2)*3^19, (1+2+3)*3^19


@pytest.mark.slow
def test_multiprocess_allreduce_exact():
    """Separate OS processes (spawned interpreters), not threads."""
    out = launch_processes(3, "ringtrain.selftest:allreduce_check", {"n": 100_003}, timeout=180)
    assert all(o["ok"] for o in out), out
    assert len({o["digest"] for o in out}) == 1
