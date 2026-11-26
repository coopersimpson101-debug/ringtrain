"""Framing, rendezvous/membership, and failure detection."""
import socket
import threading
import time

import numpy as np
import pytest

from ringtrain import wire
from ringtrain.data import global_batch_indices, shard
from ringtrain.launch import join_static
from ringtrain.rendezvous import Coordinator, RendezvousClient
from ringtrain.transport import Endpoint, GenerationAborted, PeerFailure


# ------------------------------------------------------------------- framing
def test_header_roundtrip_and_validation():
    h = wire.pack_header(wire.DATA, 3, 0xDEADBEEF12, 12345)
    assert len(h) == wire.HEADER_SIZE == 24
    assert wire.unpack_header(h) == (wire.DATA, 3, 0xDEADBEEF12, 12345)
    with pytest.raises(wire.FrameError):
        wire.unpack_header(b"XXXX" + h[4:])
    with pytest.raises(wire.FrameError):
        wire.unpack_header(wire.HEADER.pack(wire.MAGIC, 99, 0, 0, 0, 0))
    with pytest.raises(wire.FrameError):
        wire.unpack_header(wire.HEADER.pack(wire.MAGIC, wire.DATA, 0, 0, 0, wire.MAX_FRAME + 1))


def test_frames_over_a_socket_pair():
    a, b = socket.socketpair()
    wire.write_frame(a, wire.DATA, 1, 7, np.arange(5, dtype=np.float32))
    wire.send_json(a, {"op": "HB"})
    ftype, src, tag, payload = wire.read_frame(b)
    assert (ftype, src, tag) == (wire.DATA, 1, 7)
    assert np.array_equal(np.frombuffer(payload, dtype=np.float32), np.arange(5))
    assert wire.recv_json(b) == (wire.CTRL, {"op": "HB"})
    a.close()
    with pytest.raises(ConnectionError):
        wire.read_frame(b)


# ---------------------------------------------------------------- data order
def test_global_batches_do_not_depend_on_world_size():
    n, B = 1000, 96
    for step in (0, 5, 10, 11):  # step 10/11 cross an epoch boundary
        g = global_batch_indices(n, B, step, seed=3)
        for world in (1, 2, 3, 4):
            assert np.array_equal(np.concatenate([shard(g, r, world) for r in range(world)]), g)
    epoch0 = np.concatenate([global_batch_indices(n, 100, s, 3) for s in range(10)])
    assert sorted(epoch0.tolist()) == list(range(n))  # an epoch is a permutation


# ---------------------------------------------------------------- rendezvous
def _clients(coord, n, prefix="w"):
    eps = [Endpoint() for _ in range(n)]
    cls = [RendezvousClient(coord.addr, f"{prefix}{i}", eps[i].addr) for i in range(n)]
    return eps, cls


def test_coordinator_forms_group_and_keeps_rank_order_after_failure():
    coord = Coordinator(min_workers=2, max_workers=4, settle_s=0.2, hb_timeout=2.0).start()
    eps, cls = _clients(coord, 3)
    try:
        assigns = [c.wait_assignment(-1, timeout=10) for c in cls]
        assert {a["world"] for a in assigns} == {3} and len({a.gen for a in assigns}) == 1
        order = assigns[0]["ids"]
        victim = order[1]
        vi = int(victim[1:])
        cls[vi].close()  # the connection drops, as when the process is killed
        survivors = [c for i, c in enumerate(cls) if i != vi]
        new = [c.wait_assignment(assigns[0].gen, timeout=10) for c in survivors]
        assert new[0]["world"] == 2 and new[0].gen == assigns[0].gen + 1
        assert new[0]["ids"] == [i for i in order if i != victim]  # relative order preserved
        assert any(e["event"] == "abort" for e in coord.history)
    finally:
        for c in cls:
            c.close()
        for e in eps:
            e.close()
        coord.stop()


def test_late_joiner_raises_pending_flag_and_reform_scales_up():
    coord = Coordinator(min_workers=2, max_workers=4, settle_s=0.2).start()
    eps, cls = _clients(coord, 2)
    try:
        a = [c.wait_assignment(-1, timeout=10) for c in cls]
        ep3 = Endpoint()
        c3 = RendezvousClient(coord.addr, "late", ep3.addr)
        end = time.time() + 5
        while not cls[0].pending and time.time() < end:
            time.sleep(0.05)
        assert cls[0].pending == 1
        cls[0].request_reform(a[0].gen)
        b = [c.wait_assignment(a[0].gen, timeout=10) for c in cls + [c3]]
        assert {x["world"] for x in b} == {3}
        assert b[2]["ids"][-1] == "late"  # newcomers go last
        abort = [e for e in coord.history if e["event"] == "abort"][-1]
        assert abort["failure"] is False
        c3.close()
        ep3.close()
    finally:
        for c in cls:
            c.close()
        for e in eps:
            e.close()
        coord.stop()


def test_join_after_finish_is_told_finished():
    coord = Coordinator(min_workers=1, max_workers=1, settle_s=0.0).start()
    ep = Endpoint()
    c = RendezvousClient(coord.addr, "a", ep.addr)
    a = c.wait_assignment(-1, timeout=10)
    c.done(a.gen)
    time.sleep(0.2)
    ep2 = Endpoint()
    c2 = RendezvousClient(coord.addr, "b", ep2.addr)
    assert c2.wait_assignment(-1, timeout=10) is None
    for x in (c, c2):
        x.close()
    for e in (ep, ep2):
        e.close()
    coord.stop()


# ---------------------------------------------------------- failure detection
def _group(n, **kw):
    coord = Coordinator(min_workers=n, max_workers=n, settle_s=0.0).start()
    out = [None] * n

    def j(i):
        out[i] = join_static(coord.addr, f"w{i}", **kw)
    ths = [threading.Thread(target=j, args=(i,)) for i in range(n)]
    for t in ths:
        t.start()
    for t in ths:
        t.join(30)
    return coord, sorted(out, key=lambda x: x[0].rank)


def test_abrupt_disconnect_raises_peer_failure_quickly():
    coord, grp = _group(3)
    pgs = [g[0] for g in grp]
    victim = pgs[1]
    for p in victim.t.peers.values():  # drop the connections without a BYE, like a killed process
        p.sock.shutdown(socket.SHUT_RDWR)
        p.sock.close()
    t0 = time.monotonic()
    with pytest.raises(PeerFailure) as ei:
        pgs[0].t.recv(1, tag=12345, timeout=10)
    assert time.monotonic() - t0 < 2.0
    assert ei.value.peer == 1
    for pg, ep, rdv in grp:
        rdv.close()
        ep.close()
    coord.stop()


def test_silent_peer_detected_by_heartbeat_timeout():
    coord, grp = _group(2, hb_interval=0.1, hb_timeout=1.0)
    pgs = [g[0] for g in grp]
    pgs[1].t._closed = True  # stop its heartbeats but keep the socket open (a hung / partitioned node)
    t0 = time.monotonic()
    with pytest.raises(PeerFailure, match="no heartbeat"):
        pgs[0].t.recv(1, tag=1, timeout=10)
    assert 0.9 < time.monotonic() - t0 < 4.0
    for pg, ep, rdv in grp:
        rdv.close()
        ep.close()
    coord.stop()


def test_abort_wakes_a_blocked_receiver():
    coord, grp = _group(2)
    pg0 = grp[0][0]
    threading.Timer(0.3, pg0.t.abort, args=("coordinator said so",)).start()
    with pytest.raises(GenerationAborted):
        pg0.t.recv(1, tag=5, timeout=10)
    for pg, ep, rdv in grp:
        pg.close()
        rdv.close()
        ep.close()
    coord.stop()
