"""Rendezvous and membership: who is in the group, in which order, for which generation.

The coordinator is a small TCP service (``python -m ringtrain coordinator``).  It holds no
training state -- that lives in the checkpoint store -- so it can be restarted at any time; workers
reconnect and re-join.  Protocol (JSON control frames, see :mod:`ringtrain.wire`)::

    worker -> coord   JOIN      {id, addr}
    coord  -> worker  ASSIGN    {gen, rank, world, peers:[addr...], ids:[...]}
    worker -> coord   HB        {gen}                (every hb_interval)
    coord  -> worker  HB_ACK    {gen, pending}       pending = workers waiting to join
    coord  -> worker  ABORT     {gen, reason}        a member died / asked to re-form
    worker -> coord   FAILED    {gen, peer, reason}  a worker saw a peer fail first
    worker -> coord   REFORM    {gen}                graceful re-form at a step boundary
    worker -> coord   DONE      {gen}                training finished
    coord  -> worker  FINISHED  {}                   reply to a JOIN after the run finished

Generation life cycle: workers wait in a pool; once ``>= min_workers`` are waiting and no one has
joined for ``settle_s`` (or ``max_workers`` are waiting), the coordinator assigns ranks -- previous
members keep their relative order, newcomers go last -- and bumps the generation number.  A member
that disconnects or misses heartbeats for ``hb_timeout`` aborts the generation; survivors go back
to the pool and form the next generation without it.
"""
from __future__ import annotations

import json
import logging
import socket
import threading
import time

from . import wire
from .transport import parse_addr

log = logging.getLogger("ringtrain.rendezvous")


class _Member:
    def __init__(self, wid: str, addr: str, conn: socket.socket):
        self.id, self.addr, self.conn = wid, addr, conn
        self.lock = threading.Lock()
        self.last_hb = time.monotonic()
        self.joined_at = time.monotonic()
        self.alive = True

    def send(self, obj: dict) -> bool:
        try:
            with self.lock:
                wire.send_json(self.conn, obj)
            return True
        except OSError:
            self.alive = False
            return False


class Coordinator:
    def __init__(self, host: str = "127.0.0.1", port: int = 0, min_workers: int = 1, max_workers: int = 64,
                 settle_s: float = 1.0, hb_timeout: float = 8.0, event_log=None,
                 initial_workers: int | None = None, initial_wait_s: float = 120.0):
        self.min_workers, self.max_workers = min_workers, max_workers
        self.settle_s, self.hb_timeout = settle_s, hb_timeout
        # the very first generation may wait for a larger group (e.g. all 4 machines booting)
        self.initial_workers = initial_workers or min_workers
        self.initial_deadline = time.monotonic() + initial_wait_s
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        self.addr = f"{host if host != '0.0.0.0' else '127.0.0.1'}:{self.port}"
        self.lock = threading.RLock()
        self.waiting: dict[str, _Member] = {}
        self.active: list[_Member] = []
        self.gen = 0
        self.prev_order: list[str] = []
        self.last_change = time.monotonic()
        self.finished = False
        self.history: list[dict] = []  # membership timeline, for the demo plots
        self._event_log = event_log
        self._stop = threading.Event()

    # ------------------------------------------------------------ lifecycle
    def start(self) -> "Coordinator":
        threading.Thread(target=self._accept_loop, name="rdv-accept", daemon=True).start()
        threading.Thread(target=self._tick_loop, name="rdv-tick", daemon=True).start()
        return self

    def serve_forever(self) -> None:
        self.start()
        while not self._stop.is_set():
            time.sleep(0.2)

    def stop(self) -> None:
        self._stop.set()
        try:
            self.sock.close()
        except OSError:
            pass

    def _event(self, kind: str, **kw) -> None:
        ev = {"t": time.time(), "event": kind, **kw}
        self.history.append(ev)
        log.info("%s %s", kind, kw)
        if self._event_log:
            with open(self._event_log, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev) + "\n")

    # -------------------------------------------------------------- network
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            wire.tune_socket(conn)
            threading.Thread(target=self._serve_conn, args=(conn,), daemon=True).start()

    def _serve_conn(self, conn: socket.socket) -> None:
        member: _Member | None = None
        try:
            while True:
                _, msg = wire.recv_json(conn)
                op = msg.get("op")
                if op == "JOIN":
                    member = self._on_join(msg, conn)
                elif member is None:
                    continue
                elif op == "HB":
                    member.last_hb = time.monotonic()
                    with self.lock:
                        pending = len(self.waiting) if member in self.active else 0
                        gen = self.gen if member in self.active else -1
                    member.send({"op": "HB_ACK", "gen": gen, "pending": pending})
                elif op == "FAILED":
                    self._abort(int(msg.get("gen", -1)), f"{member.id} reports: {msg.get('reason', '?')}")
                elif op == "REFORM":
                    self._abort(int(msg.get("gen", -1)), f"{member.id} requested re-form (scale-up)",
                                failure=False)
                elif op == "DONE":
                    with self.lock:
                        if not self.finished:
                            self.finished = True
                            self._event("finished", gen=int(msg.get("gen", -1)))
        except (OSError, ValueError, wire.FrameError):
            pass
        finally:
            if member is not None:
                self._on_disconnect(member)
            conn.close()

    def _on_join(self, msg: dict, conn: socket.socket) -> _Member:
        m = _Member(str(msg["id"]), str(msg["addr"]), conn)
        with self.lock:
            if self.finished:
                m.send({"op": "FINISHED"})
                return m
            # a re-joining worker with the same id replaces its stale entry
            stale = self.waiting.pop(m.id, None)
            if stale is not None:
                stale.alive = False
            if any(a.id == m.id for a in self.active):
                self._abort(self.gen, f"{m.id} re-joined (it had restarted)")
            self.waiting[m.id] = m
            self.last_change = time.monotonic()
            self._event("join", id=m.id, addr=m.addr, active_gen=self.gen if self.active else None)
        return m

    def _on_disconnect(self, m: _Member) -> None:
        with self.lock:
            m.alive = False
            if self.waiting.get(m.id) is m:
                del self.waiting[m.id]
                self.last_change = time.monotonic()
                self._event("left_pool", id=m.id)
            if m in self.active and not self.finished:
                self._abort(self.gen, f"{m.id} disconnected")

    # ---------------------------------------------------------- membership
    def _abort(self, gen: int, reason: str, failure: bool = True) -> None:
        with self.lock:
            if gen != self.gen or not self.active:
                return
            survivors = [a for a in self.active if a.alive]
            self._event("abort", gen=gen, reason=reason, failure=failure,
                        survivors=[a.id for a in survivors])
            for a in survivors:
                if a.send({"op": "ABORT", "gen": gen, "reason": reason}):
                    self.waiting[a.id] = a
            self.prev_order = [a.id for a in self.active]
            self.active = []
            self.last_change = time.monotonic() if failure else self.last_change

    def _tick_loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.05)
            now = time.monotonic()
            with self.lock:
                for a in list(self.active):
                    if now - a.last_hb > self.hb_timeout:
                        a.alive = False
                        self._abort(self.gen, f"{a.id} missed heartbeats for {now - a.last_hb:.1f}s")
                        break
                for wid, w in list(self.waiting.items()):
                    if not w.alive:
                        del self.waiting[wid]
                if self.active or self.finished:
                    continue
                n = len(self.waiting)
                if self.gen == 0 and n < self.initial_workers and now < self.initial_deadline:
                    continue
                if n >= self.min_workers and (n >= self.max_workers or now - self.last_change >= self.settle_s):
                    self._form()

    def _form(self) -> None:
        order = {wid: i for i, wid in enumerate(self.prev_order)}
        members = sorted(self.waiting.values(),
                         key=lambda m: (order.get(m.id, len(order)), m.joined_at))[: self.max_workers]
        self.gen += 1
        peers = [m.addr for m in members]
        ids = [m.id for m in members]
        for rank, m in enumerate(members):
            m.last_hb = time.monotonic()
            m.send({"op": "ASSIGN", "gen": self.gen, "rank": rank, "world": len(members),
                    "peers": peers, "ids": ids})
            del self.waiting[m.id]
        self.active = members
        self._event("form", gen=self.gen, world=len(members), ids=ids)


class Assignment(dict):
    @property
    def gen(self) -> int:
        return int(self["gen"])


class RendezvousClient:
    """The worker side: joins, heartbeats, and exposes the newest assignment / abort signals."""

    def __init__(self, coord_addr: str, worker_id: str, endpoint_addr: str, hb_interval: float = 0.25,
                 on_abort=None, connect_timeout: float = 60.0):
        self.coord_addr, self.id, self.endpoint_addr = coord_addr, worker_id, endpoint_addr
        self.hb_interval = hb_interval
        self.on_abort = on_abort
        self._cv = threading.Condition()
        self._assign: Assignment | None = None
        self.aborted_gens: set[int] = set()
        self.pending = 0
        self.finished = False
        self.current_gen = -1
        self._lost = False
        host, port = parse_addr(coord_addr)
        end = time.monotonic() + connect_timeout
        while True:
            try:
                self.sock = socket.create_connection((host, port), timeout=5)
                break
            except OSError:
                if time.monotonic() > end:
                    raise
                time.sleep(0.2)
        self.sock.settimeout(None)
        wire.tune_socket(self.sock)
        self._send_lock = threading.Lock()
        self._send({"op": "JOIN", "id": worker_id, "addr": endpoint_addr})
        threading.Thread(target=self._reader, name="rdv-client", daemon=True).start()
        threading.Thread(target=self._hb_loop, name="rdv-hb", daemon=True).start()

    def _send(self, obj: dict) -> None:
        with self._send_lock:
            wire.send_json(self.sock, obj)

    def _reader(self) -> None:
        try:
            while True:
                _, msg = wire.recv_json(self.sock)
                op = msg.get("op")
                with self._cv:
                    if op == "ASSIGN":
                        self._assign = Assignment(msg)
                    elif op == "ABORT":
                        self.aborted_gens.add(int(msg["gen"]))
                        self.pending = 0
                        if self.on_abort is not None:
                            self.on_abort(int(msg["gen"]), msg.get("reason", ""))
                    elif op == "HB_ACK":
                        if int(msg.get("gen", -1)) == self.current_gen:
                            self.pending = int(msg.get("pending", 0))
                    elif op == "FINISHED":
                        self.finished = True
                    self._cv.notify_all()
        except (OSError, ValueError, wire.FrameError):
            with self._cv:
                self._lost = True
                self._cv.notify_all()
            if self.on_abort is not None:
                self.on_abort(self.current_gen, "lost the coordinator")

    def _hb_loop(self) -> None:
        while not self._lost:
            time.sleep(self.hb_interval)
            try:
                self._send({"op": "HB", "gen": self.current_gen})
            except OSError:
                return

    def wait_assignment(self, after_gen: int, timeout: float | None = None) -> Assignment | None:
        """Block until assigned to a generation newer than ``after_gen`` that is not already
        aborted.  Returns None if the run finished."""
        end = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while True:
                a = self._assign
                if a is not None and a.gen > after_gen and a.gen not in self.aborted_gens:
                    self.current_gen = a.gen
                    self.pending = 0
                    return a
                if self.finished:
                    return None
                if self._lost:
                    raise ConnectionError("lost connection to the rendezvous coordinator")
                if end is not None and time.monotonic() > end:
                    raise TimeoutError("no assignment from the coordinator")
                self._cv.wait(0.2)

    def is_aborted(self, gen: int) -> bool:
        return gen in self.aborted_gens

    def report_failure(self, gen: int, reason: str, peer: int | None = None) -> None:
        try:
            self._send({"op": "FAILED", "gen": gen, "peer": peer, "reason": reason})
        except OSError:
            pass

    def request_reform(self, gen: int) -> None:
        self._send({"op": "REFORM", "gen": gen})

    def done(self, gen: int) -> None:
        try:
            self._send({"op": "DONE", "gen": gen})
        except OSError:
            pass

    def close(self) -> None:
        self._lost = True
        try:
            self.sock.close()
        except OSError:
            pass
