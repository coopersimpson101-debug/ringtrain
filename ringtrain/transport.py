"""Threaded TCP transport: a full mesh between the ranks of one generation.

* :class:`Endpoint` owns a worker's listening socket for its whole life.  Every generation (a fixed
  membership agreed with the rendezvous coordinator) builds a new :class:`Transport` on top of it.
* Rank ``i`` dials every rank ``j < i`` and sends a HELLO ``{gen, rank}``; the endpoint's accept
  thread files incoming sockets by ``(gen, rank)`` so a slow worker never misses a connection.
* One reader thread per peer drains its socket into a tagged mailbox, so ``sendall`` can never
  deadlock against a peer that is itself blocked sending to us.  One sender thread per peer makes
  ``isend`` asynchronous, which lets the ring overlap sending one chunk with receiving the next.
* Failure detection: a peer is declared dead when its socket errors or closes (what ``SIGKILL`` /
  ``TerminateProcess`` looks like: the OS resets the connection), or when nothing at all has been
  received from it for ``hb_timeout`` seconds (what a power cut or a pulled cable looks like).
  Idle links carry a heartbeat frame every ``hb_interval`` seconds.  Any blocked ``recv`` then
  raises :class:`PeerFailure` instead of hanging.
"""
from __future__ import annotations

import collections
import json
import queue
import socket
import threading
import time

from . import wire


class PeerFailure(ConnectionError):
    """A peer in the current generation died, left, or stopped answering."""

    def __init__(self, peer: int | None, reason: str, gen: int | None = None):
        super().__init__(f"peer {peer} failed in generation {gen}: {reason}")
        self.peer, self.reason, self.gen = peer, reason, gen


class GenerationAborted(PeerFailure):
    """The coordinator (or this worker) aborted the generation; the group must re-form."""


def parse_addr(addr: str) -> tuple[str, int]:
    host, _, port = addr.rpartition(":")
    return host.strip("[]") or "127.0.0.1", int(port)


class Endpoint:
    """A worker's listening socket plus an accept thread that files connections by generation."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0, advertise: str | None = None):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(64)
        self.port = self.sock.getsockname()[1]
        self.addr = f"{advertise or (host if host != '0.0.0.0' else '127.0.0.1')}:{self.port}"
        self._pending: dict[tuple[int, int], socket.socket] = {}
        self._cv = threading.Condition()
        self._closed = False
        threading.Thread(target=self._accept_loop, name="rt-accept", daemon=True).start()

    def _accept_loop(self) -> None:
        while not self._closed:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handshake, args=(conn,), daemon=True).start()

    def _handshake(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(10)
            ftype, hello = wire.recv_json(conn)
            if ftype != wire.HELLO:
                raise wire.FrameError("expected HELLO")
            conn.settimeout(None)
            wire.tune_socket(conn)
        except (OSError, ValueError):
            conn.close()
            return
        with self._cv:
            old = self._pending.pop((int(hello["gen"]), int(hello["rank"])), None)
            if old is not None:
                old.close()
            self._pending[(int(hello["gen"]), int(hello["rank"]))] = conn
            # drop connections for generations that can no longer be used
            for key in [k for k in self._pending if k[0] < int(hello["gen"]) - 2]:
                self._pending.pop(key).close()
            self._cv.notify_all()

    def take(self, gen: int, rank: int, deadline: float, abort: threading.Event | None = None) -> socket.socket:
        with self._cv:
            while (gen, rank) not in self._pending:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise PeerFailure(rank, "did not connect in time", gen)
                if abort is not None and abort.is_set():
                    raise GenerationAborted(rank, "aborted while connecting", gen)
                self._cv.wait(min(left, 0.1))
            return self._pending.pop((gen, rank))

    def close(self) -> None:
        self._closed = True
        try:
            self.sock.close()
        except OSError:
            pass
        with self._cv:
            for s in self._pending.values():
                s.close()
            self._pending.clear()


class Handle:
    """Completion handle for :meth:`Transport.isend`."""

    __slots__ = ("_ev", "error")

    def __init__(self):
        self._ev = threading.Event()
        self.error: BaseException | None = None

    def _done(self, err: BaseException | None = None) -> None:
        self.error = err
        self._ev.set()

    def wait(self) -> None:
        self._ev.wait()
        if self.error is not None:
            raise self.error


class _Peer:
    def __init__(self, rank: int, sock: socket.socket):
        self.rank = rank
        self.sock = sock
        self.lock = threading.Lock()  # held while a frame is being written
        self.q: queue.SimpleQueue = queue.SimpleQueue()
        self.last_seen = time.monotonic()
        self.dead: str | None = None


class Transport:
    """Full-mesh tagged messaging between the ``world`` ranks of generation ``gen``."""

    def __init__(self, endpoint: Endpoint | None, gen: int, rank: int, peers: list[str],
                 connect_timeout: float = 30.0, hb_interval: float = 0.25, hb_timeout: float = 10.0):
        self.gen, self.rank, self.world = gen, rank, len(peers)
        self.hb_interval, self.hb_timeout = hb_interval, hb_timeout
        self.bytes_sent = 0
        self.msgs_sent = 0
        self._box: dict[tuple[int, int], collections.deque] = {}
        self._cv = threading.Condition()
        self._abort = threading.Event()
        self._abort_reason = ""
        self._closed = False
        self.peers: dict[int, _Peer] = {}
        deadline = time.monotonic() + connect_timeout
        try:
            for j in range(rank):  # dial lower ranks
                self.peers[j] = _Peer(j, self._dial(peers[j], deadline))
            for j in range(rank + 1, self.world):  # accept higher ranks
                if endpoint is None:
                    raise ValueError("an Endpoint is required to accept connections")
                self.peers[j] = _Peer(j, endpoint.take(gen, j, deadline, self._abort))
        except BaseException:
            for p in self.peers.values():
                p.sock.close()
            raise
        for p in self.peers.values():
            threading.Thread(target=self._reader, args=(p,), name=f"rt-recv-{p.rank}", daemon=True).start()
            threading.Thread(target=self._sender, args=(p,), name=f"rt-send-{p.rank}", daemon=True).start()
        if self.peers:
            threading.Thread(target=self._heartbeat, name="rt-heartbeat", daemon=True).start()

    # ------------------------------------------------------------------ setup
    def _dial(self, addr: str, deadline: float) -> socket.socket:
        host, port = parse_addr(addr)
        last: Exception | None = None
        while time.monotonic() < deadline:
            if self._abort.is_set():
                raise GenerationAborted(None, "aborted while connecting", self.gen)
            try:
                s = socket.create_connection((host, port), timeout=min(5.0, max(0.1, deadline - time.monotonic())))
                s.settimeout(None)
                wire.tune_socket(s)
                wire.write_frame(s, wire.HELLO, self.rank, 0,
                                 json.dumps({"gen": self.gen, "rank": self.rank}).encode())
                return s
            except OSError as e:
                last = e
                time.sleep(0.05)
        raise PeerFailure(None, f"could not connect to {addr}: {last}", self.gen)

    # --------------------------------------------------------------- threads
    def _reader(self, p: _Peer) -> None:
        def seen():
            p.last_seen = time.monotonic()
        try:
            while True:
                ftype, src, tag, payload = wire.read_frame(p.sock, seen)
                seen()
                if ftype == wire.DATA:
                    with self._cv:
                        self._box.setdefault((p.rank, tag), collections.deque()).append(payload)
                        self._cv.notify_all()
                elif ftype == wire.BYE:
                    self._mark_dead(p, "left the group")
                    return
        except (OSError, wire.FrameError) as e:
            self._mark_dead(p, f"connection lost ({type(e).__name__}: {e})")

    def _sender(self, p: _Peer) -> None:
        while True:
            item = p.q.get()
            if item is None:
                return
            tag, payload, handle = item
            if p.dead or self._closed:
                handle._done(PeerFailure(p.rank, p.dead or "transport closed", self.gen))
                continue
            try:
                with p.lock:
                    n = wire.write_frame(p.sock, wire.DATA, self.rank, tag, payload)
                self.bytes_sent += n
                self.msgs_sent += 1
                handle._done()
            except OSError as e:
                self._mark_dead(p, f"send failed ({type(e).__name__}: {e})")
                handle._done(PeerFailure(p.rank, p.dead or "send failed", self.gen))

    def _heartbeat(self) -> None:
        while not self._closed:
            time.sleep(self.hb_interval)
            now = time.monotonic()
            for p in list(self.peers.values()):
                if p.dead:
                    continue
                if now - p.last_seen > self.hb_timeout:
                    self._mark_dead(p, f"no heartbeat for {now - p.last_seen:.1f}s")
                    continue
                if p.lock.acquire(blocking=False):  # a frame in flight already proves liveness
                    try:
                        wire.write_frame(p.sock, wire.HEARTBEAT, self.rank, 0)
                    except OSError as e:
                        p.lock.release()
                        self._mark_dead(p, f"heartbeat send failed ({e})")
                        continue
                    p.lock.release()

    def _mark_dead(self, p: _Peer, reason: str) -> None:
        with self._cv:
            if p.dead is None:
                p.dead = reason
            self._cv.notify_all()
        try:
            p.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    # ------------------------------------------------------------------- API
    def isend(self, dst: int, tag: int, payload) -> Handle:
        h = Handle()
        p = self.peers[dst]
        if p.dead:
            h._done(PeerFailure(dst, p.dead, self.gen))
        else:
            p.q.put((tag, payload, h))
        return h

    def send(self, dst: int, tag: int, payload) -> None:
        self.isend(dst, tag, payload).wait()

    def recv(self, src: int, tag: int, timeout: float | None = None) -> bytearray:
        key = (src, tag)
        end = None if timeout is None else time.monotonic() + timeout
        p = self.peers[src]
        with self._cv:
            while True:
                dq = self._box.get(key)
                if dq:
                    item = dq.popleft()
                    if not dq:
                        del self._box[key]
                    return item
                if self._abort.is_set():
                    raise GenerationAborted(None, self._abort_reason, self.gen)
                if p.dead:
                    raise PeerFailure(src, p.dead, self.gen)
                if end is not None and time.monotonic() >= end:
                    raise TimeoutError(f"recv from {src} tag {tag:#x} timed out")
                self._cv.wait(0.2)

    def check(self) -> None:
        """Raise if the generation is no longer usable."""
        if self._abort.is_set():
            raise GenerationAborted(None, self._abort_reason, self.gen)
        for p in self.peers.values():
            if p.dead:
                raise PeerFailure(p.rank, p.dead, self.gen)

    def abort(self, reason: str = "aborted") -> None:
        with self._cv:
            self._abort_reason = reason
            self._abort.set()
            self._cv.notify_all()

    @property
    def aborted(self) -> bool:
        return self._abort.is_set()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for p in self.peers.values():
            if not p.dead and p.lock.acquire(timeout=1.0):
                try:
                    wire.write_frame(p.sock, wire.BYE, self.rank, 0)
                except OSError:
                    pass
                p.lock.release()
            p.q.put(None)
            try:
                p.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            p.sock.close()
