"""Length-prefixed binary frames over a TCP stream.

Every message on every ringtrain socket (peer-to-peer data, heartbeats, rendezvous control) is one
frame::

    offset size  field
    0      4     magic  b"RTF1"
    4      1     type   (DATA, HEARTBEAT, CTRL, HELLO, BYE)
    5      1     reserved (0)
    6      2     src    sender rank (uint16, little endian)
    8      8     tag    uint64 -- matches a send with the receive waiting for it
    16     8     length payload bytes (uint64)
    24     ...   payload

TCP gives us an ordered byte stream; the frame gives message boundaries, and the tag lets several
independent operations (e.g. DDP's comm thread and the training thread) share one connection
without their messages being confused.  Control payloads are JSON: nothing received from the
network is ever unpickled.
"""
from __future__ import annotations

import json
import socket
import struct

MAGIC = b"RTF1"
HEADER = struct.Struct("<4sBBHQQ")
HEADER_SIZE = HEADER.size  # 24
MAX_FRAME = 1 << 31  # 2 GiB: refuse absurd lengths from a corrupt or hostile stream

DATA, HEARTBEAT, CTRL, HELLO, BYE = 1, 2, 3, 4, 5
TYPE_NAMES = {DATA: "DATA", HEARTBEAT: "HEARTBEAT", CTRL: "CTRL", HELLO: "HELLO", BYE: "BYE"}


class FrameError(ConnectionError):
    """The byte stream is not a valid ringtrain frame stream."""


def pack_header(ftype: int, src: int, tag: int, length: int) -> bytes:
    return HEADER.pack(MAGIC, ftype, 0, src, tag, length)


def unpack_header(buf: bytes) -> tuple[int, int, int, int]:
    magic, ftype, _, src, tag, length = HEADER.unpack(buf)
    if magic != MAGIC:
        raise FrameError(f"bad magic {magic!r}")
    if ftype not in TYPE_NAMES:
        raise FrameError(f"unknown frame type {ftype}")
    if length > MAX_FRAME:
        raise FrameError(f"frame too large ({length} bytes)")
    return ftype, src, tag, length


def recv_exact(sock: socket.socket, n: int, on_progress=None) -> bytearray:
    """Read exactly ``n`` bytes or raise ConnectionError on EOF."""
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        k = sock.recv_into(view[got:], min(n - got, 1 << 22))
        if k == 0:
            raise ConnectionError("peer closed the connection")
        got += k
        if on_progress is not None:
            on_progress()
    return buf


def read_frame(sock: socket.socket, on_progress=None) -> tuple[int, int, int, bytearray]:
    ftype, src, tag, length = unpack_header(bytes(recv_exact(sock, HEADER_SIZE, on_progress)))
    payload = recv_exact(sock, length, on_progress) if length else bytearray()
    return ftype, src, tag, payload


def write_frame(sock: socket.socket, ftype: int, src: int, tag: int, payload=b"") -> int:
    """Send one frame; ``payload`` may be any buffer (bytes, memoryview of a numpy array, ...)."""
    mv = memoryview(payload).cast("B")
    sock.sendall(pack_header(ftype, src, tag, mv.nbytes))
    if mv.nbytes:
        sock.sendall(mv)
    return HEADER_SIZE + mv.nbytes


def send_json(sock: socket.socket, obj: dict, ftype: int = CTRL, src: int = 0, tag: int = 0) -> None:
    write_frame(sock, ftype, src, tag, json.dumps(obj, separators=(",", ":")).encode())


def recv_json(sock: socket.socket) -> tuple[int, dict]:
    ftype, _, _, payload = read_frame(sock)
    return ftype, (json.loads(payload) if payload else {})


def tune_socket(sock: socket.socket) -> None:
    """Low latency for small control messages, big buffers for gradient chunks."""
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        try:
            sock.setsockopt(socket.SOL_SOCKET, opt, 4 << 20)
        except OSError:  # pragma: no cover - some kernels cap this
            pass
