"""Checkpoint stores: raftstore (replicated, survives a node dying) and a local directory.

Layout in either store, for a run named ``R``::

    ringtrain/R/step-000000250     the checkpoint blob (torch.save of a dict of tensors/numbers)
    ringtrain/R/LATEST             JSON {step, key, sha256, bytes, world, gen, time}

The pointer is written only after the blob has been acknowledged, and raftstore acknowledges a PUT
only once it is committed on a majority and fsynced, so ``LATEST`` never names a blob that a
majority of storage nodes does not have.  A worker that dies half-way through an upload leaves at
most an orphaned blob; readers keep using the previous pointer.  The blob's SHA-256 is checked on
load.

The raftstore client here is a tiny stdlib HTTP client against raftstore's documented API
(``PUT/GET/DELETE /kv/<key>``; 307 -> follow the leader; 503 -> retry), so ringtrain itself has
no dependency on the raftstore package; the tests start a real 3-process raftstore cluster.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import threading
import time
import urllib.error
import urllib.request

import torch


class StoreError(RuntimeError):
    pass


# --------------------------------------------------------------- serialisation
def dumps_state(state: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def loads_state(blob: bytes) -> dict:
    # weights_only: a checkpoint can only contain tensors and plain containers, never code
    return torch.load(io.BytesIO(blob), map_location="cpu", weights_only=True)


# --------------------------------------------------------------------- stores
class KVBackend:
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes | None: ...
    def delete(self, key: str) -> None: ...


class DirBackend(KVBackend):
    """Local directory, atomic via write-to-temp + fsync + rename (tests, single machine)."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _path(self, key: str) -> str:
        return os.path.join(self.root, *key.split("/"))

    def put(self, key: str, data: bytes) -> None:
        p = self._path(key)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = f"{p}.tmp{os.getpid()}.{threading.get_ident()}"
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)

    def get(self, key: str) -> bytes | None:
        try:
            with open(self._path(key), "rb") as f:
                return f.read()
        except FileNotFoundError:
            return None

    def delete(self, key: str) -> None:
        try:
            os.remove(self._path(key))
        except FileNotFoundError:
            pass


class RaftstoreBackend(KVBackend):
    """Talks to any node of a raftstore cluster; follows leader redirects and retries while an
    election is in progress or a node is down."""

    def __init__(self, urls: str | list[str], timeout: float = 15.0, deadline_s: float = 60.0):
        if isinstance(urls, str):
            urls = [u for u in urls.split(",") if u]
        self.urls = [u.rstrip("/") if "://" in u else "http://" + u.rstrip("/") for u in urls]
        self.timeout, self.deadline_s = timeout, deadline_s
        self.preferred = self.urls[0]

    def _request(self, method: str, path: str, body: bytes | None = None) -> tuple[int, bytes]:
        end = time.monotonic() + self.deadline_s
        i, delay, last = 0, 0.05, None
        cands = [self.preferred] + [u for u in self.urls if u != self.preferred]
        while time.monotonic() < end:
            base = cands[i % len(cands)]
            req = urllib.request.Request(base + path, data=body, method=method)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 - http(s) only
                    status, data = r.status, r.read()
            except urllib.error.HTTPError as e:
                status, data = e.code, e.read()
                if status == 307:
                    hint = None
                    try:
                        hint = json.loads(data).get("leader_http")
                    except ValueError:
                        pass
                    if hint:
                        hint = hint.rstrip("/")
                        hint = hint if "://" in hint else "http://" + hint
                        if hint not in cands:
                            cands.insert(0, hint)
                        i = cands.index(hint)
                        continue
            except (urllib.error.URLError, OSError) as e:
                last = e
                i += 1
                time.sleep(delay)
                delay = min(delay * 2, 0.5)
                continue
            if status in (503, 504):
                last = StoreError(f"{method} {path}: HTTP {status} {data[:200]!r}")
                i += 1
                time.sleep(delay)
                delay = min(delay * 2, 0.5)
                continue
            self.preferred = base
            return status, data
        raise StoreError(f"raftstore unreachable for {method} {path}: {last}")

    def put(self, key: str, data: bytes) -> None:
        s, b = self._request("PUT", f"/kv/{key}", data)
        if s != 201:
            raise StoreError(f"PUT {key}: HTTP {s} {b[:200]!r}")

    def get(self, key: str) -> bytes | None:
        s, b = self._request("GET", f"/kv/{key}")
        if s == 404:
            return None
        if s != 200:
            raise StoreError(f"GET {key}: HTTP {s} {b[:200]!r}")
        return b

    def delete(self, key: str) -> None:
        s, b = self._request("DELETE", f"/kv/{key}")
        if s not in (200, 404):
            raise StoreError(f"DELETE {key}: HTTP {s} {b[:200]!r}")


def make_backend(spec: str) -> KVBackend:
    """``raftstore:http://h1:8080,http://h2:8080`` or ``dir:/path``."""
    kind, _, rest = spec.partition(":")
    if kind == "raftstore":
        return RaftstoreBackend(rest)
    if kind == "dir":
        return DirBackend(rest)
    raise ValueError(f"unknown store spec {spec!r} (want raftstore:URLS or dir:PATH)")


class CheckpointStore:
    def __init__(self, backend: KVBackend, run: str, keep: int = 3):
        self.backend, self.run, self.keep = backend, run, keep
        self._saved: list[str] = []
        self._thread: threading.Thread | None = None
        self._error: BaseException | None = None
        self.save_log: list[dict] = []

    def _key(self, step: int) -> str:
        return f"ringtrain/{self.run}/step-{step:09d}"

    @property
    def _latest_key(self) -> str:
        return f"ringtrain/{self.run}/LATEST"

    def save_blob(self, step: int, blob: bytes, meta: dict | None = None) -> dict:
        t0 = time.time()
        key = self._key(step)
        self.backend.put(key, blob)
        ptr = {"step": step, "key": key, "sha256": hashlib.sha256(blob).hexdigest(), "bytes": len(blob),
               "time": time.time(), **(meta or {})}
        self.backend.put(self._latest_key, json.dumps(ptr).encode())
        self._saved.append(key)
        while len(self._saved) > self.keep:
            old = self._saved.pop(0)
            try:
                self.backend.delete(old)
            except StoreError:
                pass
        self.save_log.append({"step": step, "bytes": len(blob), "seconds": time.time() - t0})
        return ptr

    def save(self, step: int, state: dict, meta: dict | None = None) -> dict:
        return self.save_blob(step, dumps_state(state), meta)

    def save_async(self, step: int, state: dict, meta: dict | None = None) -> None:
        """Serialise now (on the training thread, so later steps cannot race the tensors), upload
        in the background.  Raises a previous upload's error, and waits for it to finish first."""
        blob = dumps_state(state)
        self.wait()

        def run():
            try:
                self.save_blob(step, blob, meta)
            except BaseException as e:  # noqa: BLE001
                self._error = e
        self._thread = threading.Thread(target=run, name="rt-ckpt", daemon=True)
        self._thread.start()

    def wait(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        if self._error is not None:
            e, self._error = self._error, None
            raise e

    def latest_pointer(self) -> dict | None:
        raw = self.backend.get(self._latest_key)
        return json.loads(raw) if raw else None

    def load_latest_blob(self) -> tuple[dict, bytes] | None:
        ptr = self.latest_pointer()
        if ptr is None:
            return None
        blob = self.backend.get(ptr["key"])
        if blob is None:
            raise StoreError(f"LATEST points at missing {ptr['key']}")
        if hashlib.sha256(blob).hexdigest() != ptr["sha256"]:
            raise StoreError(f"checksum mismatch for {ptr['key']}")
        return ptr, blob

    def load_latest(self) -> tuple[dict, dict] | None:
        got = self.load_latest_blob()
        if got is None:
            return None
        ptr, blob = got
        return ptr, loads_state(blob)
