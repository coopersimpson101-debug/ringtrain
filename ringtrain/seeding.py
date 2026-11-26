"""Thread-safe deterministic model construction.

``torch.manual_seed(s); build()`` is not safe when several ranks build their models at the same time
in one process (the threaded test harness, or any user running ranks as threads): the global
generator is shared, so one thread's ``manual_seed`` can land between another thread's parameter
initialisations.  That produced a corrupted rank-0 initialisation on Linux/macOS CI, where thread
switches happen at different points than on Windows.

:func:`seeded` serialises construction and restores the caller's RNG state afterwards, so
``with seeded(s): build()`` gives exactly the parameters that ``torch.manual_seed(s); build()``
gives single-threaded -- bit for bit -- whatever other threads are doing with ``seeded``.
"""
from __future__ import annotations

import contextlib
import threading

import torch

_LOCK = threading.RLock()


@contextlib.contextmanager
def seeded(seed: int):
    with _LOCK:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            yield
