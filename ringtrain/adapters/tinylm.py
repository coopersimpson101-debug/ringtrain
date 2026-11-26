"""Plug-in point for tinylm (the owner's LLM trainer, developed separately).

ringtrain does not import tinylm at install time and does not assume its internals.  The contract
is: tinylm ships a module exposing a factory that returns an object implementing
:class:`ringtrain.adapters.base.Adapter`.  The default location looked up here is
``tinylm.ringtrain_adapter:make_adapter``; any other location works with
``--adapter some.module:factory``.

What the tinylm side needs to provide (about 40 lines, see docs/adapters.md)::

    class TinyLMAdapter:
        name = "tinylm"
        num_samples   = len(token_windows)          # dataset positions
        global_batch  = 64                          # sequences per optimizer step, fixed for the run
        def build_model(self, seed): ...            # tinylm's GPT, as nn.Module (nn.Sequential of
                                                    # embed / blocks / head if you want pipeline stages)
        def build_optimizer(self, model): ...       # e.g. AdamW with tinylm's schedule
        def batch(self, indices): ...               # (x, y) token tensors for these windows
        def loss_sum(self, model, batch): ...       # SUM over sequences of mean token CE, and B

    def make_adapter(**kw): return TinyLMAdapter(**kw)
"""
from __future__ import annotations

import importlib


def make_adapter(**kw):
    try:
        mod = importlib.import_module("tinylm.ringtrain_adapter")
    except ImportError as e:  # pragma: no cover - depends on tinylm being installed
        raise ImportError(
            "tinylm is not installed (or has no tinylm.ringtrain_adapter module). Install tinylm, or "
            "pass --adapter your.module:factory. See docs/adapters.md for the 5-method interface."
        ) from e
    return mod.make_adapter(**kw)
