"""Workload adapters: what to train, independent of how it is distributed.

An adapter is any object with this interface (see :class:`ringtrain.adapters.base.Adapter`)::

    name: str
    num_samples: int
    global_batch: int
    build_model(seed) -> nn.Module
    build_optimizer(model) -> torch.optim.Optimizer
    batch(indices: np.ndarray) -> Any                 # samples for these global indices
    loss_sum(model, batch) -> (loss_sum Tensor, count) # SUM over samples, not mean

Built in: ``mlp`` (synthetic teacher-student classification), ``charlm`` (a tiny character-level
transformer, the stand-in for tinylm).  Any other adapter is loaded from ``module:factory``.
"""
from __future__ import annotations

import importlib


def make_adapter(spec: str, **kw):
    if spec == "mlp":
        from .mlp import TeacherMLPAdapter
        return TeacherMLPAdapter(**kw)
    if spec == "charlm":
        from .charlm import CharLMAdapter
        return CharLMAdapter(**kw)
    if spec == "tinylm":
        from .tinylm import make_adapter as mk
        return mk(**kw)
    mod, _, fn = spec.partition(":")
    if not fn:
        raise ValueError(f"unknown adapter {spec!r}: use mlp, charlm, tinylm or module:factory")
    return getattr(importlib.import_module(mod), fn)(**kw)
