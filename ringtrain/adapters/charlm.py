"""A tiny byte-level causal transformer, the in-repo stand-in for tinylm.

The corpus is ringtrain's own Python source (so it is deterministic, licence-clean and needs no
download).  The model is an ``nn.Sequential`` -- [embedding, blocks..., head] -- so the same
object can be trained data-parallel or split into pipeline stages.  Attention is written out by
hand (QK^T / sqrt(d), causal mask, softmax, V) rather than using ``nn.MultiheadAttention``.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch
from torch import nn


def load_corpus() -> bytes:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parts = []
    for dp, _, fs in sorted(os.walk(root)):
        for f in sorted(fs):
            if f.endswith(".py"):
                with open(os.path.join(dp, f), "rb") as fh:
                    parts.append(fh.read())
    return b"\n".join(parts)


class Embed(nn.Module):
    def __init__(self, vocab: int, d: int, ctx: int):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Parameter(torch.zeros(ctx, d))
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, idx):
        return self.tok(idx) + self.pos[: idx.shape[1]]


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.h)
        mask = torch.ones(T, T, dtype=torch.bool).tril()
        att = att.masked_fill(~mask, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        return x + self.mlp(self.ln2(x))


class Head(nn.Module):
    def __init__(self, d: int, vocab: int):
        super().__init__()
        self.ln, self.out = nn.LayerNorm(d), nn.Linear(d, vocab)

    def forward(self, x):
        return self.out(self.ln(x))


def make_charlm(vocab=256, d=64, heads=4, layers=2, ctx=64) -> nn.Sequential:
    return nn.Sequential(Embed(vocab, d, ctx), *[Block(d, heads) for _ in range(layers)], Head(d, vocab))


def lm_loss(logits: torch.Tensor, targets: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """Per-sample mean token cross-entropy, then ``reduction`` over samples."""
    B, T, V = logits.shape
    per_tok = nn.functional.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T), reduction="none")
    per_sample = per_tok.view(B, T).mean(1)
    return per_sample.sum() if reduction == "sum" else per_sample.mean()


class CharLMAdapter:
    name = "charlm"

    def __init__(self, ctx: int = 64, d: int = 64, heads: int = 4, layers: int = 2, global_batch: int = 32,
                 lr: float = 3e-3):
        self.data = np.frombuffer(load_corpus(), dtype=np.uint8)
        self.ctx, self.d, self.heads, self.layers = ctx, d, heads, layers
        self.global_batch, self.lr = global_batch, lr
        self.num_samples = len(self.data) - ctx - 1

    def build_model(self, seed: int = 0):
        torch.manual_seed(seed)
        return make_charlm(256, self.d, self.heads, self.layers, self.ctx)

    def build_optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=0.0)

    def batch(self, indices):
        idx = np.asarray(indices, dtype=np.int64)[:, None] + np.arange(self.ctx + 1)[None, :]
        w = torch.from_numpy(self.data[idx].astype(np.int64))
        return w[:, :-1], w[:, 1:]

    def loss_sum(self, model, batch):
        x, y = batch
        return lm_loss(model(x), y, "sum"), int(x.shape[0])
