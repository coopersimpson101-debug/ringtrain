"""Synthetic teacher-student classification: a fixed random 'teacher' MLP labels Gaussian inputs,
a student MLP learns to imitate it.  Cheap, deterministic, and its loss curve has a clear shape,
which is what the scaling and fault-injection experiments need."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn


def make_mlp(d_in: int, hidden: int, depth: int, d_out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    last = d_in
    for _ in range(depth):
        layers += [nn.Linear(last, hidden), nn.ReLU()]
        last = hidden
    layers.append(nn.Linear(last, d_out))
    return nn.Sequential(*layers)


class TeacherMLPAdapter:
    name = "mlp"

    def __init__(self, num_samples: int = 32768, d_in: int = 32, classes: int = 10, hidden: int = 256,
                 depth: int = 3, global_batch: int = 256, lr: float = 0.05, data_seed: int = 1234,
                 momentum: float = 0.9):
        self.num_samples, self.global_batch = num_samples, global_batch
        self.d_in, self.classes, self.hidden, self.depth = d_in, classes, hidden, depth
        self.lr, self.momentum = lr, momentum
        g = torch.Generator().manual_seed(data_seed)
        self.X = torch.randn(num_samples, d_in, generator=g)
        teacher = make_mlp(d_in, 64, 2, classes)
        with torch.no_grad():
            for p in teacher.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.6)
            self.y = teacher(self.X).argmax(1)

    def build_model(self, seed: int = 0) -> nn.Module:
        torch.manual_seed(seed)
        return make_mlp(self.d_in, self.hidden, self.depth, self.classes)

    def build_optimizer(self, model):
        return torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)

    def batch(self, indices: np.ndarray):
        idx = torch.from_numpy(np.asarray(indices, dtype=np.int64))
        return self.X[idx], self.y[idx]

    def loss_sum(self, model, batch):
        x, y = batch
        return nn.functional.cross_entropy(model(x), y, reduction="sum"), int(x.shape[0])

    @torch.no_grad()
    def accuracy(self, model, n: int = 4096) -> float:
        return float((model(self.X[:n]).argmax(1) == self.y[:n]).float().mean())
