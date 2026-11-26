"""DDP vs single-process training on the same global batches."""
import numpy as np
import pytest
import torch
from torch import nn

from ringtrain.adapters.mlp import TeacherMLPAdapter
from ringtrain.data import global_batch_indices, shard
from ringtrain.ddp import DistributedDataParallel, broadcast_module
from ringtrain.launch import run_threads
from ringtrain.seeding import seeded

torch.set_num_threads(1)
AD = TeacherMLPAdapter(num_samples=2048, hidden=48, depth=2, global_batch=64, lr=0.1)
STEPS = 8


def single_process():
    model = AD.build_model(0)
    opt = AD.build_optimizer(model)
    losses = []
    for s in range(STEPS):
        idx = global_batch_indices(AD.num_samples, AD.global_batch, s, 0)
        opt.zero_grad()
        loss, _ = AD.loss_sum(model, AD.batch(idx))
        (loss / AD.global_batch).backward()
        opt.step()
        losses.append(float(loss.detach()) / AD.global_batch)
    return [p.detach().clone() for p in model.parameters()], losses


def ddp_run(pg, bucket_mb=0.004, overlap=True):
    # different init per rank on purpose: DDP's initial broadcast must fix it
    model = AD.build_model(100 + pg.rank if pg.rank else 0)
    opt = AD.build_optimizer(model)
    ddp = DistributedDataParallel(model, pg, bucket_mb=bucket_mb, average=False, overlap=overlap)
    losses = []
    launched_during_backward = 0
    for s in range(STEPS):
        idx = shard(global_batch_indices(AD.num_samples, AD.global_batch, s, 0), pg.rank, pg.world)
        opt.zero_grad()
        loss, _ = AD.loss_sum(model, AD.batch(idx))
        (loss / AD.global_batch).backward()
        launched_during_backward += ddp._next_launch  # buckets already handed to the comm thread
        ddp.finish()
        opt.step()
        m = np.array([float(loss.detach())])
        pg.allreduce(m)
        losses.append(m[0] / AD.global_batch)
    ddp.close()
    return [p.detach().clone() for p in model.parameters()], losses, len(ddp.buckets), launched_during_backward


@pytest.mark.parametrize("world", [1, 2, 3, 4])
def test_ddp_matches_single_process(world):
    ref_params, ref_losses = single_process()
    out = run_threads(world, ddp_run)
    for params, losses, nb, _ in out:
        assert nb > 1  # the tiny bucket cap really produced several buckets
        np.testing.assert_allclose(losses, ref_losses, rtol=1e-5, atol=1e-6)
        for p, q in zip(params, ref_params):
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-6)
    # every rank holds bit-identical parameters (the ring's all-gather copies, it never re-sums)
    for params, *_ in out[1:]:
        for p, q in zip(params, out[0][0]):
            assert torch.equal(p, q)


def test_overlap_launches_buckets_before_backward_finishes():
    out = run_threads(2, ddp_run)
    _, _, nbuckets, launched = out[0]
    assert launched >= STEPS * (nbuckets - 1)  # all but possibly the last bucket go out early


def test_no_overlap_gives_the_same_result():
    a = run_threads(2, lambda pg: ddp_run(pg, overlap=True))
    b = run_threads(2, lambda pg: ddp_run(pg, overlap=False))
    for p, q in zip(a[0][0], b[0][0]):
        assert torch.equal(p, q)


def test_unused_parameters_count_as_zero():
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.used, self.unused = nn.Linear(4, 1), nn.Linear(4, 1)

        def forward(self, x):
            return self.used(x)

    def fn(pg):
        with seeded(0):
            m = Two()
        ddp = DistributedDataParallel(m, pg, bucket_mb=1e-6)
        m(torch.ones(2, 4) * (pg.rank + 1)).sum().backward()
        ddp.finish()
        ddp.close()
        return m.used.weight.grad.clone(), m.unused.weight.grad.clone()
    for used, unused in run_threads(2, fn):
        assert torch.allclose(used, torch.full((1, 4), 3.0))  # mean of 2 and 4
        assert torch.count_nonzero(unused) == 0


def test_broadcast_module_makes_ranks_identical():
    def fn(pg):
        with seeded(pg.rank):
            m = nn.Sequential(nn.Linear(3, 5), nn.BatchNorm1d(5))
        m[1].running_mean.fill_(pg.rank)
        broadcast_module(pg, m, root=0)
        return [t.clone() for t in list(m.parameters()) + list(m.buffers())]
    out = run_threads(3, fn)
    for ts in out[1:]:
        for a, b in zip(ts, out[0]):
            assert torch.equal(a, b)


def test_concurrent_model_construction_is_deterministic():
    """Regression: ranks building models in parallel threads used to race on torch's global RNG
    (manual_seed in one thread between another thread's parameter inits), which corrupted rank 0's
    initialisation on Linux/macOS CI.  ``seeded`` must give the single-threaded parameters."""
    import threading
    torch.manual_seed(0)
    from ringtrain.adapters.mlp import make_mlp
    ref = [p.clone() for p in make_mlp(AD.d_in, AD.hidden, AD.depth, AD.classes).parameters()]
    for _ in range(50):
        out = {}

        def build(seed):
            out[seed] = [p.clone() for p in AD.build_model(seed).parameters()]
        ths = [threading.Thread(target=build, args=(s,)) for s in (0, 101, 102, 103)]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        assert all(torch.equal(a, b) for a, b in zip(out[0], ref))
