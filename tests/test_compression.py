"""Top-k with error feedback and PowerSGD."""
import numpy as np
import pytest
import torch

from ringtrain.adapters.mlp import TeacherMLPAdapter
from ringtrain.compression import PowerSGDReducer, Reducer, TopKReducer, make_reducer, orthonormalise_
from ringtrain.data import global_batch_indices, shard
from ringtrain.ddp import DistributedDataParallel
from ringtrain.launch import run_threads

torch.set_num_threads(1)


def _grads(rank, step=0):
    g = torch.Generator().manual_seed(rank * 100 + step)
    return [torch.randn(20, 30, generator=g), torch.randn(30, generator=g), torch.randn(4, 5, 6, generator=g)]


def _exact_sum(world, step=0):
    return [sum(_grads(r, step)[i] for r in range(world)) for i in range(3)]


def test_make_reducer():
    assert type(make_reducer("none")) is Reducer
    assert make_reducer("topk:0.05").ratio == 0.05
    assert make_reducer("powersgd:4").q == 4
    with pytest.raises(ValueError):
        make_reducer("zip")


def test_topk_full_ratio_is_exact():
    out = run_threads(3, lambda pg: TopKReducer(ratio=1.0).reduce(pg, 0, _grads(pg.rank)))
    for res in out:
        for a, b in zip(res, _exact_sum(3)):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-5)


def test_topk_error_feedback_conserves_mass():
    """Sum over steps of what was transmitted + the final residual == sum of all gradients."""
    steps = 6

    def fn(pg):
        red = TopKReducer(ratio=0.05)
        sent = None
        for s in range(steps):
            out = torch.cat([t.reshape(-1) for t in red.reduce(pg, 0, _grads(pg.rank, s))])
            sent = out if sent is None else sent + out
        resid = red.residual[0]
        all_resid = torch.from_numpy(pg.all_gather_array(resid.numpy())[0].copy()) * 0
        for r in pg.all_gather_array(resid.numpy()):
            all_resid += torch.from_numpy(r.copy())
        return sent, all_resid
    total = sum(torch.cat([t.reshape(-1) for t in _exact_sum(3, s)]) for s in range(steps))
    for sent, resid in run_threads(3, fn):
        torch.testing.assert_close(sent + resid, total, rtol=1e-4, atol=1e-4)


def test_topk_results_identical_on_every_rank():
    out = run_threads(4, lambda pg: TopKReducer(ratio=0.1).reduce(pg, 0, _grads(pg.rank)))
    for res in out[1:]:
        for a, b in zip(res, out[0]):
            assert torch.equal(a, b)


def test_orthonormalise():
    p = orthonormalise_(torch.randn(50, 4, dtype=torch.float64))
    torch.testing.assert_close(p.T @ p, torch.eye(4, dtype=torch.float64), rtol=0, atol=1e-7)


def test_powersgd_full_rank_is_exact():
    """With q = min(n, m) the projection P^ P^T is the identity on the column space: exact."""
    def fn(pg):
        return PowerSGDReducer(rank=30, min_compression=0.0).reduce(pg, 0, _grads(pg.rank))
    for res in run_threads(2, fn):
        for a, b in zip(res, _exact_sum(2)):
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_powersgd_low_rank_compresses_and_feeds_back_error():
    def fn(pg):
        red = PowerSGDReducer(rank=2, min_compression=1.0)
        b0 = pg.bytes_sent
        out = red.reduce(pg, 0, _grads(pg.rank))
        return out, pg.bytes_sent - b0, {k: v.clone() for k, v in red.err.items()}
    res = run_threads(2, fn)
    out, nbytes, err = res[0]
    assert out[1].shape == (30,) and out[2].shape == (4, 5, 6)
    torch.testing.assert_close(out[1], _exact_sum(2)[1])  # vectors are reduced exactly
    assert len(err) == 2  # two compressible matrices carry error feedback
    dense_bytes = 4 * (600 + 30 + 120)
    assert nbytes < dense_bytes  # less traffic than an exact all-reduce of the same bucket
    assert torch.equal(res[0][0][0], res[1][0][0])  # identical on both ranks


@pytest.mark.parametrize("spec", ["none", "topk:0.05", "powersgd:4"])
def test_compressed_training_converges(spec):
    ad = TeacherMLPAdapter(num_samples=4096, hidden=64, depth=2, global_batch=128, lr=0.1)

    def fn(pg):
        model = ad.build_model(0)
        opt = ad.build_optimizer(model)
        ddp = DistributedDataParallel(model, pg, make_reducer(spec), bucket_mb=0.05, average=False)
        losses = []
        for s in range(100):
            idx = shard(global_batch_indices(ad.num_samples, ad.global_batch, s, 0), pg.rank, pg.world)
            opt.zero_grad()
            loss, _ = ad.loss_sum(model, ad.batch(idx))
            (loss / ad.global_batch).backward()
            ddp.finish()
            opt.step()
            m = np.array([float(loss.detach())])
            pg.allreduce(m)
            losses.append(m[0] / ad.global_batch)
        ddp.close()
        return float(np.mean(losses[-20:])), losses[0]
    final, first = run_threads(2, fn)[0]
    assert final < 0.6 * first, (final, first)  # full results: scripts/compression_compare.py
