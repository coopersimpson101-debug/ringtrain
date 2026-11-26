"""Pipeline parallelism: schedules, and GPipe / 1F1B vs single-process training."""
import copy

import pytest
import torch
from torch import nn

from ringtrain.adapters.charlm import lm_loss, make_charlm
from ringtrain.launch import run_threads
from ringtrain.pipeline import (PipelineStage, bubble_fraction, gpipe_schedule, one_f_one_b_schedule,
                                reference_step, split_sequential)

torch.set_num_threads(1)


@pytest.mark.parametrize("sched", [gpipe_schedule, one_f_one_b_schedule])
@pytest.mark.parametrize("S,M", [(1, 1), (2, 4), (4, 4), (4, 8), (4, 2), (3, 7)])
def test_schedule_is_complete_and_causal(sched, S, M):
    for s in range(S):
        acts = sched(S, s, M)
        assert sorted(m for k, m in acts if k == "F") == list(range(M))
        assert sorted(m for k, m in acts if k == "B") == list(range(M))
        seen = set()
        for k, m in acts:
            if k == "F":
                seen.add(m)
            else:
                assert m in seen  # backward of a micro-batch only after its forward


@pytest.mark.parametrize("S,M", [(4, 8), (4, 16), (2, 8)])
def test_1f1b_bounds_in_flight_activations(S, M):
    for s in range(S):
        inflight = peak = 0
        for k, _ in one_f_one_b_schedule(S, s, M):
            inflight += 1 if k == "F" else -1
            peak = max(peak, inflight)
        assert peak == min(S - s, M)  # GPipe would hold all M
    assert bubble_fraction(4, 8) == pytest.approx(3 / 11)


def test_split_sequential_balances_and_preserves_layers():
    model = nn.Sequential(*[nn.Linear(8, 8) for _ in range(8)])
    stages = split_sequential(model, 4)
    assert [len(s) for s in stages] == [2, 2, 2, 2]
    x = torch.randn(3, 8)
    h = x
    for s in stages:
        h = s(h)
    assert torch.equal(h, model(x))


def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                         nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 4))


def _run_pipeline(schedule, model_fn, loss_fn, x, y, S, M, steps):
    full = model_fn()
    stages_ref = split_sequential(copy.deepcopy(full), S)

    def fn(pg):
        stage_mod = split_sequential(copy.deepcopy(full), S)[pg.rank]
        opt = torch.optim.SGD(stage_mod.parameters(), lr=0.05)
        st = PipelineStage(stage_mod, pg, loss_fn=loss_fn, schedule=schedule)
        losses, grads = [], None
        for i in range(steps):
            opt.zero_grad()
            loss = st.train_step(x if pg.rank == 0 else None, y if pg.rank == S - 1 else None, num_micro=M)
            if i == 0:
                grads = [p.grad.clone() for p in stage_mod.parameters()]
            opt.step()
            losses.append(loss)
        return losses, grads, [p.detach().clone() for p in stage_mod.parameters()], st.max_inflight

    out = run_threads(S, fn)
    # single-process reference with identical micro-batching
    opt = torch.optim.SGD([p for s in stages_ref for p in s.parameters()], lr=0.05)
    ref_losses, ref_grads = [], None
    for i in range(steps):
        opt.zero_grad()
        ref_losses.append(reference_step(stages_ref, x, y, loss_fn, M))
        if i == 0:
            ref_grads = [[p.grad.clone() for p in s.parameters()] for s in stages_ref]
        opt.step()
    return out, stages_ref, ref_losses, ref_grads


@pytest.mark.parametrize("schedule", ["gpipe", "1f1b"])
@pytest.mark.parametrize("S,M", [(2, 4), (4, 4), (4, 8)])
def test_pipeline_matches_single_process(schedule, S, M):
    torch.manual_seed(1)
    x, y = torch.randn(32, 16), torch.randn(32, 4)
    out, stages_ref, ref_losses, ref_grads = _run_pipeline(schedule, _mlp, nn.functional.mse_loss, x, y, S, M, 5)
    last_losses = out[S - 1][0]
    assert last_losses == pytest.approx(ref_losses, rel=1e-6, abs=1e-7)
    for s in range(S):
        _, grads, params, inflight = out[s]
        for g, rg in zip(grads, ref_grads[s]):
            torch.testing.assert_close(g, rg, rtol=1e-6, atol=1e-7)
        for p, rp in zip(params, stages_ref[s].parameters()):
            torch.testing.assert_close(p, rp.detach(), rtol=1e-5, atol=1e-6)
        if schedule == "1f1b":
            assert inflight <= S - s
        else:
            assert inflight == M


def test_pipeline_bit_exact_1f1b_on_transformer():
    """1F1B accumulates micro-batch gradients in the same order (0..M-1) as the sequential
    reference, and activations cross the wire as raw bytes, so the result is bit-for-bit equal."""
    torch.manual_seed(3)
    x = torch.randint(0, 256, (8, 16))
    y = torch.randint(0, 256, (8, 16))

    def model_fn():
        torch.manual_seed(0)
        return make_charlm(vocab=256, d=32, heads=2, layers=2, ctx=16)
    out, stages_ref, ref_losses, ref_grads = _run_pipeline("1f1b", model_fn, lm_loss, x, y, 4, 4, 3)
    assert out[3][0] == ref_losses
    for s in range(4):
        for g, rg in zip(out[s][1], ref_grads[s]):
            assert torch.equal(g, rg)
