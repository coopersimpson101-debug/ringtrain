"""dog2real integration: data-parallel PPO with dog2real's own networks, GAE and PPO loss."""
import os

import pytest
import torch

pytest.importorskip("dog2real", reason="dog2real not installed")
from dog2real.rl.normalizer import EmpiricalNormalization  # noqa: E402
from dog2real.rl.toy_env import PointMassEnv  # noqa: E402

from ringtrain.integrations.dog2real_ppo import DistributedPPOTrainer, distributed_normalizer_update  # noqa: E402
from ringtrain.launch import run_threads  # noqa: E402

torch.set_num_threads(1)
CFG = {"policy": {"actor_hidden": [32, 32], "critic_hidden": [32, 32], "init_noise_std": 0.5},
       "ppo": {"steps_per_env": 16, "num_epochs": 4, "num_minibatches": 2, "learning_rate": 3e-3}}


def test_distributed_normalizer_equals_single_update_on_concatenation():
    xs = [torch.randn(50, 3) * (r + 1) + r for r in range(3)]
    ref = EmpiricalNormalization(3)
    ref.update(torch.randn(10, 3))
    base = {k: v.clone() for k, v in ref.state_dict().items()}
    ref.update(torch.cat(xs))

    def fn(pg):
        n = EmpiricalNormalization(3)
        n.load_state_dict(base)
        distributed_normalizer_update(pg, n, xs[pg.rank])
        return n.mean.clone(), n.var.clone(), float(n.count)
    for mean, var, count in run_threads(3, fn):
        torch.testing.assert_close(mean, ref.mean, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(var, ref.var, rtol=1e-4, atol=1e-5)
        assert count == float(ref.count)


@pytest.mark.parametrize("compression", [None, "powersgd:2"])
def test_dd_ppo_keeps_ranks_identical_and_learns(compression):
    def fn(pg):
        env = PointMassEnv(num_envs=32, seed=10 + pg.rank)
        tr = DistributedPPOTrainer(pg, env, CFG, seed=0, compression=compression)
        hist = tr.learn(25)
        return tr.param_digest(), [h["train/mean_step_reward"] for h in hist], float(tr.alg.learning_rate)
    out = run_threads(2, fn)
    assert out[0][0] == out[1][0]  # bit-identical policies on every rank
    assert out[0][2] == out[1][2]  # adaptive LR moved in lock-step
    rewards = out[0][1]
    assert sum(rewards[-5:]) / 5 > sum(rewards[:5]) / 5 + 0.1  # it learns to reach the origin


DOG2REAL_ROOT = os.environ.get("DOG2REAL_ROOT")


@pytest.mark.skipif(not DOG2REAL_ROOT, reason="set DOG2REAL_ROOT to a dog2real checkout (configs/) "
                    "to run the MuJoCo quadruped smoke test")
def test_quadruped_two_iterations():
    pytest.importorskip("mujoco")
    from ringtrain.integrations.dog2real_ppo import run_entry
    args = {"dog2real_root": DOG2REAL_ROOT, "config": "smoke_test.yaml", "iterations": 2, "num_envs": 4,
            "overrides": {"policy": {"actor_hidden": [32, 32], "critic_hidden": [32, 32]}}}
    out = run_threads(2, lambda pg: run_entry(pg, args))
    assert out[0]["param_digests"][0] == out[0]["param_digests"][1]
    assert len(out[0]["history"]) == 2
