"""Data-parallel PPO for dog2real policies (DD-PPO style, Wijmans et al. ICLR 2020).

Each rank runs its own batch of vectorised MuJoCo environments with a different seed and collects
its own rollout.  The policy, value function, observation normalisers and learning rate stay
identical on every rank:

* gradients of every PPO minibatch are averaged with :class:`ringtrain.ddp.DistributedDataParallel`
  (ring all-reduce, optionally compressed) before the optimizer step, so the update is the one a
  single learner would make on the union of all ranks' minibatches;
* the running mean/variance normalisers merge their batch statistics across ranks (Chan et al.'s
  parallel variance formula on all-reduced count / sum / sum of squares);
* the KL divergence that drives dog2real's adaptive learning-rate schedule is all-reduced, so every
  rank changes the learning rate at the same minibatch.

Everything else -- the actor-critic network, GAE, the clipped surrogate, value clipping, the
environment -- is dog2real's own code, imported unchanged.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
from torch import nn

from ..collectives import ProcessGroup
from ..compression import make_reducer
from ..ddp import DistributedDataParallel, broadcast_module
from ..seeding import seeded

try:  # dog2real is an optional dependency
    from dog2real.rl.networks import ActorCritic
    from dog2real.rl.normalizer import EmpiricalNormalization
    from dog2real.rl.ppo import PPO, gaussian_kl
except ImportError as e:  # pragma: no cover
    raise ImportError("dog2real is not installed: pip install 'dog2real @ git+https://github.com/"
                      "coopersimpson101-debug/dog2real'") from e


@torch.no_grad()
def distributed_normalizer_update(pg: ProcessGroup, norm: EmpiricalNormalization, x: torch.Tensor) -> None:
    """Merge this step's observations from every rank into the running statistics, identically on
    all ranks (equivalent to ``norm.update(concat of every rank's x)``)."""
    x = x.reshape(-1, x.shape[-1]).to(torch.float64)
    stats = np.concatenate([[float(x.shape[0])], x.sum(0).numpy(), (x * x).sum(0).numpy()])
    pg.allreduce(stats)
    d = x.shape[1]
    n = stats[0]
    if n == 0:
        return
    b_mean = torch.from_numpy(stats[1:1 + d] / n)
    b_var = torch.from_numpy(np.maximum(stats[1 + d:] / n - (stats[1:1 + d] / n) ** 2, 0.0))
    tot = norm.count.double() + n
    delta = b_mean - norm.mean.double()
    new_mean = norm.mean.double() + delta * n / tot
    m2 = norm.var.double() * norm.count.double() + b_var * n + delta.pow(2) * norm.count.double() * n / tot
    norm.mean.copy_(new_mean.float())
    norm.var.copy_((m2 / tot).float())
    norm.count.copy_(tot.float())


class DistributedPPOTrainer:
    def __init__(self, pg: ProcessGroup, env, cfg: dict, seed: int = 0, compression: str | None = None):
        self.pg, self.env, self.cfg = pg, env, cfg
        pcfg = cfg.get("policy", {})
        ppo_cfg = dict(cfg.get("ppo", {}))
        self.steps_per_env = int(ppo_cfg.pop("steps_per_env", 24))
        self.asymmetric = bool(pcfg.get("asymmetric_critic", True))
        critic_dim = env.num_privileged_obs if self.asymmetric else env.num_obs
        with seeded(seed):  # same init everywhere (thread-safe); broadcast below makes it certain
            self.policy = ActorCritic(env.num_obs, critic_dim, env.num_actions,
                                      pcfg.get("actor_hidden", [256, 128, 64]),
                                      pcfg.get("critic_hidden", [256, 128, 64]),
                                      pcfg.get("activation", "elu"), pcfg.get("init_noise_std", 1.0))
        self.obs_norm = EmpiricalNormalization(env.num_obs)
        self.critic_norm = EmpiricalNormalization(critic_dim)
        self.alg = PPO(self.policy, seed=seed + 7919 * pg.rank, **ppo_cfg)
        self.alg.init_storage(self.steps_per_env, env.num_envs, env.num_obs, critic_dim, env.num_actions)
        broadcast_module(pg, self.policy)
        self.ddp = DistributedDataParallel(self.policy, pg, make_reducer(compression), bucket_mb=0.25,
                                           average=True, broadcast_init=False)
        self.iteration = 0
        self.history: list[dict] = []

    @staticmethod
    def _t(x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=torch.float32)

    def _update(self) -> dict:
        alg, pg = self.alg, self.pg
        sums = {"surrogate": 0.0, "value": 0.0, "entropy": 0.0, "kl": 0.0}
        n = 0
        for (obs, critic_obs, actions, old_logp, old_mu, old_sigma, old_v, adv, ret) in \
                alg.storage.minibatches(alg.num_minibatches, alg.num_epochs, alg.generator):
            dist = alg.policy.distribution(obs)
            logp = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1).mean()
            value = alg.policy.evaluate(critic_obs)
            with torch.no_grad():
                kl_local = gaussian_kl(old_mu, old_sigma, dist.mean, dist.stddev).mean()
            kl_arr = np.array([float(kl_local)])
            pg.allreduce(kl_arr)
            kl = kl_arr[0] / pg.world
            if alg.schedule == "adaptive" and alg.desired_kl > 0:
                if kl > 2.0 * alg.desired_kl:
                    alg._set_lr(max(alg.min_lr, alg.learning_rate / 1.5))
                elif 0.0 < kl < alg.desired_kl / 2.0:
                    alg._set_lr(min(alg.max_lr, alg.learning_rate * 1.5))
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            ratio = torch.exp(logp - old_logp)
            surr = -torch.min(ratio * adv, torch.clamp(ratio, 1 - alg.clip_param, 1 + alg.clip_param) * adv).mean()
            if alg.value_clip:
                v_clipped = old_v + (value - old_v).clamp(-alg.clip_param, alg.clip_param)
                value_loss = torch.max((value - ret).pow(2), (v_clipped - ret).pow(2)).mean()
            else:
                value_loss = (value - ret).pow(2).mean()
            loss = surr + alg.value_loss_coef * value_loss - alg.entropy_coef * entropy
            alg.optimizer.zero_grad()
            loss.backward()
            self.ddp.finish()  # ring all-reduce (average) of every gradient bucket
            nn.utils.clip_grad_norm_(alg.policy.parameters(), alg.max_grad_norm)
            alg.optimizer.step()
            sums["surrogate"] += float(surr.detach())
            sums["value"] += float(value_loss.detach())
            sums["entropy"] += float(entropy.detach())
            sums["kl"] += kl
            n += 1
        alg.storage.clear()
        out = {f"loss/{k}" if k != "kl" else "policy/kl": v / max(n, 1) for k, v in sums.items()}
        out["policy/lr"] = alg.learning_rate
        out["policy/noise_std"] = float(alg.policy.log_std.detach().exp().mean())
        return out

    def learn(self, iterations: int, log=None) -> list[dict]:
        env, pg = self.env, self.pg
        obs, priv = env.reset()
        critic = priv if self.asymmetric else obs
        n = env.num_envs
        ep_ret = np.zeros(n)
        for _ in range(iterations):
            t0 = time.time()
            step_rewards, finished = [], []
            self.policy.eval()
            for _ in range(self.steps_per_env):
                o, c = self._t(obs), self._t(critic)
                distributed_normalizer_update(pg, self.obs_norm, o)
                distributed_normalizer_update(pg, self.critic_norm, c)
                a = self.alg.act(self.obs_norm(o), self.critic_norm(c))
                obs, rew, done, info = env.step(a.numpy())
                critic = info["privileged_obs"] if self.asymmetric else obs
                self.alg.process_env_step(self._t(rew), self._t(done), self._t(info["time_outs"]))
                step_rewards.append(float(np.mean(rew)))
                ep_ret += rew
                finished += list(ep_ret[np.asarray(done, dtype=bool)])
                ep_ret[np.asarray(done, dtype=bool)] = 0.0
            collect_s = time.time() - t0
            self.alg.compute_returns(self.critic_norm(self._t(critic)))
            self.policy.train()
            t1 = time.time()
            stats = self._update()
            learn_s = time.time() - t1
            # global metrics: mean step reward over all ranks, finished-episode returns
            m = np.array([float(np.mean(step_rewards)), float(np.sum(finished)), float(len(finished)),
                          collect_s, learn_s])
            pg.allreduce(m)
            self.iteration += 1
            stats.update({"iteration": self.iteration, "world": pg.world,
                          "train/mean_step_reward": m[0] / pg.world,
                          "train/mean_return": (m[1] / m[2]) if m[2] else None,
                          "train/episodes": int(m[2]),
                          "perf/env_steps": self.steps_per_env * n * pg.world,
                          "perf/collect_s": m[3] / pg.world, "perf/learn_s": m[4] / pg.world,
                          "perf/fps": self.steps_per_env * n * pg.world / (time.time() - t0)})
            self.history.append(stats)
            if log is not None:
                log(stats)
        return self.history

    def checkpoint(self) -> dict:
        """The same keys as dog2real's ``OnPolicyRunner.save`` so ``scripts/play.py`` can load it."""
        return {"policy": self.policy.state_dict(), "obs_norm": self.obs_norm.state_dict(),
                "critic_norm": self.critic_norm.state_dict(), "optimizer": self.alg.optimizer.state_dict(),
                "learning_rate": self.alg.learning_rate, "iteration": self.iteration, "cfg": self.cfg,
                "dims": {"obs": self.env.num_obs, "critic": int(self.critic_norm.mean.numel()),
                         "act": self.env.num_actions}}

    def param_digest(self) -> float:
        return float(sum(p.detach().double().abs().sum() for p in self.policy.parameters()))


# ----------------------------------------------------------------- entry point
def load_dog2real_config(root: str, name: str = "train_walk.yaml", overrides: dict | None = None) -> dict:
    """Load a dog2real training config from a dog2real checkout at ``root`` (configs/ is not part
    of the installed package), following ``extends:`` and inlining the robot description."""
    import os

    from dog2real.config import deep_merge, load_yaml
    cfg_dir = os.path.join(root, "configs")
    cfg = load_yaml(os.path.join(cfg_dir, name))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    if isinstance(cfg.get("robot", "robot.yaml"), str):
        cfg["robot"] = load_yaml(os.path.join(cfg_dir, cfg.get("robot", "robot.yaml")))
    return cfg


def run_entry(pg: ProcessGroup, args: dict) -> dict:
    """``launch_processes`` entry: train a dog2real walking policy with N data-parallel workers.

    args: dog2real_root, config, iterations, num_envs (per worker), seed, compression,
          store (``dir:...`` / ``raftstore:...``), run, out (local .pt path for rank 0).
    """
    import os

    from dog2real.sim.env import VecQuadrupedEnv

    from ..checkpoint import CheckpointStore, make_backend
    overrides = {"env": {"num_envs": int(args.get("num_envs", 16))}}
    overrides.update(args.get("overrides", {}))
    cfg = load_dog2real_config(args["dog2real_root"], args.get("config", "train_walk.yaml"), overrides)
    seed = int(args.get("seed", cfg.get("seed", 1)))
    env = VecQuadrupedEnv(cfg, seed=seed + 1000 * pg.rank)  # different rollouts on every rank
    tr = DistributedPPOTrainer(pg, env, cfg, seed=seed, compression=args.get("compression"))
    t0 = time.time()
    log_rows = []

    def log(st):
        row = {k: st[k] for k in ("iteration", "world", "train/mean_step_reward", "train/mean_return",
                                  "train/episodes", "policy/kl", "policy/lr", "policy/noise_std",
                                  "loss/value", "perf/fps", "perf/env_steps")}
        row["wall_s"] = time.time() - t0
        log_rows.append(row)
        if pg.rank == 0:
            print(json.dumps(row), flush=True)
    tr.learn(int(args.get("iterations", 10)), log)
    digests = pg.all_gather_json(tr.param_digest())
    ck = tr.checkpoint()
    if pg.rank == 0:
        if args.get("store"):
            CheckpointStore(make_backend(args["store"]), args.get("run", "dog2real")).save(tr.iteration, ck)
        if args.get("out"):
            os.makedirs(os.path.dirname(os.path.abspath(args["out"])), exist_ok=True)
            torch.save(ck, args["out"])
    return {"rank": pg.rank, "history": log_rows if pg.rank == 0 else [], "param_digests": digests,
            "wall_s": time.time() - t0}
