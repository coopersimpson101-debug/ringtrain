# Adapters: plugging a workload into ringtrain

ringtrain separates *how* to train in parallel (transport, collectives, DDP, pipeline, elasticity,
checkpoints) from *what* to train. A workload is an **adapter**: five members, no base class needed.

```python
class MyAdapter:
    name = "my-model"
    num_samples = 1_000_000      # dataset positions; the global sampler permutes 0..num_samples-1
    global_batch = 64            # samples per optimizer step, fixed for the whole run

    def build_model(self, seed): ...            # -> nn.Module, deterministic for a seed
    def build_optimizer(self, model): ...       # -> torch.optim.Optimizer
    def batch(self, indices): ...               # samples at these global indices (numpy int64 array)
    def loss_sum(self, model, batch): ...       # -> (SUM of per-sample losses as a Tensor, count)
```

Why a *sum*: the elastic trainer divides by the global batch size, not the local one, and the
ring all-reduce sums. That makes the gradient identical whether the batch is split over 4 workers
or 3 survivors, which is what lets a run continue after a node dies without changing the maths.
The data cursor is the step number (`ringtrain/data.py`), so `batch()` must be a pure function of
the indices.

Run it with any worker:

```sh
python -m ringtrain worker --adapter my_pkg.my_module:MyAdapter --adapter-args '{"lr": 3e-4}' ...
```

`--adapter module:factory` imports `module` and calls `factory(**adapter_args)`.

## Built-in adapters

| name | what | used by |
|---|---|---|
| `mlp` | teacher-student classification on Gaussian inputs (`ringtrain/adapters/mlp.py`) | tests, scaling, fault demo, compression |
| `charlm` | byte-level causal transformer on ringtrain's own source (`ringtrain/adapters/charlm.py`) | pipeline tests and timelines; the LLM stand-in |
| `tinylm` | plug-in point for the separate tinylm project (`ringtrain/adapters/tinylm.py`) | ⏳ not yet connected |

## tinylm ⏳

tinylm (the LLM trainer) is developed separately and ringtrain does not depend on it. The
integration is designed but **not yet run**:

1. tinylm adds `tinylm/ringtrain_adapter.py` with a `make_adapter(**kw)` that returns an object
   with the five members above. `batch(indices)` maps each index to a fixed token window of the
   tokenised corpus and returns `(x, y)`; `loss_sum` returns the sum over sequences of the mean token
   cross-entropy (exactly what `ringtrain.adapters.charlm.lm_loss(..., "sum")` computes).
2. Data-parallel training then needs nothing else:
   `python -m ringtrain worker --adapter tinylm --adapter-args '{"config": "small.yaml"}' ...`
3. For pipeline parallelism, `build_model` should return an `nn.Sequential` of
   `[embedding, block_0 ... block_{L-1}, final norm + head]`, like `make_charlm`. Then
   `ringtrain.pipeline.split_sequential(model, num_stages)` balances it by parameter count and
   `PipelineStage(stage, pg, loss_fn=lm_loss, schedule="1f1b")` trains it (see
   `ringtrain.bench.pipeline_bench` for a complete loop). Tied input/output embeddings across
   stages are not supported yet (roadmap).
4. A learning-rate schedule should be a function of the step number restored from the checkpoint
   (`state["step"]`), not of wall time or of how many steps this process has run.

## dog2real

`ringtrain/integrations/dog2real_ppo.py` is not an adapter: on-policy RL collects its own data, so
it has its own loop (DD-PPO). It imports dog2real's `ActorCritic`, `RolloutStorage`, `PPO`
hyper-parameters and `VecQuadrupedEnv` unchanged and only adds the synchronisation. Run it with
`scripts/train_dog2real.py`.
