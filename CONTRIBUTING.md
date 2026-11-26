# Contributing to ringtrain

Thanks for your interest! ringtrain exists to show distributed training *written out by hand*,
so contributions should keep the core readable and dependency-light.

## Ground rules

- **No communication libraries.** PyTorch is used for tensors and autograd only; never
  `torch.distributed`, MPI, NCCL, gloo or Horovod. Every byte on the wire goes through `ringtrain/wire.py`.
- **Every collective is tested against a naive reference** (exact equality on integer-valued data),
  and every parallelism mode against single-process training.
- **Honest numbers.** Figures and tables come from `results/*.json` written by the scripts in
  `scripts/`, with the machine and date recorded. Single-host runs are labelled as such.

## Development setup

```sh
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test,plots,raftstore]" pyflakes
python -m pyflakes ringtrain tests scripts
python -m pytest -q -m "not slow and not cluster"     # threads over loopback TCP
python -m pytest -q -m "slow or cluster"              # real processes + a 3-node raftstore cluster
```

`DOG2REAL_ROOT=/path/to/dog2real python -m pytest tests/test_dog2real.py` also runs the MuJoCo
quadruped smoke test (needs `pip install -e ".[dog2real]"`).

## Pull requests

1. Open an issue first for protocol changes (frame format, rendezvous messages, checkpoint layout).
2. Keep PRs focused, with a test that fails without the change.
3. CI (lint, the test suites, the fault demo) must pass.

By contributing you agree that your contributions are licensed under the MIT License.
