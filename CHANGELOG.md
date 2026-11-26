# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Models built by several ranks at once in one process (the threaded test harness) could get a
  corrupted rank-0 initialisation: `torch.manual_seed` + construction raced on torch's global
  generator. Construction now goes through `ringtrain.seeding.seeded` (lock + `fork_rng`), which
  gives bit-identical parameters to the old single-threaded path. This was the Linux/macOS CI failure
  in `tests/test_ddp.py`.
- Worker processes no longer run interpreter finalisation after a finished run (`os._exit`), the DDP
  comm thread is joined on close, and `faulthandler` is enabled in workers; a worker had exited with
  SIGABRT ("terminate called without an active exception") on Linux CI.

### Added
- `--pause-file` for workers: hold at a step boundary while the file exists (optionally from step N).
  The elastic tests use it so kill / re-join no longer depends on process start-up time.

## [0.1.0] - 2026-11-26

### Added
- Framed TCP transport (threaded, full mesh per generation) with heartbeats and failure detection.
- Rendezvous coordinator: generations, rank assignment, abort on member death, graceful scale-up.
- Collectives: ring all-reduce (reduce-scatter + all-gather, segmented, async sends), pipelined ring
  broadcast, dissemination barrier, ring all-gather, point-to-point arrays; naive baselines.
- Data parallelism: DDP-style gradient bucketing with communication/computation overlap.
- Pipeline parallelism: GPipe and 1F1B schedules, stage splitting, single-process reference.
- Gradient compression: top-k with error feedback, PowerSGD (rank-r, warm start, error feedback).
- Checkpoints (model, optimiser, RNG, data cursor) in raftstore over its HTTP API, or a local
  directory; async upload, SHA-256-verified `LATEST` pointer, retention.
- Elastic trainer: survive a killed worker (roll back to the last checkpoint), re-join and scale up.
- Adapters: synthetic teacher-student MLP, a byte-level transformer LM, a tinylm plug-in point;
  dog2real DD-PPO integration.
- Scripts and results: fault-injection demo, 1-4 process scaling, all-reduce bandwidth,
  compression comparison, pipeline timelines, dog2real training run.

[0.1.0]: https://github.com/coopersimpson101-debug/ringtrain/releases/tag/v0.1.0
