# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
