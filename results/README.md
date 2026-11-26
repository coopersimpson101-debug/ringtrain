# results/

Raw outputs of the scripts in `scripts/`, exactly as written by them (nothing edited by hand).
Each JSON file records the machine, date and host load it was measured under. Everything here was
produced on one Windows 10 desktop with worker processes over loopback TCP, **not** on a cluster.

| file | written by | figure |
|---|---|---|
| `fault_demo/run/w*.jsonl`, `coordinator.jsonl` | `scripts/fault_demo.py` (the faulty run) | `docs/assets/fault_demo.*` |
| `fault_demo/reference/` | same script, uninterrupted reference run | grey curve in `fault_demo.*` |
| `fault_demo/summary.json`, `demo_output.txt` | same script: injected events and narration | |
| `scaling.json` | `scripts/bench_scaling.py` | `docs/assets/scaling.*` |
| `allreduce.json` | `scripts/bench_scaling.py` | `docs/assets/allreduce.*` |
| `compression.json` | `scripts/compression_compare.py` | `docs/assets/compression.*` |
| `pipeline.json` | `scripts/pipeline_timeline.py` | `docs/assets/pipeline.*` |
| `dog2real/train_history.json` | `scripts/train_dog2real.py` | `docs/assets/dog2real.*` |

`python scripts/summarise_logs.py results/fault_demo/run/w*.jsonl` prints the per-generation table.
