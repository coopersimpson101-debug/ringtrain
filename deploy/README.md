# Deploying ringtrain on the home cluster ⏳

Nothing in this directory has been run on the real machines yet. Every command below is what the
author should run; the results go in the empty tables in the main README
([Hardware validation](../README.md#hardware-validation--todo-for-the-author)).

| host | address (placeholder) | role |
|---|---|---|
| `nas` | 10.0.0.10 | raftstore node, ringtrain worker |
| `pi5` | 10.0.0.11 | raftstore node, ringtrain worker (aarch64, 4 cores) |
| `server` | 10.0.0.12 | raftstore node, **coordinator**, ringtrain worker |
| `m1` | 10.0.0.13 | ringtrain worker (macOS, arm64) |

The checkpoint store is the raftstore cluster from the sibling repo, using its own
`deploy/config/{nas,pi5,server}.json`. The coordinator holds no training state; if `server` dies the
workers keep their checkpoints in raftstore (2 of 3 nodes still form a majority), but a new
coordinator must be started before they can re-form (see the roadmap).

## 1. Install (every worker)

```sh
python3 -m venv ~/ringtrain-venv && . ~/ringtrain-venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # macOS: pip install torch
pip install "ringtrain @ git+https://github.com/coopersimpson101-debug/ringtrain"
cp deploy/cluster.env /etc/ringtrain.env   # edit RINGTRAIN_ID / RINGTRAIN_ADVERTISE per host
```

Open TCP 29400 (coordinator, on `server`) and 29500 (worker peer port) between the four hosts only.
The protocol is unauthenticated plain TCP: keep it on the LAN or a WireGuard tunnel (see SECURITY.md).

## 2. Start

```sh
# server: coordinator (systemd unit in deploy/systemd/)
python -m ringtrain coordinator --host 0.0.0.0 --port 29400 --min-workers 1 --max-workers 4 \
    --initial-workers 4 --events /var/log/ringtrain/coordinator.jsonl

# every worker (systemd on Linux, deploy/launchd/ on the M1)
python -m ringtrain worker --coordinator 10.0.0.12:29400 --id "$(hostname)" \
    --host 0.0.0.0 --port 29500 --advertise 10.0.0.1X \
    --store raftstore:http://10.0.0.10:8080,http://10.0.0.11:8080,http://10.0.0.12:8080 \
    --run mlp-scaling --adapter mlp --steps 2000 --ckpt-every 50 --log /var/log/ringtrain/worker.jsonl
```

`Restart=always` in the unit is what makes a crashed or rebooted worker *re-join* on its own.

## 3. Experiments to run (fill the README tables)

1. **Scaling across machines** -- for N in 1 2 3 4, start N workers with `--steps 200` and a fresh
   `--run scale-N`; record samples/s from the step logs (`python scripts/summarise_logs.py LOGS...`).
   Try `--compression powersgd:4` and `topk:0.01` at N = 4: on gigabit Ethernet / Wi-Fi the
   all-reduce will dominate, which is exactly where compression should pay off.
2. **Pull the plug** -- during a 2000-step run, power off the Pi 5 (not a clean shutdown). Record
   the detection time (the hb timeout, 8 s by default, applies here since no RST is sent), the
   rollback in steps, and when the remaining three resume. Then power it back on and record the
   re-join.
3. **Kill a raftstore node** during checkpointing; the run must continue.
4. **dog2real** -- `python scripts/train_dog2real.py` is single-host; for the cluster, run
   `ringtrain.launch` workers per host (see `docs/adapters.md`).

Heterogeneous hosts: the global batch is split evenly, so the slowest worker (the Pi 5) sets the
pace. Measure it; uneven sharding is on the roadmap.
