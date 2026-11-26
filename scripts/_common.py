"""Shared helpers for the experiment scripts: machine label and JSON output."""
from __future__ import annotations

import datetime
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def host_busy() -> str:
    """How busy the host already was (other jobs), so contended numbers are labelled as such."""
    try:
        if sys.platform == "win32":
            cmd = "1..3 | % { (Get-CimInstance Win32_Processor | Measure-Object LoadPercentage -Average).Average; Start-Sleep 1 }"
            out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd],
                                 capture_output=True, text=True, timeout=60).stdout.split()
            vals = [float(v) for v in out]
            return f"total CPU busy {sum(vals) / len(vals):.0f}% just before this run (other jobs on the host)"
        with open("/proc/loadavg", encoding="utf-8") as f:
            return f"load average {f.read().split()[0]} just before this run"
    except (OSError, ValueError, subprocess.SubprocessError, ZeroDivisionError):
        return "unknown"


def machine() -> dict:
    """What produced a number.  CPU model is read from the OS, never typed in."""
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                cpu = next(line.split(":", 1)[1].strip() for line in f if line.startswith("model name"))
        elif sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except (OSError, StopIteration):
        pass
    return {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
            "cpu": cpu, "logical_cpus": os.cpu_count(),
            "date": datetime.date.today().isoformat(),
            "topology": "single host, worker PROCESSES over loopback TCP (not a cluster)",
            "host_load": host_busy()}


def save(name: str, obj: dict) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
    print(f"wrote {os.path.relpath(path, ROOT)}")
    return path


CRASHES: list[str] = []


def launch_retry(n, entry, args, tries=3, timeout=1800):
    """The benchmark host is shared and short of memory; a worker process occasionally died with a
    native error (0xC000070A), even with a single worker.  Retry, and record every failed attempt in
    the output JSON instead of hiding it."""
    from ringtrain.launch import launch_processes
    for _ in range(tries):
        try:
            return launch_processes(n, entry, args, timeout=timeout)
        except RuntimeError as e:
            CRASHES.append(f"N={n} {entry} {args.get('mode', '')} {args.get('compression', '')}: {e}")
            print("retrying after:", e, flush=True)
    raise RuntimeError(f"{tries} attempts failed")
