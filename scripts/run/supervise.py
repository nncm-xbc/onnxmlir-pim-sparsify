#!/usr/bin/env python3
"""Watchdog runner: launch sparsifier experiments serially, detect stalls, and
auto-restart from the latest checkpoint.

The failure this guards against (see docs/handover.md, e16_kwon_full): a single
JAX/CUDA call wedges on a contended 8GB GPU and never returns — the process
stays "running" for days producing no output. No internal iteration cap can fix
that, so we watch progress externally: if no file under the experiment's
artifact dir changes for STALL_TIMEOUT seconds, we SIGKILL the process group and
relaunch. runner.py then auto-resumes from its newest checkpoint, so at most
``checkpoint_every`` steps of work are lost.

Usage:
    python scripts/run/supervise.py QUEUE.txt
where each non-comment line of QUEUE.txt is "MODULE CONFIG", e.g.
    sparsifier.kwon_sparsifier experiments/e16_kwon_full.json
    sparsifier.sparsifier      experiments/baseline.json

Env knobs:
    STALL_TIMEOUT  seconds of no artifact progress before kill (default 600)
    MAX_RESTARTS   relaunch attempts per experiment after the first (default 5)
    POLL           progress-poll interval in seconds (default 10)
    RESTART_SLEEP  pause between a kill and relaunch (default 15)
JAX_PLATFORMS and everything else in the environment pass through to the child.
"""
import json
import os
import signal
import subprocess
import sys
import time

STALL_TIMEOUT = float(os.environ.get('STALL_TIMEOUT', 600))
MAX_RESTARTS  = int(os.environ.get('MAX_RESTARTS', 5))
POLL          = float(os.environ.get('POLL', 10))
RESTART_SLEEP = float(os.environ.get('RESTART_SLEEP', 15))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def log(msg):
    print("[supervise %s] %s" % (time.strftime('%H:%M:%S'), msg), flush=True)


def newest_mtime(path):
    """Most recent mtime of any file under ``path`` (0.0 if none)."""
    m = 0.0
    for root, _, files in os.walk(path):
        for fn in files:
            try:
                m = max(m, os.path.getmtime(os.path.join(root, fn)))
            except OSError:
                pass
    return m


def run_one(module, config):
    name = json.load(open(os.path.join(REPO, config)))['name']
    exp_dir = os.path.join(REPO, 'artifacts', name)
    os.makedirs(exp_dir, exist_ok=True)

    for attempt in range(MAX_RESTARTS + 1):
        log("launch %s %s (attempt %d/%d)" % (module, config, attempt, MAX_RESTARTS))
        p = subprocess.Popen(
            [sys.executable, '-u', '-m', module, config],
            cwd=REPO, preexec_fn=os.setsid,
        )
        last_seen = newest_mtime(exp_dir)
        last_progress = time.time()
        killed = False
        while True:
            try:
                rc = p.wait(timeout=POLL)
            except subprocess.TimeoutExpired:
                rc = None
            if rc is not None:
                if rc == 0:
                    log("DONE %s rc=0" % config)
                    return True
                log("EXIT %s rc=%d -- will resume from checkpoint" % (config, rc))
                break
            m = newest_mtime(exp_dir)
            if m > last_seen:
                last_seen, last_progress = m, time.time()
            elif time.time() - last_progress > STALL_TIMEOUT:
                log("STALL %s (no progress for %.0fs) -- SIGKILL, will resume"
                    % (config, STALL_TIMEOUT))
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                try:
                    p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    pass
                killed = True
                break
        # ponytail: fixed retry cap. If the GPU *driver* is wedged (not just the
        # context), a bare relaunch will re-hang; giving up after MAX_RESTARTS
        # avoids an infinite kill/relaunch loop. Upgrade path: `nvidia-smi
        # --gpu-reset` between attempts if that ever becomes the common case.
        if killed:
            time.sleep(RESTART_SLEEP)
    log("GAVE UP on %s after %d restarts" % (config, MAX_RESTARTS))
    return False


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with open(sys.argv[1]) as f:
        queue = [ln.split() for ln in f
                 if ln.strip() and not ln.lstrip().startswith('#')]
    results = {}
    for module, config in queue:
        results[config] = run_one(module, config)
    log("QUEUE COMPLETE: " + ", ".join(
        "%s=%s" % (c, 'ok' if ok else 'FAILED') for c, ok in results.items()))
    sys.exit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()
