# benchmark/profile_sparsifier.py
"""Timing/memory profile + semantic fingerprint of the sparsification stack.

Profiles each algorithm component on the baseline artifact (196-10-10-10)
and records a "fingerprint" — the exact sequence of pruning decisions and
distances — so that optimizations can be verified to preserve semantics:

    python benchmark/profile_sparsifier.py --label before
    ... optimize ...
    python benchmark/profile_sparsifier.py --label after \
        --compare benchmark/profiles/before.json

Outputs benchmark/profiles/<label>.json with timings, peak memory, and
fingerprints.
"""
import argparse
import json
import os
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

OMEGA_SAMPLES = 10000
SEED          = 0
EXT_STEPS     = 30   # chained ext-search steps in the fingerprint
ADJUST_STEPS  = 3    # chained prune+adjust steps in the fingerprint


def rss_mb():
    with open('/proc/self/status') as f:
        for line in f:
            if line.startswith('VmRSS:'):
                return int(line.split()[1]) / 1024.0
    return float('nan')


class Profiled:
    """Context manager: wall time, tracemalloc peak, RSS delta."""

    def __init__(self, results, name):
        self.results, self.name = results, name

    def __enter__(self):
        self.rss0 = rss_mb()
        tracemalloc.start()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        elapsed = time.perf_counter() - self.t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.results[self.name] = {
            'wall_s':       round(elapsed, 4),
            'py_peak_mb':   round(peak / 1e6, 2),
            'rss_delta_mb': round(rss_mb() - self.rss0, 1),
        }
        print(f"  {self.name:30s} {elapsed:9.3f} s   "
              f"py-peak {peak / 1e6:8.1f} MB   rss-delta {rss_mb() - self.rss0:7.1f} MB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', required=True)
    ap.add_argument('--compare', help='previous profile JSON to diff fingerprints against')
    ap.add_argument('--skip-slow', action='store_true',
                    help='skip OBD/OBS/python-fallback (quick re-run)')
    args = ap.parse_args()

    from mlp.mlp import accuracy, load_network_params
    import sparsifier.sparsifier as ss
    from sparsifier.sparsifier import adjust, clone_network, d, prune
    from sparsifier.neuron_sparsifier import prune_neuron
    from sparsifier.magnitude_sparsifier import prune_magnitude
    from sparsifier.kwon_sparsifier import prune_kwon

    repo = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
    og_net = load_network_params(os.path.join(repo, 'artifacts', 'baseline'))
    x_test = np.genfromtxt(os.path.join(repo, 'data', 'X_test_small.csv'),
                           delimiter=',', max_rows=1000)
    y_test = np.genfromtxt(os.path.join(repo, 'data', 'Y_test_small.csv'),
                           delimiter=',', max_rows=1000)

    np.random.seed(SEED)
    omega = np.random.randint(0, 256, size=(OMEGA_SAMPLES, og_net[0].W.shape[1])).astype(np.float64)

    results, fingerprints = {}, {}
    print(f"prune_ext available: {ss._USE_EXT}")
    print(f"omega: {omega.shape}, net: {[l.W.shape for l in og_net]}")

    # Warm up JIT caches so timings measure steady-state, not compilation.
    float(d(og_net, og_net, omega))
    float(accuracy(og_net, x_test, y_test))

    # --- C++ ext candidate search, chained (search only, no adjust) ---
    net = clone_network(og_net)
    seq = []
    with Profiled(results, f'ext_search_x{EXT_STEPS}'):
        for _ in range(EXT_STEPS):
            net, meta = prune(net, og_net, omega, doAdjust=False)
            seq.append([meta.layer_idx, meta.i, meta.j, repr(meta.distance)])
    fingerprints['ext_search'] = seq

    # --- prune + adjust, chained ---
    net = clone_network(og_net)
    seq = []
    with Profiled(results, f'prune_adjust_x{ADJUST_STEPS}'):
        for _ in range(ADJUST_STEPS):
            net, meta = prune(net, og_net, omega, doAdjust=True)
            d_after = float(d(net, og_net, omega))
            seq.append([meta.layer_idx, meta.i, meta.j,
                        repr(meta.distance), repr(d_after)])
    fingerprints['prune_adjust'] = seq

    # --- adjust alone (same pruned input each time) ---
    pruned = clone_network(og_net)
    pruned[0].W[0, :], pruned[0].mask[0, :] = 0.0, 0.0
    with Profiled(results, 'adjust_single'):
        adj = adjust(pruned, og_net, omega)
    fingerprints['adjust_single'] = repr(float(d(adj, og_net, omega)))

    # --- neuron search ---
    net = clone_network(og_net)
    with Profiled(results, 'neuron_search_x3'):
        seq = []
        for _ in range(3):
            net, meta = prune_neuron(net, og_net, omega, doAdjust=False)
            seq.append([meta.layer_idx, meta.neuron_idx, repr(meta.distance)])
    fingerprints['neuron_search'] = seq

    # --- magnitude / kwon scoring (search only) ---
    with Profiled(results, 'magnitude_x3'):
        net, seq = clone_network(og_net), []
        for _ in range(3):
            net, meta = prune_magnitude(net, og_net, omega, doAdjust=False)
            seq.append([meta.layer_idx, meta.i, meta.j, repr(meta.magnitude)])
    fingerprints['magnitude'] = seq

    with Profiled(results, 'kwon_x3'):
        net, seq = clone_network(og_net), []
        for _ in range(3):
            net, meta = prune_kwon(net, og_net, omega, doAdjust=False)
            seq.append([meta.layer_idx, meta.i, meta.j, repr(meta.score)])
    fingerprints['kwon'] = seq

    if not args.skip_slow:
        from sparsifier.obd_sparsifier import prune_obd
        from sparsifier.obs_sparsifier import prune_obs

        try:
            with Profiled(results, 'obd_x1'):
                _, meta = prune_obd(clone_network(og_net), og_net, omega, doAdjust=False)
            fingerprints['obd'] = [meta.layer_idx, meta.i, meta.j, repr(meta.score)]
        except Exception as e:
            results['obd_x1'] = {'error': f'{type(e).__name__}: {e}'[:200]}
            print(f"  obd_x1 FAILED: {type(e).__name__}: {str(e)[:120]}")

        try:
            with Profiled(results, 'obs_x1'):
                _, meta = prune_obs(clone_network(og_net), og_net, omega)
            fingerprints['obs'] = [meta.layer_idx, meta.i, meta.j, repr(meta.score)]
        except Exception as e:
            results['obs_x1'] = {'error': f'{type(e).__name__}: {e}'[:200]}
            print(f"  obs_x1 FAILED: {type(e).__name__}: {str(e)[:120]}")

        # --- pure-Python fallback search (1 step) ---
        ss._USE_EXT = False
        net = clone_network(og_net)
        with Profiled(results, 'python_search_x1'):
            net, meta = prune(net, og_net, omega, doAdjust=False)
        ss._USE_EXT = True
        fingerprints['python_search'] = [meta.layer_idx, meta.i, meta.j,
                                         repr(meta.distance)]

    out_dir = os.path.join(repo, 'benchmark', 'profiles')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.label}.json')
    with open(out_path, 'w') as f:
        json.dump({'results': results, 'fingerprints': fingerprints}, f, indent=2)
    print(f"\nSaved {out_path}")

    if args.compare:
        with open(args.compare) as f:
            prev = json.load(f)
        print(f"\n=== fingerprint diff vs {args.compare} ===")
        ok = True
        for key, val in prev['fingerprints'].items():
            if key not in fingerprints:
                print(f"  {key:20s} SKIPPED in this run")
                continue
            match = fingerprints[key] == val
            ok &= match
            print(f"  {key:20s} {'IDENTICAL' if match else 'DIFFERS'}")
            if not match:
                print(f"    before: {val}")
                print(f"    after:  {fingerprints[key]}")
        print(f"\n=== timing diff ===")
        for key, val in prev['results'].items():
            if key in results:
                b, a = val['wall_s'], results[key]['wall_s']
                speedup = b / a if a > 0 else float('inf')
                print(f"  {key:30s} {b:9.3f} s -> {a:9.3f} s   ({speedup:5.1f}x)")
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
