# benchmark/correctness_check.py
"""
Correctness harness: compare exhaustive prune() against gradient-guided top-k.

Usage:
    python benchmark/correctness_check.py <net_folder> <x_test.csv> <y_test.csv> [n_steps]
"""
import sys, os, time
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from mlp.mlp import accuracy, load_network_params
from sparsifier.sparsifier import (
    clone_network, d, d_grad, make_omega, adjust,
    prune, PruneMeta,
)


def prune_gradient_topk(net, og_net, omega, top_k, doAdjust=False):
    """Gradient-guided top-k pruning. Returns (net, PruneMeta).

    One gradient call ranks all active weights by first-order importance
    |grad_W[i,j] * W[i,j]|. Only the top_k least important are evaluated exactly.
    """
    t0 = time.perf_counter()

    gradients = d_grad(net, og_net, omega)

    candidates = []
    for idx, (layer, glayer) in enumerate(zip(net, gradients)):
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                if layer.W[i, j] == 0.:
                    continue
                # First-order Taylor: zeroing W[i,j] changes d by ≈ -W[i,j]*grad[i,j]
                predicted_delta = -float(layer.W[i, j]) * float(glayer.W[i, j])
                candidates.append((predicted_delta, idx, i, j))

    candidates.sort(key=lambda x: x[0])

    minimo     = 1e16
    minimo_idx = candidates[0][1]
    minimo_i   = candidates[0][2]
    minimo_j   = candidates[0][3]

    probe_net = clone_network(net)
    for _, idx, i, j in candidates[:top_k]:
        saved_W    = probe_net[idx].W[i, j]
        saved_mask = probe_net[idx].mask[i, j]
        probe_net[idx].W[i, j]    = 0.
        probe_net[idx].mask[i, j] = 0.
        distanza = d(og_net, probe_net, omega)
        probe_net[idx].W[i, j]    = saved_W
        probe_net[idx].mask[i, j] = saved_mask
        if distanza < minimo:
            minimo     = distanza
            minimo_idx = idx
            minimo_i   = i
            minimo_j   = j
            if minimo == 0:
                break

    t_prune = time.perf_counter() - t0

    probe_net[minimo_idx].W[minimo_i, minimo_j]    = 0.
    probe_net[minimo_idx].mask[minimo_i, minimo_j] = 0.

    t_adj_start = time.perf_counter()
    if doAdjust and minimo > 0:
        probe_net = adjust(probe_net, og_net, omega)
    t_adj = time.perf_counter() - t_adj_start

    meta = PruneMeta(
        layer_idx    = minimo_idx,
        i            = minimo_i,
        j            = minimo_j,
        distanza     = float(minimo),
        prune_time_s = t_prune,
        adjust_time_s= t_adj,
    )
    return probe_net, meta


def compare_one_step(net, og_net, omega, x_test, y_test, top_k_values=(1, 5, 10, 50)):
    """
    Run exhaustive and gradient-top-k on the same network state.

    Returns dict keyed by strategy name:
      'exhaustive'      -> {candidate, time_s, d_after, acc_after}
      'gradient_top_k'  -> {candidate, time_s, d_after, acc_after, candidate_match, speedup}
    """
    results = {}

    # --- Exhaustive ---
    t0 = time.perf_counter()
    net_ex, meta_ex = prune(net, og_net, omega, doAdjust=False)
    t_ex = time.perf_counter() - t0
    results['exhaustive'] = {
        'candidate': (meta_ex.layer_idx, meta_ex.i, meta_ex.j),
        'time_s':    t_ex,
        'd_after':   float(d(net_ex, og_net, omega)),
        'acc_after': float(accuracy(net_ex, x_test, y_test)),
    }

    # --- Gradient top-k variants ---
    for k in top_k_values:
        t0 = time.perf_counter()
        net_k, meta_k = prune_gradient_topk(net, og_net, omega, top_k=k, doAdjust=False)
        t_k = time.perf_counter() - t0
        match = (
            meta_k.layer_idx == meta_ex.layer_idx and
            meta_k.i         == meta_ex.i         and
            meta_k.j         == meta_ex.j
        )
        results[f'gradient_top_{k}'] = {
            'candidate':       (meta_k.layer_idx, meta_k.i, meta_k.j),
            'time_s':          t_k,
            'd_after':         float(d(net_k, og_net, omega)),
            'acc_after':       float(accuracy(net_k, x_test, y_test)),
            'candidate_match': match,
            'speedup':         t_ex / max(t_k, 1e-9),
        }

    return results


def run_correctness_report(net_folder, x_test_path, y_test_path, n_steps=20,
                            top_k_values=(1, 5, 10, 50), omega_samples=500):
    """
    Run n_steps of exhaustive + gradient-top-k on the same network.
    Prints per-step agreement table and aggregate speedup summary.
    """
    og_net = load_network_params(net_folder)
    x_test = np.genfromtxt(x_test_path, delimiter=',', max_rows=500)
    y_test = np.genfromtxt(y_test_path, delimiter=',', max_rows=500)
    omega  = make_omega(og_net, n_samples=omega_samples)

    net = clone_network(og_net)

    print(f"\n{'Step':>4}  {'Ex(s)':>7}  " +
          "  ".join(f"top{k}(s)  match  spdup" for k in top_k_values))
    print("-" * (16 + 24 * len(top_k_values)))

    aggregate = {k: {'match': 0, 'speedup': []} for k in top_k_values}

    for step in range(n_steps):
        results = compare_one_step(net, og_net, omega, x_test, y_test,
                                   top_k_values=top_k_values)
        ex = results['exhaustive']
        row = f"{step:>4}  {ex['time_s']:>7.3f}"
        for k in top_k_values:
            r = results[f'gradient_top_{k}']
            match_str = "YES" if r['candidate_match'] else "NO "
            row += f"  {r['time_s']:>7.3f}    {match_str}  {r['speedup']:>5.1f}x"
            if r['candidate_match']:
                aggregate[k]['match'] += 1
            aggregate[k]['speedup'].append(r['speedup'])
        print(row)
        # advance net one exhaustive step to test across different sparsity levels
        net, _ = prune(net, og_net, omega, doAdjust=False)

    print(f"\n--- Aggregate over {n_steps} steps ---")
    for k in top_k_values:
        ag = aggregate[k]
        avg_sp = sum(ag['speedup']) / len(ag['speedup'])
        print(f"  top-{k:>2}: match={ag['match']}/{n_steps}  avg_speedup={avg_sp:.1f}x")


if __name__ == '__main__':
    args = sys.argv
    run_correctness_report(
        net_folder  = os.path.abspath(args[1]),
        x_test_path = os.path.abspath(args[2]),
        y_test_path = os.path.abspath(args[3]),
        n_steps     = int(args[4]) if len(args) > 4 else 20,
    )
