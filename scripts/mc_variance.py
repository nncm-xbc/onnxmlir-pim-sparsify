# mc_variance.py — Monte-Carlo estimator characterisation for the manifold
# distance d(net, og_net, Ω).
#
# Question (docs/todo.md §1 "MC-estimator characterisation"): how large must the
# Ω sample size B be for the sparsifier's per-step decision to be stable? The
# distance d is a Monte-Carlo estimate over a random Ω draw, so both its value
# for a fixed candidate and the *ranking* of candidates it induces are noisy.
#
# Design: take the trained baseline net. Build a small FIXED set of K candidate
# nets, each the baseline with exactly one prunable weight zeroed (the first K
# prunable weights, layer-major order — the same order prune() scans). For each
# B, draw `num_samples` INDEPENDENT Ω samples of size B; for every (B, sample)
# compute d(candidate, baseline, Ω) for all K candidates and rank them (rank 0 =
# smallest d = the weight prune() would remove). Downstream stats
# (var(d), rank-1 candidate, top-1 stability, pairwise Spearman between two
# independent draws, decision-agreement rate) are all derivable from this table.
#
# Output: artifacts/mc_variance/results.csv with columns EXACTLY
#     B,candidate_idx,sample_idx,d,rank
# consumed by visualize/plot_mc_variance.py (which groups by B and sample_idx,
# sorts candidates by candidate_idx to compare d across draws, and reads the
# rank-min row per sample as the top-1 pick).
#
# Usage (full campaign — run on GPU, this is the central job):
#     python scripts/mc_variance.py
# Smoke test (CPU):
#     JAX_PLATFORMS=cpu python scripts/mc_variance.py \
#         --b-list 100 300 --num-samples 3 --K 5

import argparse
import csv
import os
import sys

import numpy as np

from mlp.mlp import load_network_params, batched_predict
from sparsifier.sparsifier import make_omega, clone_network, _d_to_outputs

# Default Ω sample sizes to sweep (spec §1).
DEFAULT_B_LIST = [100, 300, 1000, 3000, 10000, 30000, 100000]


def build_candidates(baseline, K):
    """First K prunable (mask != 0) weights in layer-major order, each as a
    baseline clone with that single weight zeroed. Returns (nets, coords)."""
    nets, coords = [], []
    for idx, layer in enumerate(baseline):
        n_out, n_in = layer.W.shape
        for i in range(n_out):
            for j in range(n_in):
                if layer.mask[i, j] == 0.0:
                    continue
                net = clone_network(baseline)
                net[idx].W[i, j] = 0.0
                net[idx].mask[i, j] = 0.0
                nets.append(net)
                coords.append((idx, i, j))
                if len(nets) >= K:
                    return nets, coords
    return nets, coords


def rank_ascending(ds):
    """rank[c] = position of candidate c when sorted by d ascending (rank 0 =
    smallest d). Stable ties so a given (B, sample) is deterministic."""
    order = np.argsort(ds, kind="stable")
    ranks = np.empty(len(ds), dtype=int)
    ranks[order] = np.arange(len(ds))
    return ranks


def run(baseline_dir, out_csv, b_list, num_samples, K, seed):
    baseline = load_network_params(baseline_dir)
    candidates, coords = build_candidates(baseline, K)
    K = len(candidates)  # may be < requested if the net has fewer weights
    print("Candidates (K=%d): %s" % (K, coords))

    np.random.seed(seed)  # make_omega draws from the global numpy RNG
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["B", "candidate_idx", "sample_idx", "d", "rank"])
        for B in b_list:
            for s in range(num_samples):
                omega = make_omega(baseline, n_samples=B)
                ref_out = batched_predict(baseline, omega)  # forward once, reuse
                ds = np.array(
                    [float(_d_to_outputs(c, ref_out, omega)) for c in candidates]
                )
                ranks = rank_ascending(ds)
                for c in range(K):
                    w.writerow([B, c, s, float(ds[c]), int(ranks[c])])
            print("B=%d done (%d samples)" % (B, num_samples))
    print("Wrote", out_csv)


def _self_check():
    # rank_ascending: smallest d -> rank 0, order preserved, ties stable.
    r = rank_ascending(np.array([0.3, 0.1, 0.2]))
    assert list(r) == [2, 0, 1], r
    r = rank_ascending(np.array([5.0, 5.0, 1.0]))
    assert list(r) == [1, 2, 0], r
    print("self-check ok")


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline", default="artifacts/baseline",
                   help="dir with trained W_*/b_*.npy")
    p.add_argument("--out", default="artifacts/mc_variance/results.csv")
    p.add_argument("--b-list", type=int, nargs="+", default=DEFAULT_B_LIST)
    p.add_argument("--num-samples", type=int, default=100,
                   help="independent Ω draws per B")
    p.add_argument("--K", type=int, default=30,
                   help="number of fixed single-weight-zeroed candidates")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--self-check", action="store_true",
                   help="run the ranking self-check and exit")
    return p.parse_args(argv)


def main():
    args = _parse_args(sys.argv[1:])
    if args.self_check:
        _self_check()
        return
    run(args.baseline, args.out, args.b_list,
        args.num_samples, args.K, args.seed)


if __name__ == "__main__":
    main()
