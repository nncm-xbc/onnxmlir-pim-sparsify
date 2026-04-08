# benchmark/compare_strategies.py
"""
Run multiple pruning strategies for N steps on the same starting network.
Saves one CSV per strategy; use visualize/plot_comparison.py to overlay results.

Usage:
    python benchmark/compare_strategies.py artifacts/ \
        data/X_test_small.csv data/Y_test_small.csv \
        --steps 50 --out artifacts/comparison/

    # Run only specific strategies:
    python benchmark/compare_strategies.py artifacts/ \
        data/X_test_small.csv data/Y_test_small.csv \
        --steps 20 --out /tmp/cmp/ \
        --strategies exhaustive gradient_top_10
"""
import sys, os, csv, argparse
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from mlp.mlp import accuracy, load_network_params
from sparsifier.sparsifier import clone_network, d, make_omega, prune
from benchmark.correctness_check import prune_gradient_topk


def _make_strategies():
    """Return strategy name -> callable(net, og_net, omega) -> (net, PruneMeta)."""
    return {
        'exhaustive':      lambda net, og, om: prune(net, og, om, doAdjust=True),
        'gradient_top_1':  lambda net, og, om: prune_gradient_topk(net, og, om, top_k=1,  doAdjust=True),
        'gradient_top_10': lambda net, og, om: prune_gradient_topk(net, og, om, top_k=10, doAdjust=True),
        'gradient_top_50': lambda net, og, om: prune_gradient_topk(net, og, om, top_k=50, doAdjust=True),
    }


def run_strategy(name, strategy_fn, og_net, x_test, y_test, omega, n_steps, out_dir):
    """Run one strategy for n_steps, writing a CSV of per-step metrics."""
    net     = clone_network(og_net)
    total_W = int(sum(l.W.size for l in og_net))
    log_path = os.path.join(out_dir, f'{name}.csv')

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'NZ', 'sparsity', 'val_acc', 'd_manifold',
                         'prune_time_s', 'adjust_time_s'])
        for step in range(n_steps):
            NZ         = int(sum((l.W != 0).sum() for l in net))
            sparsity   = 1.0 - NZ / total_W
            val_acc    = float(accuracy(net, x_test, y_test))
            d_manifold = float(d(net, og_net, omega))
            net, meta  = strategy_fn(net, og_net, omega)
            writer.writerow([
                step, NZ, round(sparsity, 6), round(val_acc, 6),
                f'{d_manifold:.6e}',
                round(meta.prune_time_s,  4),
                round(meta.adjust_time_s, 4),
            ])
            f.flush()
            print(f'  [{name}] step {step:3d} | acc={val_acc:.4f} | '
                  f'search={meta.prune_time_s:.2f}s')

    print(f'Saved: {log_path}')


def main():
    strategies = _make_strategies()

    p = argparse.ArgumentParser(description='Compare pruning strategies')
    p.add_argument('net_folder')
    p.add_argument('x_test')
    p.add_argument('y_test')
    p.add_argument('--steps',      type=int, default=50)
    p.add_argument('--out',        default='artifacts/comparison/')
    p.add_argument('--strategies', nargs='+', default=list(strategies.keys()),
                   choices=list(strategies.keys()))
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    og_net = load_network_params(args.net_folder)
    x_test = np.genfromtxt(args.x_test, delimiter=',', max_rows=500)
    y_test = np.genfromtxt(args.y_test, delimiter=',', max_rows=500)
    omega  = make_omega(og_net)

    for name in args.strategies:
        print(f'\n=== Strategy: {name} ===')
        run_strategy(name, strategies[name], og_net, x_test, y_test,
                     omega, args.steps, args.out)

    print(f'\nAll results saved to {args.out}')
    print(f'Run: python visualize/plot_comparison.py {args.out}')


if __name__ == '__main__':
    main()
