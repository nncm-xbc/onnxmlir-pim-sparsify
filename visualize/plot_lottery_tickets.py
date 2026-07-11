# visualize/plot_lottery_tickets.py
"""
Lottery-ticket evidence: which weights survive across all seeds?

Each input directory must contain final-state weight files W_0.npy, W_1.npy, ...
(typically `artifacts/<exp>_seed_<n>/sparsified/`).

Usage:
    python visualize/plot_lottery_tickets.py \\
        artifacts/e10_seed_1/sparsified \\
        artifacts/e10_seed_2/sparsified \\
        artifacts/e10_seed_3/sparsified \\
        --out images/lottery/
"""
import argparse
import glob
import os
import sys

import numpy as np
from visualize.viz_common import (plt, save_fig, add_output_args,
                                  load_checkpoint_W)


def _load_masks(seed_dir: str):
    return [W != 0 for W in load_checkpoint_W(seed_dir)]


def _expand(paths):
    out = []
    for p in paths:
        hits = sorted(glob.glob(p)) or [p]
        out.extend(hits)
    return out


def plot_lottery_tickets(seed_dirs: list, out_dir: str = None,
                         show: bool = False) -> plt.Figure:
    """
    Three-panel lottery-ticket figure across N seed directories.

      [0] Histogram of survival count per weight (peak at N => strong effect).
      [1] Per-layer "core mask" (positions surviving in all N seeds).
      [2] Cumulative fraction of weights preserved by >=k seeds vs k.
    """
    seed_dirs = _expand(seed_dirs)
    if len(seed_dirs) < 2:
        raise ValueError('Need >=2 seed directories.')

    masks_per_seed = [_load_masks(d) for d in seed_dirs]
    n_layers = len(masks_per_seed[0])
    if not all(len(m) == n_layers for m in masks_per_seed):
        raise ValueError('All seeds must have the same number of layers.')

    N = len(seed_dirs)
    survival_per_layer = []  # int matrix shaped like W[l]
    for l in range(n_layers):
        stack = np.stack([m[l].astype(np.int8) for m in masks_per_seed], axis=0)
        survival_per_layer.append(stack.sum(axis=0))   # values in [0, N]

    flat_counts = np.concatenate([s.ravel() for s in survival_per_layer])

    fig = plt.figure(figsize=(16, 5 + 2 * n_layers))
    gs  = fig.add_gridspec(2, max(2, n_layers), hspace=0.5, wspace=0.35)
    fig.suptitle(
        f'Lottery-ticket Analysis  (N = {N} seeds, {n_layers} layers)',
        fontsize=13)

    # ---------------- Panel 1: histogram of survival count ----------------
    ax = fig.add_subplot(gs[0, 0])
    bins = np.arange(0, N + 2) - 0.5
    ax.hist(flat_counts, bins=bins, edgecolor='black')
    ax.set_xticks(range(0, N + 1))
    ax.set_xlabel(f'# seeds preserving the weight (out of {N})')
    ax.set_ylabel('Count of weights')
    ax.set_title('Survival-count histogram')

    # ---------------- Panel 3 (top-right): cumulative ---------------------
    ax = fig.add_subplot(gs[0, 1] if max(2, n_layers) >= 2 else gs[0, 0])
    total = flat_counts.size
    fracs = []
    for k in range(0, N + 1):
        fracs.append(float((flat_counts >= k).sum()) / total)
    ax.plot(range(0, N + 1), fracs, '-o')
    ax.set_xlabel('k (>= number of seeds)')
    ax.set_ylabel('Fraction of weights')
    ax.set_title('Fraction preserved by >= k seeds')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, N + 1))

    # ---------------- Panel 2 row: per-layer core masks -------------------
    for l in range(n_layers):
        ax = fig.add_subplot(gs[1, l])
        core = survival_per_layer[l] == N
        ax.imshow(core, aspect='auto', cmap='gray_r', vmin=0, vmax=1)
        ax.set_title(f'L{l} core mask  '
                     f'(active in all {N})\n'
                     f'{int(core.sum())} / {core.size} weights')
        ax.axis('off')

    plt.tight_layout()
    return save_fig(fig, out_dir, 'lottery_tickets.png', show=show,
                    bbox_inches='tight')


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('seed_dirs', nargs='+',
                   help='One or more final-state weight directories.')
    add_output_args(p)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    plot_lottery_tickets(args.seed_dirs, out_dir=args.out_dir,
                         show=args.out_dir is None)
