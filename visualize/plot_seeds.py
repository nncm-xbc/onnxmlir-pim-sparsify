# visualize/plot_seeds.py
"""
Multi-seed variance bands: median + min/max envelope across seeds for one
experiment configuration.

Usage:
    python visualize/plot_seeds.py \\
        artifacts/e10_seed_*/sparsified/sparsification_log.csv \\
        --label "E10 baseline" --out images/seed_variance/
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from visualize.viz_common import plt, save_fig, add_output_args


_SEED_RE = re.compile(r'seed[_-](\d+)')


def _detect_seed(csv_path: str) -> str:
    """Pull the seed value out of the filename or any parent directory.

    Returns the matched digits as a string, or 'unk' if none were found.
    """
    m = _SEED_RE.search(csv_path)
    return m.group(1) if m else 'unk'


def _expand_paths(paths: list) -> list:
    """Glob-expand each entry; preserve order; drop duplicates."""
    out, seen = [], set()
    for p in paths:
        for hit in sorted(glob.glob(p)) or [p]:
            if hit in seen:
                continue
            seen.add(hit)
            out.append(hit)
    return out


def _resample_on_sparsity(df: pd.DataFrame, grid: np.ndarray, ycol: str) -> np.ndarray:
    """Linearly interpolate `ycol` onto a common sparsity grid.

    Sparsity is monotonic non-decreasing, so np.interp is well-defined.
    Values outside the run's covered range are returned as NaN.
    """
    x = df['sparsity'].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    out = np.interp(grid, x, y, left=np.nan, right=np.nan)
    return out


def plot_seeds(csv_paths: list, label: str, out_dir: str = None,
               show: bool = False) -> plt.Figure:
    """
    Three-panel multi-seed view:
      [0] Accuracy vs sparsity — median curve, min/max band, faint per-seed lines.
      [1] d_manifold vs step  — same band style, log y.
      [2] Final-state scatter at (sparsity_final, val_acc_final).
    """
    csv_paths = _expand_paths(csv_paths)
    if not csv_paths:
        raise ValueError('No CSV paths matched.')

    runs = []  # list of (seed, DataFrame)
    for p in csv_paths:
        runs.append((_detect_seed(p), pd.read_csv(p)))

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Seed Variance — {label}  (n={len(runs)})', fontsize=13)

    # ---------------- Panel 1: accuracy vs sparsity ------------------------
    ax = axes[0]
    sparsity_max = min(float(df['sparsity'].max()) for _, df in runs)
    grid = np.linspace(0.0, sparsity_max, 200)
    acc_matrix = np.vstack([
        _resample_on_sparsity(df, grid, 'val_acc') for _, df in runs
    ])
    median = np.nanmedian(acc_matrix, axis=0) * 100
    lo = np.nanmin(acc_matrix, axis=0) * 100
    hi = np.nanmax(acc_matrix, axis=0) * 100
    ax.fill_between(grid * 100, lo, hi, alpha=0.20, color='steelblue',
                    label='min-max band')
    ax.plot(grid * 100, median, color='steelblue', linewidth=2, label='median')
    for seed, df in runs:
        ax.plot(df['sparsity'] * 100, df['val_acc'] * 100,
                alpha=0.30, linewidth=0.8, color='gray',
                label=f'seed {seed}')
    ax.set_xlabel('Sparsity (%)'); ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('Accuracy vs Sparsity')
    ax.legend(fontsize=7, loc='lower left')

    # ---------------- Panel 2: d_manifold vs step --------------------------
    ax = axes[1]
    step_max = min(int(df['step'].max()) for _, df in runs)
    step_grid = np.arange(0, step_max + 1)
    d_matrix = []
    for _, df in runs:
        d = np.interp(step_grid, df['step'].to_numpy(dtype=float),
                      df['d_manifold'].to_numpy(dtype=float),
                      left=np.nan, right=np.nan)
        d_matrix.append(d)
    d_matrix = np.vstack(d_matrix)
    d_matrix[d_matrix <= 0] = np.nan
    with np.errstate(all='ignore'):
        # All-NaN columns can arise when no seed covers a given step;
        # we just want NaN out and we ignore the warnings.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            med = np.nanmedian(d_matrix, axis=0)
            lo  = np.nanmin(d_matrix, axis=0)
            hi  = np.nanmax(d_matrix, axis=0)
    ax.fill_between(step_grid, lo, hi, alpha=0.20, color='firebrick',
                    label='min-max band')
    ax.semilogy(step_grid, med, color='firebrick', linewidth=2, label='median')
    for seed, df in runs:
        d = df['d_manifold'].replace(0, np.nan)
        ax.semilogy(df['step'], d, alpha=0.30, linewidth=0.8, color='gray',
                    label=f'seed {seed}')
    ax.set_xlabel('Step'); ax.set_ylabel('d_manifold')
    ax.set_title('Manifold Distance (log y)')
    ax.legend(fontsize=7)

    # ---------------- Panel 3: final-state scatter -------------------------
    ax = axes[2]
    finals = []
    for seed, df in runs:
        sx = float(df['sparsity'].iloc[-1]) * 100
        sy = float(df['val_acc'].iloc[-1]) * 100
        finals.append((seed, sx, sy))
        ax.scatter(sx, sy, s=60, alpha=0.7, label=f'seed {seed}')
    sx_mean = float(np.mean([f[1] for f in finals]))
    sy_mean = float(np.mean([f[2] for f in finals]))
    ax.scatter(sx_mean, sy_mean, marker='X', s=200, color='black',
               edgecolors='white', linewidths=1.5, label='mean', zorder=5)
    ax.set_xlabel('Final sparsity (%)'); ax.set_ylabel('Final val_acc (%)')
    ax.set_title('Final-state across seeds')
    ax.legend(fontsize=7)

    plt.tight_layout()
    safe = label.replace(' ', '_').replace('/', '_')
    return save_fig(fig, out_dir, f'seed_variance_{safe}.png', show=show)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv_paths', nargs='+',
                   help='List (or shell-glob) of sparsification_log.csv paths.')
    p.add_argument('--label', required=True,
                   help='Label that identifies the experiment configuration.')
    add_output_args(p)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    plot_seeds(args.csv_paths, label=args.label, out_dir=args.out_dir,
               show=args.out_dir is None)
