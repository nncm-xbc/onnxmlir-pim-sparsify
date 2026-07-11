# visualize/plot_methods.py
"""
Method-comparison overlay (3x2) — the headline experiments-chapter figure.

Reads one CSV per method from a directory; the filename (minus .csv) becomes
the legend label.

Optional colour override via a sidecar JSON file (`<csv_dir>/colors.json`):
    {"manifold": "#1f77b4", "magnitude": "#d62728", ...}

Usage:
    python visualize/plot_methods.py artifacts/comparison/ --out images/comparison/
"""
import argparse
import sys

import numpy as np

from visualize.viz_common import (plt, save_fig, add_output_args,
                                  resolve_colors, load_method_csvs)


def plot_methods(csv_dir: str, out_dir: str = None, show: bool = False) -> plt.Figure:
    """
    Six-panel overlay across all method CSVs in `csv_dir`.

    Panels (3 columns x 2 rows):
      [0,0] Accuracy vs sparsity (%)
      [0,1] d_manifold vs step  (log y)
      [0,2] NZ count vs step
      [1,0] Search time per step (prune_time_s)
      [1,1] Adjust time per step (adjust_time_s)
      [1,2] Cumulative wall-clock vs sparsity (Pareto-style)
    """
    loaded = load_method_csvs(csv_dir)
    method_names = [name for name, _ in loaded]
    colors = resolve_colors(method_names, csv_dir)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle('Method Comparison', fontsize=13)

    has_obs = False
    for name, df in loaded:
        c  = colors[name]
        kw = dict(label=name, color=c, linewidth=1.4)

        # 1. Accuracy vs sparsity
        axes[0, 0].plot(df['sparsity'] * 100, df['val_acc'] * 100, **kw)

        # 2. d_manifold vs step  (log y where positive)
        d = df['d_manifold'].replace(0, np.nan)
        axes[0, 1].semilogy(df['step'], d, **kw)

        # 3. NZ count vs step
        axes[0, 2].plot(df['step'], df['NZ'], **kw)

        # 4. Search time per step
        if 'prune_time_s' in df.columns:
            axes[1, 0].plot(df['step'], df['prune_time_s'], **kw)

        # 5. Adjust time per step
        if 'adjust_time_s' in df.columns:
            axes[1, 1].plot(df['step'], df['adjust_time_s'], **kw)
            if name.lower() == 'obs' and float(df['adjust_time_s'].sum()) == 0.0:
                has_obs = True

        # 6. Cumulative wall-clock vs sparsity
        if 'prune_time_s' in df.columns and 'adjust_time_s' in df.columns:
            cum = (df['prune_time_s'].fillna(0) + df['adjust_time_s'].fillna(0)).cumsum()
            axes[1, 2].plot(df['sparsity'] * 100, cum, **kw)

    # Cosmetics ---------------------------------------------------------------
    axes[0, 0].set_xlabel('Sparsity (%)'); axes[0, 0].set_ylabel('Val Accuracy (%)')
    axes[0, 0].set_title('Accuracy vs Sparsity')

    axes[0, 1].set_xlabel('Step'); axes[0, 1].set_ylabel('d_manifold')
    axes[0, 1].set_title('Manifold Distance (log y)')

    axes[0, 2].set_xlabel('Step'); axes[0, 2].set_ylabel('Non-zero weights')
    axes[0, 2].set_title('NZ Count vs Step')

    axes[1, 0].set_xlabel('Step'); axes[1, 0].set_ylabel('prune_time_s')
    axes[1, 0].set_title('Search Time per Step')

    axes[1, 1].set_xlabel('Step'); axes[1, 1].set_ylabel('adjust_time_s')
    title = 'Adjust Time per Step'
    if has_obs:
        title += '  (note: OBS reports 0)'
    axes[1, 1].set_title(title)

    axes[1, 2].set_xlabel('Sparsity (%)')
    axes[1, 2].set_ylabel('Cumulative wall-clock (s)')
    axes[1, 2].set_title('Compute vs Sparsity (Pareto)')

    for ax in axes.ravel():
        ax.legend(fontsize=8)

    plt.tight_layout()
    return save_fig(fig, out_dir, 'method_comparison.png', show=show)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv_dir', help='Directory containing one CSV per method.')
    add_output_args(p)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    plot_methods(args.csv_dir, out_dir=args.out_dir, show=args.out_dir is None)
