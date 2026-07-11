# visualize/plot_layer_dynamics.py
"""
Per-layer pruning timeline: stacked area + relative fraction.

Usage:
    python visualize/plot_layer_dynamics.py \\
        artifacts/baseline/sparsified/sparsification_log.csv \\
        --out images/layer_dynamics/baseline/
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from visualize.viz_common import plt, save_fig, add_output_args


# Sparsity-fraction thresholds we annotate as vertical lines on each panel.
_THRESHOLDS = [0.9, 0.5, 0.1]


def _layer_columns(df: pd.DataFrame) -> list:
    cols = [c for c in df.columns if c.startswith('layer_') and c.endswith('_NZ')]
    cols.sort(key=lambda c: int(c.split('_')[1]))
    return cols


def _first_step_below(series: pd.Series, fraction: float, baseline: float):
    """Return first step where `series <= fraction * baseline`, or None."""
    threshold = fraction * baseline
    hits = series[series <= threshold]
    return int(hits.index[0]) if len(hits) > 0 else None


def plot_layer_dynamics(csv_path: str, out_dir: str = None,
                        show: bool = False) -> plt.Figure:
    """
    Two panels:
      [0] Stacked area: NZ count per layer over steps.
      [1] Per-layer fraction of currently-active weights vs step.
    """
    df = pd.read_csv(csv_path)
    layer_cols = _layer_columns(df)
    if not layer_cols:
        raise ValueError('CSV has no `layer_*_NZ` columns.')

    steps = df['step'].to_numpy()
    nz_per_layer = np.vstack([df[c].to_numpy(dtype=float) for c in layer_cols])
    total = nz_per_layer.sum(axis=0)
    fractions = np.where(total > 0, nz_per_layer / total, 0.0)

    labels = [c.replace('layer_', 'L').replace('_NZ', '') for c in layer_cols]
    cmap = plt.cm.tab10
    colors = [cmap(i % 10) for i in range(len(layer_cols))]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))
    fig.suptitle(f'Layer Dynamics — {os.path.basename(csv_path)}', fontsize=12)

    # -------- Panel 1: stacked area ----------------------------------------
    ax = axes[0]
    ax.stackplot(steps, nz_per_layer, labels=labels, colors=colors, alpha=0.85)
    ax.set_xlabel('Step'); ax.set_ylabel('NZ count')
    ax.set_title('Per-layer NZ (stacked)')
    ax.legend(loc='upper right', fontsize=8)

    # Annotate threshold crossings on panel 1.
    for col, lbl, color in zip(layer_cols, labels, colors):
        baseline = float(df[col].iloc[0])
        if baseline <= 0:
            continue
        for fr in _THRESHOLDS:
            step = _first_step_below(df[col], fr, baseline)
            if step is None:
                continue
            ax.axvline(step, color=color, linestyle='--', alpha=0.35,
                       linewidth=0.8)
            ax.text(step, ax.get_ylim()[1] * 0.97,
                    f'{lbl}<{int(fr*100)}%',
                    rotation=90, fontsize=6, color=color, va='top', ha='right',
                    alpha=0.8)

    # -------- Panel 2: relative fraction -----------------------------------
    ax = axes[1]
    for i, (lbl, color) in enumerate(zip(labels, colors)):
        ax.plot(steps, fractions[i] * 100, label=lbl, color=color,
                linewidth=1.4)
    ax.set_xlabel('Step'); ax.set_ylabel('% of currently active weights')
    ax.set_title('Per-layer share of remaining capacity')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

    plt.tight_layout()
    return save_fig(fig, out_dir, 'layer_dynamics.png', show=show)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv_path', help='sparsification_log.csv')
    add_output_args(p)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    plot_layer_dynamics(args.csv_path, out_dir=args.out_dir,
                        show=args.out_dir is None)
