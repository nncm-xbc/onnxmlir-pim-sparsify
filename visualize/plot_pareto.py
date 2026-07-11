# visualize/plot_pareto.py
"""
Accuracy vs cumulative wall-clock — the practitioner-facing Pareto plot.

Reads one CSV per method from a directory (filename minus .csv = method label).
Optional sidecar `<csv_dir>/colors.json` overrides the default palette.

Usage:
    python visualize/plot_pareto.py artifacts/comparison/ --out images/pareto/
"""
import argparse
import sys

import numpy as np

from visualize.viz_common import (plt, save_fig, add_output_args,
                                  resolve_colors, load_method_csvs)


_MILESTONES = (0.10, 0.20, 0.30, 0.50, 0.70, 0.90)


def _envelope(curves):
    """Compute the upper-left Pareto frontier across a list of (x, y) curves.

    A point is on the frontier if no other point has both higher accuracy AND
    lower wall-clock. Returns (xs, ys) sorted by x.
    """
    pts = np.concatenate([
        np.column_stack([np.asarray(x, dtype=float),
                         np.asarray(y, dtype=float)])
        for x, y in curves if len(x) > 0
    ], axis=0)
    if pts.size == 0:
        return np.array([]), np.array([])
    order = np.argsort(pts[:, 0])
    pts = pts[order]
    xs, ys = [], []
    best = -np.inf
    for x, y in pts:
        if y > best:
            xs.append(x); ys.append(y)
            best = y
    return np.array(xs), np.array(ys)


def plot_pareto(csv_dir: str, out_dir: str = None,
                show: bool = False) -> plt.Figure:
    """
    Single-panel: x = cumulative wall-clock (search + adjust), y = val_acc;
    one curve per method, with markers at sparsity milestones; an envelope
    line shows the joint Pareto frontier.
    """
    loaded = load_method_csvs(csv_dir)
    names = [name for name, _ in loaded]
    colors = resolve_colors(names, csv_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.suptitle('Accuracy vs Cumulative Wall-clock (Pareto)', fontsize=12)

    curves = []
    for name, df in loaded:
        if 'prune_time_s' not in df.columns or 'adjust_time_s' not in df.columns:
            print(f'[skip] {name}.csv: missing timing columns')
            continue
        cum = (df['prune_time_s'].fillna(0) + df['adjust_time_s'].fillna(0)).cumsum()
        acc = df['val_acc'] * 100
        c = colors[name]
        ax.plot(cum, acc, label=name, color=c, linewidth=1.6)

        # Milestone markers.
        for m in _MILESTONES:
            close = (df['sparsity'] - m).abs()
            if close.min() <= 0.01:
                idx = int(close.idxmin())
                ax.scatter(cum.iloc[idx], acc.iloc[idx], s=40,
                           color=c, edgecolors='black', linewidths=0.6,
                           zorder=4)
                ax.annotate(f'{int(m*100)}%',
                            (cum.iloc[idx], acc.iloc[idx]),
                            xytext=(4, 4), textcoords='offset points',
                            fontsize=7, color=c)

        curves.append((cum.to_numpy(), acc.to_numpy()))

    xs, ys = _envelope(curves)
    if xs.size > 0:
        ax.plot(xs, ys, color='black', linewidth=1.2, linestyle='--',
                alpha=0.6, label='Pareto envelope')

    ax.set_xlabel('Cumulative wall-clock (s)')
    ax.set_ylabel('Val Accuracy (%)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return save_fig(fig, out_dir, 'pareto.png', show=show)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv_dir', help='Directory containing one CSV per method.')
    add_output_args(p)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    plot_pareto(args.csv_dir, out_dir=args.out_dir,
                show=args.out_dir is None)
