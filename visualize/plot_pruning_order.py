# visualize/plot_pruning_order.py
"""
Spatial-temporal view of pruning: per-layer heatmap of "step at which each
weight was pruned". Surviving weights are drawn in white.

Single-method mode:
    python visualize/plot_pruning_order.py artifacts/baseline/sparsified/sparsification_log.csv \\
        --out images/order/baseline/

Comparison mode (two CSVs, side by side):
    python visualize/plot_pruning_order.py log_a.csv log_b.csv \\
        --labels A B --out images/order/

Weight shapes are inferred from any checkpoint under
`<csv_parent>/checkpoints/step_*/W_*.npy`; if none are found, the script
falls back to inferring per-layer shape from the candidate (i, j) ranges.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from visualize.viz_common import plt, save_fig, add_output_args
import matplotlib.colors as mcolors


def _shapes_from_checkpoints(csv_path: str):
    """Look for any checkpoint dir alongside `csv_path` and return W shapes."""
    base = os.path.dirname(csv_path)
    ckpt_root = os.path.join(base, 'checkpoints')
    if not os.path.isdir(ckpt_root):
        return None
    step_dirs = sorted(d for d in os.listdir(ckpt_root)
                       if d.startswith('step_'))
    for d in step_dirs:
        full = os.path.join(ckpt_root, d)
        n = sum(1 for f in os.listdir(full)
                if f.startswith('W_') and f.endswith('.npy'))
        if n == 0:
            continue
        return [np.load(os.path.join(full, f'W_{i}.npy')).shape
                for i in range(n)]
    return None


def _shapes_from_log(df: pd.DataFrame):
    """Last-resort fallback: derive (rows, cols) per layer from candidate i/j."""
    layers = sorted(df['candidate_layer'].dropna().unique().astype(int).tolist())
    shapes = []
    for l in layers:
        sub = df[df['candidate_layer'] == l]
        rows = int(sub['candidate_i'].max()) + 1 if len(sub) else 1
        cols = int(sub['candidate_j'].max()) + 1 if len(sub) else 1
        shapes.append((rows, cols))
    return shapes


def _build_T_matrices(csv_path: str):
    """Return list of T[l] arrays (NaN where the weight survived)."""
    df = pd.read_csv(csv_path)
    needed = {'candidate_layer', 'candidate_i', 'candidate_j', 'step'}
    if not needed.issubset(df.columns):
        raise ValueError(f'CSV missing required columns: {needed - set(df.columns)}')

    shapes = _shapes_from_checkpoints(csv_path) or _shapes_from_log(df)
    T = [np.full(shape, np.nan, dtype=float) for shape in shapes]

    for _, row in df.iterrows():
        l = row['candidate_layer']
        if pd.isna(l):
            continue
        l = int(l); i = int(row['candidate_i']); j = int(row['candidate_j'])
        if l < 0 or l >= len(T):
            continue
        if i >= T[l].shape[0] or j >= T[l].shape[1]:
            continue
        if np.isnan(T[l][i, j]):
            T[l][i, j] = float(row['step'])
    return T


def _draw_heatmap(ax, T, title):
    """Heatmap of T with NaN drawn white (survivors)."""
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='white')
    masked = np.ma.array(T, mask=np.isnan(T))
    if masked.count() > 0:
        vmax = float(masked.max())
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    im = ax.imshow(masked, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest')
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def plot_pruning_order(csv_paths, labels=None, out_dir: str = None,
                       show: bool = False) -> plt.Figure:
    """Plot per-layer "death step" heatmap for one or two CSV logs."""
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
    if labels is None:
        labels = [os.path.basename(os.path.dirname(p)) or p for p in csv_paths]
    if len(labels) != len(csv_paths):
        raise ValueError('len(labels) must match len(csv_paths)')

    T_per_method = [_build_T_matrices(p) for p in csv_paths]
    n_methods = len(csv_paths)
    n_layers  = len(T_per_method[0])
    if not all(len(t) == n_layers for t in T_per_method):
        raise ValueError('All logs must have the same number of layers.')

    fig, axes = plt.subplots(n_layers, n_methods,
                             figsize=(4.0 * n_methods, 3.0 * n_layers),
                             squeeze=False)
    fig.suptitle('Pruning-order Heatmap (color = step at which weight died)',
                 fontsize=12)

    last_im = None
    for col, (lbl, Ts) in enumerate(zip(labels, T_per_method)):
        for row, T in enumerate(Ts):
            ax = axes[row][col]
            survived = int(np.isnan(T).sum())
            total    = T.size
            title = (f'{lbl}  L{row}\n'
                     f'survived {survived}/{total}')
            last_im = _draw_heatmap(ax, T, title)

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(),
                            fraction=0.025, pad=0.02)
        cbar.set_label('step pruned')

    suffix = '_'.join(labels) if len(labels) > 1 else labels[0]
    suffix = suffix.replace('/', '_').replace(' ', '_')
    return save_fig(fig, out_dir, f'pruning_order_{suffix}.png', show=show,
                    bbox_inches='tight')


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv_paths', nargs='+',
                   help='One sparsification_log.csv (single mode) '
                        'or two CSVs (comparison mode).')
    p.add_argument('--labels', nargs='+', default=None,
                   help='Label per CSV (defaults: parent directory names).')
    add_output_args(p)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    plot_pruning_order(args.csv_paths, labels=args.labels,
                       out_dir=args.out_dir, show=args.out_dir is None)
