# visualize/plot_pattern_overlap.py
"""
Sparsity-pattern overlap (Jaccard) heatmap across pruning methods.

Each input directory must contain final-state weight files W_0.npy, W_1.npy, ...
(typically `artifacts/<method>/sparsified/`).

Usage:
    python visualize/plot_pattern_overlap.py \\
        artifacts/manifold/sparsified artifacts/magnitude/sparsified \\
        --labels manifold magnitude \\
        --out images/overlap/
"""
import argparse
import os
import sys

import numpy as np
from visualize.viz_common import (plt, save_fig, add_output_args,
                                  load_checkpoint_W)


def _load_masks(method_dir: str) -> list:
    """Load all W_*.npy files in `method_dir` and return them as bool masks."""
    return [W != 0 for W in load_checkpoint_W(method_dir)]


def _jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel(); b = b.ravel()
    inter = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return inter / union if union > 0 else 1.0


def _heatmap(ax, M, labels, title):
    im = ax.imshow(M, cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                    color='white' if M[i, j] < 0.5 else 'black', fontsize=7)
    return im


def plot_pattern_overlap(method_dirs: list, labels: list = None,
                         out_dir: str = None, show: bool = False) -> plt.Figure:
    """
    Compute Jaccard overlap between final pruning masks of N methods.

    Output: one combined heatmap + one heatmap per layer.
    """
    if labels is None:
        labels = [os.path.basename(os.path.dirname(d.rstrip('/'))) or
                  os.path.basename(d.rstrip('/'))
                  for d in method_dirs]
    if len(labels) != len(method_dirs):
        raise ValueError('len(labels) must match len(method_dirs)')

    masks_per_method = [_load_masks(d) for d in method_dirs]
    n_layers = len(masks_per_method[0])
    if not all(len(m) == n_layers for m in masks_per_method):
        raise ValueError('All methods must have the same number of layers.')

    n = len(method_dirs)

    # Combined: concatenate flattened masks across layers, one vector per method.
    flat = [np.concatenate([m.ravel() for m in masks]) for masks in masks_per_method]
    combined = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            combined[i, j] = _jaccard(flat[i], flat[j])

    per_layer = []
    for l in range(n_layers):
        M = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                M[i, j] = _jaccard(masks_per_method[i][l], masks_per_method[j][l])
        per_layer.append(M)

    n_cols = 1 + n_layers
    fig, axes = plt.subplots(1, n_cols, figsize=(4.0 * n_cols, 4.5),
                             squeeze=False)
    axes = axes[0]
    fig.suptitle('Sparsity-pattern Jaccard Overlap', fontsize=12)

    last = _heatmap(axes[0], combined, labels, 'all layers (combined)')
    for l, M in enumerate(per_layer):
        last = _heatmap(axes[l + 1], M, labels, f'layer {l}')

    fig.colorbar(last, ax=axes.tolist(), fraction=0.025, pad=0.02,
                 label='Jaccard index')

    return save_fig(fig, out_dir, 'pattern_overlap.png', show=show,
                    bbox_inches='tight')


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('method_dirs', nargs='+',
                   help='One or more final-state weight directories.')
    p.add_argument('--labels', nargs='+', default=None,
                   help='Label per method directory (defaults: parent dir names).')
    add_output_args(p)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    plot_pattern_overlap(args.method_dirs, labels=args.labels,
                         out_dir=args.out_dir, show=args.out_dir is None)
