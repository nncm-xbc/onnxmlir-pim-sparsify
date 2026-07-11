# visualize/plot_confusion_evolution.py
"""
Class-wise accuracy degradation across pruning checkpoints.

Modes:
  fast (default if confusion.npy is present): for each checkpoint dir
        `step_NNNN/`, read `confusion.npy` (shape K x K) and use it directly.
  slow (`--recompute`): reload weights from each checkpoint and recompute
        the confusion matrix on a held-out test set. This requires the project
        root on sys.path so `mlp` is importable.

Output: a heatmap with rows = sparsity (%), columns = class index,
colour = per-class accuracy.

Usage:
    # Use cached confusion.npy files
    python visualize/plot_confusion_evolution.py \\
        artifacts/baseline/sparsified/checkpoints/ --out images/confusion/

    # Recompute from weights (slower)
    python visualize/plot_confusion_evolution.py \\
        artifacts/baseline/sparsified/checkpoints/ \\
        --recompute --data-dir data/ --out images/confusion/
"""
import argparse
import os
import sys

import numpy as np
from visualize.viz_common import plt, save_fig, add_output_args


def _list_checkpoints(ckpt_root: str):
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(ckpt_root)
    dirs = sorted(d for d in os.listdir(ckpt_root)
                  if d.startswith('step_') and os.path.isdir(
                      os.path.join(ckpt_root, d)))
    if not dirs:
        raise ValueError(f'No step_* dirs in: {ckpt_root}')
    out = []
    for d in dirs:
        try:
            step = int(d.split('_')[1])
        except (IndexError, ValueError):
            continue
        out.append((step, os.path.join(ckpt_root, d)))
    return out


def _checkpoint_sparsity(ckpt_dir: str) -> float:
    n  = sum(1 for f in os.listdir(ckpt_dir)
             if f.startswith('W_') and f.endswith('.npy'))
    if n == 0:
        return float('nan')
    total = 0; zeros = 0
    for i in range(n):
        W = np.load(os.path.join(ckpt_dir, f'W_{i}.npy'))
        total += W.size
        zeros += int((W == 0).sum())
    return zeros / total if total > 0 else float('nan')


def _confusion_from_cache(ckpt_dir: str):
    p = os.path.join(ckpt_dir, 'confusion.npy')
    return np.load(p) if os.path.isfile(p) else None


def _build_network(ckpt_dir: str, bias_dir: str):
    """Construct an mlp.Layer list using W from `ckpt_dir` and b from `bias_dir`.

    Checkpoints only persist W_*.npy; biases are stored once at the run root.
    """
    from mlp import mlp as mlp_mod
    n = sum(1 for f in os.listdir(ckpt_dir)
            if f.startswith('W_') and f.endswith('.npy'))
    layers = []
    for i in range(n):
        W = np.load(os.path.join(ckpt_dir, f'W_{i}.npy'))
        b = np.load(os.path.join(bias_dir, f'b_{i}.npy'))
        layers.append(mlp_mod.Layer(W=W, b=b, mask=np.ones_like(W)))
    return layers


def _confusion_recompute(ckpt_dir: str, bias_dir: str, x_test, y_true_idx):
    """Recompute confusion using the project's mlp module."""
    from mlp import mlp as mlp_mod
    network = _build_network(ckpt_dir, bias_dir)
    logits  = np.array(mlp_mod.batched_predict(network, x_test))
    pred    = np.argmax(logits, axis=1)
    K = int(max(int(y_true_idx.max()), int(pred.max())) + 1)
    M = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true_idx, pred):
        M[int(t), int(p)] += 1
    return M


def _load_test_set(data_dir: str):
    """Load test split from a CSV folder containing X_test.csv / Y_test.csv."""
    from mlp.dataset import load_dataset
    (_, _), (x_test, y_test), _ = load_dataset(data_dir, source='csv')
    if y_test.ndim == 2:
        y_idx = np.argmax(y_test, axis=1)
    else:
        y_idx = y_test.astype(int)
    return np.array(x_test), y_idx


def plot_confusion_evolution(ckpt_root: str, recompute: bool = False,
                             data_dir: str = None,
                             out_dir: str = None,
                             show: bool = False) -> plt.Figure:
    """Heatmap of per-class accuracy across sparsity levels."""
    ckpts = _list_checkpoints(ckpt_root)
    # The run root sits two levels up from `step_NNNN/`: <run>/checkpoints/.
    bias_dir = os.path.dirname(os.path.normpath(ckpt_root.rstrip('/')))

    x_test = y_idx = None
    if recompute:
        if data_dir is None:
            raise ValueError('--recompute requires --data-dir.')
        x_test, y_idx = _load_test_set(data_dir)

    rows = []  # (sparsity, per_class_accuracy_vector)
    for step, ckpt_dir in ckpts:
        M = None if recompute else _confusion_from_cache(ckpt_dir)
        if M is None:
            if not recompute:
                continue
            M = _confusion_recompute(ckpt_dir, bias_dir, x_test, y_idx)
        per_class = np.where(M.sum(axis=1) > 0,
                             np.diag(M) / np.maximum(M.sum(axis=1), 1),
                             np.nan)
        sp = _checkpoint_sparsity(ckpt_dir)
        rows.append((step, sp, per_class))

    if not rows:
        raise RuntimeError(
            'No confusion data available. Pass --recompute (with --data-dir) '
            'or run an experiment that dumps confusion.npy per checkpoint.')

    rows.sort(key=lambda r: r[0])
    K = max(r[2].size for r in rows)
    M = np.full((len(rows), K), np.nan)
    sparsity_pct = np.array([r[1] * 100 for r in rows])
    for i, (_, _, pc) in enumerate(rows):
        M[i, :pc.size] = pc

    fig, ax = plt.subplots(
        figsize=(max(8, K * 0.7), max(5, len(rows) * 0.3)))
    im = ax.imshow(M, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_xlabel('Class index')
    ax.set_ylabel('Sparsity (%)')
    ax.set_title('Per-class Accuracy vs Sparsity')

    yticks_idx = np.linspace(0, len(rows) - 1, min(len(rows), 12)).astype(int)
    ax.set_yticks(yticks_idx)
    ax.set_yticklabels([f'{sparsity_pct[i]:.1f}' for i in yticks_idx])
    ax.set_xticks(range(K))
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label='per-class accuracy')

    plt.tight_layout()
    return save_fig(fig, out_dir, 'confusion_evolution.png', show=show)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ckpt_root', help='Checkpoints directory.')
    p.add_argument('--recompute', action='store_true',
                   help='Recompute confusion from weights (requires --data-dir).')
    p.add_argument('--data-dir', default=None,
                   help='Folder with X_test.csv / Y_test.csv (used with --recompute).')
    add_output_args(p)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    plot_confusion_evolution(
        args.ckpt_root, recompute=args.recompute, data_dir=args.data_dir,
        out_dir=args.out_dir, show=args.out_dir is None,
    )
