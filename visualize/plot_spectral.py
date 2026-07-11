# visualize/plot_spectral.py
"""
Per-layer singular-value evolution and effective rank across checkpoints.

Looks for `<exp_dir>/checkpoints/step_NNNN/W_*.npy`. The checkpoint root may
be passed directly (e.g. `artifacts/<exp>/sparsified/checkpoints/`) or as the
parent `sparsified/` directory.

Usage:
    python visualize/plot_spectral.py \\
        artifacts/e05_stupidity_point/sparsified/checkpoints/ \\
        --out images/spectral/e05/
"""
import argparse
import os
import sys

import numpy as np
from visualize.viz_common import plt, save_fig, add_output_args


_RANK_EPS = (1e-1, 1e-2, 1e-3)


def _find_checkpoint_root(path: str) -> str:
    """Accept either `.../checkpoints/` or its parent `sparsified/` dir."""
    if os.path.isdir(os.path.join(path, 'checkpoints')):
        return os.path.join(path, 'checkpoints')
    return path


def _load_checkpoints(ckpt_root: str):
    """Return (steps_sorted, weights_per_step) where weights_per_step[k] is a
    list of np.ndarray (one per layer)."""
    if not os.path.isdir(ckpt_root):
        raise FileNotFoundError(ckpt_root)
    step_dirs = sorted(d for d in os.listdir(ckpt_root)
                       if d.startswith('step_'))
    if not step_dirs:
        raise ValueError(f'No step_* dirs in: {ckpt_root}')

    steps, weights = [], []
    for d in step_dirs:
        full = os.path.join(ckpt_root, d)
        n = sum(1 for f in os.listdir(full)
                if f.startswith('W_') and f.endswith('.npy'))
        if n == 0:
            continue
        try:
            step = int(d.split('_')[1])
        except (IndexError, ValueError):
            continue
        steps.append(step)
        weights.append([np.load(os.path.join(full, f'W_{i}.npy'))
                        for i in range(n)])
    if not steps:
        raise ValueError(f'No usable checkpoints in: {ckpt_root}')
    order = np.argsort(steps)
    return [steps[i] for i in order], [weights[i] for i in order]


def plot_spectral(ckpt_root: str, out_dir: str = None,
                  show: bool = False) -> plt.Figure:
    """
    For each layer, two panels:
      Left:  σ_k for all k vs step (log y).
      Right: r_eps for ε ∈ {1e-1, 1e-2, 1e-3} vs step.

    Highlights the step at which any layer's effective rank (ε=1e-2) drops
    by ≥ 50% relative to its initial value.
    """
    ckpt_root = _find_checkpoint_root(ckpt_root)
    steps, weights = _load_checkpoints(ckpt_root)
    n_layers = len(weights[0])

    # svds[l] is shape (n_steps, max_k) padded with NaN.
    max_k = [min(W.shape) for W in weights[0]]
    svds = [np.full((len(steps), max_k[l]), np.nan) for l in range(n_layers)]
    ranks = [np.full((len(steps), len(_RANK_EPS)), np.nan) for l in range(n_layers)]

    for s_idx, ws in enumerate(weights):
        for l, W in enumerate(ws):
            sv = np.linalg.svd(W, compute_uv=False)
            svds[l][s_idx, :len(sv)] = sv
            sigma1 = float(sv[0]) if sv.size > 0 else 0.0
            for e_idx, eps in enumerate(_RANK_EPS):
                ranks[l][s_idx, e_idx] = int((sv > eps * sigma1).sum())

    # Compute the global "rank-collapse" step (first step at which any layer's
    # ε=1e-2 effective rank drops by >= 50% from its initial value).
    collapse_step = None
    eps_idx = _RANK_EPS.index(1e-2)
    for l in range(n_layers):
        col = ranks[l][:, eps_idx]
        if np.all(np.isnan(col)) or col[0] <= 0:
            continue
        threshold = 0.5 * col[0]
        hits = np.where(col <= threshold)[0]
        if hits.size > 0:
            s = steps[int(hits[0])]
            if collapse_step is None or s < collapse_step:
                collapse_step = s

    fig, axes = plt.subplots(n_layers, 2,
                             figsize=(13, 3.0 * n_layers),
                             squeeze=False)
    fig.suptitle(f'Spectral Evolution  (root: {ckpt_root})', fontsize=12)

    for l in range(n_layers):
        # Left: σ_k curves.
        ax = axes[l][0]
        for k in range(svds[l].shape[1]):
            ax.semilogy(steps, svds[l][:, k], color='steelblue',
                        alpha=0.5, linewidth=0.8)
        if collapse_step is not None:
            ax.axvline(collapse_step, color='red', linestyle='--', alpha=0.7,
                       label=f'rank↓50% @ step {collapse_step}')
            ax.legend(fontsize=8)
        ax.set_xlabel('Step'); ax.set_ylabel('σ_k')
        ax.set_title(f'L{l}: singular values')

        # Right: effective rank.
        ax = axes[l][1]
        for e_idx, eps in enumerate(_RANK_EPS):
            ax.plot(steps, ranks[l][:, e_idx], '-o', markersize=3,
                    label=f'ε = {eps:g}')
        if collapse_step is not None:
            ax.axvline(collapse_step, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Step'); ax.set_ylabel('effective rank')
        ax.set_title(f'L{l}: rank_eps vs step')
        ax.legend(fontsize=8)

    plt.tight_layout()
    return save_fig(fig, out_dir, 'spectral.png', show=show,
                    bbox_inches='tight')


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('ckpt_root',
                   help='Checkpoint root (sparsified/ or sparsified/checkpoints/).')
    add_output_args(p)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    plot_spectral(args.ckpt_root, out_dir=args.out_dir,
                  show=args.out_dir is None)
