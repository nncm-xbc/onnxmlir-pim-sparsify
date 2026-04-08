# visualize/plot_weights.py
"""
Weight magnitude and sparsity mask heatmaps from checkpoint directories.

Usage:
    # Single checkpoint:
    python visualize/plot_weights.py artifacts/sparsified/checkpoints/step_0000

    # Multiple checkpoints (side-by-side comparison):
    python visualize/plot_weights.py artifacts/sparsified/checkpoints/step_0000 \
                                     artifacts/sparsified/checkpoints/step_0100 \
                                     --out artifacts/sparsified/
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_checkpoint(ckpt_dir: str) -> list:
    """
    Load all W_i.npy arrays from a checkpoint directory.

    Returns list of np.ndarray ordered by layer index.
    Raises FileNotFoundError if ckpt_dir does not exist.
    Raises ValueError if no W_*.npy files are found.
    """
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    count = sum(1 for f in os.listdir(ckpt_dir)
                if f.startswith('W_') and f.endswith('.npy'))
    if count == 0:
        raise ValueError(f"No W_*.npy files found in: {ckpt_dir}")
    return [np.load(os.path.join(ckpt_dir, f'W_{i}.npy')) for i in range(count)]


def plot_weight_heatmaps(ckpt_dirs: list, step_labels: list = None,
                         out_dir: str = None, show: bool = True) -> plt.Figure:
    """
    Produce a heatmap grid: each column = one checkpoint, each row-pair = one layer.
    Row 2*l:   weight magnitude |W| (log-scale viridis colourmap)
    Row 2*l+1: binary mask (black = pruned, white = active)

    Note: with the Agg backend (set at module level), show=True is a silent no-op.
    The returned Figure is closed (deregistered from pyplot). Use fig.savefig()
    directly on the returned object rather than pyplot state-machine functions.

    Args:
        ckpt_dirs:   list of checkpoint directory paths
        step_labels: column header labels (defaults to directory basenames)
        out_dir:     if provided, saves 'weight_heatmaps.png' here
        show:        call plt.show() (no-op with Agg backend)
    """
    if step_labels is None:
        step_labels = [os.path.basename(d.rstrip('/')) for d in ckpt_dirs]

    checkpoints = [load_checkpoint(d) for d in ckpt_dirs]
    n_layers = len(checkpoints[0])
    n_ckpts  = len(checkpoints)

    if not all(len(c) == n_layers for c in checkpoints):
        raise ValueError(
            f"All checkpoints must have the same number of layers. "
            f"Got: {[len(c) for c in checkpoints]}"
        )

    fig, axes = plt.subplots(
        n_layers * 2, n_ckpts,
        figsize=(max(3 * n_ckpts, 4), max(4 * n_layers, 4)),
        squeeze=False,
    )
    fig.suptitle('Weight Magnitude & Sparsity Pattern', fontsize=12)

    for ci, (weights, label) in enumerate(zip(checkpoints, step_labels)):
        for li, W in enumerate(weights):
            sparsity = 1.0 - float((W != 0).mean())
            vmax = float(np.abs(W).max())

            # Weight magnitude — log scale
            ax_w = axes[li * 2][ci]
            safe_vmax = max(vmax, 1e-9)
            norm = mcolors.LogNorm(vmin=max(vmax * 1e-4, 1e-12), vmax=safe_vmax)
            im = ax_w.imshow(np.abs(W), aspect='auto', cmap='viridis', norm=norm)
            ax_w.set_title(f'{label}\nL{li} |W| ({W.shape[0]}×{W.shape[1]})', fontsize=7)
            ax_w.axis('off')
            plt.colorbar(im, ax=ax_w, fraction=0.046, pad=0.02)

            # Binary mask
            ax_m = axes[li * 2 + 1][ci]
            ax_m.imshow(W != 0, aspect='auto', cmap='gray_r', vmin=0, vmax=1)
            ax_m.set_title(f'mask  sp={sparsity:.2f}', fontsize=7)
            ax_m.axis('off')

    plt.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'weight_heatmaps.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


if __name__ == '__main__':
    raw_args = sys.argv[1:]
    out_dir  = None
    dirs     = []
    i = 0
    while i < len(raw_args):
        if raw_args[i] == '--out' and i + 1 < len(raw_args):
            out_dir = raw_args[i + 1]
            i += 2
        else:
            dirs.append(raw_args[i])
            i += 1
    if not dirs:
        print("Usage: plot_weights.py <ckpt_dir> [<ckpt_dir> ...] [--out <dir>]")
        sys.exit(1)
    plot_weight_heatmaps(dirs, out_dir=out_dir, show=out_dir is None)
