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
from visualize.viz_common import plt, save_fig, load_checkpoint_W
# Backwards-compatible alias (identical signature/behavior) for external importers.
from visualize.viz_common import load_checkpoint_W as load_checkpoint
import matplotlib.colors as mcolors


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

    checkpoints = [load_checkpoint_W(d) for d in ckpt_dirs]
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
    return save_fig(fig, out_dir, 'weight_heatmaps.png', show=show,
                    bbox_inches='tight')


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
