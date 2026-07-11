# visualize/plot_pixel_saliency.py
"""
Per-pixel saliency for the input layer: shows what the (now-sparse) network
"looks at" in pixel space. Useful as a structure-vs-uniform sanity check
(e.g. against the Gaussian baseline).

W_0 is laid out as (out_neurons, in_pixels). Per-pixel saliency:
    s[j] = sum_i |W_0[i, j]|
Optional second metric:
    f[j] = (W_0[:, j] != 0).mean()

Usage:
    python visualize/plot_pixel_saliency.py \\
        artifacts/baseline/sparsified/W_0.npy \\
        artifacts/manifold/sparsified/W_0.npy \\
        --shape 14 14 --out images/saliency/
"""
import argparse
import os
import sys

import numpy as np
from visualize.viz_common import plt, save_fig, add_output_args


def _saliency(W: np.ndarray):
    s = np.abs(W).sum(axis=0)               # one value per input pixel
    f = (W != 0).mean(axis=0)               # active fraction per pixel
    return s, f


def plot_pixel_saliency(w_paths: list, shape, out_dir: str = None,
                        labels: list = None, show: bool = False) -> plt.Figure:
    """
    Two rows × N columns:
      Row 0: |W_0| sum per pixel (saliency), shared colour scale.
      Row 1: active fraction per pixel,    shared colour scale.

    `shape` is the spatial layout to reshape input pixels into (e.g. (14, 14)).
    """
    if labels is None:
        labels = []
        for p in w_paths:
            base = os.path.dirname(p)
            parent = os.path.basename(os.path.dirname(base)) or os.path.basename(base)
            labels.append(parent)
    if len(labels) != len(w_paths):
        raise ValueError('len(labels) must match len(w_paths)')

    H, Wd = int(shape[0]), int(shape[1])
    expected = H * Wd

    saliencies, fractions = [], []
    for p in w_paths:
        W = np.load(p)
        if W.shape[1] != expected:
            raise ValueError(
                f'{p}: W has {W.shape[1]} input columns but '
                f'shape={shape} expects {expected}.')
        s, f = _saliency(W)
        saliencies.append(s.reshape(H, Wd))
        fractions.append(f.reshape(H, Wd))

    s_vmax = max(float(s.max()) for s in saliencies)
    f_vmax = max(float(f.max()) for f in fractions)

    n = len(w_paths)
    fig, axes = plt.subplots(2, n, figsize=(3.6 * n, 7.2), squeeze=False)
    fig.suptitle('Input-layer Saliency  (top: |W| sum, bottom: active fraction)',
                 fontsize=12)

    for col, (s, f, lbl) in enumerate(zip(saliencies, fractions, labels)):
        ax = axes[0][col]
        im = ax.imshow(s, cmap='viridis', vmin=0.0, vmax=s_vmax)
        ax.set_title(f'{lbl}\nΣ|W|', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

        ax = axes[1][col]
        im = ax.imshow(f, cmap='magma', vmin=0.0, vmax=f_vmax)
        ax.set_title('active fraction', fontsize=9)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)

    plt.tight_layout()
    return save_fig(fig, out_dir, 'pixel_saliency.png', show=show,
                    bbox_inches='tight')


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('w_paths', nargs='+', help='One or more W_0.npy files.')
    p.add_argument('--shape', nargs=2, type=int, default=[14, 14],
                   metavar=('H', 'W'),
                   help='Image shape to reshape input pixels to. Default: 14 14.')
    p.add_argument('--labels', nargs='+', default=None,
                   help='Label per W file (defaults: parent directory names).')
    add_output_args(p)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    plot_pixel_saliency(args.w_paths, shape=args.shape, labels=args.labels,
                        out_dir=args.out_dir, show=args.out_dir is None)
