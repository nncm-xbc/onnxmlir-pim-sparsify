# visualize/viz_common.py
"""
Shared helpers for the visualize/ plot scripts.

Importing this module sets the non-interactive 'Agg' matplotlib backend and
re-exports `plt`, so callers can just do:

    from visualize.viz_common import plt, save_fig, add_output_args
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------- #
# Figure output
# --------------------------------------------------------------------------- #
def save_fig(fig, out_dir, name, show=False, dpi=150, bbox_inches=None):
    """Save `fig` to `<out_dir>/<name>` (if out_dir), optionally show, close.

    Mirrors the save/show trailer that every plot script repeated. Returns the
    figure so callers can `return save_fig(...)`.
    """
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, name)
        fig.savefig(out_path, dpi=dpi, bbox_inches=bbox_inches)
        print(f'Saved: {out_path}')
    if show:
        plt.show()
    plt.close(fig)
    return fig


# --------------------------------------------------------------------------- #
# argparse
# --------------------------------------------------------------------------- #
def add_output_args(parser):
    """Add the shared `--out` option (dest=out_dir). Returns the parser."""
    parser.add_argument('--out', dest='out_dir', default=None,
                        help='Output directory.')
    return parser


# --------------------------------------------------------------------------- #
# Method-comparison colours (shared by plot_methods / plot_pareto)
# --------------------------------------------------------------------------- #
DEFAULT_COLORS = {
    'manifold':   '#1f77b4',
    'magnitude':  '#d62728',
    'obd':        '#2ca02c',
    'obs':        '#9467bd',
    'lazarevich': '#ff7f0e',
    'kwon':       '#8c564b',
}


def resolve_colors(method_names, csv_dir):
    """Build a {method: color} mapping from a sidecar `colors.json` (optional)
    merged over DEFAULT_COLORS, with unlisted methods falling through to the
    matplotlib tab10 cycle."""
    palette = dict(DEFAULT_COLORS)
    sidecar = os.path.join(csv_dir, 'colors.json')
    if os.path.isfile(sidecar):
        with open(sidecar) as f:
            palette.update(json.load(f))

    fallback = plt.cm.tab10.colors
    fb = 0
    out = {}
    for name in method_names:
        if name in palette:
            out[name] = palette[name]
        else:
            out[name] = fallback[fb % len(fallback)]
            fb += 1
    return out


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_method_csvs(csv_dir):
    """Return [(name, DataFrame), ...] for every `*.csv` in `csv_dir`, sorted by
    filename; the name is the filename minus `.csv`."""
    csv_files = sorted(f for f in os.listdir(csv_dir) if f.endswith('.csv'))
    if not csv_files:
        raise ValueError(f'No CSV files found in: {csv_dir}')
    return [(f[:-4], pd.read_csv(os.path.join(csv_dir, f))) for f in csv_files]


def load_checkpoint_W(ckpt_dir):
    """Load all `W_i.npy` arrays from a checkpoint directory, ordered by index.

    Raises FileNotFoundError if the directory is missing, ValueError if it has
    no `W_*.npy` files.
    """
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")
    n = sum(1 for f in os.listdir(ckpt_dir)
            if f.startswith('W_') and f.endswith('.npy'))
    if n == 0:
        raise ValueError(f"No W_*.npy files found in: {ckpt_dir}")
    return [np.load(os.path.join(ckpt_dir, f'W_{i}.npy')) for i in range(n)]
