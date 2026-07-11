# visualize/plot_bias_drift.py
"""
Bias-drift over training: ||b[l] - b_0[l]||_2 per layer vs step.

Prerequisite: the input CSV must contain `bias_drift_<l>` columns (one per
layer) and optionally `bias_norm_<l>`. These are added by the logging change
referenced in `prompts/missing_tools.md` T11.

If bias-norm columns are missing, only the drift curves are plotted.

Usage:
    python visualize/plot_bias_drift.py \\
        artifacts/baseline/sparsified/sparsification_log.csv \\
        --out images/bias/baseline/
"""
import argparse
import os
import re
import sys

import pandas as pd
from visualize.viz_common import plt, save_fig, add_output_args


_DRIFT_RE = re.compile(r'^bias_drift_(\d+)$')
_NORM_RE  = re.compile(r'^bias_norm_(\d+)$')


def _layer_idx(name: str, regex):
    m = regex.match(name)
    return int(m.group(1)) if m else None


def plot_bias_drift(csv_path: str, out_dir: str = None,
                    show: bool = False) -> plt.Figure:
    """
    Single panel: one curve per layer of `||b[l] - b_0[l]||_2` vs step.
    If `bias_norm_<l>` columns exist they are overlaid as dashed lines.
    """
    df = pd.read_csv(csv_path)

    drift_cols = sorted(
        [c for c in df.columns if _DRIFT_RE.match(c)],
        key=lambda c: _layer_idx(c, _DRIFT_RE),
    )
    if not drift_cols:
        raise ValueError(
            'CSV missing `bias_drift_<l>` columns. '
            'Add the per-layer ||b - b0||_2 logging from prompts/missing_tools.md T11.')

    norm_cols = sorted(
        [c for c in df.columns if _NORM_RE.match(c)],
        key=lambda c: _layer_idx(c, _NORM_RE),
    )

    cmap = plt.cm.tab10

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle(f'Bias Drift — {os.path.basename(csv_path)}', fontsize=12)

    for col in drift_cols:
        l = _layer_idx(col, _DRIFT_RE)
        ax.plot(df['step'], df[col], color=cmap(l % 10),
                linewidth=1.4, label=f'L{l} ||b - b0||')

    for col in norm_cols:
        l = _layer_idx(col, _NORM_RE)
        ax.plot(df['step'], df[col], color=cmap(l % 10),
                linewidth=1.0, linestyle='--', alpha=0.6,
                label=f'L{l} ||b||')

    ax.set_xlabel('Step'); ax.set_ylabel('L2 norm')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return save_fig(fig, out_dir, 'bias_drift.png', show=show)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv_path', help='sparsification_log.csv')
    add_output_args(p)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    plot_bias_drift(args.csv_path, out_dir=args.out_dir,
                    show=args.out_dir is None)
