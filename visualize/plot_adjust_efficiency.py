# visualize/plot_adjust_efficiency.py
"""
How much work the adjust step does — separates search-step damage from
adjust-step rescue.

Prerequisite: the input CSV must contain `d_manifold_pre_adjust` (added by
Group E3 in `prompts/missing_experiments.md`). Without that column the script
exits with a clear message instead of guessing.

Usage:
    python visualize/plot_adjust_efficiency.py \\
        artifacts/baseline/sparsified/sparsification_log.csv \\
        --out images/adjust/baseline/
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from visualize.viz_common import plt, save_fig, add_output_args


def plot_adjust_efficiency(csv_path: str, out_dir: str = None,
                           show: bool = False) -> plt.Figure:
    """
    Three panels:
      [0] d_manifold_pre_adjust vs d_manifold (post-adjust) overlaid by step.
      [1] Ratio post / pre on log y. Closer to 0 = adjust working hard.
      [2] Cumulative d_W (weight shift due to adjust) vs cumulative pre-adjust
          damage; annotates the step where ratio crosses 0.5.
    """
    df = pd.read_csv(csv_path)
    if 'd_manifold_pre_adjust' not in df.columns:
        raise ValueError(
            'CSV missing column `d_manifold_pre_adjust`. '
            'Add the logging change from prompts/missing_experiments.md '
            'group E3 first.')

    pre  = df['d_manifold_pre_adjust'].astype(float).to_numpy()
    post = df['d_manifold'].astype(float).to_numpy()
    step = df['step'].astype(float).to_numpy()
    dW   = df['d_W'].astype(float).to_numpy()

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.where(pre > 0, post / pre, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Adjust Efficiency — {os.path.basename(csv_path)}',
                 fontsize=12)

    # ------ Panel 1: pre vs post overlaid -----------------------------------
    ax = axes[0]
    ax.plot(step, pre,  label='d_manifold_pre_adjust', color='tab:red',
            linewidth=1.2)
    ax.plot(step, post, label='d_manifold (post)',     color='tab:blue',
            linewidth=1.2)
    ax.fill_between(step, post, pre, where=(pre > post),
                    color='tab:green', alpha=0.18, label='adjust impact')
    ax.set_xlabel('Step'); ax.set_ylabel('d_manifold')
    ax.set_yscale('symlog', linthresh=1e-6)
    ax.set_title('Pre vs Post adjust')
    ax.legend(fontsize=8)

    # ------ Panel 2: ratio --------------------------------------------------
    ax = axes[1]
    ax.semilogy(step, ratio, color='tab:purple', linewidth=1.2)
    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6,
               label='ratio = 0.5')

    # Annotate first crossing (last index where ratio <= 0.5).
    crossings = np.where(ratio > 0.5)[0]
    if crossings.size > 0:
        idx = int(crossings[0])
        ax.axvline(step[idx], color='red', linestyle=':', alpha=0.7,
                   label=f'first ratio>0.5 @ step {int(step[idx])}')
    ax.set_xlabel('Step'); ax.set_ylabel('post / pre')
    ax.set_title('Ratio post/pre (log y)')
    ax.legend(fontsize=8)

    # ------ Panel 3: cumulative dW vs cumulative pre damage -----------------
    ax = axes[2]
    cum_dW   = np.cumsum(np.where(np.isnan(dW),  0, dW))
    cum_dmg  = np.cumsum(np.where(np.isnan(pre), 0, pre))
    ax.plot(cum_dmg, cum_dW, color='tab:orange', linewidth=1.4)
    ax.set_xlabel('Cumulative pre-adjust damage')
    ax.set_ylabel('Cumulative |ΔW| from adjust')
    ax.set_title('Adjust effort vs pruning damage')

    plt.tight_layout()
    return save_fig(fig, out_dir, 'adjust_efficiency.png', show=show)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv_path', help='sparsification_log.csv')
    add_output_args(p)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    plot_adjust_efficiency(args.csv_path, out_dir=args.out_dir,
                           show=args.out_dir is None)
