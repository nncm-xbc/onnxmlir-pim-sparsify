# visualize/plot_neuron_methods.py
"""
Neuron-pruning method comparison overlay.

Mirror of plot_methods.py, but for the NEURON schema
    step,neurons_pruned,total_neurons,neuron_sparsity,val_acc,d_manifold,d_W,...
(weight-schema plot_methods.py assumes `sparsity`/`NZ` columns, which neuron
runs don't have).

Reads one CSV per method from a directory; the filename (minus .csv) is the
legend label and the key into viz_common.DEFAULT_COLORS.

Usage:
    python -m visualize.plot_neuron_methods <csv_dir> --out images/neuron_comparison/
"""
import argparse
import sys

import numpy as np

from visualize.viz_common import (plt, save_fig, add_output_args,
                                  resolve_colors, load_method_csvs)


def plot_neuron_methods(csv_dir: str, out_dir: str = None,
                        show: bool = False) -> plt.Figure:
    """Two-panel overlay across neuron-pruning method CSVs in `csv_dir`.

    [0] Val accuracy (%) vs neuron sparsity (%), markers on each pruning step.
    [1] d_manifold vs step (log y).
    """
    loaded = load_method_csvs(csv_dir)
    colors = resolve_colors([name for name, _ in loaded], csv_dir)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle('Neuron-Pruning Method Comparison', fontsize=13)

    for name, df in loaded:
        kw = dict(label=name, color=colors[name], linewidth=1.6)
        axes[0].plot(df['neuron_sparsity'] * 100, df['val_acc'] * 100,
                     marker='o', markersize=3, **kw)
        d = df['d_manifold'].replace(0, np.nan)
        axes[1].semilogy(df['step'], d, marker='o', markersize=3, **kw)

    axes[0].set_xlabel('Neuron sparsity (%)')
    axes[0].set_ylabel('Val Accuracy (%)')
    axes[0].set_title('Accuracy vs Neuron Sparsity')
    axes[0].axvline(70, color='gray', linestyle=':', alpha=0.5)

    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('d_manifold')
    axes[1].set_title('Manifold Distance (log y)')

    for ax in axes:
        ax.legend(fontsize=9)

    plt.tight_layout()
    return save_fig(fig, out_dir, 'neuron_method_comparison.png', show=show)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv_dir', help='Directory with one neuron CSV per method.')
    add_output_args(p)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    plot_neuron_methods(args.csv_dir, out_dir=args.out_dir,
                        show=args.out_dir is None)
