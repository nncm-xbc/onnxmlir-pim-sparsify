# visualize/plot_comparison.py
"""
Overlay accuracy and timing curves from multiple strategy CSVs.

Usage:
    python visualize/plot_comparison.py artifacts/comparison/
    python visualize/plot_comparison.py artifacts/comparison/ --out artifacts/comparison/
"""
import sys, os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Consistent colours per strategy name
_COLORS = {
    'exhaustive':      '#1f77b4',
    'gradient_top_1':  '#d62728',
    'gradient_top_10': '#ff7f0e',
    'gradient_top_50': '#2ca02c',
}


def plot_strategy_comparison(csv_dir: str, out_dir: str = None,
                              show: bool = True) -> plt.Figure:
    """
    Read all *.csv files in csv_dir and produce a 3-panel overlay:
      [0] Accuracy vs Sparsity
      [1] Search time per step
      [2] Total time per step (search + adjust)

    Note: with the Agg backend (set at module level), show=True is a silent no-op.
    The returned Figure is closed (deregistered from pyplot); use fig.savefig() directly.
    """
    csv_files = sorted(f for f in os.listdir(csv_dir) if f.endswith('.csv'))
    if not csv_files:
        raise ValueError(f'No CSV files found in: {csv_dir}')

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Strategy Comparison', fontsize=13)

    for fname in csv_files:
        name = fname[:-4]   # strip .csv
        df   = pd.read_csv(os.path.join(csv_dir, fname))
        c    = _COLORS.get(name, None)
        kw   = dict(label=name, color=c, linewidth=1.5)

        axes[0].plot(df['sparsity'] * 100, df['val_acc'] * 100, **kw)

        if 'prune_time_s' in df.columns and 'adjust_time_s' in df.columns:
            axes[1].plot(df['step'], df['prune_time_s'], **kw)
            axes[2].plot(df['step'], df['prune_time_s'] + df['adjust_time_s'], **kw)
        else:
            axes[1].text(0.5, 0.5, f'{name}: no timing data',
                         ha='center', va='center', transform=axes[1].transAxes, fontsize=8)
            axes[2].text(0.5, 0.5, f'{name}: no timing data',
                         ha='center', va='center', transform=axes[2].transAxes, fontsize=8)

    labels = [
        ('Sparsity (%)',  'Val Accuracy (%)',    'Accuracy vs Sparsity'),
        ('Step',          'Search time (s)',      'Search Time per Step'),
        ('Step',          'Total time (s)',        'Total Time (search + adjust)'),
    ]
    for ax, (xlabel, ylabel, title) in zip(axes, labels):
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'strategy_comparison.png')
        fig.savefig(out_path, dpi=150)
        print(f'Saved: {out_path}')
    if show:
        plt.show()
    plt.close(fig)
    return fig


if __name__ == '__main__':
    raw = sys.argv[1:]
    csv_dir = raw[0]
    out_dir = None
    for i, a in enumerate(raw):
        if a == '--out' and i + 1 < len(raw):
            out_dir = raw[i + 1]
    plot_strategy_comparison(csv_dir, out_dir=out_dir, show=out_dir is None)
