# visualize/plot_run.py
"""
Static 6-panel summary plot from a sparsification log CSV.

Usage:
    python visualize/plot_run.py path/to/sparsification_log.csv [output_dir]
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — safe for headless/test environments
import matplotlib.pyplot as plt


def parse_log(log_path: str) -> pd.DataFrame:
    """Read a sparsification log CSV and return a DataFrame."""
    return pd.read_csv(log_path)


def plot_run(log_path: str, out_dir: str = None, show: bool = True) -> plt.Figure:
    """
    Produce a 6-panel summary figure from a sparsification log CSV.

    Panels:
      [0,0] Accuracy vs Sparsity
      [0,1] Manifold distance vs step (log scale)
      [0,2] Weight shift per step (||ΔW||₂)
      [1,0] Non-zero weight count vs step
      [1,1] Per-layer NZ over time (if extended log)
      [1,2] Timing breakdown: search vs adjust (if extended log)

    Note: with the Agg backend (set at module level), show=True is a silent no-op.
    Pass show=False when saving to file to be explicit.
    """
    df  = parse_log(log_path)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Sparsification Run — {os.path.basename(log_path)}', fontsize=13)

    # 1. Accuracy vs Sparsity
    ax = axes[0, 0]
    ax.plot(df['sparsity'] * 100, df['val_acc'] * 100, 'b-o', markersize=3)
    dense_acc = float(df['val_acc'].iloc[0]) * 100
    ax.axhline(dense_acc - 2, color='r', linestyle='--', alpha=0.5, label='Dense −2 pp')
    ax.set_xlabel('Sparsity (%)'); ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('Accuracy vs Sparsity'); ax.legend(fontsize=8)

    # 2. Manifold distance vs step (log scale where nonzero)
    ax = axes[0, 1]
    d_vals = df['d_manifold'].replace(0, np.nan)
    if d_vals.notna().any():
        ax.semilogy(df['step'], d_vals, 'r-', linewidth=1)
    else:
        ax.plot(df['step'], df['d_manifold'], 'r-', linewidth=1)
    ax.set_xlabel('Step'); ax.set_ylabel('d_manifold')
    ax.set_title('Manifold Distance (log scale)')

    # 3. Weight shift per step
    ax = axes[0, 2]
    ax.plot(df['step'], df['d_W'], 'g-', linewidth=1)
    ax.set_xlabel('Step'); ax.set_ylabel('||ΔW||₂')
    ax.set_title('Weight Shift per Step')

    # 4. Non-zero weight count
    ax = axes[1, 0]
    ax.plot(df['step'], df['NZ'], 'k-', linewidth=1.5)
    ax.set_xlabel('Step'); ax.set_ylabel('Non-zero weights')
    ax.set_title('Non-zero Weight Count')

    # 5. Per-layer sparsification (extended log only)
    ax = axes[1, 1]
    layer_cols = [c for c in df.columns if c.startswith('layer_') and c.endswith('_NZ')]
    if layer_cols:
        for col in layer_cols:
            label = col.replace('layer_', 'L').replace('_NZ', '')
            ax.plot(df['step'], df[col], label=label)
        ax.legend(fontsize=8)
        ax.set_xlabel('Step'); ax.set_ylabel('NZ per layer')
        ax.set_title('Per-layer Sparsification')
    else:
        ax.text(0.5, 0.5, 'No per-layer data\n(needs extended log)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Per-layer Sparsification')

    # 6. Timing breakdown (extended log only)
    ax = axes[1, 2]
    if 'prune_time_s' in df.columns:
        ax.plot(df['step'], df['prune_time_s'],  label='search', linewidth=1)
        ax.plot(df['step'], df['adjust_time_s'], label='adjust', linewidth=1)
        ax.legend(fontsize=8)
        ax.set_xlabel('Step'); ax.set_ylabel('Time (s)')
        ax.set_title('Time per Step')
    else:
        ax.text(0.5, 0.5, 'No timing data\n(needs extended log)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Time per Step')

    plt.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'run_summary.png')
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    if show:
        plt.show()
    plt.close(fig)
    return fig


if __name__ == '__main__':
    log     = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    plot_run(log, out_dir=out_dir, show=out_dir is None)
