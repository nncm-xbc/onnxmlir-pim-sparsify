"""
Live real-time dashboard for an in-progress sparsification run.
Polls the log CSV every interval_ms milliseconds using FuncAnimation.

Usage (run while sparsifier is running in another terminal):
    python visualize/live_view.py path/to/sparsification_log.csv
    python visualize/live_view.py path/to/sparsification_log.csv path/to/checkpoints/
"""
import sys
import os


def live_view(log_path: str, ckpt_base: str = None, interval_ms: int = 2000):
    """
    Open a live-updating matplotlib window polling log_path every interval_ms ms.

    Panels:
      [0,0] Accuracy vs Sparsity
      [0,1] NZ weight count vs step
      [0,2] Manifold distance vs step
      [1,0] Timing: search + adjust per step (if extended log)
      [1,1] Layer 0 weight mask from latest checkpoint (if ckpt_base given)
      [1,2] Info text panel (step, NZ, sparsity, val_acc, d_W, elapsed time)

    Args:
        log_path:    path to sparsification_log.csv (read live as it grows)
        ckpt_base:   path to checkpoints/ directory (optional, for mask panel)
        interval_ms: polling interval in milliseconds
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(15, 8))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    ax_acc  = fig.add_subplot(gs[0, 0])
    ax_nz   = fig.add_subplot(gs[0, 1])
    ax_d    = fig.add_subplot(gs[0, 2])
    ax_time = fig.add_subplot(gs[1, 0])
    ax_mask = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.axis('off')

    def _latest_checkpoint_W0(base):
        """Return W_0.npy from the most recent step_NNNN/ dir, or None."""
        if not base or not os.path.isdir(base):
            return None
        dirs = sorted([
            os.path.join(base, d) for d in os.listdir(base)
            if d.startswith('step_') and os.path.isdir(os.path.join(base, d))
        ])
        if not dirs:
            return None
        w_path = os.path.join(dirs[-1], 'W_0.npy')
        return np.load(w_path) if os.path.exists(w_path) else None

    def update(_frame):
        try:
            df = pd.read_csv(log_path)
        except Exception:
            return
        if len(df) == 0:
            return

        for ax in [ax_acc, ax_nz, ax_d, ax_time, ax_mask]:
            ax.clear()

        last = df.iloc[-1]

        # [0,0] Accuracy vs Sparsity
        ax_acc.plot(df['sparsity'] * 100, df['val_acc'] * 100,
                    'b-o', markersize=2, linewidth=1)
        ax_acc.axhline(float(df['val_acc'].iloc[0]) * 100,
                       color='gray', linestyle='--', alpha=0.5, label='Dense baseline')
        ax_acc.set_xlabel('Sparsity (%)'); ax_acc.set_ylabel('Val Acc (%)')
        ax_acc.set_title('Accuracy vs Sparsity')
        ax_acc.legend(fontsize=7)

        # [0,1] NZ vs step
        ax_nz.plot(df['step'], df['NZ'], 'g-', linewidth=1)
        ax_nz.set_xlabel('Step'); ax_nz.set_ylabel('NZ weights')
        ax_nz.set_title(
            f"Step {int(last['step'])} | NZ {int(last['NZ'])} / {int(last['total_W'])}")

        # [0,2] Manifold distance
        d_vals = df['d_manifold'].replace(0, np.nan)
        if d_vals.notna().any():
            ax_d.semilogy(df['step'], d_vals, 'r-', linewidth=1)
        ax_d.set_xlabel('Step'); ax_d.set_ylabel('d_manifold')
        ax_d.set_title('Manifold Distance')

        # [1,0] Timing
        if 'prune_time_s' in df.columns:
            ax_time.plot(df['step'], df['prune_time_s'],  label='search', linewidth=1)
            ax_time.plot(df['step'], df['adjust_time_s'], label='adjust', linewidth=1)
            ax_time.legend(fontsize=7)
            ax_time.set_xlabel('Step'); ax_time.set_ylabel('Time (s)')
            last_prune  = float(last['prune_time_s'])
            last_adjust = float(last['adjust_time_s'])
            ax_time.set_title(f'Last step: {last_prune:.2f}s + {last_adjust:.2f}s')
        else:
            ax_time.text(0.5, 0.5, 'No timing data\n(needs extended log)',
                         ha='center', va='center', transform=ax_time.transAxes)
            ax_time.set_title('Time per Step')

        # [1,1] Weight mask (latest checkpoint)
        W = _latest_checkpoint_W0(ckpt_base)
        if W is not None:
            ax_mask.imshow(W != 0, aspect='auto', cmap='gray_r', vmin=0, vmax=1)
            sp = 1.0 - float((W != 0).mean())
            ax_mask.set_title(f'L0 mask  sparsity={sp:.2f}')
        else:
            ax_mask.text(0.5, 0.5, 'No checkpoint yet\n(saved every 50 steps)',
                         ha='center', va='center', transform=ax_mask.transAxes, fontsize=9)
            ax_mask.set_title('L0 Weight Mask')
        ax_mask.axis('off')

        # [1,2] Info text
        info_lines = [
            f"step:     {int(last['step'])}",
            f"NZ:       {int(last['NZ'])} / {int(last['total_W'])}",
            f"sparsity: {float(last['sparsity'])*100:.2f}%",
            f"val_acc:  {float(last['val_acc'])*100:.2f}%",
            f"d_W:      {float(last['d_W']):.3e}",
        ]
        if 'prune_time_s' in df.columns:
            total_t = df['prune_time_s'].sum() + df['adjust_time_s'].sum()
            info_lines.append(f"elapsed:  {total_t:.1f}s")
        ax_info.clear(); ax_info.axis('off')
        ax_info.text(
            0.05, 0.95, '\n'.join(info_lines),
            transform=ax_info.transAxes,
            fontfamily='monospace', fontsize=10, verticalalignment='top',
            bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round'),
        )

        fig.suptitle(
            f'Live Sparsification — {os.path.basename(log_path)}', fontsize=12)

    ani = animation.FuncAnimation(   # noqa: F841  kept alive by plt.show()
        fig, update, interval=interval_ms, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Set interactive backend only when running as a script, not on import.
    # TkAgg works on macOS/Linux; change to 'Qt5Agg' or 'MacOSX' if unavailable.
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass   # fall back to whatever is available

    log  = sys.argv[1]
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    live_view(log, ckpt_base=ckpt)
