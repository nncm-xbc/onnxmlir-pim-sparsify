"""
Live real-time dashboard for an in-progress sparsification run.
Polls the log CSV every interval_ms milliseconds using FuncAnimation.

Usage (run while sparsifier is running in another terminal):
    python visualize/live_view.py path/to/sparsification_log.csv
    python visualize/live_view.py path/to/sparsification_log.csv path/to/checkpoints/
    python visualize/live_view.py path/to/sparsification_log.csv --target-sparsity 0.9
"""
import datetime
import sys
import os


# Window size (in steps) used for the median-per-step pace estimate.
_PACE_WINDOW = 20


def _format_seconds(s: float) -> str:
    """Pretty-print a duration in seconds as e.g. '1:12:03'."""
    if s != s or s < 0:                          # NaN guard
        return '—'
    return str(datetime.timedelta(seconds=int(s)))


def live_view(log_path: str, ckpt_base: str = None, interval_ms: int = 2000,
              target_sparsity: float = None):
    """
    Open a live-updating matplotlib window polling log_path every interval_ms ms.

    Panels (2 rows x 4 cols):
      [0,0] Accuracy vs Sparsity
      [0,1] NZ weight count vs step
      [0,2] Manifold distance vs step
      [0,3] Per-layer NZ over time              (new)
      [1,0] Timing: search + adjust per step
      [1,1] Layer 0 weight mask from latest checkpoint
      [1,2] Info text (step, NZ, sparsity, val_acc, d_W, elapsed)
      [1,3] Pace + estimated remaining time     (new)

    Args:
        log_path:        path to sparsification_log.csv (read live as it grows)
        ckpt_base:       path to checkpoints/ directory (optional, mask panel)
        interval_ms:     polling interval in milliseconds
        target_sparsity: optional target (0..1). If given, ETA is reported
                         alongside the pace estimate; otherwise only pace is shown.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(18, 8))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)
    ax_acc   = fig.add_subplot(gs[0, 0])
    ax_nz    = fig.add_subplot(gs[0, 1])
    ax_d     = fig.add_subplot(gs[0, 2])
    ax_layer = fig.add_subplot(gs[0, 3])
    ax_time  = fig.add_subplot(gs[1, 0])
    ax_mask  = fig.add_subplot(gs[1, 1])
    ax_info  = fig.add_subplot(gs[1, 2])
    ax_eta   = fig.add_subplot(gs[1, 3])
    ax_info.axis('off')
    ax_eta.axis('off')

    def _latest_checkpoint_W0(base):
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

    def _pace_estimate(df):
        """Return (median_seconds_per_step, source_steps_used)."""
        if 'prune_time_s' not in df.columns or 'adjust_time_s' not in df.columns:
            return float('nan'), 0
        per_step = (df['prune_time_s'].fillna(0)
                    + df['adjust_time_s'].fillna(0)).to_numpy(dtype=float)
        if per_step.size == 0:
            return float('nan'), 0
        window = per_step[-_PACE_WINDOW:]
        return float(np.median(window)), int(window.size)

    def update(_frame):
        try:
            df = pd.read_csv(log_path)
        except Exception:
            return
        if len(df) == 0:
            return

        for ax in [ax_acc, ax_nz, ax_d, ax_layer, ax_time, ax_mask]:
            ax.clear()

        last = df.iloc[-1]

        # [0,0] Accuracy vs Sparsity
        ax_acc.plot(df['sparsity'] * 100, df['val_acc'] * 100,
                    'b-o', markersize=2, linewidth=1)
        ax_acc.axhline(float(df['val_acc'].iloc[0]) * 100,
                       color='gray', linestyle='--', alpha=0.5,
                       label='Dense baseline')
        ax_acc.set_xlabel('Sparsity (%)'); ax_acc.set_ylabel('Val Acc (%)')
        ax_acc.set_title('Accuracy vs Sparsity')
        ax_acc.legend(fontsize=7)

        # [0,1] NZ vs step
        ax_nz.plot(df['step'], df['NZ'], 'g-', linewidth=1)
        ax_nz.set_xlabel('Step'); ax_nz.set_ylabel('NZ weights')
        ax_nz.set_title(
            f"Step {int(last['step'])} | "
            f"NZ {int(last['NZ'])} / {int(last['total_W'])}")

        # [0,2] Manifold distance
        d_vals = df['d_manifold'].replace(0, np.nan)
        if d_vals.notna().any():
            ax_d.semilogy(df['step'], d_vals, 'r-', linewidth=1)
        ax_d.set_xlabel('Step'); ax_d.set_ylabel('d_manifold')
        ax_d.set_title('Manifold Distance')

        # [0,3] Per-layer NZ over time (new)
        layer_cols = [c for c in df.columns
                      if c.startswith('layer_') and c.endswith('_NZ')]
        if layer_cols:
            for col in sorted(layer_cols,
                              key=lambda c: int(c.split('_')[1])):
                lbl = col.replace('layer_', 'L').replace('_NZ', '')
                ax_layer.plot(df['step'], df[col], label=lbl, linewidth=1.2)
            ax_layer.legend(fontsize=7)
            ax_layer.set_xlabel('Step'); ax_layer.set_ylabel('NZ per layer')
            ax_layer.set_title('Per-layer Sparsification')
        else:
            ax_layer.text(0.5, 0.5, 'No per-layer data\n(needs extended log)',
                          ha='center', va='center',
                          transform=ax_layer.transAxes, fontsize=9)
            ax_layer.set_title('Per-layer Sparsification')

        # [1,0] Timing
        if 'prune_time_s' in df.columns:
            ax_time.plot(df['step'], df['prune_time_s'],
                         label='search', linewidth=1)
            ax_time.plot(df['step'], df['adjust_time_s'],
                         label='adjust', linewidth=1)
            ax_time.legend(fontsize=7)
            ax_time.set_xlabel('Step'); ax_time.set_ylabel('Time (s)')
            ax_time.set_title(
                f'Last step: {float(last["prune_time_s"]):.2f}s + '
                f'{float(last["adjust_time_s"]):.2f}s')
        else:
            ax_time.text(0.5, 0.5, 'No timing data\n(needs extended log)',
                         ha='center', va='center',
                         transform=ax_time.transAxes)
            ax_time.set_title('Time per Step')

        # [1,1] Weight mask (latest checkpoint)
        W = _latest_checkpoint_W0(ckpt_base)
        if W is not None:
            ax_mask.imshow(W != 0, aspect='auto', cmap='gray_r',
                           vmin=0, vmax=1)
            sp = 1.0 - float((W != 0).mean())
            ax_mask.set_title(f'L0 mask  sparsity={sp:.2f}')
        else:
            ax_mask.text(0.5, 0.5, 'No checkpoint yet\n(saved every 50 steps)',
                         ha='center', va='center',
                         transform=ax_mask.transAxes, fontsize=9)
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
            total_t = (df['prune_time_s'].fillna(0).sum()
                       + df['adjust_time_s'].fillna(0).sum())
            info_lines.append(f"elapsed:  {_format_seconds(float(total_t))}")
        ax_info.clear(); ax_info.axis('off')
        ax_info.text(
            0.05, 0.95, '\n'.join(info_lines),
            transform=ax_info.transAxes,
            fontfamily='monospace', fontsize=10, verticalalignment='top',
            bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round'),
        )

        # [1,3] Pace + ETA (new)
        ax_eta.clear(); ax_eta.axis('off')
        median_step, n_used = _pace_estimate(df)
        eta_lines = [
            f"pace (median of last {n_used}):",
            f"   {median_step:.2f} s/step" if median_step == median_step else
            "   —  (no timing data)",
        ]
        if (target_sparsity is not None and median_step == median_step
                and 'total_W' in df.columns):
            current_nz = int(last['NZ'])
            total_w    = int(last['total_W'])
            target_nz  = int(round((1.0 - target_sparsity) * total_w))
            remaining_steps = max(0, current_nz - target_nz)
            eta_seconds = remaining_steps * median_step
            eta_lines.extend([
                '',
                f"target sparsity: {target_sparsity*100:.1f}%",
                f"steps remaining: {remaining_steps}",
                f"ETA:             {_format_seconds(eta_seconds)}",
            ])
        else:
            eta_lines.extend([
                '',
                '(pass --target-sparsity for ETA)',
            ])
        ax_eta.text(
            0.05, 0.95, '\n'.join(eta_lines),
            transform=ax_eta.transAxes,
            fontfamily='monospace', fontsize=10, verticalalignment='top',
            bbox=dict(facecolor='lightcyan', alpha=0.8, boxstyle='round'),
        )

        fig.suptitle(
            f'Live Sparsification — {os.path.basename(log_path)}', fontsize=12)

    ani = animation.FuncAnimation(   # noqa: F841  kept alive by plt.show()
        fig, update, interval=interval_ms, cache_frame_data=False)
    plt.tight_layout()
    plt.show()


def _parse_args(argv):
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('log_path', help='sparsification_log.csv (read live).')
    p.add_argument('ckpt_base', nargs='?', default=None,
                   help='checkpoints/ directory (optional).')
    p.add_argument('--target-sparsity', type=float, default=None,
                   help='Target sparsity in [0, 1] for ETA computation.')
    p.add_argument('--interval-ms', type=int, default=2000,
                   help='Polling interval in milliseconds (default: 2000).')
    return p.parse_args(argv)


if __name__ == '__main__':
    # Set interactive backend only when running as a script, not on import.
    # TkAgg works on macOS/Linux; change to 'Qt5Agg' or 'MacOSX' if unavailable.
    import matplotlib
    try:
        matplotlib.use('TkAgg')
    except Exception:
        pass   # fall back to default — may be non-interactive (Agg) on headless systems

    args = _parse_args(sys.argv[1:])
    live_view(
        args.log_path,
        ckpt_base=args.ckpt_base,
        interval_ms=args.interval_ms,
        target_sparsity=args.target_sparsity,
    )
