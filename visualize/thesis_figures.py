"""Thesis figure generation — vector PDFs for the experiments chapter.

Produces (into --out, default /home/simon/repos/ThesisTex/main_doc/Images/):
    fig_seed_variance.pdf      seed variance bands (e10_seed_1..4)
    fig_method_comparison.pdf  method comparison: acc-vs-sparsity + search time
    fig_stupidity.pdf          e05 stupidity-point trajectory
    fig_architecture.pdf       width / depth sweeps

PNG previews are written next to the repo in images/thesis_preview/.

Usage:
    python visualize/thesis_figures.py [--only seed,methods,stupidity,arch]
"""
import argparse
import os

import numpy as np
import pandas as pd
from visualize.viz_common import plt

REPO = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
OUT_DEFAULT = '/home/simon/repos/ThesisTex/main_doc/Images'
PREVIEW = os.path.join(REPO, 'images', 'thesis_preview')

plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.constrained_layout.use': True,
})


def _load(run, sub='sparsified', log='sparsification_log.csv'):
    p = os.path.join(REPO, 'artifacts', run, sub, log)
    return pd.read_csv(p) if os.path.exists(p) else None


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(PREVIEW, exist_ok=True)
    pdf = os.path.join(out_dir, name + '.pdf')
    fig.savefig(pdf, bbox_inches='tight')
    fig.savefig(os.path.join(PREVIEW, name + '.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', pdf)


def _seed_band(ax, dfs, col, color, label, logy=False, smooth=None):
    n = min(len(d) for d in dfs)
    steps = dfs[0]['step'].values[:n]
    Y = np.stack([d[col].values[:n] for d in dfs])
    if smooth:
        Y = np.stack([
            pd.Series(y).rolling(smooth, center=True, min_periods=1).median().values
            for y in Y
        ])
    mean, std = Y.mean(axis=0), Y.std(axis=0, ddof=1)
    ax.plot(steps, mean, color=color, lw=1.2, label=label)
    lo = mean - std
    if logy:
        lo = np.maximum(lo, np.maximum(mean * 1e-3, 1e-12))
    ax.fill_between(steps, lo, mean + std, color=color, alpha=0.25, lw=0)
    return steps, mean, std


# ---------------------------------------------------------------- seed variance
def fig_seed_variance(out_dir):
    dfs = [_load(f'e10_seed_{s}') for s in (1, 2, 3, 4)]
    dfs = [d for d in dfs if d is not None]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.5))

    _seed_band(ax1, dfs, 'val_acc', 'C0', 'mean of 4 seeds')
    for d in dfs:
        ax1.plot(d['step'], d['val_acc'], color='C0', alpha=0.25, lw=0.5)
    ax1.set_xlabel('pruning step')
    ax1.set_ylabel('validation accuracy')
    ax1.set_title('(a) accuracy', loc='left')
    ax1.legend(frameon=False, loc='lower left')

    for d in dfs:
        ax2.plot(d['step'], d['d_W'].clip(lower=1e-12), color='C3', alpha=0.12, lw=0.4)
    _seed_band(ax2, dfs, 'd_W', 'C3', 'rolling median, 4 seeds', logy=True, smooth=25)
    ax2.set_yscale('log')
    ax2.set_xlabel('pruning step')
    ax2.set_ylabel(r'$\|\Delta w\|_2$ per step')
    ax2.set_title('(b) parameter-space step size', loc='left')
    ax2.legend(frameon=False, loc='lower left')

    _save(fig, out_dir, 'fig_seed_variance')


# ------------------------------------------------------------ method comparison
METHODS = [
    # (label, run dir, subdir, log name, color)
    ('Magnitude',        'e09_magnitude_search', 'magnitude_sparsified',  'magnitude_sparsification_log.csv',  'C1'),
    ('Kwon (Fisher)',    'e12_kwon',             'kwon_sparsified',       'kwon_sparsification_log.csv',       'C2'),
    ('OBD',              'e13_obd',              'obd_sparsified',        'obd_sparsification_log.csv',        'C4'),
    ('OBS',              'e14_obs',              'obs_sparsified',        'obs_sparsification_log.csv',        'C5'),
    ('Lazarevich (data Omega)', 'e08_omega_data', 'lazarevich_sparsified', 'lazarevich_sparsification_log.csv', 'C6'),
]


def fig_method_comparison(out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.7),
                                   gridspec_kw={'width_ratios': [1.6, 1]})

    # manifold band (e10 seeds) on the sparsity axis
    dfs = [_load(f'e10_seed_{s}') for s in (1, 2, 3, 4)]
    dfs = [d for d in dfs if d is not None]
    n = min(len(d) for d in dfs)
    sp = dfs[0]['sparsity'].values[:n] * 100
    Y = np.stack([d['val_acc'].values[:n] for d in dfs])
    ax1.plot(sp, Y.mean(0), color='C0', lw=1.4, label='Manifold (ours, 4 seeds)')
    ax1.fill_between(sp, Y.mean(0) - Y.std(0, ddof=1), Y.mean(0) + Y.std(0, ddof=1),
                     color='C0', alpha=0.25, lw=0)

    # full-budget (2160-step) trajectories where available
    d05 = _load('e05_stupidity_point')
    if d05 is not None and len(d05) > 600:
        ax1.plot(d05['sparsity'] * 100, d05['val_acc'], color='C0', lw=0.9, ls='--',
                 label='Manifold, full budget (seed 0)')
    full_runs = {
        'Magnitude': ('e15_magnitude_full', 'magnitude_sparsified', 'magnitude_sparsification_log.csv'),
        'Kwon (Fisher)': ('e16_kwon_full', 'kwon_sparsified', 'kwon_sparsification_log.csv'),
    }

    bars = [('Manifold', np.concatenate([d['prune_time_s'].values for d in dfs]).mean(), 'C0')]
    for label, run, sub, log, color in METHODS:
        d = _load(run, sub, log)
        if d is None or len(d) < 5:
            print('  (skip %s — no data)' % label)
            continue
        bars.append((label.split(' ')[0], d['prune_time_s'].mean(), color))
        # prefer the full-budget trajectory for the accuracy panel if it exists
        if label in full_runs:
            dfull = _load(*full_runs[label])
            if dfull is not None and len(dfull) > 600:
                ax1.plot(dfull['sparsity'] * 100, dfull['val_acc'], color=color, lw=1.0,
                         label=label + ' (full budget)')
                continue
        ax1.plot(d['sparsity'] * 100, d['val_acc'], color=color, lw=1.0, label=label)

    ax1.set_xlabel('sparsity [%]')
    ax1.set_ylabel('validation accuracy')
    ax1.set_title('(a) accuracy vs. sparsity', loc='left')
    ax1.legend(frameon=False, fontsize=6.5, loc='lower left')

    names = [b[0] for b in bars]
    times = [b[1] for b in bars]
    colors = [b[2] for b in bars]
    ax2.bar(range(len(bars)), times, color=colors)
    ax2.set_yscale('log')
    ax2.set_xticks(range(len(bars)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.set_ylabel('mean search time per step [s]')
    ax2.set_title('(b) selection cost', loc='left')
    for x, t in enumerate(times):
        ax2.text(x, t * 1.15, f'{t:.3g}', ha='center', va='bottom', fontsize=6.5)

    _save(fig, out_dir, 'fig_method_comparison')


# ---------------------------------------------------------------- stupidity point
def _collapse_step(df):
    """First step with val_acc < 0.5 whose remaining trajectory stays below
    0.5 on average (sustained collapse). Returns (step, sparsity%) or None."""
    acc = df['val_acc'].values
    for i in range(len(acc)):
        if acc[i] < 0.5 and acc[i:].mean() < 0.5:
            return int(df['step'].values[i]), df['sparsity'].values[i] * 100
    return None


def fig_stupidity(out_dir):
    """Selector collapse overlay: manifold (e05) vs magnitude (e15) vs Kwon
    (e16), all driven to the full 2160-step budget on the same dense net."""
    runs = [
        ('Manifold search (e05)', _load('e05_stupidity_point'), 'C0'),
        ('Magnitude (e15)', _load('e15_magnitude_full', 'magnitude_sparsified',
                                  'magnitude_sparsification_log.csv'), 'C1'),
        ('Kwon / 1st-order Taylor (e16)', _load('e16_kwon_full', 'kwon_sparsified',
                                                'kwon_sparsification_log.csv'), 'C2'),
    ]
    runs = [(lab, df, c) for lab, df, c in runs if df is not None]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 4.0), sharex=True)

    ymark = {0: 0.60, 1: 0.45, 2: 0.30}
    for k, (lab, df, c) in enumerate(runs):
        ax1.plot(df['step'], df['val_acc'], color=c, lw=0.9, label=lab)
        ax2.plot(df['step'],
                 pd.Series(df['d_W'].clip(lower=1e-12)).rolling(25, center=True, min_periods=1).median(),
                 color=c, lw=0.9)
        cp = _collapse_step(df)
        if cp is not None:
            step_c, sp_c = cp
            ax1.axvline(step_c, color=c, lw=0.8, ls='--', alpha=0.8)
            ax2.axvline(step_c, color=c, lw=0.8, ls='--', alpha=0.8)
            ax1.annotate(f'step {step_c}\n({sp_c:.1f}%)',
                         xy=(step_c, ymark.get(k, 0.5)), xytext=(step_c - 60, ymark.get(k, 0.5)),
                         ha='right', va='center', fontsize=7, color=c,
                         arrowprops=dict(arrowstyle='->', lw=0.6, color=c))

    ax1.axhline(0.1, color='gray', lw=0.6, ls=':')
    ax1.text(30, 0.115, 'chance level', fontsize=6.5, color='gray')
    ax1.set_ylabel('validation accuracy')
    ax1.set_title('(a) accuracy collapse per selector (dashed: stupidity point)', loc='left')
    ax1.legend(frameon=False, fontsize=7, loc='lower left')

    ax2.set_yscale('log')
    ax2.set_ylabel(r'$\|\Delta w\|_2$ per step (rolling median)')
    ax2.set_xlabel('pruning step')
    ax2.set_title('(b) parameter-space step size', loc='left')

    # secondary x-axis: sparsity (one weight pruned per step for all runs)
    df0 = runs[0][1]
    total = df0['total_W'].iloc[0]

    def step2sp(x):
        return 100 * x / total

    def sp2step(x):
        return x * total / 100

    sec = ax1.secondary_xaxis('top', functions=(step2sp, sp2step))
    sec.set_xlabel('sparsity [%]', fontsize=8)
    sec.tick_params(labelsize=7)

    _save(fig, out_dir, 'fig_stupidity')


# ---------------------------------------------------------------- architecture
def fig_architecture(out_dir):
    """Width / depth sweep.

    NOTE (2026-06-12 verification): the archived e03_*/e04_* runs all pruned a
    byte-identical copy of the *baseline* [196,10,10,10] network — the
    configured topologies were never trained (see FABLE_RESULTS.md, Task 3).
    Curves drawn from those runs are labelled with their *actual* topology.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.5), sharey=True)

    # ----- width panel
    dfs10 = [d for d in (_load(f'e10_seed_{s}') for s in (1, 2, 3, 4)) if d is not None]
    _seed_band(ax1, dfs10, 'val_acc', 'C0', 'width 10 (baseline, 4 seeds)')
    for run, label, color, ls in [
        ('e03_width_50',  'e03_width_50 run*',  'C1', '--'),
        ('e03_width_200', 'e03_width_200 run*', 'C3', ':'),
    ]:
        d = _load(run)
        if d is not None:
            ax1.plot(d['step'], d['val_acc'], color=color, lw=1.0, ls=ls, label=label)
    ax1.set_xlabel('pruning step')
    ax1.set_ylabel('validation accuracy')
    ax1.set_title('(a) width sweep', loc='left')
    ax1.legend(frameon=False, fontsize=6.5, loc='lower left')

    # ----- depth panel
    _seed_band(ax2, dfs10, 'val_acc', 'C0', '3 layers (baseline, 4 seeds)')
    for run, label, color, ls in [
        ('e04_depth_2layer', 'e04_depth_2layer run*', 'C2', '--'),
        ('e04_depth_4layer', 'e04_depth_4layer run*', 'C5', ':'),
    ]:
        d = _load(run)
        if d is not None:
            ax2.plot(d['step'], d['val_acc'], color=color, lw=1.0, ls=ls, label=label)
    ax2.set_xlabel('pruning step')
    ax2.set_title('(b) depth sweep', loc='left')
    ax2.legend(frameon=False, fontsize=6.5, loc='lower left')

    fig.text(0.5, -0.04,
             '* archived e03/e04 runs pruned a copy of the baseline [196,10,10,10] network '
             '(intended topologies were never trained); curves show re-run variance only.',
             ha='center', fontsize=6.5, style='italic')

    _save(fig, out_dir, 'fig_architecture')


# ---------------------------------------------------------- neuron vs weight
def neuron_weight_fractions(df_neuron, run='e11_neuron_baseline'):
    """Cumulative fraction of prunable WEIGHTS zeroed after each neuron removal.

    Simulates the mask updates of sparsifier.neuron_sparsifier.prune_neuron:
    removing neuron i of layer l zeroes row W[l][i,:] and column W[l+1][:,i].
    Returns array of len(df)+1 entries: fraction after 0..N removals.
    """
    shapes = []
    k = 0
    while os.path.exists(os.path.join(REPO, 'artifacts', run, f'W_{k}.npy')):
        shapes.append(np.load(os.path.join(REPO, 'artifacts', run, f'W_{k}.npy')).shape)
        k += 1
    masks = [np.ones(s) for s in shapes]
    total = sum(m.size for m in masks)
    fr = [0.0]
    for _, row in df_neuron.iterrows():
        l, i = int(row['candidate_layer']), int(row['candidate_neuron'])
        masks[l][i, :] = 0.0
        masks[l + 1][:, i] = 0.0
        fr.append(1.0 - sum(m.sum() for m in masks) / total)
    return np.array(fr)


def fig_neuron_vs_weight(out_dir):
    """Accuracy vs cumulative fraction of prunable weights removed:
    structured (neuron, e11) vs unstructured (weight-level, e05/e15/e16)."""
    dn = _load('e11_neuron_baseline', 'neuron_sparsified', 'neuron_sparsification_log.csv')
    fig, ax = plt.subplots(figsize=(5.5, 3.0))

    for label, run, sub, log, color in [
        ('Manifold weight pruning (e05)', 'e05_stupidity_point', 'sparsified', 'sparsification_log.csv', 'C0'),
        ('Magnitude weight pruning (e15)', 'e15_magnitude_full', 'magnitude_sparsified', 'magnitude_sparsification_log.csv', 'C1'),
        ('Kwon weight pruning (e16)', 'e16_kwon_full', 'kwon_sparsified', 'kwon_sparsification_log.csv', 'C2'),
    ]:
        d = _load(run, sub, log)
        if d is not None and len(d) > 600:
            ax.plot(d['sparsity'], d['val_acc'], color=color, lw=0.9, label=label)

    if dn is not None:
        fr = neuron_weight_fractions(dn)
        # row k's val_acc is measured after k removals (logged before the k-th prune)
        ax.plot(fr[:len(dn)], dn['val_acc'], color='C3', lw=1.2, marker='o', ms=3.5,
                drawstyle='steps-post', label='Manifold neuron pruning (e11)')

    ax.axhline(0.1, color='gray', lw=0.6, ls=':')
    ax.text(0.02, 0.115, 'chance level', fontsize=6.5, color='gray')
    ax.set_xlabel('cumulative fraction of prunable weights removed')
    ax.set_ylabel('validation accuracy')
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=7, loc='upper right')
    _save(fig, out_dir, 'fig_neuron_vs_weight')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=OUT_DEFAULT)
    ap.add_argument('--only', default=None,
                    help='comma list: seed,methods,stupidity,arch,neuron')
    args = ap.parse_args()
    todo = set(args.only.split(',')) if args.only else {'seed', 'methods', 'stupidity', 'arch', 'neuron'}
    if 'seed' in todo:
        fig_seed_variance(args.out)
    if 'methods' in todo:
        fig_method_comparison(args.out)
    if 'stupidity' in todo:
        fig_stupidity(args.out)
    if 'arch' in todo:
        fig_architecture(args.out)
    if 'neuron' in todo:
        fig_neuron_vs_weight(args.out)


if __name__ == '__main__':
    main()
