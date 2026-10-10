"""Figures for the full manuscript (seed band, crossbar staircase, KMNIST Omega, neuron Omega gap).

Usage: python visualize/full_figures.py [--out DIR]
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from visualize.viz_common import plt

REPO = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(REPO, 'scripts'))
from analyze_seedband import RUNS, stats            # noqa: E402
from backend.compact import crossbar_cost            # noqa: E402

OUT_DEFAULT = '/home/simon/repos/ThesisTex-full/main_doc/Images'
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.spines.top': False,
    'axes.spines.right': False, 'figure.constrained_layout.use': True,
})
COL = {'exhaustive': 'C0', 'search': 'C0', 'obd': 'C4', 'magnitude': 'C1', 'kwon': 'C2',
       'obs': 'C5', 'scratch': '0.45'}
NAME = {'exhaustive': 'exhaustive $d_W$ search', 'obd': 'OBD with $d_W$', 'magnitude': 'magnitude',
        'kwon': 'Kwon (first-order)', 'obs': 'OBS (layer-wise, no adjustment)'}
LW = 1.5


def _csv(path):
    return pd.read_csv(os.path.join(REPO, 'artifacts', path))


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, name + '.pdf')
    fig.savefig(p, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p)


# ---------------------------------------------------------------- 1. seed band
def fig_seedband(out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.6), gridspec_kw={'width_ratios': [2, 1]})
    order = ['exhaustive', 'obd', 'magnitude', 'obs', 'kwon']
    grid = np.linspace(0, 100, 401)
    coll = {s: [] for s in order}
    for sel in order:
        Y = []
        for s, _, run, log in [r for r in RUNS if r[0] == sel]:
            df = _csv(f'{run}/{log}')
            acc = pd.Series(df.val_acc).rolling(25, center=True, min_periods=1).median().values
            Y.append(np.interp(grid, df.sparsity.values * 100, acc))
            coll[sel].append(stats(df)['collapse_sp'])
        Y = np.stack(Y)
        ax1.fill_between(grid, Y.min(0), Y.max(0), color=COL[sel], alpha=0.15, lw=0)
        ax1.plot(grid, Y.mean(0), color=COL[sel], lw=LW + 0.5, label=NAME[sel])
    ax1.set_xlabel('sparsity [%]'); ax1.set_ylabel('validation accuracy (rolling median)')
    ax1.set_xlim(0, 100); ax1.set_ylim(0, 1)
    ax1.legend(loc='lower left', frameon=False)
    ax1.text(0.02, 0.98, '(a) seed mean and min–max band, 5 seeds', transform=ax1.transAxes, va='top')
    rng = np.random.default_rng(0)
    for i, sel in enumerate(order):
        c = np.array(coll[sel])
        ax2.scatter(i + rng.uniform(-0.15, 0.15, len(c)), c, s=14, color=COL[sel], zorder=3)
        ax2.errorbar(i, c.mean(), yerr=c.std(ddof=1), fmt='_', color='k', ms=14, capsize=3, lw=1, zorder=4)
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(['exhaustive', 'OBD', 'magnitude', 'OBS', 'Kwon'], rotation=35, ha='right')
    ax2.set_ylabel('collapse sparsity [%]')
    ax2.text(0.02, 0.04, '(b) collapse per seed', transform=ax2.transAxes, fontsize=8)
    _save(fig, out_dir, 'fig_seedband')


# ----------------------------------------------------- 2. crossbar staircase
def _staircase(run_glob, n_in):
    """{seed: {crossbars: acc_first}} for every log matching run_glob."""
    out = {}
    for csv in sorted(glob.glob(os.path.join(REPO, 'artifacts', run_glob))):
        df = pd.read_csv(csv)
        xb = [crossbar_cost([n_in, int(a), int(b), 10])['crossbars']
              for a, b in zip(df.layer_0_neurons, df.layer_1_neurons)]
        df['xb'] = xb
        d = {int(k): float(g.val_acc.iloc[0]) for k, g in df.groupby('xb') if k > 0}
        out[csv.split('/')[-3]] = d
    return out


def _controls(pattern):
    """{crossbars: [test_acc per seed]} from e4x_*_<xb>xb_seed_k training logs."""
    out = {}
    for p in sorted(glob.glob(os.path.join(REPO, 'artifacts', pattern, 'training_log.csv'))):
        xb = int(p.split('_scratch_')[1].split('xb')[0])
        out.setdefault(xb, []).append(float(pd.read_csv(p).test_acc.iloc[-1]))
    return out


def _plot_stairs(ax, runs, color, label, ls='-'):
    allk = sorted({k for d in runs.values() for k in d}, reverse=True)
    means = {}
    for d in runs.values():
        ks = sorted(d, reverse=True)
        ax.plot(ks, [d[k] for k in ks], color=color, lw=0.8, ls=ls, alpha=0.45)
    for k in allk:
        v = [d[k] for d in runs.values() if k in d]
        means[k] = (np.mean(v), len(v))
    ks = [k for k in allk if means[k][1] >= 2]  # ponytail: mean marker only where >=2 seeds reached it
    ax.plot(ks, [means[k][0] for k in ks], color=color, lw=LW, ls=ls, marker='o', ms=4, label=label)
    return means


def fig_crossbar(out_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.7))
    report = {}
    for ax, n_in, tag, srch, mag, noise, ctrl, dense_runs in [
        (ax1, 196, '(a) [196,200,200,10], S=128', 'e38_w200_neuron_data_seed_*/neuron_sparsified/*.csv',
         'e38_w200_neuron_data_seed_*/neuron_magnitude_sparsified/*.csv',
         'e39_w200_neuron_noise_seed_*/neuron_sparsified/*.csv', 'e40_w200_scratch_*', 'e38_w200_neuron_data_seed_*'),
        (ax2, 784, '(b) [784,512,512,10], S=128', 'e42_m784_neuron_data_seed_*/neuron_sparsified/*.csv',
         'e42_m784_neuron_data_seed_*/neuron_magnitude_sparsified/*.csv', None, 'e43_m784_scratch_*',
         'e42_m784_neuron_data_seed_*'),
    ]:
        rep = {}
        rep['search'] = _plot_stairs(ax, _staircase(srch, n_in), COL['search'], '$d_W$ search, training-image $\\Omega$')
        rep['magnitude'] = _plot_stairs(ax, _staircase(mag, n_in), COL['magnitude'], 'magnitude')
        if noise:
            rep['noise'] = _plot_stairs(ax, _staircase(noise, n_in), COL['search'], '$d_W$ search, noise $\\Omega$', ls='--')
        c = _controls(ctrl)
        ks = sorted(c, reverse=True)
        ax.errorbar(ks, [np.mean(c[k]) for k in ks], yerr=[np.std(c[k], ddof=1) for k in ks], fmt='s',
                    color=COL['scratch'], ms=4, capsize=3, lw=1, label='trained from scratch at that shape')
        rep['scratch'] = {k: (np.mean(c[k]), len(c[k])) for k in ks}
        dense = [float(pd.read_csv(p).test_acc.iloc[-1])
                 for p in glob.glob(os.path.join(REPO, 'artifacts', dense_runs, 'training_log.csv'))]
        ax.axhline(np.mean(dense), color='k', lw=0.8, ls=':')
        ax.text(ax.get_xlim()[0], np.mean(dense) + 0.004, f'dense {np.mean(dense):.3f}', fontsize=7, ha='left')
        rep['dense'] = (np.mean(dense), len(dense))
        ax.invert_xaxis()
        ax.set_xlabel('crossbars after compaction'); ax.set_ylabel('test accuracy at first step reaching count')
        ax.set_ylim(0.6, 1.0)
        ax.text(0.02, 0.04, tag, transform=ax.transAxes, fontsize=8)
        ax.legend(loc='lower left', bbox_to_anchor=(0.0, 0.1), frameon=False, fontsize=7)
        report[tag] = rep
    _save(fig, out_dir, 'fig_crossbar_staircase')
    for tag, rep in report.items():
        print(tag)
        for k, v in rep.items():
            print('  ', k, v if not isinstance(v, dict) else {kk: (round(m, 3), n) for kk, (m, n) in sorted(v.items(), reverse=True)})


# ------------------------------------------------------------ 3. KMNIST Omega
def fig_kmnist(out_dir):
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    for path, color, ls, label in [
        ('e27_kmnist/sparsified/sparsification_log.csv', COL['exhaustive'], '--', 'exhaustive $d_W$ search, noise $\\Omega$'),
        ('e27_kmnist/obd_sparsified/obd_sparsification_log.csv', COL['obd'], '--', 'OBD with $d_W$, noise $\\Omega$'),
        ('e27_kmnist/magnitude_sparsified/magnitude_sparsification_log.csv', COL['magnitude'], '-', 'magnitude'),
        ('e27b_kmnist_trainomega/sparsified/sparsification_log.csv', COL['exhaustive'], '-', 'exhaustive $d_W$ search, training-image $\\Omega$'),
    ]:
        df = _csv(path)
        ax.plot(df.step, df.val_acc, color=color, ls=ls, lw=LW, label=label)
    ax.axhline(0.703, color='k', lw=0.8, ls=':'); ax.text(300, 0.708, 'dense 0.703', fontsize=7)
    ax.set_xlabel('pruning step (500 steps = 23.1% weight sparsity)'); ax.set_ylabel('validation accuracy')
    ax.set_xlim(0, 500); ax.set_ylim(0.45, 0.75)
    ax.legend(loc='lower left', frameon=False)
    _save(fig, out_dir, 'fig_kmnist_omega')


# ------------------------------------------------------- 4. neuron Omega gap
def fig_omega_gap(out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 2.5), sharey=True)
    for ax, tag, noise, data, dlab in [
        (axes[0], '(a) MNIST, [196,10,10,10]', 'e11_neuron_baseline', 'e33_neuron_data', 'test-row $\\Omega$'),
        (axes[1], '(b) S1 teacher data, [196,10,10,10]', 'e31_s1_neuron_noise', 'e32_s1_neuron_data', 'data $\\Omega$'),
    ]:
        for run, ls, label in [(noise, '--', 'noise $\\Omega$'), (data, '-', dlab)]:
            df = _csv(f'{run}/neuron_sparsified/neuron_sparsification_log.csv')
            ax.plot(df.neuron_sparsity * 100, df.val_acc, color=COL['search'], ls=ls, lw=LW, marker='o', ms=3, label=label)
        ax.set_xlabel('hidden neurons removed [%]'); ax.set_xlim(0, 75)
        ax.text(0.02, 0.04, tag, transform=ax.transAxes, fontsize=8)
        ax.legend(loc='upper right', frameon=False)
    axes[0].set_ylabel('validation accuracy')
    _save(fig, out_dir, 'fig_omega_gap_neuron')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--out', default=OUT_DEFAULT)
    out = ap.parse_args().out
    fig_seedband(out); fig_crossbar(out); fig_kmnist(out); fig_omega_gap(out)
