# visualize/plot_mc_variance.py
"""
Monte-Carlo characterisation: justifies the choice of B (Ω draw count) used
elsewhere in the experiments by showing how distance estimates and
top-1/ranking stability behave as B grows.

Prerequisite: experiment E2 must have produced a CSV with the schema
    B, candidate_idx, sample_idx, d, rank
(typically `artifacts/mc_variance/results.csv`).

Each (B, sample_idx) row corresponds to one independent Ω draw of size B.
For each draw, the script reads candidate distances + their rank within that
draw.

Usage:
    python visualize/plot_mc_variance.py \\
        artifacts/mc_variance/results.csv --out images/mc_variance/
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from visualize.viz_common import plt, save_fig, add_output_args


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two 1-D arrays of the same length."""
    if len(a) < 2:
        return float('nan')
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def plot_mc_variance(csv_path: str, out_dir: str = None,
                     show: bool = False) -> plt.Figure:
    """
    Three panels:
      [0] Per-B scatter of distance values for one fixed candidate, with
          mean ± std bars overlaid.
      [1] Top-1 selection stability: fraction of independent draws that
          picked the same top-1 candidate as the modal choice, vs B.
      [2] Spearman rank correlation between rankings of two independent
          draws (averaged over all unordered pairs), vs B.

    Annotates the minimum B at which top-1 stability >= 90%.
    """
    df = pd.read_csv(csv_path)
    needed = {'B', 'candidate_idx', 'sample_idx', 'd', 'rank'}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f'CSV missing columns: {missing}')

    Bs = sorted(df['B'].unique().tolist())

    # Pick the candidate with the most rows for panel 1 (most stable view).
    candidate_idx = int(df['candidate_idx'].value_counts().idxmax())
    sub_cand = df[df['candidate_idx'] == candidate_idx]

    # Panel 2: top-1 stability per B.
    stabilities = []
    for B in Bs:
        sub = df[df['B'] == B]
        # Top-1 = each sample's own min-rank candidate (grouped per sample_idx,
        # not a global rank.min() across the whole B-group which biases the
        # stability fraction when samples don't share the same min rank).
        top1_per_sample = (
            sub.loc[sub.groupby('sample_idx')['rank'].idxmin()]
            .set_index('sample_idx')['candidate_idx']
        )
        if len(top1_per_sample) == 0:
            stabilities.append(float('nan'))
            continue
        modal = top1_per_sample.value_counts().iloc[0]
        stabilities.append(float(modal) / float(len(top1_per_sample)))
    stabilities = np.array(stabilities)

    # Panel 3: average pairwise Spearman per B.
    spearmans = []
    for B in Bs:
        sub = df[df['B'] == B]
        samples = sorted(sub['sample_idx'].unique().tolist())
        rs = []
        for i, si in enumerate(samples):
            for sj in samples[i + 1:]:
                ai = (sub[sub['sample_idx'] == si]
                      .sort_values('candidate_idx')['d'].to_numpy())
                aj = (sub[sub['sample_idx'] == sj]
                      .sort_values('candidate_idx')['d'].to_numpy())
                if len(ai) == len(aj) and len(ai) >= 2:
                    rs.append(_spearman(ai, aj))
        spearmans.append(np.nanmean(rs) if rs else float('nan'))
    spearmans = np.array(spearmans)

    min_B_90 = None
    for B, s in zip(Bs, stabilities):
        if not np.isnan(s) and s >= 0.9:
            min_B_90 = B
            break

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f'Monte-Carlo Variance — {os.path.basename(csv_path)}',
                 fontsize=12)

    # --- Panel 1: scatter + mean ± std for one candidate -------------------
    ax = axes[0]
    for B in Bs:
        ds = sub_cand[sub_cand['B'] == B]['d'].to_numpy(dtype=float)
        if ds.size == 0:
            continue
        xs = np.full_like(ds, fill_value=B, dtype=float)
        ax.scatter(xs, ds, alpha=0.4, s=14, color='steelblue')
        ax.errorbar(B, ds.mean(), yerr=ds.std(),
                    fmt='o', color='black', capsize=4, markersize=4)
    ax.set_xscale('log')
    ax.set_xlabel('B'); ax.set_ylabel(f'd  (candidate {candidate_idx})')
    ax.set_title('Distance per draw (mean ± std)')

    # --- Panel 2: top-1 stability ------------------------------------------
    ax = axes[1]
    ax.semilogx(Bs, stabilities * 100, '-o', color='tab:green')
    ax.axhline(90, color='gray', linestyle='--', alpha=0.6,
               label='90% threshold')
    if min_B_90 is not None:
        ax.axvline(min_B_90, color='red', linestyle=':', alpha=0.7,
                   label=f'min B for ≥90% = {min_B_90}')
    ax.set_xlabel('B'); ax.set_ylabel('Top-1 stability (%)')
    ax.set_ylim(0, 102)
    ax.set_title('Selection stability')
    ax.legend(fontsize=8)

    # --- Panel 3: pairwise Spearman ----------------------------------------
    ax = axes[2]
    ax.semilogx(Bs, spearmans, '-o', color='tab:purple')
    ax.set_xlabel('B'); ax.set_ylabel('Spearman ρ (avg over pairs)')
    ax.set_ylim(-0.05, 1.05)
    ax.set_title('Ranking agreement between draws')

    rec = (f'Recommendation: minimum B for >90% top-1 stability = '
           f'{min_B_90 if min_B_90 is not None else "not reached"}')
    fig.text(0.5, 0.01, rec, ha='center', fontsize=10, color='darkred')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    return save_fig(fig, out_dir, 'mc_variance.png', show=show)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('csv_path', help='mc_variance/results.csv')
    add_output_args(p)
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    plot_mc_variance(args.csv_path, out_dir=args.out_dir,
                     show=args.out_dir is None)
