# visualize/compare_runs_table.py
"""
Scriptable summary table over multiple sparsification logs.

For each CSV, computes:
    dense_acc            : val_acc at row 0
    final_acc            : val_acc at the last row
    final_sparsity       : sparsity at the last row
    final_d_manifold     : d_manifold at the last row
    total_wall_clock     : sum of prune_time_s + adjust_time_s (s)
    sparsity_at_2pp_drop : highest sparsity at which val_acc >= dense_acc - 0.02

Output formats: markdown (default) or LaTeX (`--format latex`).

Usage:
    python visualize/compare_runs_table.py \\
        'artifacts/*/sparsified/sparsification_log.csv' \\
        --format markdown

Note: although the originating spec placed this tool under `analysis/`, this
project keeps all post-hoc tooling under `visualize/`.
"""
import argparse
import glob
import os
import sys

import pandas as pd


_HEADERS = (
    'run',
    'dense_acc',
    'final_acc',
    'final_sparsity',
    'final_d_manifold',
    'total_wall_clock_s',
    'sparsity_at_2pp_drop',
)


def _run_label(csv_path: str) -> str:
    """Use the run directory name (e.g. `e10_seed_1`) as the row label."""
    parts = os.path.normpath(csv_path).split(os.sep)
    if 'sparsified' in parts:
        i = parts.index('sparsified')
        if i > 0:
            return parts[i - 1]
    return os.path.splitext(os.path.basename(csv_path))[0]


def summarize(csv_path: str) -> dict:
    df = pd.read_csv(csv_path)
    if len(df) == 0:
        raise ValueError(f'Empty CSV: {csv_path}')

    dense_acc = float(df['val_acc'].iloc[0])
    final_acc = float(df['val_acc'].iloc[-1])
    final_sp  = float(df['sparsity'].iloc[-1])
    final_dm  = float(df['d_manifold'].iloc[-1])

    if 'prune_time_s' in df.columns and 'adjust_time_s' in df.columns:
        wall = float(df['prune_time_s'].fillna(0).sum()
                     + df['adjust_time_s'].fillna(0).sum())
    else:
        wall = float('nan')

    threshold = dense_acc - 0.02
    ok = df[df['val_acc'] >= threshold]
    sparsity_2pp = float(ok['sparsity'].max()) if len(ok) > 0 else float('nan')

    return {
        'run':                  _run_label(csv_path),
        'dense_acc':            dense_acc,
        'final_acc':            final_acc,
        'final_sparsity':       final_sp,
        'final_d_manifold':     final_dm,
        'total_wall_clock_s':   wall,
        'sparsity_at_2pp_drop': sparsity_2pp,
    }


def _fmt(v, key):
    if isinstance(v, float):
        if v != v:                       # NaN
            return '—'
        if 'acc' in key or 'sparsity' in key:
            return f'{v:.4f}'
        if 'd_manifold' in key:
            return f'{v:.3e}'
        if 'wall_clock' in key:
            return f'{v:.1f}'
        return f'{v:.4f}'
    return str(v)


def render_markdown(rows) -> str:
    out = []
    out.append('| ' + ' | '.join(_HEADERS) + ' |')
    out.append('|' + '|'.join(['---'] * len(_HEADERS)) + '|')
    for r in rows:
        out.append('| ' + ' | '.join(_fmt(r[h], h) for h in _HEADERS) + ' |')
    return '\n'.join(out)


def render_latex(rows) -> str:
    cols = 'l' + 'r' * (len(_HEADERS) - 1)
    out = []
    out.append(f'\\begin{{tabular}}{{{cols}}}')
    out.append('\\toprule')
    out.append(' & '.join(_HEADERS) + ' \\\\')
    out.append('\\midrule')
    for r in rows:
        out.append(' & '.join(_fmt(r[h], h) for h in _HEADERS) + ' \\\\')
    out.append('\\bottomrule')
    out.append('\\end{tabular}')
    return '\n'.join(out)


def compare_runs_table(pattern: str, fmt: str = 'markdown') -> str:
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise ValueError(f'No files matched: {pattern}')
    rows = [summarize(p) for p in paths]
    if fmt == 'latex':
        return render_latex(rows)
    return render_markdown(rows)


def _parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('pattern',
                   help='Glob pattern matching sparsification_log.csv files.')
    p.add_argument('--format', dest='fmt', default='markdown',
                   choices=('markdown', 'latex'))
    return p.parse_args(argv)


if __name__ == '__main__':
    args = _parse_args(sys.argv[1:])
    print(compare_runs_table(args.pattern, fmt=args.fmt))
