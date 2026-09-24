# analyze_seedband.py — collapse statistics for the full-budget seed band (D2).
# Definitions are exactly those of the manuscript (Sec. "deep-sparsity regime", Table collapse):
#   acc>=0.85 / acc>=0.5 : last sparsity at which val_acc reaches the threshold
#   collapse             : first step with acc < 0.5 whose remaining trajectory averages < 0.5
#   stop rule            : 25-step centred rolling median of d_W (per-step displacement) first
#                          exceeds 10x its baseline = median of that rolling median over steps 0-499
#   acc@80/90            : 25-step centred rolling-median accuracy at the step closest to 80/90 %
# usage: python scripts/analyze_seedband.py [--md]   (prints a table; --md for markdown)

import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
RUNS = [('exhaustive', s, f'e05_stupidity_point' if s == 0 else f'e25_stupidity_seed_{s}',
         'sparsified/sparsification_log.csv') for s in range(5)] + \
       [('obd', s, 'e17_obd_full' if s == 0 else f'e34_obd_full_seed_{s}',
         'obd_sparsified/obd_sparsification_log.csv') for s in range(3)]


def stats(df):
    acc, sp, step = df['val_acc'].values, df['sparsity'].values * 100, df['step'].values
    out = {'rows': len(df), 'dense': acc[0], 'acc@500': acc[500] if len(acc) > 500 else np.nan}
    for thr in (0.85, 0.5):
        idx = np.where(acc >= thr)[0]
        out[f'sp>={thr}'] = sp[idx[-1]] if len(idx) else np.nan
    out['collapse_step'] = out['collapse_sp'] = np.nan
    for i in range(len(acc)):
        if acc[i] < 0.5 and acc[i:].mean() < 0.5:
            out['collapse_step'], out['collapse_sp'] = step[i], sp[i]
            break
    dw = pd.Series(df['d_W'].values).rolling(25, center=True, min_periods=1).median().values
    base = np.median(dw[:500])
    hit = np.where(dw > 10 * base)[0]
    out['stop_step'] = step[hit[0]] if len(hit) else np.nan
    out['stop_sp'] = sp[hit[0]] if len(hit) else np.nan
    racc = pd.Series(acc).rolling(25, center=True, min_periods=1).median().values
    for t in (80, 90):
        out[f'acc@{t}'] = racc[np.argmin(np.abs(sp - t))] if sp.max() >= t else np.nan
    return out


def main():
    rows = []
    for sel, seed, run, log in RUNS:
        p = os.path.join(REPO, 'artifacts', run, log)
        if not os.path.exists(p):
            continue
        r = stats(pd.read_csv(p)); r.update(selector=sel, seed=seed, run=run); rows.append(r)
    t = pd.DataFrame(rows).set_index(['selector', 'seed'])
    cols = ['rows', 'dense', 'acc@500', 'sp>=0.85', 'sp>=0.5', 'collapse_step', 'collapse_sp',
            'stop_step', 'stop_sp', 'acc@80', 'acc@90']
    t = t[cols]
    full = t[t['rows'] >= 2160]
    agg = full.groupby('selector')[['dense', 'acc@500', 'sp>=0.85', 'sp>=0.5', 'collapse_sp',
                                   'stop_sp', 'acc@80', 'acc@90']].agg(['mean', 'std', 'min', 'max', 'count'])
    if '--md' in sys.argv:
        print(t.round(4).to_markdown()); print(); print(agg.round(4).to_markdown())
    else:
        with pd.option_context('display.width', 250, 'display.max_columns', 50):
            print(t.round(4)); print(); print(agg.round(4))


if __name__ == '__main__':
    main()
