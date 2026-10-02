# analyze_crossbar.py — accuracy vs crossbar count for the width-200 neuron-pruning runs (Block A).
# Crossbars of [196,w1,w2,10] at tile S=128: 2*ceil(w1/S) + ceil(w1/S)*ceil(w2/S) + ceil(w2/S)
# (mirrors backend.compact.crossbar_cost). Per run: accuracy at the FIRST step reaching each
# crossbar count (what compaction at that moment would deploy) and the best accuracy at it.
import glob, os, sys
import numpy as np, pandas as pd
from backend.compact import crossbar_cost
REPO = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
rows = []
for csv in sorted(glob.glob(f'{REPO}/artifacts/e3[89]_w200_*/*sparsified/*.csv')):
    run, sel = csv.split('/')[-3], csv.split('/')[-2].replace('_sparsified', '').replace('sparsified', 'manifold')
    df = pd.read_csv(csv)
    df['xb'] = [crossbar_cost([196, int(a), int(b), 10])['crossbars'] for a, b in zip(df.layer_0_neurons, df.layer_1_neurons)]
    for xb, g in df.groupby('xb'):
        rows.append(dict(run=run, selector=sel, crossbars=int(xb), first_step=int(g.step.iloc[0]),
                         shape=f"[196,{int(g.layer_0_neurons.iloc[0])},{int(g.layer_1_neurons.iloc[0])},10]",
                         acc_first=float(g.val_acc.iloc[0]), acc_best=float(g.val_acc.max())))
t = pd.DataFrame(rows)
with pd.option_context('display.width', 200, 'display.max_rows', 500):
    print(t.sort_values(['selector', 'run', 'crossbars'], ascending=[True, True, False]).to_string(index=False))
    print(); print(t.groupby(['selector', 'crossbars'])[['acc_first', 'acc_best']].agg(['mean', 'std', 'count']).round(3))
