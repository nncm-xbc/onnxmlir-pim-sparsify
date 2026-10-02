# zero_cost_fraction.py — redundancy diagnostic for a dense net: the exact removal cost
# d_W(w0, w0 - w_i e_i) of every prunable weight on the config's Omega (one exhaustive sweep),
# plus dead / always-on hidden units on that sample (Prop. kernel (2)-(3) of the thesis).
# Writes artifacts/<name>/zero_cost.json. usage: python scripts/zero_cost_fraction.py <config.json>
import json, os, sys, time
import numpy as np
from mlp.mlp import load_network_params, Layer
import mlp.mlp as _mlp
from sparsifier.sparsifier import d
from sparsifier.runner import _build_omega, load_calibration

cfg_path = os.path.abspath(sys.argv[1]); cfg = json.load(open(cfg_path))
root = os.path.dirname(os.path.dirname(cfg_path)); resolve = lambda p: os.path.join(root, p)
folder = os.path.join(root, 'artifacts', cfg['name']); sp = cfg['sparsify']
_mlp.hidden_activation = _mlp._ACTS[cfg.get('hidden_activation', 'relu')]
scale = float(cfg.get('input_scale', 1.0))
x_test = np.genfromtxt(resolve(cfg['data']['x_test']), delimiter=',', max_rows=1000) / scale
net = load_network_params(folder)
src = sp.get('omega_source', 'noise')
np.random.seed(int(sp.get('omega_seed', cfg.get('train', {}).get('seed', 0))))
omega = _build_omega(net, load_calibration(cfg, resolve, sp, src, x_test, scale), sp, src, scale=scale)

t0 = time.time(); costs = []
for l, layer in enumerate(net):
    W = np.array(layer.W); nz = np.argwhere(W != 0)
    for i, j in nz:
        W2 = W.copy(); W2[i, j] = 0.0
        probe = list(net); probe[l] = Layer(W=W2, b=layer.b, mask=layer.mask)
        costs.append((l, int(i), int(j), float(d(net, probe, omega))))
c = np.array([x[3] for x in costs])
# hidden-unit activity on the sample
h = np.asarray(omega, dtype=np.float64); units = []
for l, layer in enumerate(net[:-1]):
    z = h @ np.array(layer.W).T + np.array(layer.b)
    units.append({'layer': l, 'width': int(z.shape[1]), 'dead': int((z < 0).all(0).sum()),
                  'always_on': int((z > 0).all(0).sum())})
    h = np.maximum(z, 0)
out = {'name': cfg['name'], 'topology': cfg['topology'], 'omega_source': src, 'B': int(len(omega)),
       'N_prunable': int(len(c)), 'n_zero_cost': int((c == 0).sum()), 'zero_cost_fraction': float((c == 0).mean()),
       'cost_quantiles': {q: float(np.quantile(c, q / 100)) for q in (1, 5, 10, 25, 50, 75, 90, 99)},
       'units': units, 'sweep_seconds': round(time.time() - t0, 1)}
json.dump(out, open(os.path.join(folder, 'zero_cost.json'), 'w'), indent=2)
print(json.dumps({k: out[k] for k in ('name', 'N_prunable', 'n_zero_cost', 'zero_cost_fraction', 'units', 'sweep_seconds')}))
