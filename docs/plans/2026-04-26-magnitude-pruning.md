# Magnitude-Based Pruning Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `sparsifier/magnitude_sparsifier.py` — a weight pruner that selects `argmin |w_ij|` globally and applies the same gradient-descent adjust step as the manifold baseline, enabling direct ablation of the selection criterion.

**Architecture:** New standalone module mirroring `sparsifier/neuron_sparsifier.py`. Imports `adjust`, `clone_network`, `make_omega`, `d` from `sparsifier.sparsifier`; never imports `prune`. Selection is O(N) (read-only scan of weights), so no C++ extension needed. Log schema is identical to the manifold baseline for overlay comparison.

**Tech Stack:** Python 3, JAX/NumPy, pytest.

---

## File map

| Action | Path |
|---|---|
| Create | `sparsifier/magnitude_sparsifier.py` |
| Create | `tests/test_magnitude_sparsifier.py` |
| Modify | `experiments/e09_magnitude_search.json` |

---

### Task 1: Module skeleton + `MagnitudePruneMeta`

**Files:**
- Create: `sparsifier/magnitude_sparsifier.py`
- Create: `tests/test_magnitude_sparsifier.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_magnitude_sparsifier.py` with this content:

```python
# tests/test_magnitude_sparsifier.py
import sys, os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from mlp.mlp import Layer
from sparsifier.magnitude_sparsifier import MagnitudePruneMeta, prune_magnitude
from sparsifier.sparsifier import clone_network, d


def _tiny_net_known():
    """2→2→2 net with known magnitudes. Global minimum is |W0[0,1]|=0.1."""
    W0 = np.array([[0.5, 0.1], [0.8, 0.3]], dtype=np.float64)
    b0 = np.array([0.1, 0.2])
    W1 = np.array([[0.4, 0.6], [0.2, 0.7]], dtype=np.float64)
    b1 = np.array([0.1, 0.2])
    return [
        Layer(W=W0.copy(), b=b0.copy(), mask=np.ones((2, 2))),
        Layer(W=W1.copy(), b=b1.copy(), mask=np.ones((2, 2))),
    ]


def _tiny_net_rng42():
    """4→3→2 network, seed 42. Used for adjust tests."""
    rng = np.random.default_rng(42)
    W0 = rng.standard_normal((3, 4))
    b0 = rng.standard_normal((3,))
    W1 = rng.standard_normal((2, 3))
    b1 = rng.standard_normal((2,))
    return [
        Layer(W=W0.copy(), b=b0.copy(), mask=np.ones((3, 4))),
        Layer(W=W1.copy(), b=b1.copy(), mask=np.ones((2, 3))),
    ]


def test_meta_fields():
    meta = MagnitudePruneMeta(
        layer_idx=0, i=1, j=2, magnitude=0.05,
        prune_time_s=0.01, adjust_time_s=0.02,
    )
    assert meta.layer_idx == 0
    assert meta.i == 1
    assert meta.j == 2
    assert meta.magnitude == 0.05
    assert meta.prune_time_s == 0.01
    assert meta.adjust_time_s == 0.02
```

- [ ] **Step 2: Run to confirm it fails**

```bash
cd /home/simon/repos/onnxmlir-pim-sparsify
python -m pytest tests/test_magnitude_sparsifier.py::test_meta_fields -v
```

Expected: `ModuleNotFoundError: No module named 'sparsifier.magnitude_sparsifier'`

- [ ] **Step 3: Create the module skeleton**

Create `sparsifier/magnitude_sparsifier.py` with this content:

```python
"""Magnitude-scored weight pruning with manifold-distance adjustment.

Selection criterion: argmin |W[l][i,j]| over all active weights (globally).
Adjustment step:    gradient descent on d_W — identical to sparsifier.sparsifier.

This is the ablation baseline for the thesis: any difference in the
sparsity/accuracy/d_W trajectory vs the manifold baseline is attributable
solely to the choice of selection criterion.

Public API: :func:`prune_magnitude`, :class:`MagnitudePruneMeta`.
"""

import csv
import json
import os
import sys
import time
from typing import NamedTuple

import numpy as np

from mlp.mlp import Layer, accuracy, load_network_params
from sparsifier.sparsifier import adjust, clone_network, d, make_omega


class MagnitudePruneMeta(NamedTuple):
    layer_idx:     int
    i:             int
    j:             int
    magnitude:     float   # |W[layer_idx][i,j]| before zeroing
    prune_time_s:  float
    adjust_time_s: float


def prune_magnitude(net, og_net, omega, doAdjust=True):
    raise NotImplementedError
```

- [ ] **Step 4: Run to confirm test passes**

```bash
python -m pytest tests/test_magnitude_sparsifier.py::test_meta_fields -v
```

Expected: `PASSED`

- [ ] **Step 5: Commit**

```bash
git add sparsifier/magnitude_sparsifier.py tests/test_magnitude_sparsifier.py
git commit -m "feat: add MagnitudePruneMeta and module skeleton"
```

---

### Task 2: `prune_magnitude` — selection, zeroing, mask integrity

**Files:**
- Modify: `sparsifier/magnitude_sparsifier.py`
- Modify: `tests/test_magnitude_sparsifier.py`

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_magnitude_sparsifier.py`:

```python
def test_selects_smallest_magnitude():
    """prune_magnitude must zero the entry with the globally smallest |w|."""
    net = _tiny_net_known()
    og_net = clone_network(net)
    omega = np.random.default_rng(0).integers(0, 256, (50, 2)).astype(np.float32)

    pruned, meta = prune_magnitude(net, og_net, omega, doAdjust=False)

    # W0[0,1]=0.1 is the global minimum — must be the candidate
    assert meta.layer_idx == 0
    assert meta.i == 0
    assert meta.j == 1
    assert abs(meta.magnitude - 0.1) < 1e-9


def test_skips_pruned_weights():
    """Weights with mask=0 must never be selected even if W is smaller."""
    net = _tiny_net_known()
    # Manually prune W0[0,1]=0.1 (the global minimum) so it must be skipped
    net[0].W[0, 1] = 0.0
    net[0].mask[0, 1] = 0.0

    og_net = clone_network(net)
    omega = np.random.default_rng(0).integers(0, 256, (50, 2)).astype(np.float32)

    _, meta = prune_magnitude(net, og_net, omega, doAdjust=False)

    assert not (meta.layer_idx == 0 and meta.i == 0 and meta.j == 1), \
        "already-pruned weight must be skipped"


def test_single_weight_zeroed():
    """With doAdjust=False exactly one active weight is removed per call."""
    net = _tiny_net_rng42()
    og_net = clone_network(net)
    omega = np.random.default_rng(1).integers(0, 256, (50, 4)).astype(np.float32)

    nz_before = int(sum((l.W != 0).sum() for l in net))
    pruned, _ = prune_magnitude(net, og_net, omega, doAdjust=False)
    nz_after = int(sum((l.W != 0).sum() for l in pruned))

    assert nz_after == nz_before - 1


def test_meta_consistency():
    """meta.(layer_idx, i, j) must index the actually-pruned weight."""
    net = _tiny_net_known()
    og_net = clone_network(net)
    omega = np.random.default_rng(0).integers(0, 256, (50, 2)).astype(np.float32)

    original_val = float(net[0].W[0, 1])  # the known minimum
    pruned, meta = prune_magnitude(net, og_net, omega, doAdjust=False)

    assert pruned[meta.layer_idx].W[meta.i, meta.j] == 0.0
    assert abs(meta.magnitude - abs(original_val)) < 1e-9


def test_mask_integrity():
    """mask must be 0 at the pruned position after pruning."""
    net = _tiny_net_known()
    og_net = clone_network(net)
    omega = np.random.default_rng(0).integers(0, 256, (50, 2)).astype(np.float32)

    pruned, meta = prune_magnitude(net, og_net, omega, doAdjust=False)

    assert pruned[meta.layer_idx].mask[meta.i, meta.j] == 0.0
```

- [ ] **Step 2: Run to confirm all five new tests fail**

```bash
python -m pytest tests/test_magnitude_sparsifier.py -v
```

Expected: `test_meta_fields` PASSED; five new tests FAILED with `NotImplementedError`.

- [ ] **Step 3: Implement `prune_magnitude` (selection + zeroing, no adjust yet)**

Append to `sparsifier/magnitude_sparsifier.py` after the `MagnitudePruneMeta` definition:

```python
def prune_magnitude(net, og_net, omega, doAdjust=True):
    """Remove the smallest-magnitude active weight; then optionally adjust.

    Selection is global: argmin |W[l][i,j]| over all (l,i,j) with mask==1.
    og_net and omega are unused by selection but required by adjust().

    Returns (pruned_net, MagnitudePruneMeta).
    """
    min_mag = float('inf')
    min_layer = 0
    min_i = 0
    min_j = 0

    prune_t0 = time.perf_counter()
    for l, layer in enumerate(net):
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                if layer.mask[i, j] == 0.0:
                    continue
                mag = abs(float(layer.W[i, j]))
                if mag < min_mag:
                    min_mag = mag
                    min_layer = l
                    min_i = i
                    min_j = j
    prune_time_s = time.perf_counter() - prune_t0

    result_net = clone_network(net)
    result_net[min_layer].W[min_i, min_j] = 0.0
    result_net[min_layer].mask[min_i, min_j] = 0.0

    adjust_t0 = time.perf_counter()
    adjust_time_s = time.perf_counter() - adjust_t0

    meta = MagnitudePruneMeta(
        layer_idx=min_layer,
        i=min_i,
        j=min_j,
        magnitude=min_mag,
        prune_time_s=prune_time_s,
        adjust_time_s=adjust_time_s,
    )
    return result_net, meta
```

- [ ] **Step 4: Run tests — five selection tests must pass, adjust test not written yet**

```bash
python -m pytest tests/test_magnitude_sparsifier.py -v
```

Expected: all 6 tests PASSED (adjust test not yet added).

- [ ] **Step 5: Commit**

```bash
git add sparsifier/magnitude_sparsifier.py tests/test_magnitude_sparsifier.py
git commit -m "feat: implement prune_magnitude selection and zeroing"
```

---

### Task 3: `prune_magnitude` — adjust step

**Files:**
- Modify: `sparsifier/magnitude_sparsifier.py`
- Modify: `tests/test_magnitude_sparsifier.py`

- [ ] **Step 1: Add the failing test**

Append to `tests/test_magnitude_sparsifier.py`:

```python
def test_adjust_reduces_distance():
    """With doAdjust=True, d(og, result) <= d(og, post_zero_no_adjust)."""
    net = _tiny_net_rng42()
    og_net = clone_network(net)
    omega = np.random.default_rng(42).integers(0, 256, (200, 4)).astype(np.float32)

    pruned_no_adj, _ = prune_magnitude(net, og_net, omega, doAdjust=False)
    d_no_adj = float(d(og_net, pruned_no_adj, omega))

    pruned_adj, _ = prune_magnitude(net, og_net, omega, doAdjust=True)
    d_adj = float(d(og_net, pruned_adj, omega))

    assert d_adj <= d_no_adj + 1e-6, (
        f"adjust should not increase d: no_adj={d_no_adj:.4e}, adj={d_adj:.4e}"
    )
```

- [ ] **Step 2: Run to confirm it fails**

```bash
python -m pytest tests/test_magnitude_sparsifier.py::test_adjust_reduces_distance -v
```

Expected: FAILED — `d_adj > d_no_adj` because adjust is not yet called.

- [ ] **Step 3: Wire up `adjust()` in `prune_magnitude`**

In `sparsifier/magnitude_sparsifier.py`, replace the adjust block inside `prune_magnitude`:

Old:
```python
    adjust_t0 = time.perf_counter()
    adjust_time_s = time.perf_counter() - adjust_t0
```

New:
```python
    adjust_t0 = time.perf_counter()
    if doAdjust and min_mag > 0:
        result_net = adjust(result_net, og_net, omega)
    adjust_time_s = time.perf_counter() - adjust_t0
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/test_magnitude_sparsifier.py -v
```

Expected: all 7 tests PASSED.

- [ ] **Step 5: Commit**

```bash
git add sparsifier/magnitude_sparsifier.py tests/test_magnitude_sparsifier.py
git commit -m "feat: wire adjust() into prune_magnitude"
```

---

### Task 4: `main()` entry point

**Files:**
- Modify: `sparsifier/magnitude_sparsifier.py`

- [ ] **Step 1: Append `main()` to `sparsifier/magnitude_sparsifier.py`**

```python
########################################################################


def main():
    cfg_path = os.path.abspath(sys.argv[1])
    with open(cfg_path) as f:
        cfg = json.load(f)

    repo_root     = os.path.dirname(os.path.dirname(cfg_path))
    input_folder  = os.path.join(repo_root, 'artifacts', cfg['name'])
    output_folder = os.path.join(input_folder, 'magnitude_sparsified')

    def resolve(p):
        return os.path.join(repo_root, p)

    x_test = np.genfromtxt(resolve(cfg['data']['x_test']), delimiter=',', max_rows=1000)
    y_test = np.genfromtxt(resolve(cfg['data']['y_test']), delimiter=',', max_rows=1000)

    sp = cfg['sparsify']

    import mlp.mlp as _mlp
    _mlp.hidden_activation = _mlp._ACTS[cfg.get('hidden_activation', 'relu')]

    print("Load the parameters from the folder")
    og_net = load_network_params(input_folder)
    print("Accuracy in validation: %.4f" % float(accuracy(og_net, x_test, y_test)))
    total_W = int(sum(l.W.size for l in og_net))
    print("Total parameters: %d" % total_W)

    omega = make_omega(og_net, n_samples=sp['omega_samples'])
    print(
        "Perturbation distance (sanity check): %.4e"
        % float(d(og_net, [
            Layer(W=l.W + np.random.normal(size=l.W.shape) * 0.00001, b=l.b, mask=l.mask)
            for l in og_net
        ], omega))
    )

    print("Starting magnitude sparsification loop")
    net = clone_network(og_net)

    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, 'magnitude_sparsification_log.csv')
    with open(log_path, 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        layer_NZ_cols = ['layer_%d_NZ' % li for li in range(len(og_net))]
        header = [
            'step', 'NZ', 'total_W', 'sparsity', 'val_acc',
            'd_manifold', 'd_W',
            'prune_time_s', 'adjust_time_s',
            'candidate_layer', 'candidate_i', 'candidate_j',
        ] + layer_NZ_cols
        writer.writerow(header)

        for i in range(sp['steps']):
            NZ = int(np.sum([(l.W != 0).sum() for l in net]))
            sparsity = 1.0 - NZ / total_W
            val_acc = float(accuracy(net, x_test, y_test))
            d_manifold = float(d(net, og_net, omega))

            print(
                'step {:4d} | acc={:.4f} | NZ={:6d} | sparsity={:.4f} | d_m={:.4e}'.format(
                    i, val_acc, NZ, sparsity, d_manifold
                )
            )

            W_snapshot = [np.array(l.W).copy() for l in net]
            net, meta = prune_magnitude(net, og_net, omega, doAdjust=sp['do_adjust'])
            d_W = float(
                np.sqrt(
                    sum(
                        np.sum((np.array(l.W) - w) ** 2)
                        for l, w in zip(net, W_snapshot)
                    )
                )
            )

            layer_nz_vals = [int((l.W != 0).sum()) for l in net]
            writer.writerow([
                i, NZ, total_W, round(sparsity, 6), round(val_acc, 6),
                '{:.6e}'.format(d_manifold), '{:.6e}'.format(d_W),
                round(meta.prune_time_s, 4), round(meta.adjust_time_s, 4),
                meta.layer_idx, meta.i, meta.j,
            ] + layer_nz_vals)
            log_file.flush()

            if i % sp['checkpoint_every'] == 0:
                ckpt_dir = os.path.join(output_folder, 'checkpoints', 'step_%04d' % i)
                os.makedirs(ckpt_dir, exist_ok=True)
                for li, layer in enumerate(net):
                    np.save(os.path.join(ckpt_dir, 'W_%d.npy' % li), layer.W)

    print("Magnitude sparsification log saved to:", log_path)
    for i, l in enumerate(net):
        np.save(os.path.join(output_folder, 'W_%i.npy' % i), l.W)
        np.save(os.path.join(output_folder, 'b_%i.npy' % i), l.b)


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Verify all tests still pass**

```bash
python -m pytest tests/test_magnitude_sparsifier.py -v
```

Expected: all 7 tests PASSED.

- [ ] **Step 3: Smoke-test the entry point against the baseline artifact**

```bash
cd /home/simon/repos/onnxmlir-pim-sparsify
python -m sparsifier.magnitude_sparsifier experiments/e09_magnitude_search.json 2>&1 | head -20
```

Expected: prints "Load the parameters from the folder", "Found N layers", accuracy line, "Starting magnitude sparsification loop", then step 0 line. No stack trace.

- [ ] **Step 4: Commit**

```bash
git add sparsifier/magnitude_sparsifier.py
git commit -m "feat: add main() to magnitude_sparsifier"
```

---

### Task 5: Update `e09_magnitude_search.json`

**Files:**
- Modify: `experiments/e09_magnitude_search.json`

- [ ] **Step 1: Edit the config**

Replace the full content of `experiments/e09_magnitude_search.json` with:

```json
{
  "name": "e09_magnitude_search",
  "topology": [196, 10, 10, 10],
  "hidden_activation": "relu",
  "data": {
    "x_train": "data/X_train_small.csv",
    "y_train": "data/Y_train_small.csv",
    "x_test":  "data/X_test_small.csv",
    "y_test":  "data/Y_test_small.csv"
  },
  "train": {
    "epochs":     1000,
    "batch_size": 128,
    "seed":       0
  },
  "sparsify": {
    "steps":            500,
    "do_adjust":        true,
    "omega_samples":    10000,
    "checkpoint_every": 50
  },
  "_experiment": {
    "id": "E9",
    "description": "Magnitude-based selection criterion with the same manifold adjust step. Selects argmin |w_ij| globally at each step; adjustment is identical to the manifold baseline. Run via: python -m sparsifier.magnitude_sparsifier experiments/e09_magnitude_search.json. Any difference vs baseline is attributable solely to the selection criterion (manifold d_W vs magnitude |w|).",
    "literature": [
      "Han et al. 2015: magnitude pruning competitive when paired with retraining — this checks whether the same holds with manifold adjust instead",
      "Molchanov et al. 2019: systematic comparison of magnitude, first-order Taylor, and second-order importance metrics"
    ]
  }
}
```

- [ ] **Step 2: Verify the smoke test still resolves the config correctly**

```bash
python -c "import json; c=json.load(open('experiments/e09_magnitude_search.json')); print(c['name'], c['sparsify'])"
```

Expected: `e09_magnitude_search {'steps': 500, 'do_adjust': True, 'omega_samples': 10000, 'checkpoint_every': 50}`

- [ ] **Step 3: Run the full test suite to confirm nothing regressed**

```bash
python -m pytest tests/test_magnitude_sparsifier.py tests/test_neuron_sparsifier.py -v
```

Expected: all tests PASSED.

- [ ] **Step 4: Commit**

```bash
git add experiments/e09_magnitude_search.json
git commit -m "config: update e09 to reference magnitude_sparsifier module"
```
