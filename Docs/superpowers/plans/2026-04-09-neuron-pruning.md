# Neuron Pruning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a standalone manifold-distance-scored neuron pruning algorithm that mirrors `sparsifier/sparsifier.py` as closely as possible, enabling empirical comparison of weight vs. neuron pruning.

**Architecture:** A new `sparsifier/neuron_sparsifier.py` imports `d`, `adjust`, `clone_network`, and `make_omega` directly from `sparsifier.sparsifier` — no logic is duplicated. The core `prune_neuron` function iterates over hidden-layer neurons, scores each by `d(og_net, probe_net, omega)` (zeroing the full row/column/bias), and picks the minimum-cost candidate, then calls `adjust` unchanged. Output goes to `artifacts/<name>/neuron_sparsified/`.

**Tech Stack:** JAX (via existing mlp/sparsifier imports), NumPy, Python stdlib csv/json

**Branch:** `neuron_pruning`

---

## File Map

| Action | Path | Responsibility |
|---|---|---|
| Create | `sparsifier/neuron_sparsifier.py` | `NeuronPruneMeta`, `prune_neuron`, `main` |
| Create | `tests/test_neuron_sparsifier.py` | Unit tests for `NeuronPruneMeta` and `prune_neuron` |
| Create | `experiments/mnist_neuron.json` | Experiment config for the MNIST run |

No existing files are modified.

---

## Task 1: Scaffold `neuron_sparsifier.py` with `NeuronPruneMeta`

**Files:**
- Create: `sparsifier/neuron_sparsifier.py`
- Create: `tests/test_neuron_sparsifier.py`

- [ ] **Step 1.1: Write the failing import test**

`tests/test_neuron_sparsifier.py`:

```python
# tests/test_neuron_sparsifier.py
import sys, os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from mlp.mlp import Layer
from sparsifier.neuron_sparsifier import NeuronPruneMeta, prune_neuron


def _tiny_net():
    """4→3→2 network as Layer list with numpy float64 arrays. Seed 42."""
    rng = np.random.default_rng(42)
    W0 = rng.standard_normal((3, 4))
    b0 = rng.standard_normal((3,))
    W1 = rng.standard_normal((2, 3))
    b1 = rng.standard_normal((2,))
    return [
        Layer(W=W0.copy(), b=b0.copy(), mask=np.ones((3, 4))),
        Layer(W=W1.copy(), b=b1.copy(), mask=np.ones((2, 3))),
    ]


def test_neuron_prune_meta_fields():
    meta = NeuronPruneMeta(
        layer_idx=1, neuron_idx=2, distanza=0.5,
        prune_time_s=0.1, adjust_time_s=0.2,
    )
    assert meta.layer_idx == 1
    assert meta.neuron_idx == 2
    assert meta.distanza == 0.5
    assert meta.prune_time_s == 0.1
    assert meta.adjust_time_s == 0.2
```

- [ ] **Step 1.2: Run the test to verify it fails**

```bash
pytest tests/test_neuron_sparsifier.py::test_neuron_prune_meta_fields -v
```

Expected: `FAILED` — `ModuleNotFoundError: No module named 'sparsifier.neuron_sparsifier'`

- [ ] **Step 1.3: Create the scaffold**

`sparsifier/neuron_sparsifier.py`:

```python
# sparsifier/neuron_sparsifier.py

import csv
import json
import os
import sys
import time
from typing import NamedTuple

import numpy as np

from mlp.mlp import Layer, accuracy, load_network_params
from sparsifier.sparsifier import adjust, clone_network, d, make_omega


class NeuronPruneMeta(NamedTuple):
    layer_idx: int      # weight-matrix index l; neuron is row l.W[neuron_idx, :]
    neuron_idx: int     # neuron index i within that layer
    distanza: float     # d(og_net, probe_net, omega) at the winning candidate
    prune_time_s: float
    adjust_time_s: float
```

- [ ] **Step 1.4: Run the test to verify it passes**

```bash
pytest tests/test_neuron_sparsifier.py::test_neuron_prune_meta_fields -v
```

Expected: `PASSED`

---

## Task 2: Implement `prune_neuron` and its unit tests

**Files:**
- Modify: `sparsifier/neuron_sparsifier.py` (add `prune_neuron`)
- Modify: `tests/test_neuron_sparsifier.py` (add three tests)

- [ ] **Step 2.1: Write the three failing tests**

Append to `tests/test_neuron_sparsifier.py`:

```python
def test_prune_neuron_zeros_correct_entries():
    """Winning neuron's row, bias, and next-layer column must all be zero after pruning."""
    net = _tiny_net()
    og_net = clone_network(net)
    omega = np.random.randint(0, 256, size=(200, 4)).astype(np.float32)

    pruned, meta = prune_neuron(net, og_net, omega, doAdjust=False)

    l = meta.layer_idx
    i = meta.neuron_idx
    assert np.all(pruned[l].W[i, :] == 0.0),   "row of W[l] must be zeroed"
    assert np.all(pruned[l].mask[i, :] == 0.0), "row of mask[l] must be zeroed"
    assert pruned[l].b[i] == 0.0,               "bias must be zeroed"
    assert np.all(pruned[l+1].W[:, i] == 0.0),   "column of W[l+1] must be zeroed"
    assert np.all(pruned[l+1].mask[:, i] == 0.0), "column of mask[l+1] must be zeroed"


def test_prune_neuron_only_hidden_layers():
    """For a [4,3,2] network the only prunable layer index is 0 (the single hidden layer)."""
    net = _tiny_net()
    og_net = clone_network(net)
    omega = np.random.randint(0, 256, size=(200, 4)).astype(np.float32)

    _, meta = prune_neuron(net, og_net, omega, doAdjust=False)

    # len(net)==2, hidden layers: range(2-1) = [0] only
    assert meta.layer_idx == 0, "output layer neurons must never be candidates"


def test_prune_neuron_skips_already_zeroed():
    """A neuron whose row is already all-zero must not be selected again."""
    net = _tiny_net()
    # Manually zero neuron 0 of layer 0
    net[0].W[0, :] = 0.0
    net[0].mask[0, :] = 0.0
    net[0].b[0] = 0.0
    net[1].W[:, 0] = 0.0
    net[1].mask[:, 0] = 0.0

    og_net = clone_network(net)
    omega = np.random.randint(0, 256, size=(200, 4)).astype(np.float32)

    _, meta = prune_neuron(net, og_net, omega, doAdjust=False)

    assert not (meta.layer_idx == 0 and meta.neuron_idx == 0), \
        "already-pruned neuron must be skipped"
```

- [ ] **Step 2.2: Run the tests to verify they all fail**

```bash
pytest tests/test_neuron_sparsifier.py -v
```

Expected: `test_neuron_prune_meta_fields` PASSED, the three new tests FAILED with `ImportError` on `prune_neuron`.

- [ ] **Step 2.3: Implement `prune_neuron`**

Append to `sparsifier/neuron_sparsifier.py`:

```python
def prune_neuron(net, og_net, omega, doAdjust=True):
    """Remove the hidden neuron that minimally perturbs d(net, og_net, omega).

    Iterates over all hidden-layer neurons (rows of W[l] for l in 0..len(net)-2).
    For each candidate neuron i in layer l, temporarily zeros:
      - W[l][i, :], mask[l][i, :], b[l][i]   — outgoing connections + bias
      - W[l+1][:, i], mask[l+1][:, i]         — downstream connections
    Evaluates d(og_net, probe_net, omega), then restores and tracks the minimum.
    Applies the winner permanently and runs adjust() if doAdjust=True.

    Returns (pruned_net, NeuronPruneMeta).
    """
    if len(net) < 2:
        raise ValueError("Network must have at least 2 weight matrices (one hidden layer).")

    minimo = 1e16
    minimo_layer = 0
    minimo_neuron = 0

    probe_net = clone_network(net)
    prune_t0 = time.perf_counter()

    for l in range(len(net) - 1):          # hidden-layer indices: 0 to len(net)-2
        for i in range(net[l].W.shape[0]): # neurons = rows of W[l]
            if np.all(net[l].W[i, :] == 0.0):
                continue  # neuron already pruned — skip

            # Save entries that will be temporarily zeroed
            saved_W_row    = probe_net[l].W[i, :].copy()
            saved_mask_row = probe_net[l].mask[i, :].copy()
            saved_b        = float(probe_net[l].b[i])
            saved_W_col    = probe_net[l + 1].W[:, i].copy()
            saved_mask_col = probe_net[l + 1].mask[:, i].copy()

            # Zero the candidate neuron
            probe_net[l].W[i, :]      = 0.0
            probe_net[l].mask[i, :]   = 0.0
            probe_net[l].b[i]         = 0.0
            probe_net[l + 1].W[:, i]    = 0.0
            probe_net[l + 1].mask[:, i] = 0.0

            distanza = float(d(og_net, probe_net, omega))

            # Restore
            probe_net[l].W[i, :]      = saved_W_row
            probe_net[l].mask[i, :]   = saved_mask_row
            probe_net[l].b[i]         = saved_b
            probe_net[l + 1].W[:, i]    = saved_W_col
            probe_net[l + 1].mask[:, i] = saved_mask_col

            if distanza < minimo:
                minimo        = distanza
                minimo_layer  = l
                minimo_neuron = i

    prune_time_s = time.perf_counter() - prune_t0

    # Apply winning neuron permanently
    probe_net[minimo_layer].W[minimo_neuron, :]        = 0.0
    probe_net[minimo_layer].mask[minimo_neuron, :]     = 0.0
    probe_net[minimo_layer].b[minimo_neuron]           = 0.0
    probe_net[minimo_layer + 1].W[:, minimo_neuron]    = 0.0
    probe_net[minimo_layer + 1].mask[:, minimo_neuron] = 0.0

    adjust_t0 = time.perf_counter()
    if doAdjust and minimo > 0:
        probe_net = adjust(probe_net, og_net, omega)
    adjust_time_s = time.perf_counter() - adjust_t0

    meta = NeuronPruneMeta(
        layer_idx=minimo_layer,
        neuron_idx=minimo_neuron,
        distanza=float(minimo),
        prune_time_s=prune_time_s,
        adjust_time_s=adjust_time_s,
    )
    return probe_net, meta
```

- [ ] **Step 2.4: Run all tests to verify they pass**

```bash
pytest tests/test_neuron_sparsifier.py -v
```

Expected: all 4 tests PASSED.

---

## Task 3: Implement `main()`

**Files:**
- Modify: `sparsifier/neuron_sparsifier.py` (append `main` and `__main__` guard)

No new tests — `main` is an I/O entry point validated manually in Task 4.

- [ ] **Step 3.1: Append `main()` to `neuron_sparsifier.py`**

```python
def main():
    cfg_path = os.path.abspath(sys.argv[1])
    with open(cfg_path) as f:
        cfg = json.load(f)

    repo_root     = os.path.dirname(os.path.dirname(cfg_path))
    input_folder  = os.path.join(repo_root, 'artifacts', cfg['name'])
    output_folder = os.path.join(input_folder, 'neuron_sparsified')

    def resolve(p):
        return os.path.join(repo_root, p)

    x_test = np.genfromtxt(resolve(cfg['data']['x_test']),  delimiter=',', max_rows=1000)
    y_test = np.genfromtxt(resolve(cfg['data']['y_test']),  delimiter=',', max_rows=1000)

    sp = cfg['sparsify']

    print("Load the parameters from the folder")
    og_net = load_network_params(input_folder)
    print("Accuracy in validation: %.4f" % float(accuracy(og_net, x_test, y_test)))

    # Count total hidden neurons (rows of W[l] for l in 0..len-2)
    total_neurons = int(sum(og_net[l].W.shape[0] for l in range(len(og_net) - 1)))
    print("Total hidden neurons: %d" % total_neurons)

    omega = make_omega(og_net, n_samples=sp['omega_samples'])

    print("Starting neuron sparsification loop")
    net = clone_network(og_net)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    log_path = os.path.join(output_folder, 'neuron_sparsification_log.csv')
    with open(log_path, 'w', newline='') as log_file:
        writer = csv.writer(log_file)

        hidden_layer_indices = list(range(len(og_net) - 1))
        layer_neuron_cols = ['layer_%d_neurons' % l for l in hidden_layer_indices]
        header = [
            'step', 'neurons_pruned', 'total_neurons', 'neuron_sparsity',
            'val_acc', 'd_manifold', 'd_W',
            'prune_time_s', 'adjust_time_s',
            'candidate_layer', 'candidate_neuron',
        ] + layer_neuron_cols
        writer.writerow(header)

        neurons_pruned = 0
        for step in range(sp['steps']):
            # Count surviving neurons per hidden layer
            layer_neuron_counts = [
                int((net[l].W != 0).any(axis=1).sum())
                for l in hidden_layer_indices
            ]
            neurons_pruned_now = total_neurons - sum(layer_neuron_counts)
            neuron_sparsity    = neurons_pruned_now / total_neurons
            val_acc            = float(accuracy(net, x_test, y_test))
            d_manifold         = float(d(net, og_net, omega))

            print(
                'step {:4d} | acc={:.4f} | neurons_pruned={:4d}/{:4d} | '
                'sparsity={:.4f} | d_m={:.4e}'.format(
                    step, val_acc, neurons_pruned_now, total_neurons,
                    neuron_sparsity, d_manifold,
                )
            )

            W_snapshot = [np.array(l.W).copy() for l in net]
            net, meta  = prune_neuron(net, og_net, omega, doAdjust=sp['do_adjust'])
            d_W = float(
                np.sqrt(sum(
                    np.sum((np.array(l.W) - w) ** 2)
                    for l, w in zip(net, W_snapshot)
                ))
            )

            layer_neuron_counts_after = [
                int((net[l].W != 0).any(axis=1).sum())
                for l in hidden_layer_indices
            ]
            writer.writerow([
                step,
                total_neurons - sum(layer_neuron_counts_after),
                total_neurons,
                round(neuron_sparsity, 6),
                round(val_acc, 6),
                '{:.6e}'.format(d_manifold),
                '{:.6e}'.format(d_W),
                round(meta.prune_time_s, 4),
                round(meta.adjust_time_s, 4),
                meta.layer_idx,
                meta.neuron_idx,
            ] + layer_neuron_counts_after)
            log_file.flush()

            if step % sp['checkpoint_every'] == 0:
                ckpt_dir = os.path.join(output_folder, 'checkpoints', 'step_%04d' % step)
                os.makedirs(ckpt_dir, exist_ok=True)
                for li, layer in enumerate(net):
                    np.save(os.path.join(ckpt_dir, 'W_%d.npy' % li), layer.W)

    print('Neuron sparsification log saved to:', log_path)
    for i, layer in enumerate(net):
        np.save(os.path.join(output_folder, 'W_%i.npy' % i), layer.W)
        np.save(os.path.join(output_folder, 'b_%i.npy' % i), layer.b)


if __name__ == '__main__':
    main()
```

---

## Task 4: Experiment config and smoke validation

**Files:**
- Create: `experiments/mnist_neuron.json`

- [ ] **Step 4.1: Create the config**

`experiments/mnist_neuron.json`:

```json
{
  "name": "mnist_784_256_128_10",
  "data": {
    "x_test": "data/X_test_small.csv",
    "y_test": "data/Y_test_small.csv"
  },
  "sparsify": {
    "steps": 50,
    "do_adjust": true,
    "omega_samples": 10000,
    "checkpoint_every": 10
  }
}
```

- [ ] **Step 4.2: Run the full test suite to confirm nothing is broken**

```bash
pytest tests/test_neuron_sparsifier.py -v
```

Expected: 4 tests PASSED.

- [ ] **Step 4.3: Smoke-run the script for 3 steps**

Edit `experiments/mnist_neuron.json` temporarily, setting `"steps": 3`, then:

```bash
python3 -m sparsifier.neuron_sparsifier experiments/mnist_neuron.json
```

Expected output (values will differ):
```
Found 3 layers in artifacts/mnist_784_256_128_10
Accuracy in validation: 0.9560
Total hidden neurons: 384
Starting neuron sparsification loop
step    0 | acc=0.9560 | neurons_pruned=   0/384 | sparsity=0.0000 | d_m=0.00e+00
step    1 | acc=...
step    2 | acc=...
Neuron sparsification log saved to: artifacts/mnist_784_256_128_10/neuron_sparsified/neuron_sparsification_log.csv
```

Verify the log file exists and has 4 rows (header + 3 steps):

```bash
wc -l artifacts/mnist_784_256_128_10/neuron_sparsified/neuron_sparsification_log.csv
```

Expected: `4`

- [ ] **Step 4.4: Restore steps to 50 in the config**

Edit `experiments/mnist_neuron.json` and set `"steps": 50` again.

---

## Self-Review

**Spec coverage:**
- `NeuronPruneMeta` ✓ Task 1
- `prune_neuron` candidate loop over hidden layers only ✓ Task 2
- Skip already-zeroed neurons ✓ Task 2
- Zero row, column, bias of winning neuron ✓ Task 2
- Same `d` and `adjust` reused unchanged ✓ Task 2 (imported, not reimplemented)
- CSV log with all required columns ✓ Task 3
- Output to `neuron_sparsified/` folder ✓ Task 3
- Checkpoints every N steps ✓ Task 3
- Experiment config ✓ Task 4
- `ValueError` guard for networks with no hidden layers ✓ Task 2

**No placeholders found.**

**Type consistency:** `NeuronPruneMeta` fields used identically across Tasks 1–3. `prune_neuron` signature `(net, og_net, omega, doAdjust=True)` matches calls in Task 2 tests and Task 3 `main`.
