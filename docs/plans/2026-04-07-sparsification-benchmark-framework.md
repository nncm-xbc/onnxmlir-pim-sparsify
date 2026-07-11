# Sparsification Benchmark & Visualization Framework — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a framework that instruments the sparsification loop, verifies the correctness of approximate pruning proposals (P1–P4), and provides static + live visualizations of network sparsity.

**Architecture:** Three concerns kept separate — (1) instrumentation: `prune()` returns structured metadata so all callers can log timing and candidate info without extra JAX calls; (2) correctness harness: `benchmark/correctness_check.py` runs exhaustive vs gradient-top-k side-by-side on the same network state and reports agreement rate and speedup; (3) visualization: `visualize/` scripts read the extended CSV log and checkpoint weight files, with `live_view.py` polling the log during a live run.

**Tech Stack:** JAX, NumPy, pandas, matplotlib, pytest

---

## File Map

| File                              | Action     | Responsibility                                                                                                  |
| --------------------------------- | ---------- | --------------------------------------------------------------------------------------------------------------- |
| `sparsifier/sparsifier.py`        | **Modify** | Add `PruneMeta` NamedTuple; instrument `prune()` to return `(net, meta)`; extend log columns; checkpoint saving |
| `benchmark/__init__.py`           | **Create** | Empty                                                                                                           |
| `benchmark/correctness_check.py`  | **Create** | `prune_gradient_topk()`, `compare_one_step()` — run exhaustive vs gradient-top-k, report agreement/speedup      |
| `benchmark/compare_strategies.py` | **Create** | Run N steps of each strategy, save per-strategy CSVs for overlay comparison                                     |
| `visualize/__init__.py`           | **Create** | Empty                                                                                                           |
| `visualize/plot_run.py`           | **Create** | `parse_log()`, `plot_run()` — 6-panel static summary from a log CSV                                             |
| `visualize/plot_weights.py`       | **Create** | `load_checkpoint()`, `plot_weight_heatmaps()` — weight magnitude + mask heatmaps from .npy snapshots            |
| `visualize/plot_comparison.py`    | **Create** | `plot_strategy_comparison()` — overlay accuracy/timing curves for multiple strategy CSVs                        |
| `visualize/live_view.py`          | **Create** | `live_view()` — matplotlib FuncAnimation polling log + checkpoint dirs in real time                             |
| `tests/test_benchmark.py`         | **Create** | Tests for log parsing, `PruneMeta` structure, `per_layer_NZ`, `compare_one_step` shape                          |

---

## Task 1: Instrument `prune()` — add `PruneMeta` and timing

**Files:**

- Modify: `sparsifier/sparsifier.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_benchmark.py
import sys, os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from jax import random
from mlp.mlp import init_network_params, Layer, accuracy
from sparsifier.sparsifier import prune, make_omega, PruneMeta

def _tiny_net():
    """3→4→2 network for fast tests."""
    return init_network_params([3, 4, 2], random.PRNGKey(0))

def test_prune_returns_tuple_with_meta():
    net   = _tiny_net()
    omega = make_omega(net, n_samples=50)
    result = prune(net, net, omega, doAdjust=False)
    assert isinstance(result, tuple), "prune() must return (net, meta)"
    assert len(result) == 2
    _, meta = result
    assert isinstance(meta, PruneMeta)
    assert hasattr(meta, 'layer_idx')
    assert hasattr(meta, 'i')
    assert hasattr(meta, 'j')
    assert hasattr(meta, 'distanza')
    assert hasattr(meta, 'prune_time_s')
    assert hasattr(meta, 'adjust_time_s')
    assert meta.prune_time_s >= 0.
    assert meta.adjust_time_s >= 0.

def test_prune_meta_candidate_is_valid():
    net   = _tiny_net()
    omega = make_omega(net, n_samples=50)
    pruned_net, meta = prune(net, net, omega, doAdjust=False)
    layer = pruned_net[meta.layer_idx]
    assert layer.W[meta.i, meta.j] == 0., "chosen weight must be zeroed in result"
```

- [ ] **Step 2: Run the tests to confirm they fail**

```bash
cd /Users/simonhgt/Documents/repos/onnxmlir-pim-sparsify
python -m pytest tests/test_benchmark.py::test_prune_returns_tuple_with_meta tests/test_benchmark.py::test_prune_meta_candidate_is_valid -v
```

Expected: `FAILED — cannot unpack non-sequence Layer` or `AttributeError: PruneMeta`.

- [ ] **Step 3: Add `PruneMeta` and `import time` to `sparsifier.py`**

Add at the top of `sparsifier/sparsifier.py`, after the existing imports:

```python
import time
from typing import NamedTuple

class PruneMeta(NamedTuple):
    layer_idx:     int
    i:             int
    j:             int
    distanza:      float
    prune_time_s:  float
    adjust_time_s: float
```

- [ ] **Step 4: Instrument `prune()` to return `(net, PruneMeta)`**

Replace the existing `prune()` function body (lines 76–116 in `sparsifier/sparsifier.py`) with:

```python
def prune(net, og_net, omega, doAdjust=True):
    minimo     = 1e16
    minimo_idx = 0
    minimo_i   = 0
    minimo_j   = 0

    t_prune_start = time.perf_counter()
    probe_net   = clone_network(net)
    search_done = False
    for idx, layer in enumerate(net):
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                if layer.W[i, j] == 0.:
                    continue
                saved_W    = probe_net[idx].W[i, j]
                saved_mask = probe_net[idx].mask[i, j]
                probe_net[idx].W[i, j]    = 0.
                probe_net[idx].mask[i, j] = 0.
                distanza = d(og_net, probe_net, omega)
                probe_net[idx].W[i, j]    = saved_W
                probe_net[idx].mask[i, j] = saved_mask
                if distanza < minimo:
                    minimo     = distanza
                    minimo_idx = idx
                    minimo_i   = i
                    minimo_j   = j
                    if minimo == 0:
                        search_done = True
                        break
            if search_done:
                break
        if search_done:
            break
    t_prune_end = time.perf_counter()

    probe_net[minimo_idx].W[minimo_i, minimo_j]    = 0.
    probe_net[minimo_idx].mask[minimo_i, minimo_j] = 0.

    t_adjust_start = time.perf_counter()
    if doAdjust and minimo > 0:
        probe_net = adjust(probe_net, og_net, omega)
    t_adjust_end = time.perf_counter()

    meta = PruneMeta(
        layer_idx    = minimo_idx,
        i            = minimo_i,
        j            = minimo_j,
        distanza     = float(minimo),
        prune_time_s = t_prune_end  - t_prune_start,
        adjust_time_s= t_adjust_end - t_adjust_start,
    )
    return probe_net, meta
```

- [ ] **Step 5: Update the call site in `main()`**

In `sparsifier/sparsifier.py`, line 164, change:

```python
# Before:
net = prune(net, og_net, omega, True)

# After:
net, meta = prune(net, og_net, omega, True)
```

- [ ] **Step 6: Run tests to confirm they pass**

```bash
python -m pytest tests/test_benchmark.py::test_prune_returns_tuple_with_meta tests/test_benchmark.py::test_prune_meta_candidate_is_valid -v
```

Expected: `2 passed`.

- [ ] **Step 7: Commit**

```bash
git add sparsifier/sparsifier.py tests/test_benchmark.py
git commit -m "feat: instrument prune() to return (net, PruneMeta) with timing"
```

---

## Task 2: Extended log columns + checkpoint saving

**Files:**

- Modify: `sparsifier/sparsifier.py` (the `main()` function)
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_benchmark.py

import csv, tempfile, os

def test_extended_log_has_required_columns():
    """Extended log must contain timing + per-layer NZ + candidate columns."""
    required = {
        'step', 'NZ', 'total_W', 'sparsity', 'val_acc',
        'd_manifold', 'd_W',
        'prune_time_s', 'adjust_time_s',
        'candidate_layer', 'candidate_i', 'candidate_j',
        'layer_0_NZ',
    }
    # Write a synthetic log with those columns and verify parse_log picks them up.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sorted(required))
        writer.writeheader()
        writer.writerow({k: 0 for k in required})
        path = f.name
    try:
        import pandas as pd
        df = pd.read_csv(path)
        assert required.issubset(set(df.columns)), f"Missing: {required - set(df.columns)}"
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run test to confirm it passes immediately** (this test validates the fixture, not the code yet)

```bash
python -m pytest tests/test_benchmark.py::test_extended_log_has_required_columns -v
```

Expected: `1 passed` (the fixture itself is valid).

- [ ] **Step 3: Update `main()` in `sparsifier/sparsifier.py` to write extended columns**

Replace the `main()` log header and row-writing block with the following. Find the existing `writer.writerow(['step', ...])` line and replace from there through the end of the `with open(log_path` block:

```python
    # ------ header ------
    layer_NZ_cols = ['layer_%d_NZ' % li for li in range(len(og_net))]
    header = (
        ['step', 'NZ', 'total_W', 'sparsity', 'val_acc', 'd_manifold', 'd_W',
         'prune_time_s', 'adjust_time_s',
         'candidate_layer', 'candidate_i', 'candidate_j']
        + layer_NZ_cols
    )
    writer.writerow(header)

    for i in range(500):
        NZ         = int(np.sum([(l.W != 0).sum() for l in net]))
        sparsity   = 1.0 - NZ / total_W
        val_acc    = float(accuracy(net, x_test, y_test))
        d_manifold = float(d(net, og_net, omega))

        print("step {:4d} | acc={:.4f} | NZ={:6d} | sparsity={:.4f} | d_m={:.4e}".format(
            i, val_acc, NZ, sparsity, d_manifold))

        W_snapshot = [np.array(l.W).copy() for l in net]
        net, meta  = prune(net, og_net, omega, True)
        d_W = float(np.sqrt(sum(
            np.sum((np.array(l.W) - w) ** 2) for l, w in zip(net, W_snapshot)
        )))
        layer_nz_vals = [int((l.W != 0).sum()) for l in net]

        writer.writerow(
            [i, NZ, total_W, round(sparsity, 6), round(val_acc, 6),
             "{:.6e}".format(d_manifold), "{:.6e}".format(d_W),
             round(meta.prune_time_s, 4), round(meta.adjust_time_s, 4),
             meta.layer_idx, meta.i, meta.j]
            + layer_nz_vals
        )
        log_file.flush()

        # checkpoint every 50 steps
        if i % 50 == 0:
            ckpt_dir = os.path.join(output_folder, 'checkpoints', 'step_%04d' % i)
            os.makedirs(ckpt_dir, exist_ok=True)
            for li, layer in enumerate(net):
                np.save(os.path.join(ckpt_dir, 'W_%d.npy' % li), layer.W)
```

Also remove the now-stale `omega = make_omega(net)` line that previously appeared inside the loop, and add before the loop:

```python
    omega = make_omega(og_net)   # fixed sample set — no need to regenerate each step
```

- [ ] **Step 4: Run full sparsifier smoke test** (just 3 steps to verify the log format)

```bash
python -m sparsifier.sparsifier artifacts/ data/X_test_small.csv data/Y_test_small.csv 2>&1 | head -20
```

Expected: prints `step    0 | acc=...` and no crash.

- [ ] **Step 5: Commit**

```bash
git add sparsifier/sparsifier.py tests/test_benchmark.py
git commit -m "feat: extend sparsification log with timing, per-layer NZ, candidate info, checkpoints"
```

---

## Task 3: Correctness check harness

**Files:**

- Create: `benchmark/__init__.py`
- Create: `benchmark/correctness_check.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_benchmark.py

def test_compare_one_step_returns_expected_keys():
    """compare_one_step() must return exhaustive + gradient_top_k entries."""
    from benchmark.correctness_check import compare_one_step
    net   = _tiny_net()
    omega = make_omega(net, n_samples=50)
    x     = np.random.rand(20, 3).astype(np.float32)
    # one-hot targets for 2 classes
    y     = np.zeros((20, 2), dtype=np.float32)
    y[np.arange(20), np.random.randint(0, 2, 20)] = 1.
    result = compare_one_step(net, net, omega, x, y, top_k_values=(1, 5))
    assert 'exhaustive' in result
    assert 'gradient_top_1' in result
    assert 'gradient_top_5' in result
    for key in ('candidate', 'time_s', 'd_after', 'acc_after'):
        assert key in result['exhaustive'], f"missing '{key}' in exhaustive"
    for key in ('candidate', 'time_s', 'd_after', 'acc_after', 'candidate_match', 'speedup'):
        assert key in result['gradient_top_1'], f"missing '{key}' in gradient_top_1"

def test_gradient_topk_full_equals_exhaustive():
    """When top_k >= NZ, gradient-top-k must select the same weight as exhaustive."""
    from benchmark.correctness_check import compare_one_step
    net   = _tiny_net()
    omega = make_omega(net, n_samples=50)
    x     = np.random.rand(20, 3).astype(np.float32)
    y     = np.zeros((20, 2), dtype=np.float32)
    y[np.arange(20), np.random.randint(0, 2, 20)] = 1.
    NZ    = int(sum((l.W != 0).sum() for l in net))
    result = compare_one_step(net, net, omega, x, y, top_k_values=(NZ,))
    assert result[f'gradient_top_{NZ}']['candidate_match'], \
        "With top_k=NZ, gradient ranking must evaluate all candidates — result must match exhaustive"
```

- [ ] **Step 2: Run to confirm they fail**

```bash
python -m pytest tests/test_benchmark.py::test_compare_one_step_returns_expected_keys tests/test_benchmark.py::test_gradient_topk_full_equals_exhaustive -v
```

Expected: `ModuleNotFoundError: No module named 'benchmark'`.

- [ ] **Step 3: Create `benchmark/__init__.py`**

```python
# benchmark/__init__.py
```

(empty file)

- [ ] **Step 4: Create `benchmark/correctness_check.py`**

```python
# benchmark/correctness_check.py
"""
Correctness harness: compare exhaustive prune() against gradient-guided top-k.

Usage:
    python benchmark/correctness_check.py <net_folder> <x_test.csv> <y_test.csv> [n_steps]
"""
import sys, os, time
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from mlp.mlp import accuracy, load_network_params
from sparsifier.sparsifier import (
    clone_network, d, d_grad, make_omega, adjust,
    prune, PruneMeta
)


def prune_gradient_topk(net, og_net, omega, top_k, doAdjust=False):
    """Gradient-guided top-k pruning. Returns (net, PruneMeta)."""
    t0 = time.perf_counter()

    # One gradient call replaces O(NZ) forward passes for ranking.
    gradients = d_grad(net, og_net, omega)

    candidates = []
    for idx, (layer, glayer) in enumerate(zip(net, gradients)):
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                if layer.W[i, j] == 0.:
                    continue
                # First-order Taylor: zeroing W[i,j] changes d by ≈ -W[i,j]*grad[i,j]
                predicted_delta = -float(layer.W[i, j]) * float(glayer.W[i, j])
                candidates.append((predicted_delta, idx, i, j))

    candidates.sort(key=lambda x: x[0])

    minimo     = 1e16
    minimo_idx = candidates[0][1]
    minimo_i   = candidates[0][2]
    minimo_j   = candidates[0][3]

    probe_net = clone_network(net)
    for _, idx, i, j in candidates[:top_k]:
        saved_W    = probe_net[idx].W[i, j]
        saved_mask = probe_net[idx].mask[i, j]
        probe_net[idx].W[i, j]    = 0.
        probe_net[idx].mask[i, j] = 0.
        distanza = d(og_net, probe_net, omega)
        probe_net[idx].W[i, j]    = saved_W
        probe_net[idx].mask[i, j] = saved_mask
        if distanza < minimo:
            minimo     = distanza
            minimo_idx = idx
            minimo_i   = i
            minimo_j   = j
            if minimo == 0:
                break

    t_prune = time.perf_counter() - t0

    probe_net[minimo_idx].W[minimo_i, minimo_j]    = 0.
    probe_net[minimo_idx].mask[minimo_i, minimo_j] = 0.

    t_adj_start = time.perf_counter()
    if doAdjust and minimo > 0:
        probe_net = adjust(probe_net, og_net, omega)
    t_adj = time.perf_counter() - t_adj_start

    meta = PruneMeta(
        layer_idx    = minimo_idx,
        i            = minimo_i,
        j            = minimo_j,
        distanza     = float(minimo),
        prune_time_s = t_prune,
        adjust_time_s= t_adj,
    )
    return probe_net, meta


def compare_one_step(net, og_net, omega, x_test, y_test, top_k_values=(1, 5, 10, 50)):
    """
    Run exhaustive and gradient-top-k on the same network state.

    Returns dict keyed by strategy name:
      'exhaustive'       -> {candidate, time_s, d_after, acc_after}
      'gradient_top_k'   -> {candidate, time_s, d_after, acc_after, candidate_match, speedup}
    """
    results = {}

    # --- Exhaustive ---
    t0 = time.perf_counter()
    net_ex, meta_ex = prune(net, og_net, omega, doAdjust=False)
    t_ex = time.perf_counter() - t0
    results['exhaustive'] = {
        'candidate': (meta_ex.layer_idx, meta_ex.i, meta_ex.j),
        'time_s':    t_ex,
        'd_after':   float(d(net_ex, og_net, omega)),
        'acc_after': float(accuracy(net_ex, x_test, y_test)),
    }

    # --- Gradient top-k variants ---
    for k in top_k_values:
        t0 = time.perf_counter()
        net_k, meta_k = prune_gradient_topk(net, og_net, omega, top_k=k, doAdjust=False)
        t_k = time.perf_counter() - t0
        match = (
            meta_k.layer_idx == meta_ex.layer_idx and
            meta_k.i         == meta_ex.i         and
            meta_k.j         == meta_ex.j
        )
        results[f'gradient_top_{k}'] = {
            'candidate':       (meta_k.layer_idx, meta_k.i, meta_k.j),
            'time_s':          t_k,
            'd_after':         float(d(net_k, og_net, omega)),
            'acc_after':       float(accuracy(net_k, x_test, y_test)),
            'candidate_match': match,
            'speedup':         t_ex / max(t_k, 1e-9),
        }

    return results


def run_correctness_report(net_folder, x_test_path, y_test_path, n_steps=20,
                            top_k_values=(1, 5, 10, 50), omega_samples=500):
    """
    Run n_steps of exhaustive + gradient-top-k on the same network.
    Print per-step agreement table and aggregate speedup summary.
    """
    og_net = load_network_params(net_folder)
    x_test = np.genfromtxt(x_test_path, delimiter=',', max_rows=500)
    y_test = np.genfromtxt(y_test_path, delimiter=',', max_rows=500)
    omega  = make_omega(og_net, n_samples=omega_samples)

    net    = [l._replace(W=np.array(l.W).copy()) for l in og_net]  # working copy

    print(f"\n{'Step':>4}  {'Ex(s)':>7}  " +
          "  ".join(f"top{k}(s)  match  spdup" for k in top_k_values))
    print("-" * (16 + 24 * len(top_k_values)))

    aggregate = {k: {'match': 0, 'speedup': []} for k in top_k_values}

    for step in range(n_steps):
        results = compare_one_step(net, og_net, omega, x_test, y_test,
                                    top_k_values=top_k_values)
        ex = results['exhaustive']
        row = f"{step:>4}  {ex['time_s']:>7.3f}"
        for k in top_k_values:
            r = results[f'gradient_top_{k}']
            match_str = "YES" if r['candidate_match'] else "NO "
            row += f"  {r['time_s']:>7.3f}    {match_str}  {r['speedup']:>5.1f}x"
            if r['candidate_match']:
                aggregate[k]['match'] += 1
            aggregate[k]['speedup'].append(r['speedup'])
        print(row)
        # advance net one exhaustive step so we test different sparsity levels
        net, _ = prune(net, og_net, omega, doAdjust=False)

    print("\n--- Aggregate over", n_steps, "steps ---")
    for k in top_k_values:
        ag = aggregate[k]
        avg_sp = sum(ag['speedup']) / len(ag['speedup'])
        print(f"  top-{k:>2}: match={ag['match']}/{n_steps}  avg_speedup={avg_sp:.1f}x")


if __name__ == '__main__':
    args = sys.argv
    run_correctness_report(
        net_folder   = os.path.abspath(args[1]),
        x_test_path  = os.path.abspath(args[2]),
        y_test_path  = os.path.abspath(args[3]),
        n_steps      = int(args[4]) if len(args) > 4 else 20,
    )
```

- [ ] **Step 5: Run the tests**

```bash
python -m pytest tests/test_benchmark.py::test_compare_one_step_returns_expected_keys tests/test_benchmark.py::test_gradient_topk_full_equals_exhaustive -v
```

Expected: `2 passed`.

- [ ] **Step 6: Smoke-test the CLI**

```bash
python benchmark/correctness_check.py artifacts/ data/X_test_small.csv data/Y_test_small.csv 5
```

Expected: a table with 5 rows showing exhaustive time, per-k match (YES/NO), and speedup.

- [ ] **Step 7: Commit**

```bash
git add benchmark/__init__.py benchmark/correctness_check.py tests/test_benchmark.py
git commit -m "feat: add correctness_check harness — exhaustive vs gradient-top-k comparison"
```

---

## Task 4: Static visualizations — `plot_run.py`

**Files:**

- Create: `visualize/__init__.py`
- Create: `visualize/plot_run.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_benchmark.py

def test_parse_log_basic():
    """parse_log() must return a DataFrame with core columns and correct dtypes."""
    import pandas as pd, tempfile, csv as csv_mod
    cols = ['step','NZ','total_W','sparsity','val_acc','d_manifold','d_W',
            'prune_time_s','adjust_time_s','candidate_layer','candidate_i','candidate_j',
            'layer_0_NZ','layer_1_NZ']
    rows = [[i, 100-i, 100, i/100, 0.9, 0.0, 0.01, 0.5, 0.1, 0, i%4, i%3, 60-i, 40-i]
            for i in range(5)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        w = csv_mod.writer(f); w.writerow(cols); w.writerows(rows); path = f.name
    try:
        from visualize.plot_run import parse_log
        df = parse_log(path)
        assert list(df.columns) == cols
        assert len(df) == 5
        assert df['step'].dtype.kind == 'i' or df['step'].dtype.kind == 'f'
    finally:
        os.unlink(path)
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_benchmark.py::test_parse_log_basic -v
```

Expected: `ModuleNotFoundError: No module named 'visualize'`.

- [ ] **Step 3: Create `visualize/__init__.py`**

```python
# visualize/__init__.py
```

(empty)

- [ ] **Step 4: Create `visualize/plot_run.py`**

```python
# visualize/plot_run.py
"""
Static 6-panel summary plot from a sparsification log CSV.

Usage:
    python visualize/plot_run.py path/to/sparsification_log.csv [output_dir]
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_log(log_path: str) -> pd.DataFrame:
    return pd.read_csv(log_path)


def plot_run(log_path: str, out_dir: str = None, show: bool = True) -> plt.Figure:
    df  = parse_log(log_path)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Sparsification Run — {os.path.basename(log_path)}', fontsize=13)

    # 1. Accuracy vs Sparsity
    ax = axes[0, 0]
    ax.plot(df['sparsity'] * 100, df['val_acc'] * 100, 'b-o', markersize=3)
    dense_acc = float(df['val_acc'].iloc[0]) * 100
    ax.axhline(dense_acc - 2, color='r', linestyle='--', alpha=0.5, label='Dense −2 pp')
    ax.set_xlabel('Sparsity (%)'); ax.set_ylabel('Val Accuracy (%)')
    ax.set_title('Accuracy vs Sparsity'); ax.legend(fontsize=8)

    # 2. Manifold distance vs step
    ax = axes[0, 1]
    d_vals = df['d_manifold'].replace(0, np.nan)
    if d_vals.notna().any():
        ax.semilogy(df['step'], d_vals, 'r-', linewidth=1)
    else:
        ax.plot(df['step'], df['d_manifold'], 'r-', linewidth=1)
    ax.set_xlabel('Step'); ax.set_ylabel('d_manifold')
    ax.set_title('Manifold Distance (log scale)')

    # 3. Weight shift per step (adjust magnitude)
    ax = axes[0, 2]
    ax.plot(df['step'], df['d_W'], 'g-', linewidth=1)
    ax.set_xlabel('Step'); ax.set_ylabel('||ΔW||₂')
    ax.set_title('Weight Shift per Step')

    # 4. NZ over time
    ax = axes[1, 0]
    ax.plot(df['step'], df['NZ'], 'k-', linewidth=1.5)
    ax.set_xlabel('Step'); ax.set_ylabel('Non-zero weights')
    ax.set_title('Non-zero Weight Count')

    # 5. Per-layer sparsification (if extended log)
    ax = axes[1, 1]
    layer_cols = [c for c in df.columns if c.startswith('layer_') and c.endswith('_NZ')]
    if layer_cols:
        for col in layer_cols:
            label = col.replace('layer_', 'L').replace('_NZ', '')
            ax.plot(df['step'], df[col], label=label)
        ax.legend(fontsize=8)
        ax.set_xlabel('Step'); ax.set_ylabel('NZ per layer')
        ax.set_title('Per-layer Sparsification')
    else:
        ax.text(0.5, 0.5, 'No per-layer data\n(needs extended log)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Per-layer Sparsification')

    # 6. Timing breakdown (if extended log)
    ax = axes[1, 2]
    if 'prune_time_s' in df.columns:
        ax.plot(df['step'], df['prune_time_s'],  label='search', linewidth=1)
        ax.plot(df['step'], df['adjust_time_s'], label='adjust', linewidth=1)
        ax.legend(fontsize=8)
        ax.set_xlabel('Step'); ax.set_ylabel('Time (s)')
        ax.set_title('Time per Step')
    else:
        ax.text(0.5, 0.5, 'No timing data\n(needs extended log)',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)
        ax.set_title('Time per Step')

    plt.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'run_summary.png')
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    log     = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else None
    plot_run(log, out_dir=out_dir)
```

- [ ] **Step 5: Run the test**

```bash
python -m pytest tests/test_benchmark.py::test_parse_log_basic -v
```

Expected: `1 passed`.

- [ ] **Step 6: Smoke-test against the existing log**

```bash
python visualize/plot_run.py artifacts/sparsified/sparsification_log.csv artifacts/sparsified/
```

Expected: matplotlib window with 6 panels (or saved PNG if no display). Notes: panels 5+6 will show "No per-layer data" / "No timing data" until Task 2 has been run.

- [ ] **Step 7: Commit**

```bash
git add visualize/__init__.py visualize/plot_run.py tests/test_benchmark.py
git commit -m "feat: add visualize/plot_run.py — 6-panel static summary from sparsification log"
```

---

## Task 5: Weight heatmaps — `plot_weights.py`

**Files:**

- Create: `visualize/plot_weights.py`
- Test: `tests/test_benchmark.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_benchmark.py

def test_load_checkpoint_returns_correct_layers():
    """load_checkpoint() must return a list of numpy arrays matching saved shapes."""
    import tempfile
    from visualize.plot_weights import load_checkpoint
    shapes = [(10, 5), (3, 10), (2, 3)]
    with tempfile.TemporaryDirectory() as tmp:
        for i, shape in enumerate(shapes):
            np.save(os.path.join(tmp, f'W_{i}.npy'), np.random.randn(*shape).astype(np.float32))
        weights = load_checkpoint(tmp)
    assert len(weights) == 3
    for w, shape in zip(weights, shapes):
        assert w.shape == shape
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_benchmark.py::test_load_checkpoint_returns_correct_layers -v
```

Expected: `ModuleNotFoundError: No module named 'visualize.plot_weights'`.

- [ ] **Step 3: Create `visualize/plot_weights.py`**

```python
# visualize/plot_weights.py
"""
Weight magnitude and sparsity mask heatmaps from checkpoint directories.

Usage:
    python visualize/plot_weights.py artifacts/sparsified/checkpoints/step_0000 \
                                     artifacts/sparsified/checkpoints/step_0100 \
                                     [--out artifacts/sparsified/]
"""
import sys, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_checkpoint(ckpt_dir: str):
    """Return list of W_i.npy arrays from a checkpoint directory, ordered by layer index."""
    count = sum(1 for f in os.listdir(ckpt_dir) if f.startswith('W_') and f.endswith('.npy'))
    return [np.load(os.path.join(ckpt_dir, f'W_{i}.npy')) for i in range(count)]


def plot_weight_heatmaps(ckpt_dirs, step_labels=None, out_dir=None, show=True):
    """
    Each column = one checkpoint, each pair of rows = (|W| magnitude, mask) for one layer.

    ckpt_dirs:    list of paths to checkpoint directories
    step_labels:  list of strings for column headers (defaults to directory names)
    """
    if step_labels is None:
        step_labels = [os.path.basename(d) for d in ckpt_dirs]

    checkpoints = [load_checkpoint(d) for d in ckpt_dirs]
    n_layers = len(checkpoints[0])
    n_ckpts  = len(checkpoints)

    fig, axes = plt.subplots(
        n_layers * 2, n_ckpts,
        figsize=(max(3 * n_ckpts, 6), max(4 * n_layers, 4)),
        squeeze=False,
    )
    fig.suptitle('Weight Magnitude & Sparsity Pattern over Steps', fontsize=12)

    for ci, (weights, label) in enumerate(zip(checkpoints, step_labels)):
        for li, W in enumerate(weights):
            sparsity = 1.0 - float((W != 0).mean())

            # Row 2*li: weight magnitude (log scale)
            ax_w = axes[li * 2][ci]
            vmax = float(np.abs(W).max()) + 1e-9
            im = ax_w.imshow(
                np.abs(W), aspect='auto', cmap='viridis',
                norm=mcolors.LogNorm(vmin=1e-6, vmax=vmax),
            )
            ax_w.set_title(f'{label}\nL{li} |W| ({W.shape[0]}×{W.shape[1]})', fontsize=7)
            ax_w.axis('off')
            plt.colorbar(im, ax=ax_w, fraction=0.046, pad=0.02)

            # Row 2*li+1: binary mask (black = pruned)
            ax_m = axes[li * 2 + 1][ci]
            ax_m.imshow(W != 0, aspect='auto', cmap='gray_r', vmin=0, vmax=1)
            ax_m.set_title(f'mask  sp={sparsity:.2f}', fontsize=7)
            ax_m.axis('off')

    plt.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'weight_heatmaps.png')
        fig.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {out_path}")
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    out_dir = None
    for i, a in enumerate(sys.argv[1:]):
        if a == '--out':
            out_dir = sys.argv[i + 2]
    plot_weight_heatmaps(args, out_dir=out_dir)
```

- [ ] **Step 4: Run test**

```bash
python -m pytest tests/test_benchmark.py::test_load_checkpoint_returns_correct_layers -v
```

Expected: `1 passed`.

- [ ] **Step 5: Smoke-test** (requires having run the sparsifier at least once so `checkpoints/step_0000` exists)

```bash
python visualize/plot_weights.py artifacts/sparsified/checkpoints/step_0000
```

Expected: matplotlib window showing one column with |W| and mask panels per layer.

- [ ] **Step 6: Commit**

```bash
git add visualize/plot_weights.py tests/test_benchmark.py
git commit -m "feat: add visualize/plot_weights.py — weight magnitude and mask heatmaps from checkpoints"
```

---

## Task 6: Live view — `live_view.py`

**Files:**

- Create: `visualize/live_view.py`
- Test: `tests/test_benchmark.py` (import smoke test only)

- [ ] **Step 1: Write the smoke test**

```python
# append to tests/test_benchmark.py

def test_live_view_imports_without_error():
    """live_view.py must be importable — catches syntax errors and missing deps."""
    import importlib
    mod = importlib.import_module('visualize.live_view')
    assert hasattr(mod, 'live_view'), "live_view() function must be defined"
```

- [ ] **Step 2: Run to confirm failure**

```bash
python -m pytest tests/test_benchmark.py::test_live_view_imports_without_error -v
```

Expected: `ModuleNotFoundError: No module named 'visualize.live_view'`.

- [ ] **Step 3: Create `visualize/live_view.py`**

```python
# visualize/live_view.py
"""
Live real-time view of an in-progress sparsification run.
Polls the log CSV every interval_ms milliseconds.
Optionally shows weight mask heatmap from the latest checkpoint.

Usage:
    python visualize/live_view.py path/to/sparsification_log.csv
    python visualize/live_view.py path/to/sparsification_log.csv path/to/checkpoints/
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')          # change to 'Qt5Agg' or 'MacOSX' if TkAgg unavailable
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import pandas as pd


def _latest_checkpoint_W0(ckpt_base):
    """Return W_0 array from the most recent checkpoint dir, or None."""
    if not ckpt_base or not os.path.isdir(ckpt_base):
        return None
    dirs = sorted([
        os.path.join(ckpt_base, d) for d in os.listdir(ckpt_base)
        if d.startswith('step_') and os.path.isdir(os.path.join(ckpt_base, d))
    ])
    if not dirs:
        return None
    w_path = os.path.join(dirs[-1], 'W_0.npy')
    return np.load(w_path) if os.path.exists(w_path) else None


def live_view(log_path: str, ckpt_base: str = None, interval_ms: int = 2000):
    """
    Open a live-updating matplotlib window.
    Panels: accuracy vs sparsity, NZ vs step, d_manifold, timing, weight mask, info text.
    """
    fig = plt.figure(figsize=(15, 8))
    gs  = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)
    ax_acc  = fig.add_subplot(gs[0, 0])
    ax_nz   = fig.add_subplot(gs[0, 1])
    ax_d    = fig.add_subplot(gs[0, 2])
    ax_time = fig.add_subplot(gs[1, 0])
    ax_mask = fig.add_subplot(gs[1, 1])
    ax_info = fig.add_subplot(gs[1, 2])
    ax_info.axis('off')

    def update(_frame):
        try:
            df = pd.read_csv(log_path)
        except Exception:
            return
        if len(df) == 0:
            return

        for ax in [ax_acc, ax_nz, ax_d, ax_time, ax_mask]:
            ax.clear()

        last = df.iloc[-1]

        # Accuracy vs sparsity
        ax_acc.plot(df['sparsity'] * 100, df['val_acc'] * 100, 'b-o', markersize=2, linewidth=1)
        ax_acc.axhline(float(df['val_acc'].iloc[0]) * 100, color='gray', linestyle='--', alpha=0.5)
        ax_acc.set_xlabel('Sparsity (%)'); ax_acc.set_ylabel('Val Acc (%)')
        ax_acc.set_title('Accuracy vs Sparsity')

        # NZ vs step
        ax_nz.plot(df['step'], df['NZ'], 'g-', linewidth=1)
        ax_nz.set_xlabel('Step'); ax_nz.set_ylabel('NZ weights')
        ax_nz.set_title(f'Step {int(last["step"])} | NZ {int(last["NZ"])} / {int(last["total_W"])}')

        # d_manifold
        d_vals = df['d_manifold'].replace(0, np.nan)
        if d_vals.notna().any():
            ax_d.semilogy(df['step'], d_vals, 'r-', linewidth=1)
        ax_d.set_xlabel('Step'); ax_d.set_ylabel('d_manifold')
        ax_d.set_title('Manifold Distance')

        # Timing
        if 'prune_time_s' in df.columns:
            ax_time.plot(df['step'], df['prune_time_s'],  label='search', linewidth=1)
            ax_time.plot(df['step'], df['adjust_time_s'], label='adjust', linewidth=1)
            ax_time.legend(fontsize=7)
            ax_time.set_xlabel('Step'); ax_time.set_ylabel('Time (s)')
            ax_time.set_title(f"Last: {float(last['prune_time_s']):.2f}s + {float(last['adjust_time_s']):.2f}s")
        else:
            ax_time.text(0.5, 0.5, 'No timing data', ha='center', va='center',
                         transform=ax_time.transAxes)
            ax_time.set_title('Time per Step')

        # Weight mask (latest checkpoint)
        W = _latest_checkpoint_W0(ckpt_base)
        if W is not None:
            ax_mask.imshow(W != 0, aspect='auto', cmap='gray_r', vmin=0, vmax=1)
            sp = 1.0 - float((W != 0).mean())
            ax_mask.set_title(f'L0 mask  sp={sp:.2f}')
        else:
            ax_mask.text(0.5, 0.5, 'No checkpoint yet\n(saved every 50 steps)',
                         ha='center', va='center', transform=ax_mask.transAxes, fontsize=9)
            ax_mask.set_title('L0 Weight Mask')
        ax_mask.axis('off')

        # Info panel
        info = (
            f"step:     {int(last['step'])}\n"
            f"NZ:       {int(last['NZ'])} / {int(last['total_W'])}\n"
            f"sparsity: {float(last['sparsity'])*100:.2f}%\n"
            f"val_acc:  {float(last['val_acc'])*100:.2f}%\n"
            f"d_W:      {float(last['d_W']):.3e}\n"
        )
        if 'prune_time_s' in df.columns:
            total_t = df['prune_time_s'].sum() + df['adjust_time_s'].sum()
            info += f"total_t:  {total_t:.1f}s\n"
        ax_info.clear(); ax_info.axis('off')
        ax_info.text(0.05, 0.95, info, transform=ax_info.transAxes,
                     fontfamily='monospace', fontsize=10, verticalalignment='top',
                     bbox=dict(facecolor='lightyellow', alpha=0.8, boxstyle='round'))

    ani = animation.FuncAnimation(
        fig, update, interval=interval_ms, cache_frame_data=False
    )
    fig.suptitle(f'Live Sparsification — {os.path.basename(log_path)}', fontsize=12)
    plt.show()


if __name__ == '__main__':
    log  = sys.argv[1]
    ckpt = sys.argv[2] if len(sys.argv) > 2 else None
    live_view(log, ckpt_base=ckpt)
```

- [ ] **Step 4: Run the smoke test**

```bash
python -m pytest tests/test_benchmark.py::test_live_view_imports_without_error -v
```

Expected: `1 passed`.

- [ ] **Step 5: Manual integration test** (open a second terminal and start a sparsifier run, then open the live view)

Terminal 1:

```bash
python -m sparsifier.sparsifier artifacts/ data/X_test_small.csv data/Y_test_small.csv
```

Terminal 2 (while Terminal 1 is running):

```bash
python visualize/live_view.py artifacts/sparsified/sparsification_log.csv \
                               artifacts/sparsified/checkpoints/
```

Expected: matplotlib window opens, updates every 2 seconds as new rows are appended to the log.

- [ ] **Step 6: Commit**

```bash
git add visualize/live_view.py tests/test_benchmark.py
git commit -m "feat: add visualize/live_view.py — real-time sparsification dashboard"
```

---

## Task 7: Strategy comparison — `compare_strategies.py` + `plot_comparison.py`

**Files:**

- Create: `benchmark/compare_strategies.py`
- Create: `visualize/plot_comparison.py`

- [ ] **Step 1: Create `benchmark/compare_strategies.py`**

```python
# benchmark/compare_strategies.py
"""
Run multiple pruning strategies for N steps on the same starting network.
Saves one CSV per strategy; use visualize/plot_comparison.py to overlay plots.

Usage:
    python benchmark/compare_strategies.py artifacts/ data/X_test_small.csv data/Y_test_small.csv \
        --steps 50 --out artifacts/comparison/
"""
import sys, os, time, csv, argparse
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from mlp.mlp import accuracy, load_network_params
from sparsifier.sparsifier import clone_network, d, make_omega, prune
from benchmark.correctness_check import prune_gradient_topk


STRATEGIES = {
    'exhaustive':       lambda net, og, om: prune(net, og, om, doAdjust=True),
    'gradient_top_1':   lambda net, og, om: prune_gradient_topk(net, og, om, top_k=1,  doAdjust=True),
    'gradient_top_10':  lambda net, og, om: prune_gradient_topk(net, og, om, top_k=10, doAdjust=True),
    'gradient_top_50':  lambda net, og, om: prune_gradient_topk(net, og, om, top_k=50, doAdjust=True),
}


def run_strategy(strategy_name, strategy_fn, og_net, x_test, y_test, omega, n_steps, out_dir):
    net = clone_network(og_net)
    total_W = int(sum(l.W.size for l in og_net))
    log_path = os.path.join(out_dir, f'{strategy_name}.csv')

    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'NZ', 'sparsity', 'val_acc', 'd_manifold',
                         'prune_time_s', 'adjust_time_s'])
        for i in range(n_steps):
            NZ         = int(sum((l.W != 0).sum() for l in net))
            sparsity   = 1.0 - NZ / total_W
            val_acc    = float(accuracy(net, x_test, y_test))
            d_manifold = float(d(net, og_net, omega))
            net, meta  = strategy_fn(net, og_net, omega)
            writer.writerow([
                i, NZ, round(sparsity, 6), round(val_acc, 6),
                "{:.6e}".format(d_manifold),
                round(meta.prune_time_s, 4),
                round(meta.adjust_time_s, 4),
            ])
            f.flush()
            print(f"  [{strategy_name}] step {i:3d} | acc={val_acc:.4f} | "
                  f"t={meta.prune_time_s:.2f}s")

    print(f"Saved: {log_path}")
    return log_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('net_folder')
    p.add_argument('x_test')
    p.add_argument('y_test')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--out',   default='artifacts/comparison/')
    p.add_argument('--strategies', nargs='+',
                   default=list(STRATEGIES.keys()),
                   choices=list(STRATEGIES.keys()))
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    og_net = load_network_params(args.net_folder)
    x_test = np.genfromtxt(args.x_test, delimiter=',', max_rows=500)
    y_test = np.genfromtxt(args.y_test, delimiter=',', max_rows=500)
    omega  = make_omega(og_net)

    for name in args.strategies:
        print(f"\n=== Strategy: {name} ===")
        run_strategy(name, STRATEGIES[name], og_net, x_test, y_test, omega, args.steps, args.out)

    print(f"\nAll results in {args.out}. Run:")
    print(f"  python visualize/plot_comparison.py {args.out}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Create `visualize/plot_comparison.py`**

```python
# visualize/plot_comparison.py
"""
Overlay accuracy, sparsity, and timing curves from multiple strategy CSVs.

Usage:
    python visualize/plot_comparison.py artifacts/comparison/ [--out artifacts/comparison/]
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLORS = {
    'exhaustive':      '#1f77b4',
    'gradient_top_1':  '#d62728',
    'gradient_top_10': '#ff7f0e',
    'gradient_top_50': '#2ca02c',
}


def plot_strategy_comparison(csv_dir: str, out_dir: str = None, show: bool = True):
    csv_files = sorted([
        f for f in os.listdir(csv_dir) if f.endswith('.csv')
    ])
    if not csv_files:
        print("No CSV files found in", csv_dir); return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('Strategy Comparison', fontsize=13)

    for fname in csv_files:
        name = fname.replace('.csv', '')
        df   = pd.read_csv(os.path.join(csv_dir, fname))
        c    = COLORS.get(name, None)
        kw   = dict(label=name, color=c, linewidth=1.5)

        axes[0].plot(df['sparsity'] * 100, df['val_acc'] * 100, **kw)
        axes[1].plot(df['step'], df['prune_time_s'], **kw)
        axes[2].plot(df['step'], df['prune_time_s'] + df['adjust_time_s'], **kw)

    for ax, (xlabel, ylabel, title) in zip(axes, [
        ('Sparsity (%)', 'Val Accuracy (%)', 'Accuracy vs Sparsity'),
        ('Step',         'Search time (s)',  'Search Time per Step'),
        ('Step',         'Total time (s)',   'Total Time per Step (search + adjust)'),
    ]):
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, 'strategy_comparison.png')
        fig.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
    if show:
        plt.show()
    return fig


if __name__ == '__main__':
    csv_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else csv_dir
    plot_strategy_comparison(csv_dir, out_dir=out_dir)
```

- [ ] **Step 3: Smoke-test comparison run** (20 steps, 4 strategies)

```bash
python benchmark/compare_strategies.py artifacts/ \
    data/X_test_small.csv data/Y_test_small.csv \
    --steps 20 --out artifacts/comparison/
```

Expected: 4 CSV files written to `artifacts/comparison/`, each with 20 rows. Terminal shows per-step timing for each strategy.

- [ ] **Step 4: Smoke-test comparison plot**

```bash
python visualize/plot_comparison.py artifacts/comparison/
```

Expected: 3-panel matplotlib window with accuracy and timing curves for all 4 strategies overlaid. The exhaustive strategy should be slowest; `gradient_top_1` should be fastest; all should produce similar accuracy curves at low sparsity.

- [ ] **Step 5: Commit**

```bash
git add benchmark/compare_strategies.py visualize/plot_comparison.py
git commit -m "feat: add strategy comparison harness and overlay plot"
```

---

## Self-Review

**Spec coverage:**

- Performance benchmarking: covered by Task 7 (compare_strategies.py) and extended log timing in Task 2.
- Correctness verification: covered by Task 3 (correctness_check.py) — agreement rate + speedup table.
- Static visualizations: covered by Tasks 4 and 5 (plot_run.py, plot_weights.py).
- Live view: covered by Task 6 (live_view.py).
- Memory of proposals: saved via memory files outside the repo (done before plan write).

**Placeholder scan:** None found — all steps contain complete code.

**Type consistency:**

- `prune()` returns `(net, PruneMeta)` in Task 1; all callers in Tasks 2, 3, 7 unpack this correctly.
- `prune_gradient_topk()` in `correctness_check.py` returns the same `(net, PruneMeta)` signature as `prune()` — used correctly in `compare_strategies.py`.
- `parse_log()` in `plot_run.py` returns `pd.DataFrame` — used correctly in tests.
- `load_checkpoint()` in `plot_weights.py` returns `list[np.ndarray]` — used correctly in tests and `live_view.py`.
