---
title: Magnitude-Based Pruning Baseline (Cycle 1 of 5)
date: 2026-04-26
status: approved
---

# Magnitude-Based Weight Pruning — Design Spec

## Purpose

Add a **magnitude-based pruning baseline** that uses the same gradient-descent adjustment step as the manifold method. The only difference from the manifold baseline (`sparsifier/sparsifier.py`) is the selection criterion:

| Method | Selection criterion | Adjustment step |
|---|---|---|
| Manifold (thesis) | argmin d_W(og_net, prune_one(net, l,i,j), omega) | gradient descent on d_W |
| **Magnitude (this spec)** | **argmin \|W[l][i,j]\|** | gradient descent on d_W (identical) |

Any difference in the sparsity / accuracy / d_W trajectory between the two is attributable **solely** to the choice of selection criterion — isolating the claim that the d_W criterion is better than the magnitude heuristic. Corresponds to experiment config `experiments/e09_magnitude_search.json`.

Literature context: Han et al. (2015) show magnitude pruning is competitive when paired with retraining. This experiment checks whether the same holds when paired with the manifold adjust step instead.

---

## New file

`sparsifier/magnitude_sparsifier.py`

Mirrors the structure of `sparsifier/neuron_sparsifier.py`. Imports the following from `sparsifier.sparsifier` (manifold pipeline is the canonical source):

- `adjust`
- `clone_network`
- `make_omega`
- `d`

Does **not** import `prune` — the selection step is reimplemented with a different scoring function. Does **not** use the C++ extension `prune_ext` (magnitude selection is O(N) per step, no acceleration needed).

Run via: `python -m sparsifier.magnitude_sparsifier <config.json>`

---

## Public API

```python
class MagnitudePruneMeta(NamedTuple):
    layer_idx:     int
    i:             int
    j:             int
    magnitude:     float   # |W[layer_idx][i,j]| before zeroing
    prune_time_s:  float
    adjust_time_s: float


def prune_magnitude(net, og_net, omega, doAdjust=True):
    """Remove the smallest-magnitude active weight; then adjust.

    1. Find argmin |W[l][i,j]| over all (l,i,j) with mask == 1.
       Selection is global across all layers (same scope as the manifold method).
    2. Zero W[l][i,j] and mask[l][i,j] permanently.
    3. If doAdjust and pruned magnitude > 0, run adjust(net, og_net, omega).

    Returns (pruned_net, MagnitudePruneMeta).
    """
```

`og_net` and `omega` are unused by selection but required by `adjust()`. The signature matches `prune_neuron` for symmetry.

---

## Algorithm

```
for each active weight (l, i, j)  [mask[l][i,j] == 1]:
    track (l, i, j) with smallest |W[l][i,j]|

zero W[winner] and mask[winner]

if doAdjust and magnitude_winner > 0:
    net = adjust(net, og_net, omega)   # identical code path as manifold baseline
```

Selection scope is **global** — the same candidate pool as the manifold method. This is required for fair ablation: the only axis of variation is the scoring function, not the candidate set.

---

## `main()` entry point

Mirrors `neuron_sparsifier.main()`:

- Read config from `sys.argv[1]`.
- Load dense network from `artifacts/<cfg['name']>/`.
- Set `mlp.hidden_activation` from `cfg['hidden_activation']` **before any JAX trace**.
- Build `omega = make_omega(og_net, n_samples=cfg['sparsify']['omega_samples'])`.
- Output folder: `artifacts/<cfg['name']>/magnitude_sparsified/` (parallel to `sparsified/` and `neuron_sparsified/`).
- Run `cfg['sparsify']['steps']` iterations of `prune_magnitude(net, og_net, omega, doAdjust=cfg['sparsify']['do_adjust'])`.
- Save weight checkpoints every `cfg['sparsify']['checkpoint_every']` steps under `magnitude_sparsified/checkpoints/step_NNNN/`.
- At completion, write final `W_i.npy`, `b_i.npy` to `magnitude_sparsified/`.

---

## Log format

Output file: `artifacts/<name>/magnitude_sparsified/magnitude_sparsification_log.csv`

Columns — **identical to `sparsifier.sparsifier`'s log**:

```
step, NZ, total_W, sparsity, val_acc, d_manifold, d_W,
prune_time_s, adjust_time_s,
candidate_layer, candidate_i, candidate_j,
layer_0_NZ, layer_1_NZ, ...
```

- `d_manifold` — computed each step as `d(net, og_net, omega)` **before** pruning, same definition as baseline.
- `d_W` — L2 weight change per step (same definition as baseline).

Identical schema enables downstream comparison plots to overlay manifold vs magnitude trajectories without special-casing.

---

## Tests

File: `tests/test_magnitude_sparsifier.py`

| # | Test | What it checks |
|---|---|---|
| 1 | `test_selects_smallest_magnitude` | On a tiny net with known magnitudes, `prune_magnitude(..., doAdjust=False)` zeros exactly the entry with smallest `|w|` |
| 2 | `test_skips_pruned_weights` | A weight with `mask=0, W=0` is never selected even when its entry would be "smallest" |
| 3 | `test_single_weight_zeroed` | With `doAdjust=False`, exactly one entry differs between input and output (NZ count grows by exactly 1) |
| 4 | `test_adjust_reduces_distance` | With `doAdjust=True` and a non-zero pruned weight, `d(og_net, result, omega) ≤ d(og_net, post_zero_no_adjust, omega)` |
| 5 | `test_meta_consistency` | `meta.layer_idx, meta.i, meta.j` index into the actually-pruned weight; `meta.magnitude == |w_pre_zero|` |
| 6 | `test_mask_integrity` | `result[meta.layer_idx].mask[meta.i, meta.j] == 0` after pruning |

---

## Config update

`experiments/e09_magnitude_search.json`:

- Remove `"search": "magnitude"` from the `sparsify` block (the new module is the dispatch; no field needed).
- Update `_experiment.description` to reference `python -m sparsifier.magnitude_sparsifier`.

---

## Out of scope

- `run_experiments.sh` — untouched. Magnitude experiments are invoked manually.
- `prune_ext` C++ extension — not used.
- Architecture sweeps, seed variants — experimental coverage decided by user at run time.
- Comparison / plotting code — downstream of having both logs.
- OBD, OBS, Lazarevich, Kwon baselines — separate cycles (2–5).
