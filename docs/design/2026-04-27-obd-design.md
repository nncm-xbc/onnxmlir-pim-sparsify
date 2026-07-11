---
title: OBD Baseline — Optimal Brain Damage
date: 2026-04-27
status: approved
---

# OBD Baseline Design Spec

## Purpose

Implement LeCun, Denker & Solla (1989) OBD (`lecun1989optimal`) as a post-training baseline using d_W as the objective. OBD uses the **diagonal of the Hessian** to score weight saliency, capturing second-order curvature.

| Axis | Thesis (manifold) | OBD baseline |
|---|---|---|
| Selection | argmin d_W (exact) | argmin (1/2)·H_ii·w_ij² |
| Adjustment | gradient descent on d_W | gradient descent on d_W (identical) |
| Cost per step | O(N·B·C) | O(N²) Hessian + O(N) scan |

Score: `score(l,i,j) = 0.5 · H_ii · w_ij²` where `H_ii = ∂²d_W/∂W[l][i,j]²`.

Note (ReLU identity): For ReLU networks, `H_ii = 2·(∂d_W/∂w_i)²` because `relu'' = 0` in JAX's convention. Therefore OBD score = `(∂d_W/∂w_i)² · w_i²` = Kwon score for ReLU. The two methods diverge for tanh/sigmoid activations (experiments e06_act_tanh, e06_act_sigmoid) where the cross-term `(f_k-f*_k)·∂²f_k/∂w_i²` is non-zero.

## New file

`sparsifier/obd_sparsifier.py`

Hessian computation: flatten all W matrices to a vector, call `jax.hessian(d_of_W_flat)(W_flat)`, take diagonal. Feasible for N ≤ ~5000 (baseline N=2160, width_50 N≈12500 — note may be slow/memory-heavy for width_50, will OOM for width_200).

Public API:
```python
class OBDPruneMeta(NamedTuple):
    layer_idx: int; i: int; j: int
    score: float     # 0.5 * H_ii * w_ij²
    prune_time_s: float; adjust_time_s: float

def prune_obd(net, og_net, omega, doAdjust=True):
    ...
```

Output: `artifacts/<name>/obd_sparsified/obd_sparsification_log.csv`
Config: `experiments/e13_obd.json`
