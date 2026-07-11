---
title: Kwon 2022 Baseline — Fisher-Weighted Importance
date: 2026-04-27
status: approved
---

# Kwon 2022 Baseline Design Spec

## Purpose

Adapt Kwon et al. 2022 (`kwon2022fastposttrainingpruningframework`) to the MLP post-training setting. The original method targets transformers; the core idea — ranking weights by Fisher-weighted importance and pruning globally — transfers directly.

| Axis | Thesis (manifold) | Kwon baseline |
|---|---|---|
| Selection | argmin d_W (exact, O(N·B·C)) | argmin (∂d_W/∂w_ij)²·w_ij² (O(B·C) gradient + O(N) scan) |
| Adjustment | gradient descent on d_W | gradient descent on d_W (identical) |
| Cost per step | O(N·B·C) — N forward passes | O(B·C) — one gradient call |

Score formula: `score(l,i,j) = (∂d_W/∂W[l][i,j])² · W[l][i,j]²`

Interpretable as the first-order Taylor approximation of the change in d_W when w_ij → 0, squared and weighted by w² for scale invariance (Fisher information style).

## New file

`sparsifier/kwon_sparsifier.py`

Imports `d_grad`, `adjust`, `clone_network`, `d`, `make_omega` from `sparsifier.sparsifier`.

Public API:

```python
class KwonPruneMeta(NamedTuple):
    layer_idx: int; i: int; j: int
    score: float          # (∂d_W/∂w_ij)² · w_ij²
    prune_time_s: float; adjust_time_s: float

def prune_kwon(net, og_net, omega, doAdjust=True):
    ...
```

Output folder: `artifacts/<name>/kwon_sparsified/`
Log: `kwon_sparsification_log.csv` — same columns as manifold log.
Config: `experiments/e12_kwon.json`
