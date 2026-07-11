---
title: OBS Baseline — Optimal Brain Surgeon
date: 2026-04-27
status: approved
---

# OBS Baseline Design Spec

## Purpose

Implement Hassibi & Stork (1993) OBS (`hassibi1993second`) as a post-training baseline using d_W as the objective. OBS is strictly stronger than OBD: it uses the full Hessian inverse to both score saliency AND compute the closed-form optimal weight correction after each removal.

| Axis | Thesis (manifold) | OBS baseline |
|---|---|---|
| Selection | argmin d_W (exact) | argmin w_q²/(2·[H⁻¹]_qq) |
| Adjustment | gradient descent on d_W | closed-form: δw = −(w_q/[H⁻¹]_qq)·H⁻¹[:,q] |
| Cost per step | O(N·B·C) | O(N²) Hessian + O(N³) inversion |

OBS does **not** call `adjust()`. Its weight update is:
`δw_flat = -(w_q / H_inv[q,q]) · H_inv[:, q]`

Applied after zeroing w_q, with mask re-enforced (pruned positions zeroed after update). Hessian regularised: `H_reg = H + 1e-6·I` (pinv for numerical stability).

## New file

`sparsifier/obs_sparsifier.py`

Public API:
```python
class OBSPruneMeta(NamedTuple):
    layer_idx: int; i: int; j: int
    score: float    # w_q²/(2·H_inv[q,q])
    prune_time_s: float

def prune_obs(net, og_net, omega):
    ...
```

No `doAdjust` parameter — OBS always applies its own weight correction.
Output: `artifacts/<name>/obs_sparsified/obs_sparsification_log.csv`
Config: `experiments/e14_obs.json`
