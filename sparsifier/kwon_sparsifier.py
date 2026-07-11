"""Kwon et al. 2022 baseline — Fisher-weighted importance pruning.

Adapts the importance-scoring idea from Kwon et al. 2022
(kwon2022fastposttrainingpruningframework) to the MLP post-training setting.
The original paper targets transformers; the core mechanism — ranking weights
by a Fisher-information-style score and pruning the least important globally —
transfers directly to weight-level MLP pruning.

Selection criterion:
    score(l,i,j) = (∂d_W/∂W[l][i,j])² · W[l][i,j]²

Interpretation: squared gradient weighted by squared weight magnitude.  This
is a first-order approximation of the squared change in d_W when w_ij → 0.

The adjustment step is identical to the manifold baseline (gradient descent on
d_W), isolating the contribution of the faster scoring function.  Cost per step:
O(B·C) for one gradient call + O(N) scan, vs O(N·B·C) for the manifold baseline.

Public API: :func:`prune_kwon`, :class:`KwonPruneMeta`.
"""

import time
from typing import NamedTuple

import numpy as np

from sparsifier.sparsifier import adjust, clone_network, d_grad


class KwonPruneMeta(NamedTuple):
    layer_idx:     int
    i:             int
    j:             int
    score:         float   # (∂d_W/∂w_ij)² · w_ij²
    prune_time_s:  float
    adjust_time_s: float


def prune_kwon(net, og_net, omega, doAdjust=True):
    """Remove the weight with the smallest Fisher-weighted importance score.

    1. Compute gradient g = ∂d_W/∂W via one d_grad call.
    2. score(l,i,j) = g[l][i,j]² · W[l][i,j]²  for mask==1 entries.
    3. argmin score globally → winning (l,i,j).
    4. Zero W[l][i,j] and mask[l][i,j].
    5. If doAdjust, run adjust(net, og_net, omega).

    Returns (pruned_net, KwonPruneMeta).
    """
    prune_t0 = time.perf_counter()

    grads = d_grad(net, og_net, omega)

    min_score = float('inf')
    min_layer = 0
    min_i     = 0
    min_j     = 0

    # Vectorised argmin; ties resolve to the first index in row-major scan
    # order (np.argmin), matching an explicit per-entry scan.
    for l, (layer, glayer) in enumerate(zip(net, grads)):
        scores = np.asarray(glayer.W) ** 2 * np.asarray(layer.W) ** 2
        scores[np.asarray(layer.mask) == 0.0] = np.inf
        flat  = int(np.argmin(scores))
        score = float(scores.flat[flat])
        if score < min_score:
            min_score = score
            min_layer = l
            min_i, min_j = (int(v) for v in np.unravel_index(flat, scores.shape))

    prune_time_s = time.perf_counter() - prune_t0

    result_net = clone_network(net)
    result_net[min_layer].W[min_i, min_j] = 0.0
    result_net[min_layer].mask[min_i, min_j] = 0.0

    adjust_t0 = time.perf_counter()
    if doAdjust:
        result_net = adjust(result_net, og_net, omega)
    adjust_time_s = time.perf_counter() - adjust_t0

    meta = KwonPruneMeta(
        layer_idx=min_layer,
        i=min_i,
        j=min_j,
        score=min_score,
        prune_time_s=prune_time_s,
        adjust_time_s=adjust_time_s,
    )
    return result_net, meta


########################################################################


def main():
    import sys

    from sparsifier.runner import run_sparsifier
    run_sparsifier(
        sys.argv[1],
        lambda net, og, om, doAdjust, activations: prune_kwon(net, og, om, doAdjust=doAdjust),
        output_subdir='kwon_sparsified',
        log_name='kwon_sparsification_log.csv',
        loop_label='Kwon sparsification',
    )


if __name__ == '__main__':
    main()
