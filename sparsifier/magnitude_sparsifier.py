"""Magnitude-scored weight pruning with manifold-distance adjustment.

Selection criterion: argmin |W[l][i,j]| over all active weights (globally).
Adjustment step:    gradient descent on d_W — identical to sparsifier.sparsifier.

This is the ablation baseline for the thesis: any difference in the
sparsity/accuracy/d_W trajectory vs the manifold baseline is attributable
solely to the choice of selection criterion.

Public API: :func:`prune_magnitude`, :class:`MagnitudePruneMeta`.
"""

import time
from typing import NamedTuple

import numpy as np

from sparsifier.sparsifier import adjust, clone_network


class MagnitudePruneMeta(NamedTuple):
    layer_idx:     int
    i:             int
    j:             int
    magnitude:     float   # |W[layer_idx][i,j]| before zeroing
    prune_time_s:  float
    adjust_time_s: float


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
    # Vectorised argmin; ties resolve to the first index in row-major scan
    # order (np.argmin), matching an explicit per-entry scan.
    for l, layer in enumerate(net):
        mags = np.abs(np.asarray(layer.W, dtype=np.float64))
        mags[np.asarray(layer.mask) == 0.0] = np.inf
        flat = int(np.argmin(mags))
        mag  = float(mags.flat[flat])
        if mag < min_mag:
            min_mag = mag
            min_layer = l
            min_i, min_j = (int(v) for v in np.unravel_index(flat, mags.shape))
    prune_time_s = time.perf_counter() - prune_t0

    result_net = clone_network(net)
    result_net[min_layer].W[min_i, min_j] = 0.0
    result_net[min_layer].mask[min_i, min_j] = 0.0

    adjust_t0 = time.perf_counter()
    if doAdjust and min_mag > 0:
        result_net = adjust(result_net, og_net, omega)
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


########################################################################


def main():
    import sys

    from sparsifier.runner import run_sparsifier
    run_sparsifier(
        sys.argv[1],
        lambda net, og, om, doAdjust, activations: prune_magnitude(net, og, om, doAdjust=doAdjust),
        output_subdir='magnitude_sparsified',
        log_name='magnitude_sparsification_log.csv',
        loop_label='Magnitude sparsification',
    )


if __name__ == '__main__':
    main()
