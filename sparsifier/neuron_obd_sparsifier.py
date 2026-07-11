"""OBD-scored *neuron* pruning — structured counterpart to
:mod:`sparsifier.obd_sparsifier`.

Lifts the weight-level Optimal Brain Damage saliency
``score(l,i,j) = 0.5 · H_ii · w_ij²`` (H = diagonal Hessian of d_W) to
neuron granularity. A "neuron" is defined exactly as in
:mod:`sparsifier.neuron_sparsifier`: hidden neuron ``i`` of weight matrix
``l`` owns its incoming row ``W[l][i, :]`` **and** its outgoing column
``W[l+1][:, i]``.

Aggregation: **sum** of the per-weight OBD scores over the neuron's active
weights. Sum (not mean) because each per-weight OBD score already estimates
that weight's contribution to the rise in d_W when removed, so their sum is
the diagonal (independent-weights) estimate of the total d_W increase from
removing the whole neuron — the natural neuron saliency.

Selection: argmin of the summed saliency — the neuron whose removal is
predicted to perturb d_W least is pruned. The diagonal Hessian machinery is
reused verbatim from :func:`sparsifier.obd_sparsifier._diag_hessian`.
Removal + adjust are identical to :mod:`sparsifier.neuron_sparsifier`.

Public API: :func:`prune_neuron_obd`, :class:`NeuronPruneMeta`.
"""

import sys
import time

import numpy as np

from sparsifier.sparsifier import adjust, clone_network
from sparsifier.obd_sparsifier import _diag_hessian
from sparsifier.neuron_magnitude_sparsifier import (
    NeuronPruneMeta, _remove_neuron, run_neuron_sparsifier,
)


def _neuron_score(score_layers, net, l, i):
    """Sum per-weight OBD saliencies over neuron i's active weights
    (incoming row of layer l + outgoing column of layer l+1)."""
    row_m = np.asarray(net[l].mask[i, :]) != 0.0
    col_m = np.asarray(net[l + 1].mask[:, i]) != 0.0
    return float(
        np.asarray(score_layers[l][i, :])[row_m].sum()
        + np.asarray(score_layers[l + 1][:, i])[col_m].sum()
    )


def prune_neuron_obd(net, og_net, omega, doAdjust=True):
    """Remove the hidden neuron with the smallest summed OBD saliency; then adjust.

    Returns (pruned_net, NeuronPruneMeta).
    """
    prune_t0 = time.perf_counter()

    H_diag_layers = _diag_hessian(net, og_net, omega)
    score_layers = [
        0.5 * np.asarray(h) * np.asarray(l.W) ** 2
        for l, h in zip(net, H_diag_layers)
    ]

    best_score = float('inf')
    best_l, best_i = 0, 0
    for l in range(len(net) - 1):
        for i in range(net[l].W.shape[0]):
            if np.all(net[l].mask[i, :] == 0.0):
                continue
            score = _neuron_score(score_layers, net, l, i)
            if score < best_score:
                best_score, best_l, best_i = score, l, i

    prune_time_s = time.perf_counter() - prune_t0

    result_net = clone_network(net)
    _remove_neuron(result_net, best_l, best_i)

    adjust_t0 = time.perf_counter()
    if doAdjust:
        result_net = adjust(result_net, og_net, omega)
    adjust_time_s = time.perf_counter() - adjust_t0

    return result_net, NeuronPruneMeta(
        layer_idx=best_l, neuron_idx=best_i, score=best_score,
        prune_time_s=prune_time_s, adjust_time_s=adjust_time_s,
    )


def _selfcheck():
    from mlp.mlp import Layer
    from sparsifier.sparsifier import make_omega
    rng = np.random.default_rng(0)
    def layer(nout, nin):
        return Layer(W=rng.standard_normal((nout, nin)),
                     b=rng.standard_normal(nout), mask=np.ones((nout, nin)))
    net = [layer(3, 4), layer(3, 3), layer(2, 3)]
    omega = make_omega(net, n_samples=50)
    pruned, meta = prune_neuron_obd(net, net, omega, doAdjust=False)
    # A real hidden neuron is selected and fully removed.
    assert meta.layer_idx in (0, 1) and 0 <= meta.neuron_idx < 3
    l, i = meta.layer_idx, meta.neuron_idx
    assert np.all(pruned[l].W[i, :] == 0) and np.all(pruned[l + 1].W[:, i] == 0)
    assert np.isfinite(meta.score)
    print("neuron_obd selfcheck OK:", meta)


def main():
    run_neuron_sparsifier(
        sys.argv[1], prune_neuron_obd,
        output_subdir='neuron_obd_sparsified',
        log_name='neuron_sparsification_log.csv',
        loop_label='Neuron OBD sparsification',
    )


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--selfcheck':
        _selfcheck()
    else:
        main()
