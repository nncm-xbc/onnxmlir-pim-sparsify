"""OBS-scored *neuron* pruning — structured counterpart to
:mod:`sparsifier.obs_sparsifier`.

Lifts the weight-level Optimal Brain Surgeon saliency
``score(q) = w_q² / (2 · [H⁻¹]_qq)`` (H = full Hessian of d_W) to neuron
granularity. A "neuron" is defined exactly as in
:mod:`sparsifier.neuron_sparsifier`: hidden neuron ``i`` of weight matrix
``l`` owns its incoming row ``W[l][i, :]`` **and** its outgoing column
``W[l+1][:, i]``.

Aggregation: **sum** of the per-weight OBS saliencies over the neuron's
active weights (same rationale as the OBD variant: each per-weight OBS score
estimates that weight's removal cost, so the sum is the neuron's saliency).

CAVEAT — this is the *diagonal* group approximation. Exact group-OBS for
removing a weight set S uses the block sub-inverse,
``½ · w_Sᵀ (H⁻¹_SS)⁻¹ w_S``, which accounts for coupling *between* the
neuron's own weights; summing the individual ``w_q²/(2·[H⁻¹]_qq)`` terms
ignores that intra-neuron coupling. We deliberately keep the diagonal sum
(a) to stay a drop-in swap of the selection criterion only, and (b) because
the compensating weight update here is the shared gradient-descent
``adjust()`` (as in neuron_sparsifier), NOT the OBS closed-form update — so
the block-inverse correction is never applied anyway. The weight-level OBS's
active-set / near-singular-H fallback (negative H⁻¹ diagonal → skip) is
carried over: weights with non-positive ``[H⁻¹]_qq`` are excluded from a
neuron's score.

Selection: argmin of the summed saliency. The full-Hessian machinery is
reused verbatim from :func:`sparsifier.obs_sparsifier._hessian_and_inv`.

Public API: :func:`prune_neuron_obs`, :class:`NeuronPruneMeta`.
"""

import sys
import time

import numpy as np

from sparsifier.sparsifier import adjust, clone_network
from sparsifier.obs_sparsifier import _hessian_and_inv
from sparsifier.neuron_magnitude_sparsifier import (
    NeuronPruneMeta, _remove_neuron, run_neuron_sparsifier,
)


def _score_and_valid_layers(net, og_net, omega):
    """Per-weight OBS saliency (0 where invalid) and a boolean validity mask,
    both reshaped to per-layer arrays matching W. A weight is valid iff its
    mask is active AND its H⁻¹ diagonal entry is positive."""
    _, H_inv, W_shapes, W_flat = _hessian_and_inv(net, og_net, omega)
    diag = np.diag(H_inv)
    valid_flat = diag > 0.0
    score_flat = np.zeros_like(W_flat)
    score_flat[valid_flat] = W_flat[valid_flat] ** 2 / (2.0 * diag[valid_flat])

    layer_sizes = [s[0] * s[1] for s in W_shapes]
    offsets = np.cumsum([0] + layer_sizes)
    score_layers, valid_layers = [], []
    for l, shape in enumerate(W_shapes):
        sl = slice(offsets[l], offsets[l + 1])
        score_layers.append(score_flat[sl].reshape(shape))
        active = np.asarray(net[l].mask) != 0.0
        valid_layers.append(valid_flat[sl].reshape(shape) & active)
    return score_layers, valid_layers


def prune_neuron_obs(net, og_net, omega, doAdjust=True):
    """Remove the hidden neuron with the smallest summed OBS saliency; then adjust.

    Note: uses the shared adjust() for compensation (like neuron_sparsifier),
    not the OBS closed-form group update — see the module caveat.
    Returns (pruned_net, NeuronPruneMeta).
    """
    prune_t0 = time.perf_counter()

    score_layers, valid_layers = _score_and_valid_layers(net, og_net, omega)

    best_score = float('inf')
    best_l, best_i = 0, 0
    for l in range(len(net) - 1):
        for i in range(net[l].W.shape[0]):
            if np.all(net[l].mask[i, :] == 0.0):
                continue
            row_v = np.asarray(valid_layers[l][i, :])
            col_v = np.asarray(valid_layers[l + 1][:, i])
            if not (row_v.any() or col_v.any()):
                continue  # no valid (positive-H⁻¹) weight — skip this neuron
            score = float(
                np.asarray(score_layers[l][i, :])[row_v].sum()
                + np.asarray(score_layers[l + 1][:, i])[col_v].sum()
            )
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
    pruned, meta = prune_neuron_obs(net, net, omega, doAdjust=False)
    assert meta.layer_idx in (0, 1) and 0 <= meta.neuron_idx < 3
    l, i = meta.layer_idx, meta.neuron_idx
    assert np.all(pruned[l].W[i, :] == 0) and np.all(pruned[l + 1].W[:, i] == 0)
    assert np.isfinite(meta.score)
    print("neuron_obs selfcheck OK:", meta)


def main():
    run_neuron_sparsifier(
        sys.argv[1], prune_neuron_obs,
        output_subdir='neuron_obs_sparsified',
        log_name='neuron_sparsification_log.csv',
        loop_label='Neuron OBS sparsification',
    )


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--selfcheck':
        _selfcheck()
    else:
        main()
