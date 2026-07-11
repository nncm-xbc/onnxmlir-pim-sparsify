"""Manifold-distance-scored weight pruning.

Implements iterative post-training sparsification: at each step the weight
that minimally perturbs the network's behaviour on a sample Ω from the
input manifold is removed (and optionally compensated by ``adjust``).

Distance is measured as ``d(net_1, net_2, omega) = sum((y1 - y2)**2)`` on
the shared sample, so the metric is differentiable and JIT-compiled via
``jax.jit`` (and ``jax.grad`` for the gradient-guided strategies in
:mod:`benchmark.correctness_check`).

Public API: :func:`prune`, :func:`adjust`, :func:`make_omega`, :func:`d`,
:func:`d_grad`, :func:`clone_network`, :class:`PruneMeta`.
"""

import os
import time
from typing import NamedTuple

import jax

from mlp.mlp import *

try:
    import prune_ext as _prune_ext

    _USE_EXT = True
except ImportError:
    _USE_EXT = False


class PruneMeta(NamedTuple):
    layer_idx: int
    i: int
    j: int
    distance: float
    prune_time_s: float
    adjust_time_s: float


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Build a random sample from the input domain Ω (uniform pixel noise).
# input_dim is inferred from the first layer weight matrix shape (out, in).
def make_omega(network, n_samples=10000):
    input_dim = network[0].W.shape[1]
    return np.random.randint(0, 256, size=(n_samples, input_dim)).astype(np.float64)


# Estimate the manifold distance between two networks by comparing
# their outputs over a shared sample from Ω.
def _d_impl(net_1, net_2, omega):
    return jnp.sum((batched_predict(net_1, omega) - batched_predict(net_2, omega)) ** 2)


d = jax.jit(_d_impl)
d_grad = jax.jit(jax.grad(_d_impl))


# Same distance, but against precomputed reference outputs: callers that
# evaluate many networks against one fixed network (prune candidates, adjust
# iterations) compute the reference forward pass once instead of per call.
def _d_to_outputs_impl(net, ref_out, omega):
    return jnp.sum((batched_predict(net, omega) - ref_out) ** 2)


_d_to_outputs = jax.jit(_d_to_outputs_impl)
_d_val_grad_to_outputs = jax.jit(jax.value_and_grad(_d_to_outputs_impl))


# construct a copy of the network
def clone_network(network):
    return [
        Layer(
            W=np.array(l.W).copy(), b=np.array(l.b).copy(), mask=np.array(l.mask).copy()
        )
        for l in network
    ]


def _zero_weight(layer, i, j):
    new_W = np.array(layer.W).copy()
    new_W[i, j] = 0.0
    new_mask = np.array(layer.mask).copy()
    new_mask[i, j] = 0.0
    return Layer(W=new_W, b=layer.b, mask=new_mask)


# Adjust Function : minimize the distance on the plane locally isomorphic to the manifold of parameters.
# given
#       - a network not adjusted
#       - a network that we would like to mimic semantically
#
# Returns
#       - a network, with the same sparsity pattern of the
#         input network but optimized to have minimal local distance
#         in the manifold of parameters with respect to the cmp_net


def adjust(net, cmp_net, omega, max_iters=500, tol=1e-9):
    net = clone_network(net)
    alfa = 1e-11
    cmp_out = batched_predict(cmp_net, omega)  # fixed target — forward it once
    curr_val, gradiente = _d_val_grad_to_outputs(net, cmp_out, omega)
    for _ in range(max_iters):
        if alfa <= 1e-14:
            break
        new_net = [
            Layer(W=l.W - alfa * g.W, b=l.b - alfa * g.b, mask=l.mask)
            for l, g in zip(net, gradiente)
        ]
        new_val, new_grad = _d_val_grad_to_outputs(new_net, cmp_out, omega)
        if new_val < curr_val:
            if curr_val - new_val < tol * curr_val:
                net = new_net
                break
            net, curr_val, gradiente = new_net, new_val, new_grad
            alfa = min(alfa * 1.2, 1.0)
        else:
            alfa *= 0.5
    return net


# Prune Function : function for increasing the sparsity pattern of a network
# given
#       - a network
#
# The function computes every possible variation of the initial network obtainable
# by, for each variation, putting a different weight to 0.
# The optimal candidate is the variation that minimizes the local distance with the
# original network.
#
# returns
#       - a network, with increased sparsity and optionally adjusted to minimize
#         distance to the original network in parameter space
#
def prune(net, og_net, omega, activations=None, doAdjust=True):
    if activations is None:
        activations = ["relu"] * (len(net) - 1) + ["linear"]

    min_dist = 1e16
    min_dist_idx = 0
    min_dist_i = 0
    min_dist_j = 0

    probe_net = clone_network(net)
    prune_t0 = time.perf_counter()

    if _USE_EXT:
        og_outputs = np.array(batched_predict(og_net, omega), dtype=np.float64)
        layers_ext = [
            (
                np.asarray(l.W, dtype=np.float64),
                np.asarray(l.b, dtype=np.float64),
                np.asarray(l.mask, dtype=np.float64),
                act,
            )
            for l, act in zip(net, activations)
        ]
        omega_f64 = np.asarray(omega, dtype=np.float64)
        min_dist_idx, min_dist_i, min_dist_j, min_dist = _prune_ext.find_best_candidate(
            layers_ext, og_outputs, omega_f64
        )
        min_dist = float(min_dist)
    else:
        # NOTE: keep the two-network d() here (not _d_to_outputs with a
        # precomputed og forward): evaluating both networks in one JIT
        # program makes a zero-effect candidate compare *exactly* equal to
        # the original, which the min_dist == 0 early exit depends on.
        search_done = False
        for idx, layer in enumerate(net):
            for i in range(layer.W.shape[0]):
                for j in range(layer.W.shape[1]):
                    if layer.mask[i, j] == 0.0:
                        continue
                    # zero candidate in-place, evaluate, restore
                    saved_W = probe_net[idx].W[i, j]
                    saved_mask = probe_net[idx].mask[i, j]
                    probe_net[idx].W[i, j] = 0.0
                    probe_net[idx].mask[i, j] = 0.0
                    distance = d(og_net, probe_net, omega)
                    probe_net[idx].W[i, j] = saved_W
                    probe_net[idx].mask[i, j] = saved_mask

                    if distance < min_dist:
                        min_dist = distance
                        min_dist_idx = idx
                        min_dist_i = i
                        min_dist_j = j
                        if min_dist == 0:  # weight does not affect distance — exit early
                            search_done = True
                            break
                if search_done:
                    break
            if search_done:
                break

    prune_time_s = time.perf_counter() - prune_t0

    # apply the winning zero permanently
    probe_net[min_dist_idx].W[min_dist_i, min_dist_j] = 0.0
    probe_net[min_dist_idx].mask[min_dist_i, min_dist_j] = 0.0
    adjust_t0 = time.perf_counter()
    if doAdjust and min_dist > 0:
        probe_net = adjust(probe_net, og_net, omega)
    adjust_time_s = time.perf_counter() - adjust_t0

    meta = PruneMeta(
        layer_idx=min_dist_idx,
        i=min_dist_i,
        j=min_dist_j,
        distance=float(min_dist),
        prune_time_s=prune_time_s,
        adjust_time_s=adjust_time_s,
    )
    return probe_net, meta


########################################################################################################################################


def main():
    import sys

    from sparsifier.runner import run_sparsifier
    run_sparsifier(
        sys.argv[1],
        lambda net, og, om, doAdjust, activations: prune(net, og, om, activations=activations, doAdjust=doAdjust),
        output_subdir='sparsified',
        log_name='sparsification_log.csv',
        loop_label='sparsification',
    )


if __name__ == "__main__":
    main()
