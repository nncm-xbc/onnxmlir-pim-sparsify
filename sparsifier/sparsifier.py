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

import csv
import json
import os
import sys
import time
from typing import NamedTuple

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

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
    return np.random.randint(0, 256, size=(n_samples, input_dim)).astype(np.float32)


# Estimate the manifold distance between two networks by comparing
# their outputs over a shared sample from Ω.
def _d_impl(net_1, net_2, omega):
    return jnp.sum((batched_predict(net_1, omega) - batched_predict(net_2, omega)) ** 2)


d = jax.jit(_d_impl)
d_grad = jax.jit(jax.grad(_d_impl))
_d_val_grad = jax.jit(jax.value_and_grad(_d_impl))


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
    curr_val, gradiente = _d_val_grad(net, cmp_net, omega)
    for _ in range(max_iters):
        if alfa <= 1e-14:
            break
        new_net = [
            Layer(W=l.W - alfa * g.W, b=l.b - alfa * g.b, mask=l.mask)
            for l, g in zip(net, gradiente)
        ]
        new_val, new_grad = _d_val_grad(new_net, cmp_net, omega)
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
        og_outputs = np.array(batched_predict(og_net, omega), dtype=np.float32)
        layers_ext = [
            (
                np.asarray(l.W, dtype=np.float32),
                np.asarray(l.b, dtype=np.float32),
                np.asarray(l.mask, dtype=np.float32),
                act,
            )
            for l, act in zip(net, activations)
        ]
        omega_f32 = np.asarray(omega, dtype=np.float32)
        min_dist_idx, min_dist_i, min_dist_j, min_dist = _prune_ext.find_best_candidate(
            layers_ext, og_outputs, omega_f32
        )
        min_dist = float(min_dist)
    else:
        search_done = False
        for idx, layer in enumerate(net):
            for i in range(layer.W.shape[0]):
                for j in range(layer.W.shape[1]):
                    if layer.W[i, j] == 0.0:
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
    cfg_path = os.path.abspath(sys.argv[1])
    with open(cfg_path) as f:
        cfg = json.load(f)

    # paths in config are relative to the repo root (parent of experiments/)
    repo_root     = os.path.dirname(os.path.dirname(cfg_path))
    input_folder  = os.path.join(repo_root, 'artifacts', cfg['name'])
    output_folder = os.path.join(input_folder, 'sparsified')

    def resolve(p):
        return os.path.join(repo_root, p)

    x_test = np.genfromtxt(resolve(cfg['data']['x_test']), delimiter=",", max_rows=1000)
    y_test = np.genfromtxt(resolve(cfg['data']['y_test']), delimiter=",", max_rows=1000)

    sp = cfg['sparsify']

    # Set hidden activation before the first JAX trace.
    import mlp.mlp as _mlp
    _mlp.hidden_activation = _mlp._ACTS[cfg.get('hidden_activation', 'relu')]
    # Activation list for the C++ extension (one entry per layer).
    act_name   = cfg.get('hidden_activation', 'relu')
    activations = [act_name] * (len(cfg['topology']) - 1) + ['linear']

    print("Load the parameters from the folder")
    og_net = load_network_params(input_folder)
    print("Accuracy in validation: %.4f" % float(accuracy(og_net, x_test, y_test)))
    total_W = int(sum(l.W.size for l in og_net))
    print("Total parameters: %d" % total_W)
    perturbed_net = [
        Layer(W=l.W + np.random.normal(size=l.W.shape) * 0.00001, b=l.b, mask=l.mask)
        for l in og_net
    ]
    omega = make_omega(og_net, n_samples=sp['omega_samples'])
    print(
        "Perturbation distance (sanity check): %.4e"
        % float(d(og_net, perturbed_net, omega))
    )

    print("Starting sparsification loop")
    print("At every iteration the network gets:")
    print("\t 1. Pruned   — remove the least influential parameter")
    print("\t 2. Adjusted — compensate via remaining parameters")
    net = clone_network(og_net)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    log_path = os.path.join(output_folder, "sparsification_log.csv")
    with open(log_path, "w", newline="") as log_file:
        writer = csv.writer(log_file)
        layer_NZ_cols = ["layer_%d_NZ" % li for li in range(len(og_net))]
        header = [
            "step",
            "NZ",
            "total_W",
            "sparsity",
            "val_acc",
            "d_manifold",
            "d_W",
            "prune_time_s",
            "adjust_time_s",
            "candidate_layer",
            "candidate_i",
            "candidate_j",
        ] + layer_NZ_cols
        writer.writerow(header)

        for i in range(sp['steps']):
            NZ = int(np.sum([(l.W != 0).sum() for l in net]))
            sparsity = 1.0 - NZ / total_W
            val_acc = float(accuracy(net, x_test, y_test))
            d_manifold = float(d(net, og_net, omega))

            print(
                "step {:4d} | acc={:.4f} | NZ={:6d} | sparsity={:.4f} | d_m={:.4e}".format(
                    i, val_acc, NZ, sparsity, d_manifold
                )
            )

            W_snapshot = [np.array(l.W).copy() for l in net]
            net, meta = prune(net, og_net, omega, activations=activations, doAdjust=sp['do_adjust'])
            d_W = float(
                np.sqrt(
                    sum(
                        np.sum((np.array(l.W) - w) ** 2)
                        for l, w in zip(net, W_snapshot)
                    )
                )
            )

            layer_nz_vals = [int((l.W != 0).sum()) for l in net]
            writer.writerow(
                [
                    i,
                    NZ,
                    total_W,
                    round(sparsity, 6),
                    round(val_acc, 6),
                    "{:.6e}".format(d_manifold),
                    "{:.6e}".format(d_W),
                    round(meta.prune_time_s, 4),
                    round(meta.adjust_time_s, 4),
                    meta.layer_idx,
                    meta.i,
                    meta.j,
                ]
                + layer_nz_vals
            )
            log_file.flush()

            # save weight snapshots every N steps for heatmap visualizations
            if i % sp['checkpoint_every'] == 0:
                ckpt_dir = os.path.join(output_folder, "checkpoints", "step_%04d" % i)
                os.makedirs(ckpt_dir, exist_ok=True)
                for li, layer in enumerate(net):
                    np.save(os.path.join(ckpt_dir, "W_%d.npy" % li), layer.W)

    print("Sparsification log saved to:", log_path)

    for i, l in enumerate(net):
        np.save(os.path.join(output_folder, "W_%i.npy" % i), l.W)
        np.save(os.path.join(output_folder, "b_%i.npy" % i), l.b)


if __name__ == "__main__":
    main()
