"""Manifold-distance-scored *neuron* pruning — structured counterpart to :mod:`sparsifier.sparsifier`.

Prunes entire hidden neurons (row of ``W[l]`` + column of ``W[l+1]`` +
bias ``b[l][i]``) rather than individual weights. Reuses :func:`d` and
:func:`adjust` from :mod:`sparsifier.sparsifier` so results are directly
comparable with weight-level pruning.

Public API: :func:`prune_neuron`, :class:`NeuronPruneMeta`.
"""

import csv
import json
import os
import sys
import time
from typing import NamedTuple

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import numpy as np

from mlp.mlp import Layer, accuracy, load_network_params
from sparsifier.sparsifier import adjust, clone_network, d, make_omega


class NeuronPruneMeta(NamedTuple):
    layer_idx: int      # weight-matrix index l; neuron is row W[l][neuron_idx, :]
    neuron_idx: int     # neuron index i within that layer
    distance: float     # d(og_net, probe_net, omega) at the winning candidate
    prune_time_s: float
    adjust_time_s: float


def prune_neuron(net, og_net, omega, doAdjust=True):
    """Remove the hidden neuron that minimally perturbs d(net, og_net, omega).

    Iterates over all hidden-layer neurons (rows of W[l] for l in 0..len(net)-2).
    For each candidate neuron i in layer l, temporarily zeros:
      - W[l][i, :], mask[l][i, :], b[l][i]   — outgoing connections + bias
      - W[l+1][:, i], mask[l+1][:, i]         — downstream connections
    Evaluates d(og_net, probe_net, omega), then restores and tracks the minimum.
    Applies the winner permanently and runs adjust() if doAdjust=True.

    Returns (pruned_net, NeuronPruneMeta).
    """
    if len(net) < 2:
        raise ValueError("Network must have at least 2 weight matrices (one hidden layer).")

    min_dist = 1e16
    min_dist_layer = 0
    min_dist_neuron = 0

    probe_net = clone_network(net)
    prune_t0 = time.perf_counter()

    for l in range(len(net) - 1):           # hidden-layer indices: 0 to len(net)-2
        for i in range(net[l].W.shape[0]):  # neurons = rows of W[l]
            if np.all(net[l].W[i, :] == 0.0):
                continue  # neuron already pruned — skip

            # Save entries that will be temporarily zeroed
            saved_W_row    = probe_net[l].W[i, :].copy()
            saved_mask_row = probe_net[l].mask[i, :].copy()
            saved_b        = float(probe_net[l].b[i])
            saved_W_col    = probe_net[l + 1].W[:, i].copy()
            saved_mask_col = probe_net[l + 1].mask[:, i].copy()

            # Zero the candidate neuron
            probe_net[l].W[i, :]        = 0.0
            probe_net[l].mask[i, :]     = 0.0
            probe_net[l].b[i]           = 0.0
            probe_net[l + 1].W[:, i]    = 0.0
            probe_net[l + 1].mask[:, i] = 0.0

            distance = float(d(og_net, probe_net, omega))

            # Restore
            probe_net[l].W[i, :]        = saved_W_row
            probe_net[l].mask[i, :]     = saved_mask_row
            probe_net[l].b[i]           = saved_b
            probe_net[l + 1].W[:, i]    = saved_W_col
            probe_net[l + 1].mask[:, i] = saved_mask_col

            if distance < min_dist:
                min_dist        = distance
                min_dist_layer  = l
                min_dist_neuron = i

    prune_time_s = time.perf_counter() - prune_t0

    # Apply winning neuron permanently
    probe_net[min_dist_layer].W[min_dist_neuron, :]        = 0.0
    probe_net[min_dist_layer].mask[min_dist_neuron, :]     = 0.0
    probe_net[min_dist_layer].b[min_dist_neuron]           = 0.0
    probe_net[min_dist_layer + 1].W[:, min_dist_neuron]    = 0.0
    probe_net[min_dist_layer + 1].mask[:, min_dist_neuron] = 0.0

    adjust_t0 = time.perf_counter()
    if doAdjust and min_dist > 0:
        probe_net = adjust(probe_net, og_net, omega)
    adjust_time_s = time.perf_counter() - adjust_t0

    meta = NeuronPruneMeta(
        layer_idx=min_dist_layer,
        neuron_idx=min_dist_neuron,
        distance=float(min_dist),
        prune_time_s=prune_time_s,
        adjust_time_s=adjust_time_s,
    )
    return probe_net, meta


########################################################################


def main():
    cfg_path = os.path.abspath(sys.argv[1])
    with open(cfg_path) as f:
        cfg = json.load(f)

    repo_root     = os.path.dirname(os.path.dirname(cfg_path))
    input_folder  = os.path.join(repo_root, 'artifacts', cfg['name'])
    output_folder = os.path.join(input_folder, 'neuron_sparsified')

    def resolve(p):
        return os.path.join(repo_root, p)

    x_test = np.genfromtxt(resolve(cfg['data']['x_test']), delimiter=',', max_rows=1000)
    y_test = np.genfromtxt(resolve(cfg['data']['y_test']), delimiter=',', max_rows=1000)

    sp = cfg['sparsify']

    print("Load the parameters from the folder")
    og_net = load_network_params(input_folder)
    print("Accuracy in validation: %.4f" % float(accuracy(og_net, x_test, y_test)))

    # Total hidden neurons = rows of W[l] for l in 0..len-2
    hidden_layer_indices = list(range(len(og_net) - 1))
    total_neurons = int(sum(og_net[l].W.shape[0] for l in hidden_layer_indices))
    print("Total hidden neurons: %d" % total_neurons)

    omega = make_omega(og_net, n_samples=sp['omega_samples'])
    print(
        "Perturbation distance (sanity check): %.4e"
        % float(d(og_net, [
            Layer(W=l.W + np.random.normal(size=l.W.shape) * 0.00001, b=l.b, mask=l.mask)
            for l in og_net
        ], omega))
    )

    print("Starting neuron sparsification loop")
    print("At every iteration the network gets:")
    print("\t 1. Pruned   — remove the least influential neuron")
    print("\t 2. Adjusted — compensate via remaining parameters")
    net = clone_network(og_net)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    log_path = os.path.join(output_folder, 'neuron_sparsification_log.csv')
    with open(log_path, 'w', newline='') as log_file:
        writer = csv.writer(log_file)

        layer_neuron_cols = ['layer_%d_neurons' % l for l in hidden_layer_indices]
        header = [
            'step', 'neurons_pruned', 'total_neurons', 'neuron_sparsity',
            'val_acc', 'd_manifold', 'd_W',
            'prune_time_s', 'adjust_time_s',
            'candidate_layer', 'candidate_neuron',
        ] + layer_neuron_cols
        writer.writerow(header)

        for step in range(sp['steps']):
            # Surviving neurons per hidden layer = rows with at least one non-zero weight
            layer_neuron_counts = [
                int((net[l].W != 0).any(axis=1).sum())
                for l in hidden_layer_indices
            ]
            neurons_pruned  = total_neurons - sum(layer_neuron_counts)
            neuron_sparsity = neurons_pruned / total_neurons
            val_acc         = float(accuracy(net, x_test, y_test))
            d_manifold      = float(d(net, og_net, omega))

            print(
                "step {:4d} | acc={:.4f} | neurons_pruned={:4d}/{:4d} | "
                "sparsity={:.4f} | d_m={:.4e}".format(
                    step, val_acc, neurons_pruned, total_neurons,
                    neuron_sparsity, d_manifold,
                )
            )

            W_snapshot = [np.array(l.W).copy() for l in net]
            net, meta  = prune_neuron(net, og_net, omega, doAdjust=sp['do_adjust'])
            d_W = float(
                np.sqrt(sum(
                    np.sum((np.array(l.W) - w) ** 2)
                    for l, w in zip(net, W_snapshot)
                ))
            )

            layer_neuron_counts_after = [
                int((net[l].W != 0).any(axis=1).sum())
                for l in hidden_layer_indices
            ]
            writer.writerow([
                step,
                total_neurons - sum(layer_neuron_counts_after),
                total_neurons,
                round(neuron_sparsity, 6),
                round(val_acc, 6),
                '{:.6e}'.format(d_manifold),
                '{:.6e}'.format(d_W),
                round(meta.prune_time_s, 4),
                round(meta.adjust_time_s, 4),
                meta.layer_idx,
                meta.neuron_idx,
            ] + layer_neuron_counts_after)
            log_file.flush()

            if step % sp['checkpoint_every'] == 0:
                ckpt_dir = os.path.join(output_folder, 'checkpoints', 'step_%04d' % step)
                os.makedirs(ckpt_dir, exist_ok=True)
                for li, layer in enumerate(net):
                    np.save(os.path.join(ckpt_dir, 'W_%d.npy' % li), layer.W)

    print("Neuron sparsification log saved to:", log_path)
    for i, layer in enumerate(net):
        np.save(os.path.join(output_folder, 'W_%i.npy' % i), layer.W)
        np.save(os.path.join(output_folder, 'b_%i.npy' % i), layer.b)


if __name__ == '__main__':
    main()
