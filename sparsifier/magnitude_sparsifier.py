"""Magnitude-scored weight pruning with manifold-distance adjustment.

Selection criterion: argmin |W[l][i,j]| over all active weights (globally).
Adjustment step:    gradient descent on d_W — identical to sparsifier.sparsifier.

This is the ablation baseline for the thesis: any difference in the
sparsity/accuracy/d_W trajectory vs the manifold baseline is attributable
solely to the choice of selection criterion.

Public API: :func:`prune_magnitude`, :class:`MagnitudePruneMeta`.
"""

import csv
import json
import os
import sys
import time
from typing import NamedTuple

import numpy as np

from mlp.mlp import Layer, accuracy, load_network_params
from sparsifier.sparsifier import adjust, clone_network, d, make_omega


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
    for l, layer in enumerate(net):
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                if layer.mask[i, j] == 0.0:
                    continue
                mag = abs(float(layer.W[i, j]))
                if mag < min_mag:
                    min_mag = mag
                    min_layer = l
                    min_i = i
                    min_j = j
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
    cfg_path = os.path.abspath(sys.argv[1])
    with open(cfg_path) as f:
        cfg = json.load(f)

    repo_root     = os.path.dirname(os.path.dirname(cfg_path))
    input_folder  = os.path.join(repo_root, 'artifacts', cfg['name'])
    output_folder = os.path.join(input_folder, 'magnitude_sparsified')

    def resolve(p):
        return os.path.join(repo_root, p)

    x_test = np.genfromtxt(resolve(cfg['data']['x_test']), delimiter=',', max_rows=1000)
    y_test = np.genfromtxt(resolve(cfg['data']['y_test']), delimiter=',', max_rows=1000)

    sp = cfg['sparsify']

    import mlp.mlp as _mlp
    _mlp.hidden_activation = _mlp._ACTS[cfg.get('hidden_activation', 'relu')]

    print("Load the parameters from the folder")
    og_net = load_network_params(input_folder)
    print("Accuracy in validation: %.4f" % float(accuracy(og_net, x_test, y_test)))
    total_W = int(sum(l.W.size for l in og_net))
    print("Total parameters: %d" % total_W)

    omega = make_omega(og_net, n_samples=sp['omega_samples'])
    print(
        "Perturbation distance (sanity check): %.4e"
        % float(d(og_net, [
            Layer(W=l.W + np.random.normal(size=l.W.shape) * 0.00001, b=l.b, mask=l.mask)
            for l in og_net
        ], omega))
    )

    print("Starting magnitude sparsification loop")
    net = clone_network(og_net)

    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, 'magnitude_sparsification_log.csv')
    with open(log_path, 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        layer_NZ_cols = ['layer_%d_NZ' % li for li in range(len(og_net))]
        header = [
            'step', 'NZ', 'total_W', 'sparsity', 'val_acc',
            'd_manifold', 'd_W',
            'prune_time_s', 'adjust_time_s',
            'candidate_layer', 'candidate_i', 'candidate_j',
        ] + layer_NZ_cols
        writer.writerow(header)

        for i in range(sp['steps']):
            NZ = int(np.sum([(l.W != 0).sum() for l in net]))
            sparsity = 1.0 - NZ / total_W
            val_acc = float(accuracy(net, x_test, y_test))
            d_manifold = float(d(net, og_net, omega))

            print(
                'step {:4d} | acc={:.4f} | NZ={:6d} | sparsity={:.4f} | d_m={:.4e}'.format(
                    i, val_acc, NZ, sparsity, d_manifold
                )
            )

            W_snapshot = [np.array(l.W).copy() for l in net]
            net, meta = prune_magnitude(net, og_net, omega, doAdjust=sp['do_adjust'])
            d_W = float(
                np.sqrt(
                    sum(
                        np.sum((np.array(l.W) - w) ** 2)
                        for l, w in zip(net, W_snapshot)
                    )
                )
            )

            layer_nz_vals = [int((l.W != 0).sum()) for l in net]
            writer.writerow([
                i, NZ, total_W, round(sparsity, 6), round(val_acc, 6),
                '{:.6e}'.format(d_manifold), '{:.6e}'.format(d_W),
                round(meta.prune_time_s, 4), round(meta.adjust_time_s, 4),
                meta.layer_idx, meta.i, meta.j,
            ] + layer_nz_vals)
            log_file.flush()

            if i % sp['checkpoint_every'] == 0:
                ckpt_dir = os.path.join(output_folder, 'checkpoints', 'step_%04d' % i)
                os.makedirs(ckpt_dir, exist_ok=True)
                for li, layer in enumerate(net):
                    np.save(os.path.join(ckpt_dir, 'W_%d.npy' % li), layer.W)

    print("Magnitude sparsification log saved to:", log_path)
    for i, l in enumerate(net):
        np.save(os.path.join(output_folder, 'W_%i.npy' % i), l.W)
        np.save(os.path.join(output_folder, 'b_%i.npy' % i), l.b)


if __name__ == '__main__':
    main()
