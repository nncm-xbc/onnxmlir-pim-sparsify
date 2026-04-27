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

import csv
import json
import os
import sys
import time
from typing import NamedTuple

import numpy as np

from mlp.mlp import Layer, accuracy, load_network_params
from sparsifier.sparsifier import adjust, clone_network, d, d_grad, make_omega


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

    for l, (layer, glayer) in enumerate(zip(net, grads)):
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                if layer.mask[i, j] == 0.0:
                    continue
                score = float(glayer.W[i, j]) ** 2 * float(layer.W[i, j]) ** 2
                if score < min_score:
                    min_score = score
                    min_layer = l
                    min_i     = i
                    min_j     = j

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
    cfg_path = os.path.abspath(sys.argv[1])
    with open(cfg_path) as f:
        cfg = json.load(f)

    repo_root     = os.path.dirname(os.path.dirname(cfg_path))
    input_folder  = os.path.join(repo_root, 'artifacts', cfg['name'])
    output_folder = os.path.join(input_folder, 'kwon_sparsified')

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

    print("Starting Kwon sparsification loop")
    net = clone_network(og_net)

    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, 'kwon_sparsification_log.csv')
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
            sparsity   = 1.0 - NZ / total_W
            val_acc    = float(accuracy(net, x_test, y_test))
            d_manifold = float(d(net, og_net, omega))

            print(
                'step {:4d} | acc={:.4f} | NZ={:6d} | sparsity={:.4f} | d_m={:.4e}'.format(
                    i, val_acc, NZ, sparsity, d_manifold
                )
            )

            W_snapshot = [np.array(l.W).copy() for l in net]
            net, meta  = prune_kwon(net, og_net, omega, doAdjust=sp['do_adjust'])
            d_W = float(
                np.sqrt(sum(
                    np.sum((np.array(l.W) - w) ** 2)
                    for l, w in zip(net, W_snapshot)
                ))
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

    print("Kwon sparsification log saved to:", log_path)
    for i, l in enumerate(net):
        np.save(os.path.join(output_folder, 'W_%i.npy' % i), l.W)
        np.save(os.path.join(output_folder, 'b_%i.npy' % i), l.b)


if __name__ == '__main__':
    main()
