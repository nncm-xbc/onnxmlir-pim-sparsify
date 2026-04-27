"""Lazarevich et al. 2021 baseline — data-driven Omega.

Identical to the manifold sparsifier (sparsifier.sparsifier) except that the
Monte Carlo sample Omega is drawn from the *real* test-image distribution
instead of uniform pixel noise.  Any difference in the sparsity/accuracy
trajectory vs the manifold baseline is attributable solely to the choice of
Omega distribution (real MNIST images vs uniform noise).

Public API: :func:`main` (entry point only — reuses prune/adjust from
sparsifier.sparsifier directly).
"""

import csv
import json
import os
import sys

import numpy as np

from mlp.mlp import Layer, accuracy, load_network_params
from sparsifier.sparsifier import adjust, clone_network, d, make_omega, prune


def main():
    cfg_path = os.path.abspath(sys.argv[1])
    with open(cfg_path) as f:
        cfg = json.load(f)

    repo_root     = os.path.dirname(os.path.dirname(cfg_path))
    input_folder  = os.path.join(repo_root, 'artifacts', cfg['name'])
    output_folder = os.path.join(input_folder, 'lazarevich_sparsified')

    def resolve(p):
        return os.path.join(repo_root, p)

    x_test = np.genfromtxt(resolve(cfg['data']['x_test']), delimiter=',', max_rows=1000)
    y_test = np.genfromtxt(resolve(cfg['data']['y_test']), delimiter=',', max_rows=1000)

    sp = cfg['sparsify']

    import mlp.mlp as _mlp
    _mlp.hidden_activation = _mlp._ACTS[cfg.get('hidden_activation', 'relu')]

    # Activation list for C++ extension (one entry per layer).
    act_name    = cfg.get('hidden_activation', 'relu')
    activations = [act_name] * (len(cfg['topology']) - 1) + ['linear']

    print("Load the parameters from the folder")
    og_net = load_network_params(input_folder)
    print("Accuracy in validation: %.4f" % float(accuracy(og_net, x_test, y_test)))
    total_W = int(sum(l.W.size for l in og_net))
    print("Total parameters: %d" % total_W)

    # --- Key difference from manifold sparsifier ---
    # Use real calibration images (x_test, float32) as Omega instead of
    # uniform random noise.  This matches Lazarevich et al.'s use of an
    # unlabelled calibration set.
    n_omega = min(sp['omega_samples'], len(x_test))
    omega   = x_test[:n_omega].astype(np.float32)
    print("Omega: %d real calibration images (data-driven)" % n_omega)

    print(
        "Perturbation distance (sanity check): %.4e"
        % float(d(og_net, [
            Layer(W=l.W + np.random.normal(size=l.W.shape) * 0.00001, b=l.b, mask=l.mask)
            for l in og_net
        ], omega))
    )

    print("Starting Lazarevich sparsification loop")
    net = clone_network(og_net)

    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, 'lazarevich_sparsification_log.csv')
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
            sparsity  = 1.0 - NZ / total_W
            val_acc   = float(accuracy(net, x_test, y_test))
            d_manifold = float(d(net, og_net, omega))

            print(
                'step {:4d} | acc={:.4f} | NZ={:6d} | sparsity={:.4f} | d_m={:.4e}'.format(
                    i, val_acc, NZ, sparsity, d_manifold
                )
            )

            W_snapshot = [np.array(l.W).copy() for l in net]
            net, meta = prune(net, og_net, omega, activations=activations,
                              doAdjust=sp['do_adjust'])
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

    print("Lazarevich sparsification log saved to:", log_path)
    for i, l in enumerate(net):
        np.save(os.path.join(output_folder, 'W_%i.npy' % i), l.W)
        np.save(os.path.join(output_folder, 'b_%i.npy' % i), l.b)


if __name__ == '__main__':
    main()
