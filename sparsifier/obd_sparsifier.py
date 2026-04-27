"""OBD baseline — Optimal Brain Damage (LeCun, Denker & Solla 1989).

Uses the diagonal of the Hessian of d_W w.r.t. all weight parameters to
score each weight's saliency:

    score(l,i,j) = 0.5 · H_ii · W[l][i,j]²
    H_ii = ∂²d_W / ∂W[l][i,j]²

Selection: argmin score over active weights (global).
Adjustment: same gradient-descent adjust() as the manifold baseline.

Note (ReLU identity): for ReLU networks, relu''=0 by JAX convention, so
  H_ii = 2·(∂d_W/∂w_i)²  and  score_OBD = score_Kwon.
They diverge for tanh/sigmoid architectures (cross-term is non-zero).

Complexity per step: O(N²) — jax.hessian materialises an N×N matrix.
Feasibility: N=2160 (baseline) → ~18 MB; N≈12500 (width_50) → ~625 MB.
Width_200 will OOM — document as limitation.

Public API: :func:`prune_obd`, :class:`OBDPruneMeta`.
"""

import csv
import json
import os
import sys
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mlp.mlp import Layer, accuracy, batched_predict, load_network_params
from sparsifier.sparsifier import adjust, clone_network, d, make_omega


class OBDPruneMeta(NamedTuple):
    layer_idx:     int
    i:             int
    j:             int
    score:         float   # 0.5 * H_ii * w_ij²
    prune_time_s:  float
    adjust_time_s: float


def _diag_hessian(net, og_net, omega):
    """Diagonal of Hessian of d_W w.r.t. all W parameters.

    Flattens all weight matrices to a single vector, computes the full
    Hessian via jax.hessian, returns its diagonal reshaped to match each
    layer's W matrix.

    og_net outputs are precomputed outside the differentiated function to
    avoid differentiating through the constant reference network.

    Returns list of arrays (one per layer) with shape matching W.
    """
    W_shapes = [l.W.shape for l in net]
    W_flat   = jnp.concatenate([jnp.array(l.W).ravel() for l in net])

    og_out = jax.lax.stop_gradient(batched_predict(og_net, omega))

    def d_of_W_flat(w_flat):
        idx      = 0
        new_net  = []
        for layer, shape in zip(net, W_shapes):
            n     = shape[0] * shape[1]
            W_new = w_flat[idx:idx + n].reshape(shape)
            idx  += n
            new_net.append(Layer(W=W_new, b=layer.b, mask=layer.mask))
        pred = batched_predict(new_net, omega)
        return jnp.sum((pred - og_out) ** 2)

    H      = np.array(jax.hessian(d_of_W_flat)(W_flat))
    H_diag = np.diag(H)

    idx    = 0
    result = []
    for shape in W_shapes:
        n = shape[0] * shape[1]
        result.append(H_diag[idx:idx + n].reshape(shape))
        idx += n
    return result


def prune_obd(net, og_net, omega, doAdjust=True):
    """Remove the weight with the smallest OBD saliency; then adjust.

    1. Compute diagonal Hessian H_ii = ∂²d_W/∂w_ij².
    2. score(l,i,j) = 0.5 · H_ii · w_ij²  for mask==1 entries.
    3. argmin score globally.
    4. Zero W[l][i,j] and mask[l][i,j].
    5. If doAdjust, run adjust(net, og_net, omega).

    Returns (pruned_net, OBDPruneMeta).
    """
    prune_t0 = time.perf_counter()

    H_diag_layers = _diag_hessian(net, og_net, omega)

    min_score = float('inf')
    min_layer = 0
    min_i     = 0
    min_j     = 0

    for l, (layer, h_layer) in enumerate(zip(net, H_diag_layers)):
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                if layer.mask[i, j] == 0.0:
                    continue
                score = 0.5 * float(h_layer[i, j]) * float(layer.W[i, j]) ** 2
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

    meta = OBDPruneMeta(
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
    output_folder = os.path.join(input_folder, 'obd_sparsified')

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

    print("Starting OBD sparsification loop")
    net = clone_network(og_net)

    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, 'obd_sparsification_log.csv')
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
            net, meta  = prune_obd(net, og_net, omega, doAdjust=sp['do_adjust'])
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

    print("OBD sparsification log saved to:", log_path)
    for i, l in enumerate(net):
        np.save(os.path.join(output_folder, 'W_%i.npy' % i), l.W)
        np.save(os.path.join(output_folder, 'b_%i.npy' % i), l.b)


if __name__ == '__main__':
    main()
