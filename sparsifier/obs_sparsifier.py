"""OBS baseline — Optimal Brain Surgeon (Hassibi & Stork 1993).

OBS is strictly stronger than OBD: it uses the full inverse Hessian to both
score saliency and compute the closed-form optimal weight correction after
each removal.

Selection criterion:
    score(q) = w_q² / (2 · [H⁻¹]_qq)

Closed-form weight update (no gradient-descent adjust):
    δw = -(w_q / [H⁻¹]_qq) · H⁻¹[:, q]

Hessian regularised:  H_reg = H + λI,  λ=1e-4  (guards against near-singular
H when d_W ≈ 0 at the initial state).  Mask is re-enforced after the weight
update to prevent un-pruning already-zeroed weights.

Complexity per step: O(N²) Hessian + O(N³) inversion.  Same feasibility
limits as OBD (baseline N=2160 is fine; width_200 will OOM).

Public API: :func:`prune_obs`, :class:`OBSPruneMeta`.
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
from sparsifier.sparsifier import clone_network, d, make_omega


class OBSPruneMeta(NamedTuple):
    layer_idx:    int
    i:            int
    j:            int
    score:        float   # w_q² / (2·H_inv[q,q])
    prune_time_s: float


_LAMBDA_REG = 1e-4   # Hessian regularisation strength


def _hessian_and_inv(net, og_net, omega):
    """Full Hessian of d_W w.r.t. flattened W, and its regularised inverse.

    Returns (H_flat, H_inv_flat, W_shapes) where H_flat and H_inv_flat are
    (N,N) arrays and W_shapes is the list of per-layer weight matrix shapes
    for back-mapping indices.
    """
    W_shapes = [l.W.shape for l in net]
    W_flat   = jnp.concatenate([jnp.array(l.W).ravel() for l in net])
    N        = len(W_flat)

    og_out = jax.lax.stop_gradient(batched_predict(og_net, omega))

    def d_of_W_flat(w_flat):
        idx     = 0
        new_net = []
        for layer, shape in zip(net, W_shapes):
            n     = shape[0] * shape[1]
            W_new = w_flat[idx:idx + n].reshape(shape)
            idx  += n
            new_net.append(Layer(W=W_new, b=layer.b, mask=layer.mask))
        pred = batched_predict(new_net, omega)
        return jnp.sum((pred - og_out) ** 2)

    H_jax  = jax.hessian(d_of_W_flat)(W_flat)
    H      = np.array(H_jax)
    H_reg  = H + _LAMBDA_REG * np.eye(N)
    H_inv  = np.linalg.inv(H_reg)

    return H, H_inv, W_shapes, np.array(W_flat)


def prune_obs(net, og_net, omega):
    """Remove the weight with the smallest OBS saliency; apply closed-form update.

    1. Compute H and H_inv = (H + λI)⁻¹.
    2. score(q) = w_q² / (2·H_inv[q,q]) for mask==1 entries.
    3. argmin score globally → q*.
    4. Apply weight update: δw = -(w_q* / H_inv[q*,q*]) · H_inv[:,q*].
    5. Zero w_q* (mask[q*] = 0).
    6. Re-enforce mask (zero any updates to already-pruned positions).

    Returns (updated_net, OBSPruneMeta).
    Note: no doAdjust parameter — OBS always applies its own weight correction.
    """
    prune_t0 = time.perf_counter()

    H, H_inv, W_shapes, W_flat = _hessian_and_inv(net, og_net, omega)

    # Build flat index → (layer, i, j) map for active weights
    idx      = 0
    flat_idx_to_loc = {}  # flat_index → (layer, i, j)
    for l, shape in enumerate(W_shapes):
        for i in range(shape[0]):
            for j in range(shape[1]):
                flat_idx_to_loc[idx] = (l, i, j)
                idx += 1

    N = len(W_flat)

    # Build flat active-weight mask
    active_flat = np.zeros(N, dtype=bool)
    idx = 0
    for layer in net:
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                if layer.mask[i, j] != 0.0:
                    active_flat[idx] = True
                idx += 1

    # Score each active weight; only consider entries with positive H_inv diagonal
    min_score = float('inf')
    best_q    = -1   # -1 = no valid OBS candidate found yet
    for q in range(N):
        if not active_flat[q]:
            continue
        h_inv_qq = float(H_inv[q, q])
        if h_inv_qq <= 0:
            continue  # negative diagonal: saddle-point artefact, skip
        score = float(W_flat[q]) ** 2 / (2.0 * h_inv_qq)
        if score < min_score:
            min_score = score
            best_q    = q

    if best_q == -1:
        # Degenerate fallback: no valid OBS candidate (H ill-conditioned).
        # Fall back to magnitude selection — zero-only, no weight update.
        min_mag = float('inf')
        for q in range(N):
            if not active_flat[q]:
                continue
            mag = abs(float(W_flat[q]))
            if mag < min_mag:
                min_mag = mag
                best_q  = q

    min_layer, min_i, min_j = flat_idx_to_loc[best_q]
    prune_time_s = time.perf_counter() - prune_t0

    h_inv_qq = float(H_inv[best_q, best_q])
    w_q      = float(W_flat[best_q])

    # Closed-form weight update: δw = -(w_q / H_inv[q,q]) · H_inv[:,q]
    # Skip the update if H_inv diagonal is non-positive (fallback case).
    if h_inv_qq > 0:
        delta_w = -(w_q / h_inv_qq) * H_inv[:, best_q]
        if not np.isfinite(delta_w).all():
            delta_w = np.zeros(N)   # guard against inf/nan from poor conditioning
    else:
        delta_w = np.zeros(N)

    # Apply update, zero the pruned weight, re-enforce masks
    result_net = clone_network(net)
    idx = 0
    for l, layer in enumerate(result_net):
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                if layer.mask[i, j] != 0.0:
                    layer.W[i, j] += float(delta_w[idx])
                idx += 1

    # Zero the pruned weight and its mask
    result_net[min_layer].W[min_i, min_j] = 0.0
    result_net[min_layer].mask[min_i, min_j] = 0.0

    # Re-enforce: any weight whose mask is 0 must remain 0
    for layer in result_net:
        layer.W[layer.mask == 0.0] = 0.0

    meta = OBSPruneMeta(
        layer_idx=min_layer,
        i=min_i,
        j=min_j,
        score=min_score,
        prune_time_s=prune_time_s,
    )
    return result_net, meta


########################################################################


def main():
    cfg_path = os.path.abspath(sys.argv[1])
    with open(cfg_path) as f:
        cfg = json.load(f)

    repo_root     = os.path.dirname(os.path.dirname(cfg_path))
    input_folder  = os.path.join(repo_root, 'artifacts', cfg['name'])
    output_folder = os.path.join(input_folder, 'obs_sparsified')

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

    print("Starting OBS sparsification loop")
    net = clone_network(og_net)

    os.makedirs(output_folder, exist_ok=True)
    log_path = os.path.join(output_folder, 'obs_sparsification_log.csv')
    with open(log_path, 'w', newline='') as log_file:
        writer = csv.writer(log_file)
        layer_NZ_cols = ['layer_%d_NZ' % li for li in range(len(og_net))]
        header = [
            'step', 'NZ', 'total_W', 'sparsity', 'val_acc',
            'd_manifold', 'd_W',
            'prune_time_s',
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
            net, meta  = prune_obs(net, og_net, omega)
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
                round(meta.prune_time_s, 4),
                meta.layer_idx, meta.i, meta.j,
            ] + layer_nz_vals)
            log_file.flush()

            if i % sp['checkpoint_every'] == 0:
                ckpt_dir = os.path.join(output_folder, 'checkpoints', 'step_%04d' % i)
                os.makedirs(ckpt_dir, exist_ok=True)
                for li, layer in enumerate(net):
                    np.save(os.path.join(ckpt_dir, 'W_%d.npy' % li), layer.W)

    print("OBS sparsification log saved to:", log_path)
    for i, l in enumerate(net):
        np.save(os.path.join(output_folder, 'W_%i.npy' % i), l.W)
        np.save(os.path.join(output_folder, 'b_%i.npy' % i), l.b)


if __name__ == '__main__':
    main()
