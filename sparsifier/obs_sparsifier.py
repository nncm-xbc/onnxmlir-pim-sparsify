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

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mlp.mlp import Layer, batched_predict
from sparsifier.sparsifier import clone_network


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

    # Chunked Hessian: jax.hessian vmaps all N basis vectors at once, which
    # OOMs the GPU (N x B x out intermediate). Compute H column-blocks via
    # vmapped Hessian-vector products in chunks instead (H is symmetric, so
    # the HVP results can be stacked as rows). Minimal fix, 2026-06-12.
    grad_f = jax.grad(d_of_W_flat)
    _hvp_chunk = jax.jit(jax.vmap(lambda v: jax.jvp(grad_f, (W_flat,), (v,))[1]))
    _CHUNK = 240  # divides 2160; bounded memory per chunk
    rows = []
    for s in range(0, N, _CHUNK):
        basis = np.zeros((min(_CHUNK, N - s), N), dtype=np.array(W_flat).dtype)
        for k in range(basis.shape[0]):
            basis[k, s + k] = 1.0
        rows.append(np.array(_hvp_chunk(jnp.array(basis))))
    H = np.concatenate(rows, axis=0)
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

    N           = len(W_flat)
    active_flat = np.concatenate([np.asarray(l.mask).ravel() != 0.0 for l in net])
    if not active_flat.any():
        raise ValueError("No non-zero weights left — network is fully pruned.")

    # Score active weights with positive H_inv diagonal (negative diagonal:
    # saddle-point artefact). Ties resolve to the first flat index
    # (np.argmin), matching an explicit scan.
    diag      = np.diag(H_inv)
    scores    = np.asarray(W_flat) ** 2 / (2.0 * diag)
    valid     = active_flat & (diag > 0)
    min_score = float('inf')
    if valid.any():
        best_q    = int(np.argmin(np.where(valid, scores, np.inf)))
        min_score = float(scores[best_q])
    else:
        # Degenerate fallback: no valid OBS candidate (H ill-conditioned).
        # Fall back to magnitude selection — zero-only, no weight update.
        mags   = np.where(active_flat, np.abs(np.asarray(W_flat)), np.inf)
        best_q = int(np.argmin(mags))

    # Map flat index back to (layer, i, j)
    layer_sizes = [s[0] * s[1] for s in W_shapes]
    offsets     = np.cumsum([0] + layer_sizes)
    min_layer   = int(np.searchsorted(offsets, best_q, side='right') - 1)
    min_i, min_j = (int(v) for v in
                    np.unravel_index(best_q - offsets[min_layer], W_shapes[min_layer]))
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

    # Apply update (active weights only), zero the pruned weight, re-enforce masks
    result_net = clone_network(net)
    idx = 0
    for layer in result_net:
        dw = delta_w[idx:idx + layer.W.size].reshape(layer.W.shape)
        layer.W[...] += np.where(np.asarray(layer.mask) != 0.0, dw, 0.0)
        idx += layer.W.size

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
    import sys

    from sparsifier.runner import run_sparsifier
    run_sparsifier(
        sys.argv[1],
        lambda net, og, om, doAdjust, activations: prune_obs(net, og, om),
        output_subdir='obs_sparsified',
        log_name='obs_sparsification_log.csv',
        has_adjust_time=False,   # OBS applies its own closed-form correction, no adjust step
        loop_label='OBS sparsification',
    )


if __name__ == '__main__':
    main()
