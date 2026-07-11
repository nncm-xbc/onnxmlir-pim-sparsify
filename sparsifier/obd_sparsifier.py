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

Complexity per step: N Hessian-vector products, evaluated in fixed-size
chunks — peak memory O(chunk · |omega|) regardless of N, so no full N×N
matrix is ever materialised.

Public API: :func:`prune_obd`, :class:`OBDPruneMeta`.
"""

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from mlp.mlp import Layer, batched_predict
from sparsifier.sparsifier import adjust, clone_network


class OBDPruneMeta(NamedTuple):
    layer_idx:     int
    i:             int
    j:             int
    score:         float   # 0.5 * H_ii * w_ij²
    prune_time_s:  float
    adjust_time_s: float


def _diag_hessian(net, og_net, omega):
    """Diagonal of Hessian of d_W w.r.t. all W parameters.

    Uses N forward-over-reverse JVP calls (one per parameter) instead of
    jax.hessian so the full N×N matrix is never allocated.
    Peak memory: O(N) instead of O(N²).

    Returns list of arrays (one per layer) with shape matching W.
    """
    W_shapes = [l.W.shape for l in net]
    W_flat   = jnp.concatenate([jnp.array(l.W).ravel() for l in net])
    N        = int(W_flat.shape[0])

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

    grad_f    = jax.grad(d_of_W_flat)
    hvp_batch = jax.jit(jax.vmap(lambda v: jax.jvp(grad_f, (W_flat,), (v,))[1]))

    # Batch the N Hessian-vector products in chunks: same math as one HVP per
    # parameter, but ~chunk-size fewer dispatches. Chunk bounds peak memory
    # (each tangent carries its own forward/backward activations over omega).
    chunk   = 64
    W_dtype = np.array(W_flat).dtype
    H_diag  = np.zeros(N, dtype=W_dtype)
    for start in range(0, N, chunk):
        idxs = np.arange(start, min(start + chunk, N))
        E    = np.zeros((len(idxs), N), dtype=W_dtype)
        E[np.arange(len(idxs)), idxs] = 1.0
        hvp_rows = hvp_batch(jnp.array(E))          # row k = H @ e_{idxs[k]}
        H_diag[idxs] = np.asarray(hvp_rows)[np.arange(len(idxs)), idxs]

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

    # Vectorised argmin; ties resolve to the first index in row-major scan
    # order (np.argmin), matching an explicit per-entry scan.
    for l, (layer, h_layer) in enumerate(zip(net, H_diag_layers)):
        scores = 0.5 * np.asarray(h_layer) * np.asarray(layer.W) ** 2
        scores[np.asarray(layer.mask) == 0.0] = np.inf
        flat  = int(np.argmin(scores))
        score = float(scores.flat[flat])
        if score < min_score:
            min_score = score
            min_layer = l
            min_i, min_j = (int(v) for v in np.unravel_index(flat, scores.shape))

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
    import sys

    from sparsifier.runner import run_sparsifier
    run_sparsifier(
        sys.argv[1],
        lambda net, og, om, doAdjust, activations: prune_obd(net, og, om, doAdjust=doAdjust),
        output_subdir='obd_sparsified',
        log_name='obd_sparsification_log.csv',
        loop_label='OBD sparsification',
    )


if __name__ == '__main__':
    main()
