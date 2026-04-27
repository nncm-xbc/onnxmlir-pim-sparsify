# tests/test_obs_sparsifier.py
import sys, os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from mlp.mlp import Layer
from sparsifier.obs_sparsifier import OBSPruneMeta, prune_obs
from sparsifier.sparsifier import clone_network, d


def _tiny_net_rng42():
    rng = np.random.default_rng(42)
    W0 = rng.standard_normal((3, 4))
    b0 = rng.standard_normal((3,))
    W1 = rng.standard_normal((2, 3))
    b1 = rng.standard_normal((2,))
    return [
        Layer(W=W0.copy(), b=b0.copy(), mask=np.ones((3, 4))),
        Layer(W=W1.copy(), b=b1.copy(), mask=np.ones((2, 3))),
    ]


def test_meta_fields():
    meta = OBSPruneMeta(layer_idx=0, i=2, j=1, score=0.01, prune_time_s=0.05)
    assert meta.layer_idx == 0
    assert meta.score == 0.01


def test_single_weight_zeroed():
    """Exactly one weight is zeroed per prune_obs call."""
    net    = _tiny_net_rng42()
    og_net = clone_network(net)
    omega  = np.random.default_rng(1).integers(0, 256, (30, 4)).astype(np.float32)

    nz_before = int(sum((l.W != 0).sum() for l in net))
    pruned, _ = prune_obs(net, og_net, omega)
    nz_after  = int(sum((l.W != 0).sum() for l in pruned))

    assert nz_after == nz_before - 1


def test_skips_pruned_weights():
    net = _tiny_net_rng42()
    net[0].W[0, 0] = 0.0
    net[0].mask[0, 0] = 0.0

    og_net = clone_network(net)
    omega  = np.random.default_rng(0).integers(0, 256, (30, 4)).astype(np.float32)

    _, meta = prune_obs(net, og_net, omega)
    assert not (meta.layer_idx == 0 and meta.i == 0 and meta.j == 0)


def test_mask_integrity():
    """The pruned weight and all previously-pruned weights must have mask==0."""
    net    = _tiny_net_rng42()
    og_net = clone_network(net)
    omega  = np.random.default_rng(0).integers(0, 256, (30, 4)).astype(np.float32)

    pruned, meta = prune_obs(net, og_net, omega)
    assert pruned[meta.layer_idx].mask[meta.i, meta.j] == 0.0
    assert pruned[meta.layer_idx].W[meta.i, meta.j] == 0.0


def test_previously_pruned_weights_stay_zero():
    """OBS weight update must not un-prune already-zeroed weights."""
    net = _tiny_net_rng42()
    # Pre-prune a weight
    net[0].W[0, 0] = 0.0
    net[0].mask[0, 0] = 0.0

    og_net = clone_network(net)
    omega  = np.random.default_rng(0).integers(0, 256, (30, 4)).astype(np.float32)

    pruned, _ = prune_obs(net, og_net, omega)
    assert pruned[0].W[0, 0] == 0.0,   "OBS update must not restore pre-pruned weight"
    assert pruned[0].mask[0, 0] == 0.0


def test_obs_reduces_distance():
    """After OBS weight update, d(og_net, result) <= d(og_net, post_zero_only)."""
    net    = _tiny_net_rng42()
    og_net = clone_network(net)
    omega  = np.random.default_rng(42).integers(0, 256, (80, 4)).astype(np.float32)

    pruned, meta = prune_obs(net, og_net, omega)

    # Baseline: just zero the weight, no OBS update
    from sparsifier.sparsifier import clone_network as cn
    zero_only = cn(net)
    zero_only[meta.layer_idx].W[meta.i, meta.j] = 0.0
    zero_only[meta.layer_idx].mask[meta.i, meta.j] = 0.0

    d_obs  = float(d(og_net, pruned,    omega))
    d_zero = float(d(og_net, zero_only, omega))

    assert d_obs <= d_zero + 1e-5, (
        f"OBS closed-form update should not increase d vs zero-only: "
        f"obs={d_obs:.4e}, zero={d_zero:.4e}"
    )
