# tests/test_magnitude_sparsifier.py
import sys, os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from mlp.mlp import Layer
from sparsifier.magnitude_sparsifier import MagnitudePruneMeta, prune_magnitude
from sparsifier.sparsifier import clone_network, d


def _tiny_net_known():
    """2→2→2 net with known magnitudes. Global minimum is |W0[0,1]|=0.1."""
    W0 = np.array([[0.5, 0.1], [0.8, 0.3]], dtype=np.float64)
    b0 = np.array([0.1, 0.2])
    W1 = np.array([[0.4, 0.6], [0.2, 0.7]], dtype=np.float64)
    b1 = np.array([0.1, 0.2])
    return [
        Layer(W=W0.copy(), b=b0.copy(), mask=np.ones((2, 2))),
        Layer(W=W1.copy(), b=b1.copy(), mask=np.ones((2, 2))),
    ]


def _tiny_net_rng42():
    """4→3→2 network, seed 42. Used for adjust tests."""
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
    meta = MagnitudePruneMeta(
        layer_idx=0, i=1, j=2, magnitude=0.05,
        prune_time_s=0.01, adjust_time_s=0.02,
    )
    assert meta.layer_idx == 0
    assert meta.i == 1
    assert meta.j == 2
    assert meta.magnitude == 0.05
    assert meta.prune_time_s == 0.01
    assert meta.adjust_time_s == 0.02


def test_selects_smallest_magnitude():
    """prune_magnitude must zero the entry with the globally smallest |w|."""
    net = _tiny_net_known()
    og_net = clone_network(net)
    omega = np.random.default_rng(0).integers(0, 256, (50, 2)).astype(np.float32)

    pruned, meta = prune_magnitude(net, og_net, omega, doAdjust=False)

    # W0[0,1]=0.1 is the global minimum — must be the candidate
    assert meta.layer_idx == 0
    assert meta.i == 0
    assert meta.j == 1
    assert abs(meta.magnitude - 0.1) < 1e-9


def test_skips_pruned_weights():
    """Weights with mask=0 must never be selected even if W is smaller."""
    net = _tiny_net_known()
    # Manually prune W0[0,1]=0.1 (the global minimum) so it must be skipped
    net[0].W[0, 1] = 0.0
    net[0].mask[0, 1] = 0.0

    og_net = clone_network(net)
    omega = np.random.default_rng(0).integers(0, 256, (50, 2)).astype(np.float32)

    _, meta = prune_magnitude(net, og_net, omega, doAdjust=False)

    assert not (meta.layer_idx == 0 and meta.i == 0 and meta.j == 1), \
        "already-pruned weight must be skipped"


def test_single_weight_zeroed():
    """With doAdjust=False exactly one active weight is removed per call."""
    net = _tiny_net_rng42()
    og_net = clone_network(net)
    omega = np.random.default_rng(1).integers(0, 256, (50, 4)).astype(np.float32)

    nz_before = int(sum((l.W != 0).sum() for l in net))
    pruned, _ = prune_magnitude(net, og_net, omega, doAdjust=False)
    nz_after = int(sum((l.W != 0).sum() for l in pruned))

    assert nz_after == nz_before - 1


def test_meta_consistency():
    """meta.(layer_idx, i, j) must index the actually-pruned weight."""
    net = _tiny_net_known()
    og_net = clone_network(net)
    omega = np.random.default_rng(0).integers(0, 256, (50, 2)).astype(np.float32)

    original_val = float(net[0].W[0, 1])  # the known minimum
    pruned, meta = prune_magnitude(net, og_net, omega, doAdjust=False)

    assert pruned[meta.layer_idx].W[meta.i, meta.j] == 0.0
    assert abs(meta.magnitude - abs(original_val)) < 1e-9


def test_mask_integrity():
    """mask must be 0 at the pruned position after pruning."""
    net = _tiny_net_known()
    og_net = clone_network(net)
    omega = np.random.default_rng(0).integers(0, 256, (50, 2)).astype(np.float32)

    pruned, meta = prune_magnitude(net, og_net, omega, doAdjust=False)

    assert pruned[meta.layer_idx].mask[meta.i, meta.j] == 0.0


def test_adjust_reduces_distance():
    """With doAdjust=True, d(og, result) <= d(og, post_zero_no_adjust)."""
    net = _tiny_net_rng42()
    og_net = clone_network(net)
    omega = np.random.default_rng(42).integers(0, 256, (200, 4)).astype(np.float32)

    pruned_no_adj, _ = prune_magnitude(net, og_net, omega, doAdjust=False)
    d_no_adj = float(d(og_net, pruned_no_adj, omega))

    pruned_adj, _ = prune_magnitude(net, og_net, omega, doAdjust=True)
    d_adj = float(d(og_net, pruned_adj, omega))

    assert d_adj <= d_no_adj + 1e-6, (
        f"adjust should not increase d: no_adj={d_no_adj:.4e}, adj={d_adj:.4e}"
    )
