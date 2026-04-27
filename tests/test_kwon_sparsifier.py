# tests/test_kwon_sparsifier.py
import sys, os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from mlp.mlp import Layer
from sparsifier.kwon_sparsifier import KwonPruneMeta, prune_kwon
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
    meta = KwonPruneMeta(layer_idx=0, i=1, j=2, score=0.03,
                         prune_time_s=0.01, adjust_time_s=0.02)
    assert meta.layer_idx == 0
    assert meta.score == 0.03


def test_single_weight_zeroed():
    net    = _tiny_net_rng42()
    og_net = clone_network(net)
    omega  = np.random.default_rng(1).integers(0, 256, (50, 4)).astype(np.float32)

    nz_before = int(sum((l.W != 0).sum() for l in net))
    pruned, _ = prune_kwon(net, og_net, omega, doAdjust=False)
    nz_after  = int(sum((l.W != 0).sum() for l in pruned))

    assert nz_after == nz_before - 1


def test_skips_pruned_weights():
    net = _tiny_net_rng42()
    net[0].W[0, 0] = 0.0
    net[0].mask[0, 0] = 0.0

    og_net = clone_network(net)
    omega  = np.random.default_rng(0).integers(0, 256, (50, 4)).astype(np.float32)

    _, meta = prune_kwon(net, og_net, omega, doAdjust=False)
    assert not (meta.layer_idx == 0 and meta.i == 0 and meta.j == 0)


def test_mask_integrity():
    net    = _tiny_net_rng42()
    og_net = clone_network(net)
    omega  = np.random.default_rng(0).integers(0, 256, (50, 4)).astype(np.float32)

    pruned, meta = prune_kwon(net, og_net, omega, doAdjust=False)
    assert pruned[meta.layer_idx].mask[meta.i, meta.j] == 0.0


def test_adjust_reduces_distance():
    net    = _tiny_net_rng42()
    og_net = clone_network(net)
    omega  = np.random.default_rng(42).integers(0, 256, (200, 4)).astype(np.float32)

    pruned_no_adj, _ = prune_kwon(net, og_net, omega, doAdjust=False)
    d_no_adj = float(d(og_net, pruned_no_adj, omega))

    pruned_adj, _ = prune_kwon(net, og_net, omega, doAdjust=True)
    d_adj = float(d(og_net, pruned_adj, omega))

    assert d_adj <= d_no_adj + 1e-6, (
        f"adjust should not increase d: no_adj={d_no_adj:.4e}, adj={d_adj:.4e}"
    )
