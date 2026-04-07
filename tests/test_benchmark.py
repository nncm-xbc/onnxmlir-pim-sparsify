import sys, os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from jax import random
from mlp.mlp import init_network_params, Layer
from sparsifier.sparsifier import prune, make_omega, PruneMeta

def _tiny_net():
    """3→4→2 network for fast tests."""
    return init_network_params([3, 4, 2], random.PRNGKey(0))

def test_prune_returns_tuple_with_meta():
    net   = _tiny_net()
    omega = make_omega(net, n_samples=50)
    result = prune(net, net, omega, doAdjust=False)
    assert isinstance(result, tuple), "prune() must return (net, meta)"
    assert len(result) == 2
    _, meta = result
    assert isinstance(meta, PruneMeta)
    assert meta.prune_time_s  >= 0.
    assert meta.adjust_time_s >= 0.
    assert 0 <= meta.layer_idx < len(result[0])
    assert meta.i >= 0
    assert meta.j >= 0

def test_prune_meta_candidate_is_valid():
    net   = _tiny_net()
    # use a distinct og_net so the search loop meaningfully ranks candidates
    og_net = [Layer(W=np.array(l.W) + np.random.default_rng(42).normal(size=l.W.shape).astype(np.float32) * 0.01,
                    b=l.b, mask=l.mask) for l in net]
    omega = make_omega(net, n_samples=50)
    pruned_net, meta = prune(net, og_net, omega, doAdjust=False)
    assert 0 <= meta.layer_idx < len(pruned_net)
    layer = pruned_net[meta.layer_idx]
    assert 0 <= meta.i < layer.W.shape[0]
    assert 0 <= meta.j < layer.W.shape[1]
    assert layer.W[meta.i, meta.j] == 0., "chosen weight must be zeroed in result"
