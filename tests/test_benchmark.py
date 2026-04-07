import sys, os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from jax import random
from mlp.mlp import init_network_params
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
    assert hasattr(meta, 'layer_idx')
    assert hasattr(meta, 'i')
    assert hasattr(meta, 'j')
    assert hasattr(meta, 'distanza')
    assert hasattr(meta, 'prune_time_s')
    assert hasattr(meta, 'adjust_time_s')
    assert meta.prune_time_s >= 0.
    assert meta.adjust_time_s >= 0.

def test_prune_meta_candidate_is_valid():
    net   = _tiny_net()
    omega = make_omega(net, n_samples=50)
    pruned_net, meta = prune(net, net, omega, doAdjust=False)
    layer = pruned_net[meta.layer_idx]
    assert layer.W[meta.i, meta.j] == 0., "chosen weight must be zeroed in result"
