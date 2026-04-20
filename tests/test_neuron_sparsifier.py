# tests/test_neuron_sparsifier.py
import sys, os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest

from mlp.mlp import Layer
from sparsifier.neuron_sparsifier import NeuronPruneMeta, prune_neuron
from sparsifier.sparsifier import clone_network


def _tiny_net():
    """4→3→2 network as Layer list with numpy float64 arrays. Seed 42."""
    rng = np.random.default_rng(42)
    W0 = rng.standard_normal((3, 4))
    b0 = rng.standard_normal((3,))
    W1 = rng.standard_normal((2, 3))
    b1 = rng.standard_normal((2,))
    return [
        Layer(W=W0.copy(), b=b0.copy(), mask=np.ones((3, 4))),
        Layer(W=W1.copy(), b=b1.copy(), mask=np.ones((2, 3))),
    ]


def test_neuron_prune_meta_fields():
    meta = NeuronPruneMeta(
        layer_idx=1, neuron_idx=2, distanza=0.5,
        prune_time_s=0.1, adjust_time_s=0.2,
    )
    assert meta.layer_idx == 1
    assert meta.neuron_idx == 2
    assert meta.distanza == 0.5
    assert meta.prune_time_s == 0.1
    assert meta.adjust_time_s == 0.2


def test_prune_neuron_zeros_correct_entries():
    """Winning neuron's row, bias, and next-layer column must all be zero after pruning."""
    net = _tiny_net()
    og_net = clone_network(net)
    omega = np.random.randint(0, 256, size=(200, 4)).astype(np.float32)

    pruned, meta = prune_neuron(net, og_net, omega, doAdjust=False)

    l = meta.layer_idx
    i = meta.neuron_idx
    assert np.all(pruned[l].W[i, :] == 0.0),    "row of W[l] must be zeroed"
    assert np.all(pruned[l].mask[i, :] == 0.0),  "row of mask[l] must be zeroed"
    assert pruned[l].b[i] == 0.0,                "bias must be zeroed"
    assert np.all(pruned[l+1].W[:, i] == 0.0),   "column of W[l+1] must be zeroed"
    assert np.all(pruned[l+1].mask[:, i] == 0.0), "column of mask[l+1] must be zeroed"


def test_prune_neuron_only_hidden_layers():
    """For a [4,3,2] network the only prunable layer index is 0 (the single hidden layer)."""
    net = _tiny_net()
    og_net = clone_network(net)
    omega = np.random.randint(0, 256, size=(200, 4)).astype(np.float32)

    _, meta = prune_neuron(net, og_net, omega, doAdjust=False)

    # len(net)==2, hidden layers: range(2-1) = [0] only
    assert meta.layer_idx == 0, "output layer neurons must never be candidates"


def test_prune_neuron_skips_already_zeroed():
    """A neuron whose row is already all-zero must not be selected again."""
    net = _tiny_net()
    # Manually zero neuron 0 of layer 0
    net[0].W[0, :] = 0.0
    net[0].mask[0, :] = 0.0
    net[0].b[0] = 0.0
    net[1].W[:, 0] = 0.0
    net[1].mask[:, 0] = 0.0

    og_net = clone_network(net)
    omega = np.random.randint(0, 256, size=(200, 4)).astype(np.float32)

    _, meta = prune_neuron(net, og_net, omega, doAdjust=False)

    assert not (meta.layer_idx == 0 and meta.neuron_idx == 0), \
        "already-pruned neuron must be skipped"
