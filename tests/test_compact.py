# tests/test_compact.py
import sys, os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import json

import numpy as np

import mlp.mlp as mlp
from mlp.mlp import Layer, load_network_params
from backend.compact import compact_network, crossbar_cost, save_compacted, topology_of
from sparsifier.neuron_sparsifier import prune_neuron


def _net(topo, seed=0):
    rng = np.random.default_rng(seed)
    return [Layer(W=rng.standard_normal((o, i)), b=rng.standard_normal(o), mask=np.ones((o, i)))
            for i, o in zip(topo[:-1], topo[1:])]


def test_compact_after_real_neuron_pruning_preserves_function(tmp_path):
    """Prune two neurons with the actual pruner, compact, outputs must match (to fp summation order)."""
    og = _net([6, 5, 4, 3])
    omega = np.random.default_rng(1).standard_normal((50, 6))
    net = og
    for _ in range(2):
        net, _ = prune_neuron(net, og, omega, doAdjust=True)

    comp = compact_network(net)
    assert sum(topology_of(comp)[1:-1]) == sum(topology_of(og)[1:-1]) - 2
    assert topology_of(comp)[0] == 6 and topology_of(comp)[-1] == 3

    X = np.random.default_rng(2).standard_normal((64, 6))
    np.testing.assert_allclose(np.asarray(mlp.batched_predict(comp, X)),
                               np.asarray(mlp.batched_predict(net, X)), rtol=1e-12, atol=1e-12)

    # save_compacted round-trips through the standard params loader.
    save_compacted(net, str(tmp_path), crossbar_size=4)
    loaded = load_network_params(str(tmp_path / "compact"))
    assert topology_of(loaded) == topology_of(comp)
    report = json.load(open(tmp_path / "compact" / "crossbar_cost.json"))
    assert report["topology_after"] == topology_of(comp)


def test_dense_net_is_unchanged():
    net = _net([6, 5, 4, 3])
    assert topology_of(compact_network(net)) == [6, 5, 4, 3]


def test_crossbar_cost_matches_backend_formula():
    # width-200 control from the PIM report: 10 crossbars dense, 4 after pruning to tile boundaries.
    assert crossbar_cost([196, 200, 200, 10])["crossbars"] == 10
    assert crossbar_cost([196, 128, 106, 10]) == {"crossbars": 4, "cores": 3, "mvmuls": 4}
