# tests/test_prune_ext.py
import sys, os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pytest
import prune_ext  # ModuleNotFoundError until built — intentional at this stage

from mlp.mlp import Layer, batched_predict


def _tiny_net_arrays():
    """4 → 3 → 2 network as (W, b, mask, activation_name) float32 tuples. Seed 42."""
    rng = np.random.default_rng(42)
    W0 = rng.standard_normal((3, 4)).astype(np.float32)
    b0 = rng.standard_normal((3,)).astype(np.float32)
    W1 = rng.standard_normal((2, 3)).astype(np.float32)
    b1 = rng.standard_normal((2,)).astype(np.float32)
    m0 = np.ones((3, 4), dtype=np.float32)
    m1 = np.ones((2, 3), dtype=np.float32)
    return [(W0, b0, m0, "relu"), (W1, b1, m1, "linear")]


def _as_layer_list(layers):
    return [Layer(W=W.copy(), b=b.copy(), mask=m.copy()) for W, b, m, _ in layers]


def _serial_find_best(layers, og_outputs, omega):
    """Brute-force serial argmin — reference for correctness test."""
    net = _as_layer_list(layers)
    best_dist = float("inf")
    best = (0, 0, 0)
    for li, (W, b, m, _) in enumerate(layers):
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                if W[i, j] == 0.0:
                    continue
                saved = net[li].W[i, j]
                net[li].W[i, j] = 0.0
                out = np.array(
                    batched_predict(net, omega.astype(np.float64)), dtype=np.float32
                )
                net[li].W[i, j] = saved
                dist = float(np.sum((out - og_outputs) ** 2))
                if dist < best_dist:
                    best_dist = dist
                    best = (li, i, j)
    return best[0], best[1], best[2], best_dist


def test_find_best_candidate_matches_serial():
    """C++ argmin must agree with brute-force serial search on a tiny network."""
    layers = _tiny_net_arrays()
    omega  = np.random.default_rng(0).integers(0, 256, (80, 4)).astype(np.float32)
    net    = _as_layer_list(layers)
    og_outputs = np.array(
        batched_predict(net, omega.astype(np.float64)), dtype=np.float32
    )

    ref_layer, ref_i, ref_j, ref_dist = _serial_find_best(layers, og_outputs, omega)
    cpp_layer, cpp_i, cpp_j, cpp_dist = prune_ext.find_best_candidate(
        layers, og_outputs, omega
    )

    assert (cpp_layer, cpp_i, cpp_j) == (ref_layer, ref_i, ref_j), (
        f"argmin mismatch: C++ ({cpp_layer},{cpp_i},{cpp_j}) "
        f"vs reference ({ref_layer},{ref_i},{ref_j})"
    )
    # Standard atol + rtol form: handles near-zero ref_dist without amplifying
    # float32 vs float64 rounding residuals in the denominator.
    assert abs(cpp_dist - ref_dist) <= 1e-5 + 5e-3 * abs(ref_dist), (
        f"distance mismatch: C++ {cpp_dist:.6e} vs reference {ref_dist:.6e}"
    )


def test_find_best_candidate_skips_zero_weights():
    """Weights already zeroed must never be selected as candidates."""
    layers = _tiny_net_arrays()
    W0, b0, m0, act0 = layers[0]
    W0 = W0.copy(); m0 = m0.copy()
    W0[0, 0] = 0.0; m0[0, 0] = 0.0  # manually prune weight (0,0) of layer 0
    layers[0] = (W0, b0, m0, act0)

    omega = np.random.default_rng(1).integers(0, 256, (40, 4)).astype(np.float32)
    net = _as_layer_list(layers)
    og_outputs = np.array(
        batched_predict(net, omega.astype(np.float64)), dtype=np.float32
    )

    cpp_layer, cpp_i, cpp_j, _ = prune_ext.find_best_candidate(
        layers, og_outputs, omega
    )
    assert not (cpp_layer == 0 and cpp_i == 0 and cpp_j == 0), (
        "find_best_candidate selected an already-zero weight"
    )


def test_unknown_activation_raises():
    """An unregistered activation name must raise an exception mentioning the name."""
    layers = _tiny_net_arrays()
    W0, b0, m0, _ = layers[0]
    layers[0] = (W0, b0, m0, "swish")  # not in registry

    omega = np.random.default_rng(2).integers(0, 256, (10, 4)).astype(np.float32)
    og_outputs = np.zeros((10, 2), dtype=np.float32)

    with pytest.raises(Exception, match="swish"):
        prune_ext.find_best_candidate(layers, og_outputs, omega)
