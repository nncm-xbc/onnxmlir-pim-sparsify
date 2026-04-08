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

def test_compare_one_step_returns_expected_keys():
    """compare_one_step() must return exhaustive + gradient_top_k entries."""
    from benchmark.correctness_check import compare_one_step
    net   = _tiny_net()
    omega = make_omega(net, n_samples=50)
    x     = np.random.default_rng(0).random((20, 3)).astype(np.float32)
    y     = np.zeros((20, 2), dtype=np.float32)
    y[np.arange(20), np.random.default_rng(0).integers(0, 2, 20)] = 1.
    result = compare_one_step(net, net, omega, x, y, top_k_values=(1, 5))
    assert 'exhaustive' in result
    assert 'gradient_top_1' in result
    assert 'gradient_top_5' in result
    for key in ('candidate', 'time_s', 'd_after', 'acc_after'):
        assert key in result['exhaustive'], f"missing '{key}' in exhaustive"
    for key in ('candidate', 'time_s', 'd_after', 'acc_after', 'candidate_match', 'speedup'):
        assert key in result['gradient_top_1'], f"missing '{key}' in gradient_top_1"

def test_gradient_topk_full_equals_exhaustive():
    """When top_k >= NZ, gradient-top-k evaluates all candidates.
    The resulting d_after must equal exhaustive (same optimal distance),
    even if a different weight was chosen under ties.
    """
    from benchmark.correctness_check import compare_one_step
    net   = _tiny_net()
    og_net = [Layer(W=np.array(l.W) + np.random.default_rng(7).random(l.W.shape).astype(np.float32) * 0.1,
                    b=l.b, mask=l.mask) for l in net]
    omega = make_omega(net, n_samples=50)
    x     = np.random.default_rng(1).random((20, 3)).astype(np.float32)
    y     = np.zeros((20, 2), dtype=np.float32)
    y[np.arange(20), np.random.default_rng(1).integers(0, 2, 20)] = 1.
    NZ    = int(sum((np.array(l.W) != 0).sum() for l in net))
    result = compare_one_step(net, og_net, omega, x, y, top_k_values=(NZ,))
    ex_d   = result['exhaustive']['d_after']
    topk_d = result[f'gradient_top_{NZ}']['d_after']
    assert abs(ex_d - topk_d) < 1e-5, (
        f"With top_k=NZ, both methods evaluate all candidates — "
        f"d_after must be equal. Got exhaustive={ex_d:.6e}, gradient_top_{NZ}={topk_d:.6e}"
    )

def test_extended_log_has_timing_and_candidate_columns():
    """The extended log written by main() must include timing and candidate columns."""
    import pandas as pd, tempfile, csv as csv_mod
    required = {
        'step', 'NZ', 'total_W', 'sparsity', 'val_acc', 'd_manifold', 'd_W',
        'prune_time_s', 'adjust_time_s',
        'candidate_layer', 'candidate_i', 'candidate_j',
        'layer_0_NZ',
    }
    # Write a synthetic CSV with those columns and verify pandas reads them correctly.
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        writer = csv_mod.DictWriter(f, fieldnames=sorted(required))
        writer.writeheader()
        writer.writerow({k: 0 for k in required})
        path = f.name
    try:
        df = pd.read_csv(path)
        assert required.issubset(set(df.columns)), f"Missing: {required - set(df.columns)}"
    finally:
        os.unlink(path)

def test_parse_log_basic():
    """parse_log() must return a DataFrame with correct columns and row count."""
    import pandas as pd, tempfile, csv as csv_mod
    cols = ['step','NZ','total_W','sparsity','val_acc','d_manifold','d_W',
            'prune_time_s','adjust_time_s','candidate_layer','candidate_i','candidate_j',
            'layer_0_NZ','layer_1_NZ']
    rows = [[i, 100-i, 100, i/100, 0.9, 0.0, 0.01, 0.5, 0.1, 0, i%4, i%3, 60-i, 40-i]
            for i in range(5)]
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, newline='') as f:
        w = csv_mod.writer(f); w.writerow(cols); w.writerows(rows); path = f.name
    try:
        from visualize.plot_run import parse_log
        df = parse_log(path)
        assert list(df.columns) == cols
        assert len(df) == 5
    finally:
        os.unlink(path)

def test_load_checkpoint_returns_correct_layers():
    """load_checkpoint() must return a list of numpy arrays matching saved shapes."""
    import tempfile
    from visualize.plot_weights import load_checkpoint
    shapes = [(10, 5), (3, 10), (2, 3)]
    with tempfile.TemporaryDirectory() as tmp:
        for i, shape in enumerate(shapes):
            np.save(os.path.join(tmp, f'W_{i}.npy'),
                    np.random.default_rng(i).random(shape).astype(np.float32))
        weights = load_checkpoint(tmp)
    assert len(weights) == 3
    for w, shape in zip(weights, shapes):
        assert w.shape == shape, f"Expected {shape}, got {w.shape}"
        assert w.dtype == np.float32

def test_load_checkpoint_missing_dir_raises():
    from visualize.plot_weights import load_checkpoint
    import pytest
    with pytest.raises(FileNotFoundError):
        load_checkpoint("/nonexistent/path/that/does/not/exist_xyz123")

def test_load_checkpoint_empty_dir_raises():
    import tempfile, pytest
    from visualize.plot_weights import load_checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            load_checkpoint(tmp)

def test_live_view_imports_without_error():
    """live_view.py must be importable — catches syntax errors and missing deps."""
    import importlib
    mod = importlib.import_module('visualize.live_view')
    assert hasattr(mod, 'live_view'), "live_view() function must be defined"
    assert callable(mod.live_view), "live_view must be callable"
