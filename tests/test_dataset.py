# tests/test_dataset.py
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mlp.dataset import load_dataset, DatasetStats

def make_csv_dataset(tmp_path, n_train=20, n_test=10, n_features=4, n_classes=3):
    """Write a minimal CSV dataset to tmp_path."""
    rng = np.random.default_rng(0)
    x_tr = rng.random((n_train, n_features)).astype(np.float32)
    y_tr_labels = rng.integers(0, n_classes, n_train)
    y_tr = (y_tr_labels[:, None] == np.arange(n_classes)).astype(np.float32)
    x_te = rng.random((n_test, n_features)).astype(np.float32)
    y_te_labels = rng.integers(0, n_classes, n_test)
    y_te = (y_te_labels[:, None] == np.arange(n_classes)).astype(np.float32)

    np.savetxt(str(tmp_path / "X_train.csv"), x_tr, delimiter=",")
    np.savetxt(str(tmp_path / "Y_train.csv"), y_tr, delimiter=",")
    np.savetxt(str(tmp_path / "X_test.csv"),  x_te, delimiter=",")
    np.savetxt(str(tmp_path / "Y_test.csv"),  y_te, delimiter=",")
    return str(tmp_path)

def test_csv_shapes(tmp_path):
    folder = make_csv_dataset(tmp_path, n_train=20, n_test=10, n_features=4, n_classes=3)
    (x_tr, y_tr), (x_te, y_te), stats = load_dataset(folder, source="csv")
    assert x_tr.shape == (20, 4)
    assert y_tr.shape == (20, 3)
    assert x_te.shape == (10, 4)
    assert y_te.shape == (10, 3)

def test_csv_stats(tmp_path):
    folder = make_csv_dataset(tmp_path, n_train=20, n_test=10, n_features=4, n_classes=3)
    (x_tr, _), _, stats = load_dataset(folder, source="csv")
    assert stats.n_classes  == 3
    assert stats.n_features == 4
    assert stats.input_min  == pytest.approx(float(x_tr.min()))
    assert stats.input_max  == pytest.approx(float(x_tr.max()))

def test_csv_label_encoded_y(tmp_path):
    """Single-column Y (label-encoded) is converted to one-hot."""
    rng = np.random.default_rng(1)
    x = rng.random((15, 3)).astype(np.float32)
    y = rng.integers(0, 4, 15).astype(np.float32)
    np.savetxt(str(tmp_path / "X_train.csv"), x, delimiter=",")
    np.savetxt(str(tmp_path / "Y_train.csv"), y, delimiter=",")
    np.savetxt(str(tmp_path / "X_test.csv"),  x, delimiter=",")
    np.savetxt(str(tmp_path / "Y_test.csv"),  y, delimiter=",")
    (_, y_tr), _, stats = load_dataset(str(tmp_path), source="csv")
    assert y_tr.ndim == 2
    assert y_tr.shape[1] == 3   # 3 unique classes in seed 1: {0, 2, 3}
    assert stats.n_classes == 3

def test_csv_label_encoded_noncontiguous(tmp_path):
    """n_classes is inferred from unique values, not max+1."""
    rng = np.random.default_rng(2)
    x = rng.random((20, 3)).astype(np.float32)
    # Non-contiguous labels: only classes 0, 2, 5 — max+1 would give 6, correct is 3
    y = np.array([0, 2, 5, 0, 2, 5, 0, 2, 5, 0, 2, 5, 0, 2, 5, 0, 2, 5, 0, 2], dtype=np.float32)
    np.savetxt(str(tmp_path / "X_train.csv"), x, delimiter=",")
    np.savetxt(str(tmp_path / "Y_train.csv"), y, delimiter=",")
    np.savetxt(str(tmp_path / "X_test.csv"),  x[:5], delimiter=",")
    np.savetxt(str(tmp_path / "Y_test.csv"),  y[:5], delimiter=",")
    (_, y_tr), _, stats = load_dataset(str(tmp_path), source="csv")
    assert stats.n_classes == 3
    assert y_tr.shape[1] == 3

def test_unknown_source_raises():
    with pytest.raises(ValueError, match="source"):
        load_dataset(".", source="hdf5")
