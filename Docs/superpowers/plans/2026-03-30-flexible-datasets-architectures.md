# Flexible Datasets and Architectures — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove all MNIST- and MLP-specific hardcoded constants so the pipeline infers architecture and dataset properties from files.

**Architecture:** Two new modules (`mlp/topology.py`, `mlp/dataset.py`) centralise all topology CSV parsing and dataset loading. `mlp/mlp.py` gains a configurable activation system. All scripts import from these modules instead of duplicating logic.

**Tech Stack:** JAX, NumPy, PyTorch/torchvision (optional), pytest

---

## Spec Correction

The design spec states activations list is `len(sizes)` long. The correct cardinality is **`len(sizes)-1`** — one activation per weight matrix (layer), not per node. A 4-entry topology `[196, 10, 10, 10]` has 3 weight matrices and 3 activations `[relu, relu, linear]`. The CSV:

```
196,10,10,10
relu,relu,linear
```

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `mlp/topology.py` | **Create** | `load_topology`, `save_topology`, activation validation |
| `mlp/dataset.py` | **Create** | `DatasetStats`, `load_dataset` (CSV + PyTorch) |
| `mlp/mlp.py` | **Modify** | `Layer` class (custom, JAX-registered), `ACTIVATIONS` dict, updated `predict`, `loss`, `init_network_params`, `load_network_params`, `update` |
| `mlp/__init__.py` | **Modify** | Add `ACTIVATIONS` to re-exports |
| `sparsifier/sparsifier.py` | **Modify** | `make_omega` takes range args, `clone_network`/`adjust` preserve activation, `main` uses `load_dataset` |
| `scripts/dataeng.py` | **Modify** | Fix `14*14` bug, derive `w` from topology, guard non-square |
| `scripts/train.py` | **Modify** | Use `load_dataset`, `load_topology`, `save_topology`; remove `**2` quirk |
| `backend/compiler.py` | **Modify** | `Program.__init__` uses `load_topology`; fix index access `p[0]`→`p.W` |
| `data/network_topology.csv` | **Modify** | `14, 10, 10, 10` → `196,10,10,10\nrelu,relu,linear` |
| `tests/test_topology.py` | **Create** | Unit tests for `topology.py` |
| `tests/test_dataset.py` | **Create** | Unit tests for `dataset.py` |
| `tests/test_mlp.py` | **Create** | Unit tests for `mlp.py` changes |

---

## Task 1: `mlp/topology.py`

**Files:**
- Create: `mlp/topology.py`
- Create: `tests/test_topology.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_topology.py`:

```python
# tests/test_topology.py
import os, tempfile, pytest
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mlp.topology import load_topology, save_topology

def write_csv(tmp_path, content):
    p = tmp_path / "topology.csv"
    p.write_text(content)
    return str(p)

def test_load_single_row_defaults(tmp_path):
    """Single-row CSV: activations default to relu...relu, linear."""
    path = write_csv(tmp_path, "196,64,10\n")
    sizes, acts = load_topology(path)
    assert sizes == [196, 64, 10]
    assert acts == ["relu", "relu", "linear"]

def test_load_two_row(tmp_path):
    """Two-row CSV: activations are read from second row."""
    path = write_csv(tmp_path, "196,64,10\ntanh,tanh,linear\n")
    sizes, acts = load_topology(path)
    assert sizes == [196, 64, 10]
    assert acts == ["tanh", "tanh", "linear"]

def test_load_mismatched_lengths_raises(tmp_path):
    """Activations row length must equal len(sizes)-1."""
    path = write_csv(tmp_path, "196,64,10\nrelu\n")
    with pytest.raises(ValueError, match="length"):
        load_topology(path)

def test_load_unknown_activation_raises(tmp_path):
    """Unknown activation names raise ValueError with helpful message."""
    path = write_csv(tmp_path, "196,64,10\nrelu,swish\n")
    with pytest.raises(ValueError, match="swish"):
        load_topology(path)

def test_save_load_roundtrip(tmp_path):
    """save_topology → load_topology is a round-trip."""
    path = str(tmp_path / "topo.csv")
    save_topology(path, [196, 64, 10], ["tanh", "linear"])
    sizes, acts = load_topology(path)
    assert sizes == [196, 64, 10]
    assert acts == ["tanh", "linear"]

def test_load_four_layer(tmp_path):
    """Four-layer topology with three activations."""
    path = write_csv(tmp_path, "196,10,10,10\nrelu,relu,linear\n")
    sizes, acts = load_topology(path)
    assert sizes == [196, 10, 10, 10]
    assert acts == ["relu", "relu", "linear"]
```

- [ ] **Step 2: Run tests — expect import error**

```bash
cd /Users/simonhgt/Documents/repos/onnxmlir-pim-sparsify
python -m pytest tests/test_topology.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'mlp.topology'`

- [ ] **Step 3: Create `mlp/topology.py`**

```python
# mlp/topology.py

_VALID_ACTIVATIONS = {"relu", "tanh", "sigmoid", "linear"}


def load_topology(csv_path: str) -> tuple:
    """Load a topology CSV. Returns (sizes, activations).

    CSV format:
        Row 1: comma-separated actual layer sizes, e.g. 196,64,10
        Row 2: optional activations, one per weight matrix (len(sizes)-1 entries)
                e.g. relu,linear   for a 3-entry sizes row

    Defaults when row 2 is absent: relu for all hidden layers, linear for last.
    """
    rows = []
    with open(csv_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append([tok.strip() for tok in line.split(",")])

    sizes = [int(x) for x in rows[0]]
    n_layers = len(sizes) - 1

    if len(rows) >= 2:
        activations = rows[1]
        if len(activations) != n_layers:
            raise ValueError(
                f"Activations row has {len(activations)} entries but topology has "
                f"{n_layers} weight matrices (len(sizes)-1). They must match."
            )
        for act in activations:
            if act not in _VALID_ACTIVATIONS:
                raise ValueError(
                    f"Unknown activation '{act}'. "
                    f"Supported: {sorted(_VALID_ACTIVATIONS)}"
                )
    else:
        activations = ["relu"] * (n_layers - 1) + ["linear"]

    return sizes, activations


def save_topology(csv_path: str, sizes: list, activations: list) -> None:
    """Write a two-row topology CSV."""
    with open(csv_path, "w") as f:
        f.write(",".join(str(s) for s in sizes) + "\n")
        f.write(",".join(activations) + "\n")
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
python -m pytest tests/test_topology.py -v
```

Expected: 6 tests pass.

- [ ] **Step 5: Commit**

```bash
git add mlp/topology.py tests/test_topology.py
git commit -m "feat: add mlp/topology.py — shared topology CSV parser with activation support"
```

---

## Task 2: `mlp/dataset.py` — CSV mode

**Files:**
- Create: `mlp/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_dataset.py`:

```python
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
    assert y_tr.shape[1] == 4   # 4 classes
    assert stats.n_classes == 4

def test_unknown_source_raises():
    with pytest.raises(ValueError, match="source"):
        load_dataset(".", source="hdf5")
```

- [ ] **Step 2: Run tests — expect import error**

```bash
python -m pytest tests/test_dataset.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'mlp.dataset'`

- [ ] **Step 3: Create `mlp/dataset.py` (CSV mode only)**

```python
# mlp/dataset.py

from typing import NamedTuple
import numpy as np


class DatasetStats(NamedTuple):
    input_min:  float
    input_max:  float
    n_classes:  int
    n_features: int


def load_dataset(path: str, source: str = "csv"):
    """Load a dataset. Returns ((x_train, y_train), (x_test, y_test), DatasetStats).

    source="csv"     — path is a folder with X_train.csv, Y_train.csv,
                       X_test.csv, Y_test.csv
    source="pytorch" — path is a torchvision dataset name, e.g. "MNIST"
    """
    if source == "csv":
        return _load_csv(path)
    elif source == "pytorch":
        return _load_pytorch(path)
    else:
        raise ValueError(f"Unknown source '{source}'. Use 'csv' or 'pytorch'.")


def _one_hot(labels: np.ndarray, n_classes: int) -> np.ndarray:
    return (labels[:, None] == np.arange(n_classes)).astype(np.float32)


def _load_csv(folder: str):
    import os

    x_train = np.genfromtxt(os.path.join(folder, "X_train.csv"), delimiter=",")
    y_train = np.genfromtxt(os.path.join(folder, "Y_train.csv"), delimiter=",")
    x_test  = np.genfromtxt(os.path.join(folder, "X_test.csv"),  delimiter=",")
    y_test  = np.genfromtxt(os.path.join(folder, "Y_test.csv"),  delimiter=",")

    x_train = x_train.astype(np.float32)
    x_test  = x_test.astype(np.float32)

    # Infer n_classes; convert label-encoded Y to one-hot if needed
    if y_train.ndim == 1 or (y_train.ndim == 2 and y_train.shape[1] == 1):
        y_train = y_train.ravel()
        y_test  = y_test.ravel()
        n_classes = len(np.unique(y_train))
        y_train = _one_hot(y_train.astype(int), n_classes)
        y_test  = _one_hot(y_test.astype(int),  n_classes)
    else:
        n_classes = y_train.shape[1]
        y_train = y_train.astype(np.float32)
        y_test  = y_test.astype(np.float32)

    stats = DatasetStats(
        input_min  = float(x_train.min()),
        input_max  = float(x_train.max()),
        n_classes  = n_classes,
        n_features = x_train.shape[1],
    )
    return (x_train, y_train), (x_test, y_test), stats


def _load_pytorch(dataset_name: str):
    import torch
    import torchvision
    import torchvision.transforms as transforms

    transform = transforms.Compose([transforms.ToTensor()])

    dataset_cls = getattr(torchvision.datasets, dataset_name)
    train_ds = dataset_cls(root="/tmp/torchvision_data", train=True,  download=True, transform=transform)
    test_ds  = dataset_cls(root="/tmp/torchvision_data", train=False, download=True, transform=transform)

    def to_arrays(ds):
        loader = torch.utils.data.DataLoader(ds, batch_size=len(ds), shuffle=False)
        X, y = next(iter(loader))
        X = X.numpy().reshape(len(ds), -1).astype(np.float32)
        return X, y.numpy()

    x_train, y_train_labels = to_arrays(train_ds)
    x_test,  y_test_labels  = to_arrays(test_ds)

    n_classes = len(np.unique(y_train_labels))
    y_train = _one_hot(y_train_labels, n_classes)
    y_test  = _one_hot(y_test_labels,  n_classes)

    stats = DatasetStats(
        input_min  = 0.0,
        input_max  = 1.0,
        n_classes  = n_classes,
        n_features = x_train.shape[1],
    )
    return (x_train, y_train), (x_test, y_test), stats
```

- [ ] **Step 4: Run tests — expect all pass**

```bash
python -m pytest tests/test_dataset.py -v
```

Expected: 4 tests pass. (PyTorch mode is implemented in `_load_pytorch` but has no unit test here — it requires `torchvision` and a network download. Test manually with `load_dataset("MNIST", source="pytorch")` if torchvision is available.)

- [ ] **Step 5: Commit**

```bash
git add mlp/dataset.py tests/test_dataset.py
git commit -m "feat: add mlp/dataset.py — CSV and PyTorch dataset loader with DatasetStats"
```

---

## Task 3: `mlp/mlp.py` — Layer class, ACTIVATIONS, predict, loss

This is the most significant change. The `Layer` NamedTuple is replaced by a custom class with JAX pytree registration so that the `activation` string field is treated as auxiliary (non-differentiable) data — not a JAX array leaf.

**Files:**
- Modify: `mlp/mlp.py`
- Create: `tests/test_mlp.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_mlp.py`:

```python
# tests/test_mlp.py
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import jax
import jax.numpy as jnp
from jax import random, grad

from mlp.mlp import (
    Layer, ACTIVATIONS, init_network_params, load_network_params,
    predict, batched_predict, loss, accuracy, update,
)


# ── Layer ──────────────────────────────────────────────────────────────────────

def test_layer_default_activation():
    l = Layer(W=np.zeros((3, 2)), b=np.zeros(3), mask=np.ones((3, 2)))
    assert l.activation == "relu"

def test_layer_custom_activation():
    l = Layer(W=np.zeros((3, 2)), b=np.zeros(3), mask=np.ones((3, 2)), activation="tanh")
    assert l.activation == "tanh"

def test_layer_preserves_through_grad():
    """JAX grad must not crash when network contains activation strings."""
    net = init_network_params([4, 3, 2], ["relu", "linear"], random.PRNGKey(0))
    x = np.ones((5, 4), dtype=np.float32)
    y = np.array([[1, 0], [0, 1], [1, 0], [0, 1], [1, 0]], dtype=np.float32)
    # This would crash if activation strings are exposed as JAX array leaves
    grads = grad(loss)(net, x, y)
    assert grads[0].W.shape == (3, 4)


# ── ACTIVATIONS registry ───────────────────────────────────────────────────────

def test_activations_has_required_keys():
    for key in ["relu", "tanh", "sigmoid", "linear"]:
        assert key in ACTIVATIONS

def test_relu_activation():
    f = ACTIVATIONS["relu"]
    assert float(f(jnp.array(-1.0))) == 0.0
    assert float(f(jnp.array(2.0)))  == 2.0

def test_linear_activation():
    f = ACTIVATIONS["linear"]
    assert float(f(jnp.array(-3.5))) == pytest.approx(-3.5)


# ── predict ────────────────────────────────────────────────────────────────────

def test_predict_linear_final_returns_raw_logits():
    """With linear final layer, predict returns raw logits (no softmax applied)."""
    # Construct Layer directly with numpy arrays so values are predictable
    W    = np.array([[1., 0.], [0., 1.], [0., 0.]], dtype=np.float32)
    b    = np.zeros(3, dtype=np.float32)
    mask = np.ones((3, 2), dtype=np.float32)
    net  = [Layer(W=W, b=b, mask=mask, activation="linear")]
    x    = np.array([2.0, 3.0], dtype=np.float32)
    out  = predict(net, x)
    # W @ x + b = [2., 3., 0.] — no softmax
    np.testing.assert_allclose(np.array(out), np.array([2., 3., 0.]), atol=1e-5)

def test_predict_tanh_hidden():
    """tanh activation produces values in (-1, 1) after hidden layer."""
    net = init_network_params([4, 8, 2], ["tanh", "linear"], random.PRNGKey(1))
    x = np.random.randn(4).astype(np.float32) * 10   # large inputs
    out = predict(net, x)
    assert out.shape == (2,)


# ── init_network_params ────────────────────────────────────────────────────────

def test_init_stores_activations():
    sizes = [196, 64, 10]
    acts  = ["tanh", "linear"]
    net   = init_network_params(sizes, acts, random.PRNGKey(0))
    assert len(net) == 2
    assert net[0].activation == "tanh"
    assert net[1].activation == "linear"

def test_init_weight_shapes():
    net = init_network_params([196, 64, 10], ["relu", "linear"], random.PRNGKey(0))
    assert net[0].W.shape == (64, 196)
    assert net[1].W.shape == (10, 64)


# ── update ─────────────────────────────────────────────────────────────────────

def test_update_preserves_activations():
    net   = init_network_params([4, 3, 2], ["relu", "linear"], random.PRNGKey(0))
    x     = np.ones((5, 4), dtype=np.float32)
    y     = np.array([[1,0],[0,1],[1,0],[0,1],[1,0]], dtype=np.float32)
    net2  = update(net, x, y)
    assert net2[0].activation == "relu"
    assert net2[1].activation == "linear"
```

- [ ] **Step 2: Run tests — expect failures**

```bash
python -m pytest tests/test_mlp.py -v 2>&1 | head -30
```

Expected: various failures (missing `ACTIVATIONS`, wrong `init_network_params` signature, etc.)

- [ ] **Step 3: Rewrite `mlp/mlp.py`**

Replace the entire file:

```python
# mlp/mlp.py — JAX MLP primitives: parameter init, forward pass, training utilities

import jax
import jax.numpy as jnp
from jax import grad, jit, lax, vmap
from jax import random
import numpy as np
import os

# ── Activation registry ────────────────────────────────────────────────────────
# To add a new activation: add one entry here. No other changes needed.

ACTIVATIONS = {
    "relu":    lambda x: jnp.maximum(0, x),
    "tanh":    jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
    "linear":  lambda x: x,
}


# ── Layer ──────────────────────────────────────────────────────────────────────
# Custom class (not NamedTuple) so JAX treats `activation` as static auxiliary
# data rather than a differentiable array leaf.

class Layer:
    """One dense layer: weight matrix W, bias b, sparsity mask, activation name."""
    def __init__(self, W, b, mask, activation="relu"):
        self.W          = W
        self.b          = b
        self.mask       = mask
        self.activation = activation

    def __repr__(self):
        return (f"Layer(W={self.W.shape}, b={self.b.shape}, "
                f"activation={self.activation!r})")


# Register as JAX pytree: W, b, mask are differentiable leaves;
# activation is static auxiliary data.
jax.tree_util.register_pytree_node(
    Layer,
    flatten_func   = lambda l: ((l.W, l.b, l.mask), l.activation),
    unflatten_func = lambda act, children: Layer(*children, activation=act),
)


# ── Parameter initialisation ───────────────────────────────────────────────────

def random_layer_params(m, n, key, scale=1e-2):
    """Random (W, b) for a layer mapping m inputs to n outputs."""
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_params(sizes, activations, key):
    """Randomly initialise a network.

    sizes:       list of layer widths, e.g. [196, 64, 10]
    activations: list of activation names, length len(sizes)-1,
                 e.g. ["relu", "linear"]
    """
    keys   = random.split(key, len(sizes))
    layers = []
    for (m, n), k, act in zip(zip(sizes[:-1], sizes[1:]), keys, activations):
        w, b = random_layer_params(m, n, k)
        layers.append(Layer(W=w, b=b, mask=np.ones((n, m)), activation=act))
    return layers


def load_network_params(folder, activations=None):
    """Load weight files W_i.npy / b_i.npy from a folder.

    activations: optional list of activation names (len = number of files found).
                 Defaults to relu for hidden layers, linear for the last.
    """
    count = sum(1 for f in os.listdir(folder) if f.startswith("W_") and f.endswith(".npy"))
    print(f"Found {count} layers in {folder}")

    if activations is None:
        activations = ["relu"] * (count - 1) + ["linear"]

    layers = []
    for i in range(count):
        W = np.load(os.path.join(folder, f"W_{i}.npy"))
        b = np.load(os.path.join(folder, f"b_{i}.npy"))
        layers.append(Layer(W=W, b=b, mask=np.ones_like(W), activation=activations[i]))
    return layers


# ── Forward pass ───────────────────────────────────────────────────────────────

def predict(network, image):
    """Forward pass. Returns raw layer outputs (no implicit softmax)."""
    activations = image
    for layer in network:
        outputs    = jnp.dot(lax.stop_gradient(layer.mask) * layer.W, activations) + layer.b
        activations = ACTIVATIONS[layer.activation](outputs)
    return activations


batched_predict = vmap(predict, in_axes=(None, 0))


# ── Training utilities ─────────────────────────────────────────────────────────

def one_hot(x, k, dtype=jnp.float32):
    """One-hot encode x into k classes."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


def accuracy(network, images, targets):
    target_class    = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(network, images), axis=1)
    return jnp.mean(predicted_class == target_class)


def loss(network, images, targets):
    """Softmax cross-entropy loss. Assumes linear final layer (raw logits)."""
    log_probs = jax.nn.log_softmax(batched_predict(network, images))
    return -jnp.mean(log_probs * targets)


@jit
def update(network, x, y, step_size=0.01):
    grads = grad(loss)(network, x, y)
    return [Layer(W=l.W - step_size * g.W,
                  b=l.b - step_size * g.b,
                  mask=l.mask,
                  activation=l.activation)
            for l, g in zip(network, grads)]
```

- [ ] **Step 4: Update `mlp/__init__.py`** — add `ACTIVATIONS`, remove `relu` (now only in `ACTIVATIONS` dict)

```python
from mlp.mlp import (
    Layer,
    ACTIVATIONS,
    random_layer_params,
    init_network_params,
    load_network_params,
    predict,
    batched_predict,
    one_hot,
    accuracy,
    loss,
    update,
)
```

- [ ] **Step 5: Run tests — expect all pass**

```bash
python -m pytest tests/test_mlp.py -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add mlp/mlp.py mlp/__init__.py tests/test_mlp.py
git commit -m "feat: flexible activations in Layer — custom JAX pytree class + ACTIVATIONS registry"
```

---

## Task 4: `sparsifier/sparsifier.py`

The `activation` field must be preserved wherever new `Layer` objects are constructed. `make_omega` must accept input range parameters instead of hardcoding `[0, 256)`.

**Files:**
- Modify: `sparsifier/sparsifier.py`

- [ ] **Step 1: Rewrite `sparsifier/sparsifier.py`**

```python
# sparsifier/sparsifier.py

from mlp.mlp import *
from mlp.topology import load_topology
from mlp.dataset import load_dataset
import jax
import sys
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


def make_omega(network, input_min, input_max, n_samples=10_000):
    """Sample n_samples random inputs from [input_min, input_max].

    input_dim is inferred from the first layer weight matrix shape (out, in).
    input_min / input_max should come from DatasetStats to avoid MNIST assumptions.
    """
    input_dim = network[0].W.shape[1]
    lo, hi    = int(input_min), int(input_max) + 1
    return np.random.randint(lo, hi, size=(n_samples, input_dim)).astype(np.float32)


def d(net_1, net_2, omega):
    """Semantic distance between two networks measured over omega."""
    return jnp.sum(
        (batched_predict(net_1, omega)
       - batched_predict(net_2, omega)) ** 2
    )
d = jax.jit(d)
d_grad = jax.jit(jax.grad(d))


def clone_network(network):
    return [Layer(W=np.array(l.W).copy(),
                  b=np.array(l.b).copy(),
                  mask=np.array(l.mask).copy(),
                  activation=l.activation)
            for l in network]


def _zero_weight(layer, i, j):
    new_W    = np.array(layer.W).copy();    new_W[i, j]    = 0.
    new_mask = np.array(layer.mask).copy(); new_mask[i, j] = 0.
    return Layer(W=new_W, b=layer.b, mask=new_mask, activation=layer.activation)


def adjust(net, cmp_net, omega):
    net  = clone_network(net)
    alfa = 1e-11
    while alfa > 1e-14:
        gradiente = d_grad(net, cmp_net, omega)
        new_net   = [Layer(W=l.W - alfa * g.W,
                           b=l.b - alfa * g.b,
                           mask=l.mask,
                           activation=l.activation)
                     for l, g in zip(net, gradiente)]
        if d(new_net, cmp_net, omega) < d(net, cmp_net, omega):
            net   = new_net
            alfa *= 1.001
        else:
            alfa *= 0.5
    return net


def prune(net, og_net, omega, doAdjust=True):
    minimo     = 1e16
    minimo_idx = 0
    minimo_i   = 0
    minimo_j   = 0

    probe_net   = clone_network(net)
    search_done = False
    for idx, layer in enumerate(net):
        for i in range(layer.W.shape[0]):
            for j in range(layer.W.shape[1]):
                if layer.W[i, j] == 0.:
                    continue
                saved_W    = probe_net[idx].W[i, j]
                saved_mask = probe_net[idx].mask[i, j]
                probe_net[idx].W[i, j]    = 0.
                probe_net[idx].mask[i, j] = 0.
                distanza = d(og_net, probe_net, omega)
                probe_net[idx].W[i, j]    = saved_W
                probe_net[idx].mask[i, j] = saved_mask

                if distanza < minimo:
                    minimo     = distanza
                    minimo_idx = idx
                    minimo_i   = i
                    minimo_j   = j
                    if minimo == 0:
                        search_done = True
                        break
            if search_done:
                break
        if search_done:
            break

    probe_net[minimo_idx].W[minimo_i, minimo_j]    = 0.
    probe_net[minimo_idx].mask[minimo_i, minimo_j] = 0.
    if doAdjust and minimo > 0:
        probe_net = adjust(probe_net, og_net, omega)
    return probe_net


def main():
    args          = sys.argv
    params_folder = os.path.abspath(args[1])
    dataset_path  = os.path.abspath(args[2])
    dataset_source = args[3] if len(args) > 3 else "csv"
    output_folder  = params_folder + "/sparsified"

    print("Load the validation set")
    (_, _), (x_test, y_test), stats = load_dataset(dataset_path, source=dataset_source)

    # Load activations from topology CSV if present alongside the weights
    topo_csv = os.path.join(params_folder, "network_topology.csv")
    if os.path.exists(topo_csv):
        _, activations = load_topology(topo_csv)
    else:
        activations = None

    print("Load the parameters from the folder")
    og_net = load_network_params(params_folder, activations=activations)
    print()
    print("Accuracy on validation set:", accuracy(og_net, x_test, y_test))

    omega = make_omega(og_net, stats.input_min, stats.input_max)
    perturbed_net = [Layer(W=l.W + np.random.normal(size=l.W.shape) * .00001,
                           b=l.b, mask=l.mask, activation=l.activation)
                     for l in og_net]
    print("Distance sanity check:", d(og_net, perturbed_net, omega))

    print("Starting sparsification loop")
    net = clone_network(og_net)
    for i in range(500):
        NZ = np.sum([(l.W != 0).sum() for l in net])
        print("validation accuracy = %.3f" % accuracy(net, x_test, y_test),
              " | non zero elements = %d" % NZ)
        omega = make_omega(net, stats.input_min, stats.input_max)
        net = prune(net, og_net, omega, True)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, l in enumerate(net):
        np.save(output_folder + "/W_%i.npy" % i, l.W)
        np.save(output_folder + "/b_%i.npy" % i, l.b)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the sparsifier imports cleanly**

```bash
python -c "from sparsifier.sparsifier import make_omega, clone_network, adjust, prune; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Run all tests to check nothing broke**

```bash
python -m pytest tests/ -v
```

Expected: all existing tests pass.

- [ ] **Step 4: Commit**

```bash
git add sparsifier/sparsifier.py
git commit -m "fix: sparsifier — preserve activation in Layer construction, parametrize make_omega range"
```

---

## Task 5: `scripts/dataeng.py`

Fix the `14*14` hardcode and derive `w` from the topology CSV. Add a guard for non-image (non-square) input dimensions.

**Files:**
- Modify: `scripts/dataeng.py`

- [ ] **Step 1: Rewrite `scripts/dataeng.py`**

```python
# dataeng.py
# Produces a smaller dataset from MNIST by resizing images to the resolution
# specified in the first entry of the network topology CSV.
# For non-image datasets, use load_dataset directly from mlp/dataset.py.

import numpy as np
import os
import sys
from math import sqrt, isclose
from PIL import Image

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from mlp.topology import load_topology

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
_data_dir     = os.path.realpath(os.path.join(__location__, '..', 'data'))

args         = sys.argv
topology_csv = _data_dir + "/" + args[1]

sizes, _ = load_topology(topology_csv)
input_dim = sizes[0]

w_float = sqrt(input_dim)
w = int(round(w_float))
if not isclose(w * w, input_dim):
    sys.exit(
        f"Error: input_dim={input_dim} is not a perfect square. "
        f"dataeng.py only supports image datasets with square spatial dimensions. "
        f"For non-image datasets, prepare X_train.csv / Y_train.csv directly."
    )


def load_data():
    data_train        = np.genfromtxt(_data_dir + '/mnist_train.csv', delimiter=',')
    x_train, y_train  = data_train[:,1:], data_train[:,0]
    data_test         = np.genfromtxt(_data_dir + '/mnist_test.csv',  delimiter=',')
    x_test,  y_test   = data_test[:,1:], data_test[:,0]
    x_train           = x_train.reshape(-1, 28, 28)
    x_test            = x_test.reshape(-1, 28, 28)
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()

onehot = lambda y: np.concatenate([
    (y == c)[:, None] * 1.
    for c in np.arange(int(y_train.max()) + 1)
], axis=1)

x_train_small = np.array([np.array(Image.fromarray(x).resize((w, w))) for x in x_train]).reshape(-1, w * w)
x_test_small  = np.array([np.array(Image.fromarray(x).resize((w, w))) for x in x_test]).reshape(-1,  w * w)
y_train_small = onehot(y_train)
y_test_small  = onehot(y_test)

np.savetxt(_data_dir + "/X_train.csv", x_train_small, delimiter=",")
np.savetxt(_data_dir + "/Y_train.csv", y_train_small, delimiter=",")
np.savetxt(_data_dir + "/X_test.csv",  x_test_small,  delimiter=",")
np.savetxt(_data_dir + "/Y_test.csv",  y_test_small,  delimiter=",")

print("Operation complete")
print("X_train.shape = %s" % str(x_train_small.shape))
print("Y_train.shape = %s" % str(y_train_small.shape))
print("X_test.shape  = %s" % str(x_test_small.shape))
print("Y_test.shape  = %s" % str(y_test_small.shape))
```

Note: output files are now named `X_train.csv` (not `X_train_small.csv`) to match the `load_dataset` convention.

- [ ] **Step 2: Verify import**

```bash
python -c "
import sys, os
sys.path.insert(0, '.')
from mlp.topology import load_topology
print('topology import OK')
"
```

Expected: `topology import OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/dataeng.py
git commit -m "fix: dataeng.py — derive w from topology CSV, fix w*w reshape bug, guard non-square input"
```

---

## Task 6: `scripts/train.py`

Remove the `layer_sizes[0]**2` quirk, use `load_topology` and `load_dataset`, infer `n_classes` from data.

**Files:**
- Modify: `scripts/train.py`

- [ ] **Step 1: Rewrite `scripts/train.py`**

```python
# train.py — trains a simple MLP classifier

import sys
import os
sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

import time
import numpy as np
from jax import random

from mlp.mlp       import init_network_params, accuracy, update
from mlp.topology  import load_topology, save_topology
from mlp.dataset   import load_dataset

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
_data_dir    = os.path.realpath(os.path.join(__location__, '..', 'data'))


def main():
    args           = sys.argv
    output_folder  = os.path.abspath(args[1])
    topology_csv   = os.path.abspath(args[2])
    dataset_path   = os.path.abspath(args[3]) if len(args) > 3 else _data_dir
    dataset_source = args[4] if len(args) > 4 else "csv"

    print("Selected output folder: %s" % output_folder)

    sizes, activations = load_topology(topology_csv)
    print("Topology: %s" % sizes)
    print("Activations: %s" % activations)

    (x_train, y_train), (x_test, y_test), stats = load_dataset(dataset_path, source=dataset_source)

    print("Data loaded.")
    print("\tX_train.shape = %s" % str(x_train.shape))
    print("\tX_test.shape  = %s" % str(x_test.shape))
    print("\tY_train.shape = %s" % str(y_train.shape))
    print("\tY_test.shape  = %s" % str(y_test.shape))

    if sizes[0] != stats.n_features:
        sys.exit(
            f"Topology mismatch: topology first layer is {sizes[0]} "
            f"but dataset has {stats.n_features} features."
        )
    if sizes[-1] != stats.n_classes:
        sys.exit(
            f"Topology mismatch: topology output layer is {sizes[-1]} "
            f"but dataset has {stats.n_classes} classes."
        )

    batch_epochs = 100
    num_epochs   = batch_epochs * 10
    batch_size   = 128

    print("Training loop started")
    print("-" * 57)

    network = init_network_params(sizes, activations, random.PRNGKey(0))
    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(10):
            batch = np.random.choice(len(x_train), size=500)
            network = update(network, x_train[batch], y_train[batch])

        if epoch % batch_epochs == 0:
            epoch_time = time.time() - start_time
            print("\t\tEpoch {} of {} in {:0.2f} sec".format(epoch, num_epochs, epoch_time))
            print("\t\tTraining accuracy {:0.5f}".format(accuracy(network, x_train, y_train)))
            print("\t\tTest accuracy     {:0.5f}".format(accuracy(network, x_test,  y_test)))
            print("-" * 57)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, layer in enumerate(network):
        np.save(output_folder + "/W_%i.npy" % i, layer.W)
        np.save(output_folder + "/b_%i.npy" % i, layer.b)

    # Save topology alongside weights so downstream scripts can load activations
    save_topology(os.path.join(output_folder, "network_topology.csv"), sizes, activations)

    print("\nModel saved to %s" % output_folder)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import**

```bash
python -c "
import sys; sys.path.insert(0, '.')
import scripts.train  # just check imports, don't run main
" 2>&1 | grep -v "^$" | head -5
```

Expected: no import errors.

- [ ] **Step 3: Run all unit tests**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add scripts/train.py
git commit -m "fix: train.py — remove layer_sizes**2 quirk, use load_topology/load_dataset, validate shapes"
```

---

## Task 7: `backend/compiler.py`

Switch `Program.__init__` topology loading to `load_topology`. Fix index access `p[0]` / `p[1]` that breaks with the new `Layer` class.

**Files:**
- Modify: `backend/compiler.py` (only the `Program.__init__` method)

- [ ] **Step 1: Read the relevant section**

Read lines 23–50 of `backend/compiler.py` to confirm the exact code to change (already read above but confirm nothing has changed).

- [ ] **Step 2: Edit `backend/compiler.py`**

In `Program.__init__`, change the `if type(topology) == str:` branch from:

```python
if type(topology) == str: # passing file name
    params = load_network_params(topology)

    self.W = []
    self.b = []

    # "cast" the storage format
    for p in params:
        self.W.append(p[0])
        self.b.append(p[1])

    self.activation_functions_list = [ 'RELU' for w in self.W[:-1] ] + ['LINEAR']
    self.topology = [ w.shape[1] for w in self.W]  + [self.W[-1].shape[0]]
```

to:

```python
if type(topology) == str: # passing folder name
    import os
    from mlp.topology import load_topology as _load_topology

    self.W = []
    self.b = []

    # Load activations from topology CSV if saved alongside the weights
    topo_csv = os.path.join(topology, "network_topology.csv")
    if os.path.exists(topo_csv):
        _, csv_acts = _load_topology(topo_csv)
        activations = csv_acts
    else:
        activations = None  # load_network_params will apply defaults

    params = load_network_params(topology, activations=activations)

    for p in params:
        self.W.append(p.W)
        self.b.append(p.b)

    # Map lowercase activation names to the compiler's uppercase convention
    _act_map = {"relu": "RELU", "linear": "LINEAR", "tanh": "TANH", "sigmoid": "SIGMOID"}
    csv_activations = [l.activation for l in params]
    self.activation_functions_list = [_act_map.get(a, "LINEAR") for a in csv_activations]
    self.topology = [w.shape[1] for w in self.W] + [self.W[-1].shape[0]]
```

- [ ] **Step 3: Verify compiler imports cleanly**

```bash
python -c "from backend.compiler import Program; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Run all unit tests**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add backend/compiler.py
git commit -m "fix: compiler Program — use load_topology, fix Layer index access p[0]->p.W"
```

---

## Task 8: Update topology CSV and rename data files

**Files:**
- Modify: `data/network_topology.csv`

- [ ] **Step 1: Update `data/network_topology.csv`**

Replace the current contents (`14, 10, 10, 10`) with:

```
196,10,10,10
relu,relu,linear
```

(196 = 14², actual input dimension. 3 layers → 3 activations.)

- [ ] **Step 2: Verify topology loads correctly**

```bash
python -c "
from mlp.topology import load_topology
sizes, acts = load_topology('data/network_topology.csv')
print('sizes:', sizes)
print('activations:', acts)
"
```

Expected:
```
sizes: [196, 10, 10, 10]
activations: ['relu', 'relu', 'linear']
```

- [ ] **Step 3: Rename existing processed CSV files** (if present) to match the new naming convention

```bash
cd /Users/simonhgt/Documents/repos/onnxmlir-pim-sparsify/data
for f in X_train_small.csv Y_train_small.csv X_test_small.csv Y_test_small.csv; do
    new="${f/_small/}"
    [ -f "$f" ] && mv "$f" "$new" && echo "Renamed $f -> $new" || echo "Not found: $f"
done
```

- [ ] **Step 4: Run all tests**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add data/network_topology.csv data/
git commit -m "fix: update network_topology.csv to actual dimensions (196), rename processed CSVs"
```

---

## Task 9: End-to-end smoke test

Verify the full pipeline runs without errors using existing data.

**Files:** none — verification only

- [ ] **Step 1: Run the full test suite**

```bash
python -m pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Verify sparsifier imports and make_omega**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from sparsifier.sparsifier import make_omega
from mlp.mlp import init_network_params
from jax import random
net = init_network_params([196, 10, 10, 10], ['relu','relu','linear'], random.PRNGKey(0))
omega = make_omega(net, 0.0, 255.0, n_samples=100)
print('make_omega shape:', omega.shape)
assert omega.shape == (100, 196)
print('OK')
"
```

Expected:
```
make_omega shape: (100, 196)
OK
```

- [ ] **Step 3: Verify train.py help text (no crash on import)**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from mlp.topology import load_topology
from mlp.dataset  import load_dataset, DatasetStats
from mlp.mlp      import Layer, ACTIVATIONS, init_network_params
print('All imports OK')
print('ACTIVATIONS keys:', list(ACTIVATIONS.keys()))
"
```

Expected:
```
All imports OK
ACTIVATIONS keys: ['relu', 'tanh', 'sigmoid', 'linear']
```

- [ ] **Step 4: If processed data exists, run a quick training smoke test**

```bash
python scripts/train.py /tmp/smoke_test data/network_topology.csv 2>&1 | tail -5
```

Expected: training loop runs, model saved to `/tmp/smoke_test/`.

- [ ] **Step 5: Final commit**

```bash
git add .
git commit -m "chore: end-to-end smoke test verified — flexible datasets and architectures complete"
```
