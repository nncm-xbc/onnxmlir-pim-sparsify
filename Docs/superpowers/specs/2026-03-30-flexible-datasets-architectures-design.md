# Design: Flexible Datasets and Architectures

**Date:** 2026-03-30
**Branch:** refactor
**Status:** Approved

---

## Goal

Remove all MNIST- and MLP-specific hardcoded constants from the codebase. Everything that can be inferred from the network weights or dataset files must be inferred. Architectural choices that cannot be inferred (activation functions) are specified in the topology CSV with sensible defaults.

---

## Scope

- `mlp/mlp.py` — flexible activations
- `mlp/topology.py` — new: topology CSV parser (shared by all scripts)
- `mlp/dataset.py` — new: dataset loader (CSV and PyTorch)
- `scripts/train.py` — remove hardcoded paths, onehot arity, topology quirk
- `scripts/dataeng.py` — fix `14*14` bug, guard non-square input dims
- `sparsifier/sparsifier.py` — remove hardcoded `[0, 256)` from `make_omega`
- `backend/compiler.py` — switch `Program` to use shared topology parser

Out of scope: CNN support, training hyperparameter config, `backend/network.py`.

---

## 1. Topology CSV Format

**New format** — two rows, activations optional:

```
196,64,10
relu,relu,linear
```

- Row 1: actual layer sizes (input dim is the real flattened size, e.g. `196` not `14`).
- Row 2 (optional): per-layer activation name. Defaults: `relu` for all hidden layers, `linear` for the last layer.
- Supported activations: `relu`, `tanh`, `sigmoid`, `linear`.
- The old single-row format (`14,64,10` with no activations row) is still valid and parsed correctly with defaults applied.

**Breaking change:** the first entry is now the actual input dimension, not its square root. `dataeng.py` derives the image side length as `w = int(round(sqrt(sizes[0])))`.

---

## 2. `mlp/topology.py` (new)

```python
def load_topology(csv_path) -> tuple[list[int], list[str]]:
    """Returns (sizes, activations).
    sizes:       e.g. [196, 64, 10]
    activations: e.g. ["relu", "relu", "linear"] — always len(sizes)
    """

def save_topology(csv_path, sizes: list[int], activations: list[str]) -> None:
    """Writes a two-row topology CSV."""
```

- Validates all activation names against `mlp.ACTIVATIONS` registry; raises `ValueError` with a clear message on unknown names.
- Ensures `len(activations) == len(sizes)`.
- All scripts that currently call `np.genfromtxt` on a topology CSV switch to `load_topology`.

---

## 3. `mlp/dataset.py` (new)

```python
class DatasetStats(NamedTuple):
    input_min:  float
    input_max:  float
    n_classes:  int
    n_features: int

def load_dataset(
    path: str,
    source: str = "csv",          # "csv" | "pytorch"
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], DatasetStats]:
    """Returns ((x_train, y_train), (x_test, y_test), stats)."""
```

**CSV mode:**
- `path` is a folder containing `X_train.csv`, `Y_train.csv`, `X_test.csv`, `Y_test.csv`.
- Returns numpy arrays; no shape or value assumptions.
- `n_classes` inferred from Y: if Y has multiple columns, assumed one-hot and `n_classes = Y.shape[1]`; if Y has a single column, assumed label-encoded and `n_classes = len(np.unique(Y))`. The loader always returns Y in one-hot form regardless of input format.
- `n_features` from number of columns in X.
- `input_min/max` computed from `x_train`.

**PyTorch mode:**
- `path` is a dataset name string (e.g. `"MNIST"`, `"CIFAR10"`).
- Downloads via `torchvision.datasets`, flattens spatial dims, normalizes to `[0, 1]`, one-hot encodes labels.
- Returns same format as CSV mode.
- `input_min = 0.0`, `input_max = 1.0` (post-normalization).

---

## 4. `mlp/mlp.py` Changes

### Layer NamedTuple

```python
class Layer(NamedTuple):
    W:          np.ndarray
    b:          np.ndarray
    mask:       np.ndarray
    activation: str = "relu"
```

### Activation registry

```python
ACTIVATIONS = {
    "relu":    lambda x: jnp.maximum(0, x),
    "tanh":    jnp.tanh,
    "sigmoid": jax.nn.sigmoid,
    "linear":  lambda x: x,
}
```

Adding a new activation requires only one dict entry here.

### `predict` changes

- Reads `ACTIVATIONS[layer.activation]` per layer instead of hardcoding ReLU.
- Removes the hardcoded `logsumexp` softmax on the final layer — the final layer uses its activation from the `Layer` record (defaults to `"linear"`).

### `init_network_params`

```python
def init_network_params(sizes: list[int], activations: list[str], key) -> list[Layer]:
```

Gains an `activations` parameter. Stores activation name in each `Layer`.

### `load_network_params`

```python
def load_network_params(folder: str, activations: list[str] | None = None) -> list[Layer]:
```

- If `activations` is `None`, defaults to `relu`...`relu`, `linear` (backward compatible with old weight folders).
- Callers that have a topology CSV pass the activations list from `load_topology`.

### `update`

```python
def update(network, x, y, step_size=0.01):
```

`step_size` moves from module-level global to a parameter with the existing value as default.

---

## 5. Script Changes

### `scripts/train.py`

- CLI: `python3 scripts/train.py <output_folder> <topology_csv> [--dataset-path <path>] [--dataset-source csv|pytorch]`
  - `--dataset-path` defaults to `data/` (relative to repo root); `--dataset-source` defaults to `csv`.
- `load_data()` removed; replaced by `load_dataset(path, source)`.
- `layer_sizes[0] = layer_sizes[0]**2` removed.
- `onehot` uses `stats.n_classes`.
- After training, calls `save_topology(output_folder + "/network_topology.csv", sizes, activations)` so downstream scripts have the topology available in the weights folder.
- Validates `sizes[0] == stats.n_features`; errors clearly if they don't match.

### `scripts/dataeng.py`

- Reads topology via `load_topology`; `w = int(round(sqrt(sizes[0])))`.
- Fixes `reshape(-1, 14*14)` → `reshape(-1, w*w)`.
- Adds guard: if `sizes[0]` is not a perfect square, exits with a clear error (non-image datasets don't use this script).

### `sparsifier/sparsifier.py`

- `make_omega(network, input_min, input_max, n_samples=10000)` — replaces hardcoded `[0, 256)`.
- `main()` CLI: `python3 -m sparsifier.sparsifier <params_folder> <dataset_path> [--dataset-source csv|pytorch]`
- Calls `load_dataset` to get test split and stats; passes `stats.input_min/max` to `make_omega`.
- Loads activations from `<params_folder>/network_topology.csv` if present, falls back to defaults.

### `backend/compiler.py`

- `Program.__init__` switches topology loading to `load_topology` (when loading from a folder that has a `network_topology.csv`).
- No other changes to compiler logic.

---

## 6. File Layout After Changes

```
mlp/
  mlp.py          # Layer (+ activation field), ACTIVATIONS registry, predict, update
  topology.py     # NEW: load_topology, save_topology
  dataset.py      # NEW: load_dataset, DatasetStats
  __init__.py
sparsifier/
  sparsifier.py   # make_omega takes input_min/max
scripts/
  train.py        # uses load_dataset, load_topology, save_topology
  dataeng.py      # fixes 14*14 bug, uses load_topology
  test.py         # unchanged
backend/
  compiler.py     # Program uses load_topology
  network.py      # unchanged
```

---

## 7. Backward Compatibility

**Breaking change — topology CSV first value:** The old convention stored the square root of the input dimension (e.g. `14` for a 196-input network). The new convention stores the actual input dimension (`196`). Any existing `data/network_topology.csv` and topology files in `artifacts/` must be updated before running the new scripts. This is a one-time manual edit.

- Old single-row topology CSVs (no activations row) still parse; activations default to `relu`/`linear`.
- Old weight folders loaded without a topology CSV default to `relu`/`linear` activations.
- The `update()` signature change is backward compatible (default value preserved).
