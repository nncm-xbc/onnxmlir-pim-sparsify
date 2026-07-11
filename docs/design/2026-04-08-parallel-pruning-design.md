# Parallel Pruning Search — Design Spec
**Date:** 2026-04-08
**Branch:** `benchmark_framework`

## Problem

`prune()` in `sparsifier/sparsifier.py` performs a serial argmin search over all
non-zero weights. For each candidate `(layer_idx, i, j)` it zeroes the weight, calls
`d(og_net, probe_net, omega)` (a JAX JIT-compiled forward pass over 10 000 samples),
and restores it. Each evaluation is fully independent — the loop is embarrassingly
parallel. On large networks (e.g. 784 → 512 → 256 → 10, ~535 000 candidates) this
is the dominant runtime cost.

Additionally, the current `d()` call recomputes `batched_predict(og_net, omega)` for
every candidate even though `og_net` never changes during the search. This is pure
redundant work.

## Goal

Replace the serial search loop with a C++/OpenMP parallel implementation exposed as a
Python extension module (`prune_ext`). The extension is optional: `sparsifier.py` falls
back to the existing serial loop when the module is not installed.

## Non-Goals

- Replacing `adjust()` — it stays in Python/JAX.
- CUDA support (deferred).
- Batched JAX vmap approach (deferred).
- Changes to `Layer`, `mlp.py`, or any module outside `sparsifier/` and `prune_ext/`.

---

## File Layout

```
prune_ext/                     ← new directory at repo root
├── CMakeLists.txt
├── activations.hpp            ← activation registry: string → function pointer
├── forward.hpp                ← single-sample forward pass + log-softmax
└── prune_ext.cpp              ← pybind11 module: find_best_candidate()

sparsifier/sparsifier.py       ← two targeted changes (import guard + prune() patch)
```

---

## C++ Module Interface

```python
# called from Python after importing prune_ext
layer_idx, i, j, min_dist = prune_ext.find_best_candidate(
    layers,      # list of (W, b, mask, activation_name) tuples — numpy float32
    og_outputs,  # np.float32 [n_samples, n_classes] — precomputed in Python
    omega,       # np.float32 [n_samples, input_dim]
)
```

`og_outputs` must be computed by the caller before invoking `find_best_candidate`.
In `sparsifier.py` this is:
```python
og_outputs = np.array(batched_predict(og_net, omega), dtype=np.float32)
```

---

## Data Flow

```
Python
  og_outputs = batched_predict(og_net, omega)   # ONE JAX call, shape [S, C]
  layers = [(W, b, mask, activ), ...]           # numpy float32, zero-copy into C++

C++ (prune_ext.find_best_candidate)
  1. Unpack pybind11 buffer views → LayerData structs (raw pointers, no copy)
  2. Build flat candidate list: [(layer_idx, i, j) for all W[i,j] != 0]
  3. Pre-allocate per-thread scratch buffers (size = max_layer_width × 2)
  4. #pragma omp parallel for schedule(dynamic, 32) over candidates
       each thread:
         forward_pass(layers, omega[s], zero_layer=layer_idx, zero_i=i, zero_j=j)
           → log-softmax output
         SSE vs og_outputs[s], accumulated over all S samples
         track thread-local (min_dist, layer_idx, i, j)
  5. #pragma omp critical → global reduction to argmin
  6. Return (layer_idx, i, j, min_dist) as Python tuple

Python
  Apply the winning zero; call adjust() as before.
```

---

## `activations.hpp`

Static registry mapping activation name strings to `float(*)(float)` function pointers.
Lookup throws `std::invalid_argument` on unknown names.

Supported activations: `relu`, `tanh`, `sigmoid`, `linear`.

**To add a new activation:** add one entry to the static map in `get_activation()`.
No other files need to change.

```cpp
using ActivFn = float (*)(float);

inline ActivFn get_activation(const std::string& name) {
    static const std::unordered_map<std::string, ActivFn> reg = {
        {"relu",    [](float x) -> float { return x > 0.f ? x : 0.f; }},
        {"tanh",    [](float x) -> float { return std::tanh(x); }},
        {"sigmoid", [](float x) -> float { return 1.f / (1.f + std::exp(-x)); }},
        {"linear",  [](float x) -> float { return x; }},
        // add new activations here ↑
    };
    auto it = reg.find(name);
    if (it == reg.end())
        throw std::invalid_argument("Unknown activation: " + name);
    return it->second;
}
```

---

## `forward.hpp`

### `LayerData` struct

```cpp
struct LayerData {
    const float* W;       // [n_out × n_in] row-major
    const float* b;       // [n_out]
    const float* mask;    // [n_out × n_in]
    int n_out, n_in;
    ActivFn activation;
    bool is_last;         // true → log-softmax output instead of activation
};
```

### `forward_sse_one()`

Runs the forward pass for a single input sample, substituting `0` for weight
`layers[zero_layer].W[zero_i * n_in + zero_j]` during computation.

Returns the SSE contribution of this sample: `Σ(out_k - og_out_k)²`.

```cpp
float forward_sse_one(
    const std::vector<LayerData>& layers,
    const float* input,          // [input_dim]
    const float* og_out_row,     // [n_classes]
    int zero_layer, int zero_i, int zero_j,
    float* buf                   // thread-local scratch, size >= max_width * 2
);
```

Implementation notes:
- Uses two alternating sub-buffers within `buf` (ping-pong) to avoid heap allocation.
- Log-softmax on last layer: subtract `max` before `exp` for numerical stability.
- Applies mask × W for each layer (matching JAX's `lax.stop_gradient(mask) * W`).

---

## `prune_ext.cpp`

Single pybind11 module with one exported function `find_best_candidate`.

Steps:
1. Unpack `py::list layers` → `std::vector<LayerData>`.
2. Build `candidates` vector of `{layer_idx, i, j}` structs for all non-zero weights.
3. Compute `max_width` across all layers for scratch buffer sizing.
4. Run OpenMP parallel search with thread-local argmin + `#pragma omp critical` reduction.
5. Return `py::tuple(layer_idx, i, j, min_dist)`.

When compiled without OpenMP (`USE_OPENMP` not defined), the `#pragma omp` directives
are absent and the loop runs serially in C++ (still faster than Python due to
`og_outputs` precomputation and no JAX dispatch overhead).

---

## `sparsifier/sparsifier.py` Changes

### Change 1 — Import guard (module top)

```python
try:
    import prune_ext as _prune_ext
    _USE_EXT = True
except ImportError:
    _USE_EXT = False
```

### Change 2 — `prune()` signature and body

```python
def prune(net, og_net, omega, activations=None, doAdjust=True):
```

`activations` defaults to `["relu"] * (len(net) - 1) + ["linear"]`, matching the
current hardcoded behaviour in `mlp.py`'s `predict()`. Callers that load topology from
CSV can pass the activation list explicitly.

When `_USE_EXT` is `True`:
- Precompute `og_outputs = np.array(batched_predict(og_net, omega), dtype=np.float32)`
- Build `layers` list of `(W, b, mask, activ)` tuples as `np.float32` arrays
- Call `_prune_ext.find_best_candidate(layers, og_outputs, np.asarray(omega, np.float32))`
- Receive `(minimo_idx, minimo_i, minimo_j, minimo)` and proceed to the apply/adjust phase

When `_USE_EXT` is `False`, the existing serial loop runs unchanged.

---

## Build System (`prune_ext/CMakeLists.txt`)

```cmake
cmake_minimum_required(VERSION 3.15)
project(prune_ext LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
find_package(OpenMP)

pybind11_add_module(prune_ext prune_ext.cpp)

if(OpenMP_CXX_FOUND)
    target_link_libraries(prune_ext PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(prune_ext PRIVATE USE_OPENMP)
    message(STATUS "OpenMP found — parallel search enabled")
else()
    message(WARNING "OpenMP not found — serial C++ search")
endif()

target_compile_options(prune_ext PRIVATE -O3 -march=native)
```

**Build instructions:**
```bash
cd prune_ext
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cp build/prune_ext*.so ../   # or install to site-packages
```

---

## Correctness Invariant

`find_best_candidate()` must return the same `(layer_idx, i, j)` as the serial Python
loop for any deterministic input. The `og_outputs` precomputation is mathematically
equivalent to the original `d()` call because `og_net` is never modified during the
search. The SSE computed in C++ must match `float(d(og_net, probe_net, omega))` to
within float32 precision.

---

## What Does Not Change

- `adjust()` — unchanged, still runs in Python/JAX after the search.
- `d()`, `batched_predict()`, `Layer`, `mlp.py` — untouched.
- The serial Python loop — preserved as a fallback; no behaviour change when
  `prune_ext` is not installed.
- `main()` in `sparsifier.py` — no changes needed; it calls `prune()` by keyword.
