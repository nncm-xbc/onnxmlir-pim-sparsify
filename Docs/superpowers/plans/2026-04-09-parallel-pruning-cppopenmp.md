# Parallel Pruning Search — C++/OpenMP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the serial argmin search in `prune()` with a C++/OpenMP extension module (`prune_ext`) that evaluates all weight candidates in parallel, with a pure-Python fallback when the module is not built.

**Architecture:** A new `prune_ext/` directory holds four files: an activation registry header, a forward-pass header, a pybind11 module source, and a CMakeLists.txt. `sparsifier.py` gains an import guard and a fast path inside `prune()` that calls `prune_ext.find_best_candidate()`; the existing serial loop is preserved as the fallback. `og_outputs` is precomputed once per prune step (eliminating redundant recomputation of the original network's outputs for every candidate).

**Tech Stack:** C++17, pybind11, OpenMP, CMake ≥ 3.15, NumPy, JAX, pytest

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `prune_ext/activations.hpp` | **Create** | Registry mapping activation name strings to `float(*)(float)` function pointers |
| `prune_ext/forward.hpp` | **Create** | `LayerData` struct; `log_softmax_inplace()`; `forward_sse_one()` — single-sample forward pass with one weight zeroed |
| `prune_ext/prune_ext.cpp` | **Create** | pybind11 module; `find_best_candidate()` with OpenMP parallel for + critical reduction |
| `prune_ext/CMakeLists.txt` | **Create** | Build system: pybind11, optional OpenMP, `-O3 -march=native`, output `.so` to repo root |
| `tests/test_prune_ext.py` | **Create** | Three tests: argmin correctness vs serial reference, zero-weight skipping, unknown activation raises |
| `sparsifier/sparsifier.py` | **Modify** | Import guard; `activations` parameter on `prune()`; C++ fast path with `og_outputs` precomputation |

---

## Task 1: Write `tests/test_prune_ext.py`

**Files:**
- Create: `tests/test_prune_ext.py`

- [ ] **Step 1: Write the three tests**

```python
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
    assert abs(cpp_dist - ref_dist) / (abs(ref_dist) + 1e-9) < 5e-3, (
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
```

- [ ] **Step 2: Verify the tests fail with `ModuleNotFoundError`**

```bash
cd /Users/simonhgt/Documents/repos/onnxmlir-pim-sparsify
pytest tests/test_prune_ext.py -v 2>&1 | head -20
```

Expected output: `ModuleNotFoundError: No module named 'prune_ext'` (collection error — all three tests fail to import). This confirms the interface is defined before any implementation exists.

- [ ] **Step 3: Commit**

```bash
git add tests/test_prune_ext.py
git commit -m "test: add prune_ext interface tests (failing — module not yet built)"
```

---

## Task 2: Create `prune_ext/activations.hpp`

**Files:**
- Create: `prune_ext/activations.hpp`

- [ ] **Step 1: Create the file**

```cpp
// prune_ext/activations.hpp
#pragma once
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

using ActivFn = float (*)(float);

// Returns the activation function pointer for the given name.
// Throws std::invalid_argument for unknown names.
// To add a new activation: insert one entry into the static map below.
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
        throw std::invalid_argument("Unknown activation function: \"" + name + "\"");
    return it->second;
}
```

- [ ] **Step 2: Commit**

```bash
git add prune_ext/activations.hpp
git commit -m "feat: add prune_ext/activations.hpp — activation function registry"
```

---

## Task 3: Create `prune_ext/forward.hpp`

**Files:**
- Create: `prune_ext/forward.hpp`

- [ ] **Step 1: Create the file**

```cpp
// prune_ext/forward.hpp
#pragma once
#include <algorithm>
#include <cmath>
#include <vector>
#include "activations.hpp"

struct LayerData {
    const float* W;    // row-major [n_out, n_in]
    const float* b;    // [n_out]
    const float* mask; // row-major [n_out, n_in]
    int n_out, n_in;
    ActivFn activation;
    bool is_last;      // true → apply log-softmax instead of activation
};

// Numerically stable log-softmax in-place on buf[0..n).
inline void log_softmax_inplace(float* buf, int n) {
    float max_val = buf[0];
    for (int k = 1; k < n; k++)
        if (buf[k] > max_val) max_val = buf[k];
    float sum_exp = 0.f;
    for (int k = 0; k < n; k++) sum_exp += std::exp(buf[k] - max_val);
    float log_sum = max_val + std::log(sum_exp);
    for (int k = 0; k < n; k++) buf[k] -= log_sum;
}

// Forward pass for one input sample with one candidate weight treated as zero.
//
// layers:              network layer descriptors (read-only, shared across threads)
// input:               sample row, length layers[0].n_in
// og_out_row:          precomputed original-network output for this sample,
//                      length layers.back().n_out
// zero_layer/i/j:      candidate weight index to substitute with 0.f
// buf:                 thread-local scratch, must be >= max_width * 2 floats
// max_width:           max n_out across all layers (caller computes once)
//
// Returns the SSE contribution of this sample: Σ(output_k − og_out_row_k)²
//
// Ping-pong buffer: layer li writes to buf + (li%2)*max_width, reads from
// the opposite half (or from `input` for the first layer). No heap allocation.
inline float forward_sse_one(
    const std::vector<LayerData>& layers,
    const float* input,
    const float* og_out_row,
    int zero_layer, int zero_i, int zero_j,
    float* buf,
    int max_width
) {
    const float* src = input;
    for (int li = 0; li < (int)layers.size(); li++) {
        const LayerData& ld = layers[li];
        float* dst = buf + (li % 2) * max_width;
        for (int o = 0; o < ld.n_out; o++) {
            float acc = ld.b[o];
            for (int in_idx = 0; in_idx < ld.n_in; in_idx++) {
                float w = ld.W[o * ld.n_in + in_idx];
                if (li == zero_layer && o == zero_i && in_idx == zero_j)
                    w = 0.f;
                acc += ld.mask[o * ld.n_in + in_idx] * w * src[in_idx];
            }
            dst[o] = acc;
        }
        if (ld.is_last)
            log_softmax_inplace(dst, ld.n_out);
        else
            for (int o = 0; o < ld.n_out; o++) dst[o] = ld.activation(dst[o]);
        src = dst;
    }
    float sse = 0.f;
    int n_out = layers.back().n_out;
    for (int k = 0; k < n_out; k++) {
        float diff = src[k] - og_out_row[k];
        sse += diff * diff;
    }
    return sse;
}
```

- [ ] **Step 2: Commit**

```bash
git add prune_ext/forward.hpp
git commit -m "feat: add prune_ext/forward.hpp — LayerData struct and forward_sse_one"
```

---

## Task 4: Create `prune_ext/prune_ext.cpp`

**Files:**
- Create: `prune_ext/prune_ext.cpp`

- [ ] **Step 1: Create the file**

```cpp
// prune_ext/prune_ext.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <limits>
#include <stdexcept>
#include <vector>

#include "forward.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using Arr = py::array_t<float, py::array::c_style | py::array::forcecast>;

struct Candidate {
    int layer_idx, i, j;
};

py::tuple find_best_candidate(py::list layers_list, Arr og_outputs, Arr omega) {
    int n_layers = (int)layers_list.size();
    if (n_layers == 0)
        throw std::invalid_argument("layers list is empty");

    // Keep numpy buffer objects alive while C++ holds raw pointers into them.
    std::vector<Arr> W_arrs, b_arrs, mask_arrs;
    W_arrs.reserve(n_layers);
    b_arrs.reserve(n_layers);
    mask_arrs.reserve(n_layers);

    std::vector<LayerData> layers;
    layers.reserve(n_layers);

    for (int li = 0; li < n_layers; li++) {
        py::tuple t = layers_list[li].cast<py::tuple>();
        W_arrs.push_back(t[0].cast<Arr>());
        b_arrs.push_back(t[1].cast<Arr>());
        mask_arrs.push_back(t[2].cast<Arr>());
        std::string act_name = t[3].cast<std::string>();

        auto W_info    = W_arrs.back().request();
        auto b_info    = b_arrs.back().request();
        auto mask_info = mask_arrs.back().request();

        LayerData ld;
        ld.W         = static_cast<const float*>(W_info.ptr);
        ld.b         = static_cast<const float*>(b_info.ptr);
        ld.mask      = static_cast<const float*>(mask_info.ptr);
        ld.n_out     = (int)W_info.shape[0];
        ld.n_in      = (int)W_info.shape[1];
        ld.is_last   = (li == n_layers - 1);
        ld.activation = get_activation(act_name);  // throws for unknown names
        layers.push_back(ld);
    }

    auto og_info  = og_outputs.request();
    auto om_info  = omega.request();
    int n_samples = (int)og_info.shape[0];
    int n_classes = (int)og_info.shape[1];
    int input_dim = (int)om_info.shape[1];
    const float* og_ptr = static_cast<const float*>(og_info.ptr);
    const float* om_ptr = static_cast<const float*>(om_info.ptr);

    // Build flat candidate list: every (layer_idx, i, j) where W[i,j] != 0.
    std::vector<Candidate> candidates;
    for (int li = 0; li < n_layers; li++) {
        const LayerData& ld = layers[li];
        for (int i = 0; i < ld.n_out; i++)
            for (int j = 0; j < ld.n_in; j++)
                if (ld.W[i * ld.n_in + j] != 0.f)
                    candidates.push_back({li, i, j});
    }
    if (candidates.empty())
        throw std::runtime_error(
            "No non-zero weights found — network is fully pruned.");

    int n_candidates = (int)candidates.size();
    int max_width    = 0;
    for (const auto& ld : layers)
        max_width = std::max(max_width, ld.n_out);

    int   best_layer = candidates[0].layer_idx;
    int   best_i     = candidates[0].i;
    int   best_j     = candidates[0].j;
    float best_dist  = std::numeric_limits<float>::max();

#ifdef USE_OPENMP
    #pragma omp parallel
    {
        int   local_layer = candidates[0].layer_idx;
        int   local_i     = candidates[0].i;
        int   local_j     = candidates[0].j;
        float local_min   = std::numeric_limits<float>::max();
        std::vector<float> buf(max_width * 2);

        #pragma omp for schedule(dynamic, 32)
        for (int c = 0; c < n_candidates; c++) {
            const Candidate& cand = candidates[c];
            float dist = 0.f;
            for (int s = 0; s < n_samples; s++)
                dist += forward_sse_one(
                    layers,
                    om_ptr + s * input_dim,
                    og_ptr + s * n_classes,
                    cand.layer_idx, cand.i, cand.j,
                    buf.data(), max_width);
            if (dist < local_min) {
                local_min   = dist;
                local_layer = cand.layer_idx;
                local_i     = cand.i;
                local_j     = cand.j;
            }
        }

        #pragma omp critical
        if (local_min < best_dist) {
            best_dist  = local_min;
            best_layer = local_layer;
            best_i     = local_i;
            best_j     = local_j;
        }
    }
#else
    std::vector<float> buf(max_width * 2);
    for (int c = 0; c < n_candidates; c++) {
        const Candidate& cand = candidates[c];
        float dist = 0.f;
        for (int s = 0; s < n_samples; s++)
            dist += forward_sse_one(
                layers,
                om_ptr + s * input_dim,
                og_ptr + s * n_classes,
                cand.layer_idx, cand.i, cand.j,
                buf.data(), max_width);
        if (dist < best_dist) {
            best_dist  = dist;
            best_layer = cand.layer_idx;
            best_i     = cand.i;
            best_j     = cand.j;
        }
    }
#endif

    return py::make_tuple(best_layer, best_i, best_j, best_dist);
}

PYBIND11_MODULE(prune_ext, m) {
    m.doc() = "Parallel pruning candidate search using OpenMP.";
    m.def(
        "find_best_candidate", &find_best_candidate,
        py::arg("layers"), py::arg("og_outputs"), py::arg("omega"),
        R"doc(
Find the single non-zero weight whose removal minimises SSE distance to og_outputs.

Args:
    layers:     list of (W, b, mask, activation_name) tuples (numpy float32).
                W shape [n_out, n_in], b [n_out], mask [n_out, n_in].
                Activation names: "relu", "tanh", "sigmoid", "linear".
                The last layer always applies log-softmax regardless of activation_name.
    og_outputs: precomputed original-network outputs, float32 [n_samples, n_classes].
                Compute once with: np.array(batched_predict(og_net, omega), dtype=np.float32)
    omega:      input sample matrix, float32 [n_samples, input_dim].

Returns:
    (layer_idx, i, j, min_distance) as a Python tuple.
        )doc");
}
```

- [ ] **Step 2: Commit**

```bash
git add prune_ext/prune_ext.cpp
git commit -m "feat: add prune_ext/prune_ext.cpp — pybind11 module with OpenMP parallel search"
```

---

## Task 5: Create `prune_ext/CMakeLists.txt` and build

**Files:**
- Create: `prune_ext/CMakeLists.txt`

- [ ] **Step 1: Create CMakeLists.txt**

```cmake
cmake_minimum_required(VERSION 3.15)
project(prune_ext LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(pybind11 REQUIRED)
find_package(OpenMP)

pybind11_add_module(prune_ext prune_ext.cpp)

# Place the .so in the repo root so `import prune_ext` works without
# modifying PYTHONPATH — consistent with how other modules are imported.
set_target_properties(prune_ext PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/.."
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(prune_ext PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(prune_ext PRIVATE USE_OPENMP)
    message(STATUS "OpenMP found — parallel search enabled")
else()
    message(WARNING "OpenMP not found — serial C++ search (still faster than Python due to og_outputs precomputation)")
endif()

target_compile_options(prune_ext PRIVATE -O3 -march=native)
```

- [ ] **Step 2: Build the extension**

```bash
cd /Users/simonhgt/Documents/repos/onnxmlir-pim-sparsify/prune_ext
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -- -j$(nproc)
```

Expected: a file matching `prune_ext*.so` appears in the repo root
(`/Users/simonhgt/Documents/repos/onnxmlir-pim-sparsify/`).

Verify:
```bash
ls /Users/simonhgt/Documents/repos/onnxmlir-pim-sparsify/prune_ext*.so
```

- [ ] **Step 3: Run the tests — expect all three to pass**

```bash
cd /Users/simonhgt/Documents/repos/onnxmlir-pim-sparsify
pytest tests/test_prune_ext.py -v
```

Expected output:
```
tests/test_prune_ext.py::test_find_best_candidate_matches_serial  PASSED
tests/test_prune_ext.py::test_find_best_candidate_skips_zero_weights PASSED
tests/test_prune_ext.py::test_unknown_activation_raises           PASSED
3 passed
```

- [ ] **Step 4: Commit**

```bash
cd /Users/simonhgt/Documents/repos/onnxmlir-pim-sparsify
git add prune_ext/CMakeLists.txt
git commit -m "feat: add prune_ext/CMakeLists.txt — pybind11 + OpenMP build"
```

---

## Task 6: Modify `sparsifier/sparsifier.py`

**Files:**
- Modify: `sparsifier/sparsifier.py`

- [ ] **Step 1: Add import guard at the top of the file (after existing imports)**

Add these lines immediately after the last `import` statement at the top of `sparsifier/sparsifier.py`:

```python
try:
    import prune_ext as _prune_ext
    _USE_EXT = True
except ImportError:
    _USE_EXT = False
```

- [ ] **Step 2: Replace the `prune()` function**

Replace the entire `prune()` function (lines 88–141) with:

```python
def prune(net, og_net, omega, activations=None, doAdjust=True):
    if activations is None:
        activations = ["relu"] * (len(net) - 1) + ["linear"]

    minimo     = 1e16
    minimo_idx = 0
    minimo_i   = 0
    minimo_j   = 0

    probe_net = clone_network(net)
    prune_t0  = time.perf_counter()

    if _USE_EXT:
        og_outputs = np.array(batched_predict(og_net, omega), dtype=np.float32)
        layers_ext = [
            (np.asarray(l.W,    dtype=np.float32),
             np.asarray(l.b,    dtype=np.float32),
             np.asarray(l.mask, dtype=np.float32),
             act)
            for l, act in zip(net, activations)
        ]
        omega_f32 = np.asarray(omega, dtype=np.float32)
        minimo_idx, minimo_i, minimo_j, minimo = _prune_ext.find_best_candidate(
            layers_ext, og_outputs, omega_f32)
        minimo = float(minimo)
    else:
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

    prune_time_s = time.perf_counter() - prune_t0

    # Apply the winning zero permanently and optionally adjust.
    probe_net[minimo_idx].W[minimo_i, minimo_j]    = 0.
    probe_net[minimo_idx].mask[minimo_i, minimo_j] = 0.
    adjust_t0 = time.perf_counter()
    if doAdjust and minimo > 0:
        probe_net = adjust(probe_net, og_net, omega)
    adjust_time_s = time.perf_counter() - adjust_t0

    meta = PruneMeta(
        layer_idx=minimo_idx,
        i=minimo_i,
        j=minimo_j,
        distanza=float(minimo),
        prune_time_s=prune_time_s,
        adjust_time_s=adjust_time_s,
    )
    return probe_net, meta
```

Also add `batched_predict` to the import from `mlp.mlp` at the top of `sparsifier.py` — the existing `from mlp.mlp import *` already covers this, so no change needed there.

- [ ] **Step 3: Run the existing benchmark tests to verify no regression**

```bash
cd /Users/simonhgt/Documents/repos/onnxmlir-pim-sparsify
pytest tests/test_benchmark.py -v
```

Expected: all tests that passed before still pass.

- [ ] **Step 4: Run the full test suite**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add sparsifier/sparsifier.py
git commit -m "feat: wire prune_ext into prune() — parallel search with Python fallback"
```
