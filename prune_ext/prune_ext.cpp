// prune_ext/prune_ext.cpp
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include "forward.hpp"

namespace py = pybind11;
using Arr = py::array_t<double, py::array::c_style | py::array::forcecast>;

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
        ld.W          = static_cast<const double*>(W_info.ptr);
        ld.b          = static_cast<const double*>(b_info.ptr);
        ld.mask       = static_cast<const double*>(mask_info.ptr);
        ld.n_out      = (int)W_info.shape[0];
        ld.n_in       = (int)W_info.shape[1];
        ld.is_last    = (li == n_layers - 1);
        ld.activation = get_activation(act_name);  // throws for unknown names
        layers.push_back(ld);
    }

    auto og_info  = og_outputs.request();
    auto om_info  = omega.request();
    int n_samples = (int)og_info.shape[0];
    int n_classes = (int)og_info.shape[1];
    int input_dim = (int)om_info.shape[1];
    const double* og_ptr = static_cast<const double*>(og_info.ptr);
    const double* om_ptr = static_cast<const double*>(om_info.ptr);

    // Build flat candidate list: every (layer_idx, i, j) where mask != 0.
    std::vector<Candidate> candidates;
    for (int li = 0; li < n_layers; li++) {
        const LayerData& ld = layers[li];
        for (int i = 0; i < ld.n_out; i++)
            for (int j = 0; j < ld.n_in; j++)
                if (ld.mask[(size_t)i * ld.n_in + j] != 0.0)
                    candidates.push_back({li, i, j});
    }
    if (candidates.empty())
        throw std::runtime_error(
            "No non-zero weights found — network is fully pruned.");

    int64_t n_candidates = (int64_t)candidates.size();
    int max_width = 0;
    for (const auto& ld : layers)
        max_width = std::max(max_width, ld.n_out);

    // Precompute per-sample activations and baseline SSE of the unmodified
    // network once; candidate evaluation reuses everything upstream of the
    // zeroed weight instead of re-running the full forward pass.
    ForwardCache cache;
    cache.z.resize(n_layers);
    cache.a.resize(n_layers);
    for (int li = 0; li < n_layers; li++) {
        cache.z[li].resize((size_t)n_samples * layers[li].n_out);
        cache.a[li].resize((size_t)n_samples * layers[li].n_out);
    }
    cache.base_sse.resize(n_samples);

#ifdef USE_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int s = 0; s < n_samples; s++)
        cache_forward_one(layers, om_ptr + (size_t)s * input_dim,
                          og_ptr + (size_t)s * n_classes, s, cache);

    // Deterministic reduction: lexicographic min over (distance, candidate
    // index), so exact ties resolve to the first candidate in scan order —
    // identical to the serial and pure-Python searches regardless of thread
    // scheduling.
    int64_t best_cand = 0;
    double  best_dist = std::numeric_limits<double>::max();

#ifdef USE_OPENMP
    #pragma omp parallel
#endif
    {
        int64_t local_cand = n_candidates;
        double  local_min  = std::numeric_limits<double>::max();
        std::vector<double> buf((size_t)max_width * 2);

#ifdef USE_OPENMP
        #pragma omp for schedule(dynamic, 32)
#endif
        for (int64_t c = 0; c < n_candidates; c++) {
            const Candidate& cand = candidates[c];
            double dist = 0.0;
            for (int s = 0; s < n_samples; s++)
                dist += candidate_sse_one(
                    layers, cache,
                    om_ptr + (size_t)s * input_dim,
                    og_ptr + (size_t)s * n_classes,
                    s, cand.layer_idx, cand.i, cand.j,
                    buf.data(), max_width);
            if (dist < local_min || (dist == local_min && c < local_cand)) {
                local_min  = dist;
                local_cand = c;
            }
        }

#ifdef USE_OPENMP
        #pragma omp critical
#endif
        if (local_min < best_dist ||
            (local_min == best_dist && local_cand < best_cand)) {
            best_dist = local_min;
            best_cand = local_cand;
        }
    }

    const Candidate& best = candidates[best_cand];
    return py::make_tuple(best.layer_idx, best.i, best.j, best_dist);
}

PYBIND11_MODULE(prune_ext, m) {
    m.doc() = "Parallel pruning candidate search using OpenMP.";
    m.def(
        "find_best_candidate", &find_best_candidate,
        py::arg("layers"), py::arg("og_outputs"), py::arg("omega"),
        R"doc(
Find the single non-zero weight whose removal minimises SSE distance to og_outputs.

Args:
    layers:     list of (W, b, mask, activation_name) tuples (numpy float64).
                W shape [n_out, n_in], b [n_out], mask [n_out, n_in].
                Activation names: "relu", "tanh", "sigmoid", "linear".
                The last layer always applies log-softmax regardless of activation_name.
    og_outputs: precomputed original-network outputs, float64 [n_samples, n_classes].
                Compute once with: np.array(batched_predict(og_net, omega), dtype=np.float64)
    omega:      input sample matrix, float64 [n_samples, input_dim].

Returns:
    (layer_idx, i, j, min_distance) as a Python tuple.
    Ties on min_distance resolve to the first candidate in
    (layer, row, column) scan order, deterministically.
        )doc");
}
