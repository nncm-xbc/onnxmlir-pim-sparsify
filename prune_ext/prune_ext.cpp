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
        ld.W          = static_cast<const float*>(W_info.ptr);
        ld.b          = static_cast<const float*>(b_info.ptr);
        ld.mask       = static_cast<const float*>(mask_info.ptr);
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
