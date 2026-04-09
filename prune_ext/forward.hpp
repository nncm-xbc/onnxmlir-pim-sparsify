// prune_ext/forward.hpp
#pragma once
#include <cmath>
#include <vector>
#include "activations.hpp"

struct LayerData {
    const float* W;    // row-major [n_out, n_in]
    const float* b;    // [n_out]
    const float* mask; // row-major [n_out, n_in]
    int n_out, n_in;
    ActivFn activation;
    bool is_last;      // log-softmax instead of activation f
};

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
// layers:            network layer descriptors (read-only, shared across threads)
// input:             sample row, length layers[0].n_in
// og_out_row:        precomputed original-network output for this sample,
//                    length layers.back().n_out
// zero_layer/i/j:    candidate weight index to substitute with 0.f
// buf:               thread-local buffer, must be >= max_width * 2 floats
// max_width:         max n_out across all layers (caller computes once)
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
