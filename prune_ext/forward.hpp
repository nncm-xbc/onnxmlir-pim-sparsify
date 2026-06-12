// prune_ext/forward.hpp
#pragma once
#include <cmath>
#include <cstddef>
#include <vector>
#include "activations.hpp"

struct LayerData {
    const double* W;    // row-major [n_out, n_in]
    const double* b;    // [n_out]
    const double* mask; // row-major [n_out, n_in]
    int n_out, n_in;
    ActivFn activation;
    bool is_last;      // log-softmax instead of activation f
};

inline void log_softmax_inplace(double* buf, int n) {
    double max_val = buf[0];
    for (int k = 1; k < n; k++)
        if (buf[k] > max_val) max_val = buf[k];
    double sum_exp = 0.0;
    for (int k = 0; k < n; k++) sum_exp += std::exp(buf[k] - max_val);
    double log_sum = max_val + std::log(sum_exp);
    for (int k = 0; k < n; k++) buf[k] -= log_sum;
}

// Pre-activation of unit `i`: b[i] + Σ_k mask[i,k]·W[i,k]·src[k], with the
// weight at column `zero_j` treated as 0.0 (pass -1 for no substitution).
// The masked multiply-add sequence is identical with and without the
// substitution (adding ±0.0 leaves the accumulation unchanged), so cached
// and candidate evaluations stay bit-compatible.
inline double row_preact(const LayerData& ld, int i, const double* src, int zero_j) {
    const double* Wrow = ld.W    + (size_t)i * ld.n_in;
    const double* Mrow = ld.mask + (size_t)i * ld.n_in;
    double acc = ld.b[i];
    for (int k = 0; k < ld.n_in; k++) {
        double w = Wrow[k];
        if (k == zero_j) w = 0.0;
        acc += Mrow[k] * w * src[k];
    }
    return acc;
}

inline double sse_row(const double* out, const double* og_out_row, int n) {
    double sse = 0.0;
    for (int k = 0; k < n; k++) {
        double diff = out[k] - og_out_row[k];
        sse += diff * diff;
    }
    return sse;
}

// Per-sample forward-pass cache of the *unmodified* candidate network:
// pre-activations z, post-activations a (post log-softmax for the last
// layer) for every layer, plus each sample's baseline SSE vs og_outputs.
// Computed once per find_best_candidate call; lets candidate evaluation
// reuse everything upstream of the zeroed weight.
struct ForwardCache {
    std::vector<std::vector<double>> z;  // [layer][sample * n_out + o]
    std::vector<std::vector<double>> a;  // [layer][sample * n_out + o]
    std::vector<double> base_sse;        // [sample]
};

// Fill the cache row for one sample. og_out_row is the precomputed
// original-network output for this sample (length layers.back().n_out).
inline void cache_forward_one(
    const std::vector<LayerData>& layers,
    const double* input,
    const double* og_out_row,
    int s,
    ForwardCache& cache
) {
    const double* src = input;
    for (size_t li = 0; li < layers.size(); li++) {
        const LayerData& ld = layers[li];
        double* z_row = cache.z[li].data() + (size_t)s * ld.n_out;
        double* a_row = cache.a[li].data() + (size_t)s * ld.n_out;
        for (int o = 0; o < ld.n_out; o++)
            z_row[o] = row_preact(ld, o, src, -1);
        if (ld.is_last) {
            for (int o = 0; o < ld.n_out; o++) a_row[o] = z_row[o];
            log_softmax_inplace(a_row, ld.n_out);
        } else {
            for (int o = 0; o < ld.n_out; o++) a_row[o] = ld.activation(z_row[o]);
        }
        src = a_row;
    }
    const LayerData& last = layers.back();
    cache.base_sse[s] = sse_row(
        cache.a.back().data() + (size_t)s * last.n_out, og_out_row, last.n_out);
}

// SSE contribution of one sample with weight (zero_layer, zero_i, zero_j)
// treated as zero. Reuses the cache: layers before the candidate are never
// recomputed; only row zero_i of the candidate layer plus the downstream
// layers are. Two exact shortcuts return the cached baseline SSE outright:
//   * the candidate weight's input is 0.0 (zeroing it changes nothing), or
//   * the candidate unit's post-activation is unchanged (e.g. dead ReLU).
//
// buf: thread-local scratch, >= 2 * max_width doubles.
inline double candidate_sse_one(
    const std::vector<LayerData>& layers,
    const ForwardCache& cache,
    const double* input,
    const double* og_out_row,
    int s,
    int zero_layer, int zero_i, int zero_j,
    double* buf,
    int max_width
) {
    const LayerData& ld = layers[zero_layer];
    const double* src = (zero_layer == 0)
        ? input
        : cache.a[zero_layer - 1].data() + (size_t)s * layers[zero_layer - 1].n_out;

    if (src[zero_j] == 0.0)
        return cache.base_sse[s];

    double zi = row_preact(ld, zero_i, src, zero_j);

    if (ld.is_last) {
        const double* z_row = cache.z[zero_layer].data() + (size_t)s * ld.n_out;
        for (int k = 0; k < ld.n_out; k++) buf[k] = z_row[k];
        buf[zero_i] = zi;
        log_softmax_inplace(buf, ld.n_out);
        return sse_row(buf, og_out_row, ld.n_out);
    }

    double ai = ld.activation(zi);
    const double* a_row = cache.a[zero_layer].data() + (size_t)s * ld.n_out;
    if (ai == a_row[zero_i])
        return cache.base_sse[s];

    double* dst = buf + (zero_layer % 2) * max_width;  // keep ping-pong parity
    for (int k = 0; k < ld.n_out; k++) dst[k] = a_row[k];
    dst[zero_i] = ai;

    const double* cur = dst;
    for (size_t li = zero_layer + 1; li < layers.size(); li++) {
        const LayerData& nl = layers[li];
        double* nxt = buf + (li % 2) * max_width;
        for (int o = 0; o < nl.n_out; o++)
            nxt[o] = row_preact(nl, o, cur, -1);
        if (nl.is_last)
            log_softmax_inplace(nxt, nl.n_out);
        else
            for (int o = 0; o < nl.n_out; o++) nxt[o] = nl.activation(nxt[o]);
        cur = nxt;
    }
    return sse_row(cur, og_out_row, layers.back().n_out);
}
