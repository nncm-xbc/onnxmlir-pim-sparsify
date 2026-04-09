// prune_ext/activations.hpp
#pragma once
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

using ActivFn = float (*)(float);

// Returns the activation function pointer for the given name.
inline ActivFn get_activation(const std::string& name) {
    static const std::unordered_map<std::string, ActivFn> reg = {
        {"relu",    [](float x) -> float { return x > 0.f ? x : 0.f; }},
        {"tanh",    [](float x) -> float { return std::tanh(x); }},
        {"sigmoid", [](float x) -> float { return 1.f / (1.f + std::exp(-x)); }},
        {"linear",  [](float x) -> float { return x; }},
        // add new activations here
    };
    auto it = reg.find(name);
    if (it == reg.end())
        throw std::invalid_argument("Unknown activation function: \"" + name + "\"");
    return it->second;
}
