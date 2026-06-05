// prune_ext/activations.hpp
#pragma once
#include <cmath>
#include <stdexcept>
#include <string>
#include <unordered_map>

using ActivFn = double (*)(double);

// Returns the activation function pointer for the given name.
inline ActivFn get_activation(const std::string& name) {
    static const std::unordered_map<std::string, ActivFn> reg = {
        {"relu",    [](double x) -> double { return x > 0.0 ? x : 0.0; }},
        {"tanh",    [](double x) -> double { return std::tanh(x); }},
        {"sigmoid", [](double x) -> double { return 1.0 / (1.0 + std::exp(-x)); }},
        {"linear",  [](double x) -> double { return x; }},
        // add new activations here
    };
    auto it = reg.find(name);
    if (it == reg.end())
        throw std::invalid_argument("Unknown activation function: \"" + name + "\"");
    return it->second;
}
