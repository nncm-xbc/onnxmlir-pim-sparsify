---
title: Lazarevich 2021 Baseline — Data-Driven Omega
date: 2026-04-27
status: approved
---

# Lazarevich Baseline Design Spec

## Purpose

Implement the Lazarevich et al. 2021 baseline (`lazarevich2021posttrainingdeepneuralnetwork`) for direct comparison with the thesis method. The key claim to test: does using **real unlabelled calibration data** as the Monte Carlo sample Ω matter, vs using **uniform random noise** as the thesis does?

| Axis | Thesis (manifold) | Lazarevich baseline |
|---|---|---|
| Selection criterion | argmin d_W (exhaustive) | argmin d_W (exhaustive) |
| Adjustment | gradient descent on d_W | gradient descent on d_W |
| Ω distribution | Uniform(0,255)^d | Real MNIST images (x_test) |

Everything is identical except the source of Ω. Any difference in accuracy/sparsity curves is attributable solely to the data distribution used for the manifold distance estimate.

## New file

`sparsifier/lazarevich_sparsifier.py`

Imports `prune`, `adjust`, `clone_network`, `d` from `sparsifier.sparsifier`. The only substantive difference from `sparsifier.sparsifier.main()` is how omega is constructed:

- Manifold: `omega = make_omega(og_net, n_samples=sp['omega_samples'])` — uniform random pixel noise
- Lazarevich: `omega = x_test[:min(sp['omega_samples'], len(x_test))]` — real MNIST images (float32, normalised to [0,255])

Output folder: `artifacts/<name>/lazarevich_sparsified/`
Log file: `lazarevich_sparsification_log.csv` — identical columns to manifold log.
Config: `experiments/e08_omega_data.json` (existing stub, description updated).

Run via: `python -m sparsifier.lazarevich_sparsifier experiments/e08_omega_data.json`

## Out of scope
- Layer-wise reconstruction objectives (Lazarevich's actual internal mechanism)
- Changes to prune() or adjust()
