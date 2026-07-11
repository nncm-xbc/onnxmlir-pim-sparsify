# Manifold-Based Neural Network Sparsification with an ARM Compiler

Post-training **sparsification** of MLPs grounded in manifold geometry, plus an
**optimizing compiler** that lowers the resulting sparse networks to ARMv7 assembly.
The target is inference on memory-bandwidth-bound hardware — in particular
Processing-In-Memory (PIM) devices. MSc thesis, Politecnico di Milano (HPC Engineering).

## Idea in one paragraph

An already-trained network can be made sparser **without retraining** if you measure how
much a weight matters by its effect on the network's *function*, not its magnitude. Parameter
space is isomorphic to ℝ^N, and the map from parameters to functions induces a semantic distance

$$d_{\mathscr W}(w, w') = \mathbb{E}_{x \sim \mathcal U(\Omega)}\big[\lVert \mathcal F(w)(x) - \mathcal F(w')(x)\rVert^2\big]$$

which is differentiable (ReLU nets are piecewise linear). The core algorithm greedily removes the
weight whose zeroing least increases $d_{\mathscr W}$, then runs a gradient **adjust** step on the
surviving weights (mask fixed) to compensate. Full derivation: [`docs/algorithm.md`](docs/algorithm.md).

## What's here

- **`sparsifier/`** — the sparsification framework. A shared driver (`runner.py`) runs the
  prune→adjust→log loop; each method supplies only its candidate-scoring rule:
  | module | method |
  |---|---|
  | `sparsifier.py` | manifold-distance greedy prune (the core method) + `prune`/`adjust`/`d` primitives |
  | `magnitude_sparsifier.py` | magnitude pruning baseline |
  | `kwon_sparsifier.py` | Kwon 2022 Fisher-importance |
  | `obd_sparsifier.py` / `obs_sparsifier.py` | Optimal Brain Damage / Surgeon |
  | `lazarevich_sparsifier.py` | manifold prune with Ω drawn from data |
  | `neuron_sparsifier.py` | structured (whole-neuron) pruning |
- **`backend/`** — IR generation, register/memory allocation (memory placement via simulated
  annealing), and ARMv7-A VFP assembly emission. Only non-zero weights emit instructions.
- **`prune_ext/`** — C++/pybind extension that accelerates candidate evaluation in the hot loop.
- **`mlp/`** — JAX MLP primitives (forward, training, parameter I/O, topology).
- **`scripts/`** — entry points: `dataeng.py` (build dataset), `train.py`, `test.py`; `scripts/run/` holds batch experiment shell drivers.
- **`experiments/`** — one JSON config per experiment (topology, data, train + sparsify params). Configs drive train and sparsify.
- **`benchmark/`** — correctness checks, strategy comparison, and the sparsifier profiler.
- **`visualize/`** — plotting scripts (shared helpers in `viz_common.py`) + a live training view.
- **`tests/`** — pytest suite. **`docs/`** — all documentation (start at [`docs/index.md`](docs/index.md)).

## Setup

```bash
pip install -r requirements.txt
pip install -e .        # puts the packages on the path; no PYTHONPATH juggling
```

JAX defaults to CPU; for GPU swap `jaxlib` → `jax[cuda12]` in `requirements.txt`.
The `prune_ext` extension is optional (pure-Python fallback exists); build it with `prune_ext/build.sh`.

## Pipeline

All commands run from the repo root. Trained params live in `artifacts/<name>/`; processed data in `data/`.

```bash
# 1. (optional) build a downsized MNIST split for a topology
python scripts/dataeng.py network_topology.csv

# 2. (optional) train a dense MLP from an experiment config
python scripts/train.py experiments/baseline.json

# 3. sparsify — pick any method module; config sets steps, Ω samples, adjust, etc.
python -m sparsifier.sparsifier experiments/baseline.json      # core manifold method
python -m sparsifier.magnitude_sparsifier experiments/e15_magnitude_full.json
#   → writes sparsified params + a per-step CSV log under artifacts/<name>/<method>_sparsified/

# 4. compile a sparse network to ARMv7 assembly (+ .onnx/.pt for cross-check)
python -m backend.compiler artifacts/<name>/sparsified out/model

# 5. evaluate a .pt / .onnx / params folder
python scripts/test.py <model> data/X_test_small.csv data/Y_test_small.csv
```

Batch drivers for the full experiment matrix are in `scripts/run/`. Plots:
`python visualize/plot_run.py artifacts/<name>/<method>_sparsified/<log>.csv`.

## Tests

```bash
python -m pytest -q
```

## Documentation

See [`docs/index.md`](docs/index.md) — algorithm & math, thesis roadmap, the open-work/experiment
backlog ([`docs/todo.md`](docs/todo.md)), design specs, notebooks, and the Lean stability proofs.
