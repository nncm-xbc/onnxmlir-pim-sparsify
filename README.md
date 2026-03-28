# Manifold-Based Neural Network Sparsification with an Optimizing ARM Compiler

Master's thesis. The project develops a **post-training sparsification algorithm** grounded in manifold geometry, together with an **optimizing compiler** that translates the resulting sparse networks into ARM assembly code. The target application is inference on resource-constrained architectures where memory bandwidth is the primary bottleneck, including Processing-In-Memory (PIM) devices.

---

## Overview

Modern neural network pruning methods are largely training-time: they guide the training procedure toward sparse solutions. This work takes a different approach and asks how an already-trained network can be made sparser **without retraining**, while preserving its function as closely as possible.

The answer is grounded in a geometric observation. The space of network parameters $\mathscr{W}$ is isomorphic to $\mathbb{R}^N$, and the map $\mathcal{F}: \mathscr{W} \to \mathscr{F}$ that converts parameters into functions induces a natural **semantic distance** between nearby networks:

$$
d_{\mathscr{W}}(w, w') = \mathbb{E}_{x \sim \mathcal{U}(\Omega)}\left[\|\mathcal{F}(w)(x) - \mathcal{F}(w')(x)\|^2\right]
$$

Because $\mathcal{F}$ is differentiable almost everywhere (ReLU networks are piecewise linear), this distance is differentiable with respect to the parameters. The sparsification algorithm exploits this: at each step it removes the weight whose zeroing minimally perturbs the function, then uses gradient descent on $d_{\mathscr{W}}$ to adjust the remaining weights to compensate. The full mathematical development is in [`Docs/Sparsifying algorithm, mathematics and explanation.md`](Docs/Sparsifying%20algorithm%2C%20mathematics%20and%20explanation.md).

---

## What has been implemented

### Sparsification algorithm (`sparsifier/`)

A greedy prune-and-adjust loop operating on a trained dense MLP:

- **Prune step** — evaluates $d_{\mathscr{W}}$ for every candidate single-weight removal and selects the one with minimal impact. Cost is $O(N^{(k)} \cdot B \cdot C)$ per iteration where $N^{(k)}$ is the current non-zero count, $B$ the Monte Carlo batch size, and $C$ the cost of one forward pass.
- **Adjust step** — runs gradient descent on $d_{\mathscr{W}}$ with the updated sparsity mask fixed, using an adaptive step size. The mask enforces the sparsity constraint automatically: masked parameters receive zero gradient and are never updated.
- Implemented in JAX for vectorised inference and automatic differentiation.
- Validated on MNIST: a three-layer MLP retains **~95% test accuracy** after sparsification on a 14×14 input.

### Compiler (`backend/`)

An optimising compiler that takes a sparse MLP (as weight matrices) and produces ARM assembly code. The compilation pipeline has two main phases:

1. **IR generation** — the network is lowered to a lightweight tree-based intermediate representation. Only non-zero weights generate instructions, directly exploiting the sparsity.

2. **Register and memory allocation** — a two-stage optimisation targeting minimal data movement:
   - *Register allocation* tracks the lifetime of every intermediate value and assigns ARM registers to minimise spills.
   - *Memory allocation* places values that must cross layer boundaries using a moving-window density metric, optimised by a **simulated annealing** procedure. The goal is to keep the number of memory transfers $\mathcal{O}(R)$ where $R$ is the register count — independent of the network width.

The compiler outputs an ARM assembly file and two interface descriptors (input and output masks) specifying how a peripheral driver should handle memory and registers to feed data into and read results from the network. The output has been validated with the **Unicorn ARM emulator**.

### Formal proofs (`Docs/Proofs/`)

A Lean 4 formalisation is in progress, targeting three theoretical results:

- **Sparsification stability** — bounding $d_{\mathscr{F}}$ between the sparsified outputs of two close initialisations after $K$ iterations.
- **Stupidity point** — the existence of a critical sparsity level $s^*$ beyond which the network cannot maintain its function regardless of adjustment.
- **Weight pruning vs. neuron pruning** — a formal comparison of the two strategies within the manifold framework.

Proof strategy and background are documented in [`Docs/Proofs/Sparsification_Stability_Approaches.md`](Docs/Proofs/Sparsification_Stability_Approaches.md).

---

## Repository structure

```
mlp/            Pure JAX MLP primitives (forward pass, training, parameter I/O)
sparsifier/     Manifold-based sparsification algorithm
backend/        IR, register/memory allocation, ARM code generation
scripts/        Entry-point scripts: dataeng.py, train.py, test.py
data/           MNIST source CSVs and processed dataset splits
artifacts/      Trained parameters, sparsified parameters, compiled outputs
Docs/
  Sparsifying algorithm, mathematics and explanation.md   Mathematical foundations
  Interfaces Explanation.pdf                              Memory allocation spec
  Proofs/                                                 Lean 4 formalisation
  Notebooks/                                              Development notebooks
```

---

## Running the pipeline

All commands are run from the repository root. Pre-trained parameters and a processed dataset are already present in `artifacts/` and `data/` respectively, so steps 1 and 2 are optional.

**1. Prepare the dataset** *(optional — only needed to change the network topology)*

```bash
python3 scripts/dataeng.py network_topology.csv
```

Reads `data/mnist_train.csv` and `data/mnist_test.csv`, resizes images to the resolution specified in `data/network_topology.csv`, and writes the processed splits back to `data/`. The topology file stores the first-layer size as its square root (e.g. `14` for a 196-input layer) to enforce a square input shape.

**2. Train a dense network** *(optional)*

```bash
python3 scripts/train.py <output_folder> data/network_topology.csv
```

Trains a dense MLP and saves weights and biases (`W_i.npy`, `b_i.npy`) to `<output_folder>`.

**3. Sparsify**

```bash
python3 -m sparsifier.sparsifier <params_folder> data/X_test_small.csv data/Y_test_small.csv
```

Runs the prune-and-adjust loop for 500 iterations. The validation set is printed at each step for monitoring only — it does not influence the pruning. Sparsified parameters are saved to `<params_folder>/sparsified/`.

**4. Compile**

```bash
python3 -m backend.compiler <params_folder>/sparsified <output_name>
```

Produces `<output_name>` (ARM assembly), `<output_name>_exe` (executable form), and `<output_name>.onnx` / `<output_name>.pt` (for cross-validation).

**5. Evaluate**

```bash
python3 scripts/test.py <model_file> data/X_test_small.csv data/Y_test_small.csv
```

Accepts `.pt`, `.onnx`, or a compiled parameter folder.

---

## Dependencies

```bash
pip install -r requirements.txt
```

See `requirements.txt` for the full list. JAX defaults to CPU; for GPU replace `jaxlib` with `jax[cuda12]`.
