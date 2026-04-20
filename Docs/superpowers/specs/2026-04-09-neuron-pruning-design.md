# Neuron Pruning — Design Spec
**Date:** 2026-04-09
**Branch:** `neuron_pruning`
**Status:** Approved

---

## Goal

Extend the existing manifold-based sparsification algorithm from weight pruning (zeroing individual `W[i,j]` entries) to neuron pruning (zeroing entire hidden neurons). Implement as a standalone `sparsifier/neuron_sparsifier.py` for empirical comparison against weight pruning. No changes to the existing weight pruning code.

---

## Scope

- **In scope:** structured neuron removal for hidden layers, same `d_W` scoring, same `adjust` step, standalone script, CSV logging, experiment config
- **Out of scope:** Lean 4 proof work, benchmark harness integration (deferred pending meaningful results), input/output layer pruning, `prune_ext` C++ acceleration (initially)

---

## Data Structures

`Layer(W, b, mask)` is **unchanged**. Neuron pruning is expressed entirely through the existing mask:

Pruning hidden neuron `i` in layer `l` sets:
- `mask[l][i, :] = 0` — entire output row (neuron's outgoing weights)
- `mask[l+1][:, i] = 0` — corresponding column in next layer (downstream dependency)
- `W[l][i, :] = 0`, `b[l][i] = 0` — actual values zeroed
- `W[l+1][:, i] = 0`

The bias `b[l][i]` is zeroed because with `W[l][i,:] = 0` and `W[l+1][:,i] = 0`, neuron `i`'s output has no downstream effect — its bias is a dead constant. Zeroing it is mathematically equivalent and cleaner.

**No changes to `mlp/mlp.py`.** The `predict` function already applies `lax.stop_gradient(mask) * W` elementwise; a fully-zeroed row/column is automatically dead.

---

## New NamedTuple

```python
class NeuronPruneMeta(NamedTuple):
    layer_idx: int       # which hidden layer (absolute index into net)
    neuron_idx: int      # which neuron i within that layer
    distanza: float      # d(og_net, probe_net, omega) at minimum
    prune_time_s: float
    adjust_time_s: float
```

---

## Algorithm: `prune_neuron(net, og_net, omega, doAdjust=True)`

Direct analogue of `prune()` in `sparsifier.py`:

1. Clone `net` into `probe_net`
2. Iterate over **hidden layers only** — indices `1` to `len(net) - 2` inclusive
3. For each layer `l`, iterate over neurons `i` (rows of `W[l]`)
4. Skip neuron `i` if `W[l][i, :]` is already all-zero (already pruned)
5. Temporarily zero `probe_net[l].W[i,:]`, `probe_net[l].mask[i,:]`, `probe_net[l].b[i]`, `probe_net[l+1].W[:,i]`, `probe_net[l+1].mask[:,i]`
6. Evaluate `d(og_net, probe_net, omega)` — **same `d` function, imported from `sparsifier.py`**
7. Restore all modified entries
8. Track `(l, i, distanza)` minimum
9. Apply winning neuron permanently
10. If `doAdjust and distanza > 0`: run `adjust(probe_net, og_net, omega)` — **same `adjust` function, imported from `sparsifier.py`**
11. Return `(net, NeuronPruneMeta(...))`

### Why `d` and `adjust` are unchanged

`d` is a purely functional metric — it measures output divergence over Ω regardless of pruning granularity. Neuron pruning produces a more constrained parameter point (codimension O(n_in + n_out) rather than 1), but the scoring question is identical: *how much does this network deviate from the original on inputs?*

`adjust` minimises `d(current, og, omega)` by gradient descent subject to the mask constraint. `lax.stop_gradient(mask)` in `predict` ensures pruned entries receive zero gradient automatically, so they never move — this works for a full zeroed row+column exactly as it does for a single zeroed weight.

### Candidate selection scope

- **Hidden layers only:** `net[1]` through `net[len(net)-2]`
- Input neurons (columns of `W[0]`) and output neurons (rows of `W[-1]`) are never candidates

---

## Imports

`make_omega`, `clone_network`, `d`, `d_grad`, `adjust` are **imported from `sparsifier.py`** — no duplication.

---

## Config Format

```json
{
  "name": "mnist_784_256_128_10",
  "sparsify": {
    "mode": "neuron",
    "steps": 50,
    "omega_samples": 10000,
    "do_adjust": true,
    "checkpoint_every": 10
  },
  "data": {
    "x_test": "data/x_test.csv",
    "y_test": "data/y_test.csv"
  }
}
```

Output goes to `artifacts/<name>/neuron_sparsified/` — parallel to `sparsified/`.

---

## Log CSV: `neuron_sparsification_log.csv`

| column | meaning |
|---|---|
| `step` | pruning iteration |
| `neurons_pruned` | cumulative neurons removed |
| `total_neurons` | total hidden neurons at initialisation |
| `neuron_sparsity` | `neurons_pruned / total_neurons` |
| `val_acc` | validation accuracy |
| `d_manifold` | `d(net, og_net, omega)` at step start |
| `d_W` | Euclidean weight change from adjust step |
| `prune_time_s` | time for candidate search |
| `adjust_time_s` | time for adjust step |
| `candidate_layer` | absolute layer index of pruned neuron |
| `candidate_neuron` | neuron index `i` within that layer |
| `layer_<l>_neurons` | per-layer surviving neuron count (one col per hidden layer) |

---

## Invocation

```bash
python3 -m sparsifier.neuron_sparsifier experiments/mnist_neuron.json
```

---

## File Changes

| file | change |
|---|---|
| `sparsifier/neuron_sparsifier.py` | **new** — full standalone script |
| `experiments/mnist_neuron.json` | **new** — example config |
| `mlp/mlp.py` | none |
| `sparsifier/sparsifier.py` | none |
| `benchmark/` | none (deferred) |

---

## Future Work (deferred)

- C++ `prune_ext` acceleration for the neuron candidate loop
- Benchmark harness integration (`"strategy": "neuron"`) if initial results are promising
- Lean 4 formalisation in `Docs/Proofs/weightVSneuron.lean`
- MoM comparison plots once multiple runs exist
