---
title: "Thesis Sparsifier to onnx-mlir PIM Backend — Integration Report"
date: 2026-08-23
type: report
status: draft
project: post-training-sparsification
tags:
  - thesis
  - pim
  - onnx-mlir
  - sparsification
  - crossbar
  - hardware-mapping
  - experiment
aliases:
  - PIM integration report
  - Where is the PIM edge
related:
  - "[[Manifold sparsification]]"
  - "[[onnx-mlir PIM backend]]"
  - "[[Crossbar cost model]]"
  - "[[Neuron pruning]]"
  - "[[Activation bit-width]]"
sources:
  - onnxmlir-pim-sparsify (thesis repo)
  - onnx-mlir-priv (PIM fork, branch `pim`)
---

# Thesis Sparsifier to onnx-mlir PIM Backend — Integration Report

> [!warning] Status of the numbers
> This report did not compile onnx-mlir and did not run the simulator.
> The `pimcomp` and `pimsim-nn` submodules are empty in the working tree.
> All hardware numbers come from two sources. First, the code of both repos.
> Second, a Python model that repeats the compiler and simulator cost formulas.
> The accuracy numbers come from real runs on CPU. Check the numbers before you cite them.

## 1. Purpose

This report answers one question. How can the thesis help the onnx-mlir PIM
backend, and where is a real advantage?

The thesis makes a neural network sparse after training. It removes weights that
change the network function the least. It measures the change with a distance in
function space (the "manifold distance"). The target hardware is a
Processing-In-Memory (PIM) accelerator with analog crossbars.

The onnx-mlir fork (`onnx-mlir-priv`, branch `pim`) is a compiler. It lowers an
ONNX model to instructions for a ReRAM crossbar PIM device. It also has a
cycle-accurate simulator (`pimsim-nn`).

## 2. Method and limits

The work used four steps.

1. Read the code of both repos.
2. Run a multi-agent review. The review made 12 integration ideas and tested
   each one against the code.
3. Build one new tool in the thesis repo.
4. Run one control experiment on real data.

Limits of the evidence:

- The compiler and simulator were not built or run.
- The experiment used one dense network (seed 0) and one small data set.
- The data set is a 14x14 MNIST subset.
- The runs used the CPU, because the GPU stalls (see the branch handover).

## 3. The two code bases

**Thesis repo — what it produces.**

- The core method removes single weights. The result is *unstructured* sparsity.
  The zeros sit at any position inside a dense matrix.
- Separate scripts remove whole neurons (`sparsifier/neuron_*.py`). The result is
  *structured* sparsity. Each removed neuron zeroes one full row of `W[l]`, the
  bias, and one full column of `W[l+1]`.
- Both methods write dense `W_i.npy` files. The zeros stay in the files. The
  shape does not change.
- The compiler backend (`backend/compiler.py`) also exports the network to ONNX.
  It emits `Gemm` + `Relu` operations.

**PIM fork — how it maps a matrix to hardware.**

- The lowering path is: ONNX → Spatial dialect → PIM dialect → per-core JSON.
- One `Gemm` becomes a grid of square crossbar tiles.
- The simulator charges a fixed cost for each matrix-vector multiply (MVM).

## 4. Main finding — the PIM cost depends only on the shape

The compiler computes the hardware cost from the tensor **shape** only. It never
reads the weight values.

For each `Gemm`, the compiler divides the weight matrix into square tiles. Each
tile has the size `crossbar_size` by `crossbar_size`. The compiler counts the
tiles with a ceiling division:

```
input_tiles  = ceil(K / crossbar_size)
output_tiles = ceil(M / crossbar_size)
crossbars    = input_tiles * output_tiles
mvmuls       = input_tiles * output_tiles
cores        = ceil(input_tiles / crossbars_per_core) * output_tiles
```

Two facts confirm this:

- The command `grep -rniE "sparse|sparsit" src/Accelerators/PIM/` finds nothing.
- The simulator never receives the weight values. It charges the same cost for a
  full tile, a 1%-full tile, and an all-zero tile.

One crossbar of size 128x128 at 8-bit costs about **1305 cycles and 24490 pJ**.
This value does not change with the weight values.

## 5. Result 1 — unstructured pruning gives no hardware benefit

The core method makes unstructured sparsity. On a crossbar this saves nothing.

The reason is simple. The compiler still writes every tile. The device still
programs every cell. The device still reads every ADC column. A single zero
inside a tile changes no cost.

A whole 128x128 tile becomes all-zero only if all 16384 cells are zero. At 90%
unstructured sparsity the chance is about `0.9^16384`, which is near zero.

> [!important] Statement
> The thesis core method (unstructured manifold pruning) gives a **zero** hardware
> gain on this PIM backend. The gain was real only on the thesis ARM backend,
> which skips each zero weight at the instruction level.

## 6. Result 2 — the pruning criterion cannot change the hardware cost

The crossbar count depends on the layer **width** through the ceiling division.
It does not depend on *which* neuron you remove.

So the criterion does not matter to the hardware. Manifold, magnitude, OBD, and
OBS all give the same hardware, if they remove the same number of neurons from
the same layers. The advantage of the thesis method must come from **accuracy at
a given shape**, not from a special hardware effect.

## 7. The correct target — prune what the hardware charges for

The greedy search of the thesis is general. It answers one question at each step:
which unit can I remove now at the least cost to the network function?

Weights were the wrong unit. The hardware does not charge per weight. Point the
search at the units that the hardware **does** charge for.

| Hardware unit | Set by | Used today? | Lever for the thesis |
|---|---|---|---|
| Crossbar tile | `ceil(width / crossbar_size)` | no | remove whole neurons to a tile boundary |
| Activation bit-width (`ibiw`) | `ceil(ibiw / dac_resolution)` | **no** — hardcoded to 8 | choose bits per layer with the same search |
| ADC column stripe (16 columns) | `ceil(128 / 16) = 8` | no | remove aligned 16-column stripes |
| Input feature (row of `W[0]`) | `ceil(K / crossbar_size)` | no | drop dead input pixels |

## 8. New tool — `backend/compact.py`

**Problem.** The neuron sparsifiers zero a neuron but keep the shape. The
compiler reads the old shape. So it allocates the old hardware. The structured
sparsity is lost at the shape boundary.

**Solution.** `compact.py` removes the dead hidden neurons. It writes a smaller
dense network. The new network computes the **same** function, bit for bit.

Rule for safe removal: remove a hidden neuron only if its fan-out column in the
next `W` is all-zero. Then nothing reads its output. The removal cannot change
the result, for any activation. The neuron sparsifiers zero both sides, so every
pruned neuron is safe.

The tool also has `crossbar_cost()`. This function repeats the compiler formula.

**Checks that passed:**

- Self-check: `[8,6,5,3] → [8,5,4,3]`, output equal bit for bit, crossbars 10 → 7.
- Real net (e11): `[196,10,10,10] → [196,3,2,10]`. At `crossbar_size=128` the
  crossbars stay 4 → 4. Reason: 10 and 3 both need one 128-tile. This small net
  sits below the crossbar granularity. This is honest evidence for the null case.
- End-to-end export with **no compiler change**: compact folder → the existing
  `Program` and `torch_to_onnx` → ONNX with `Gemm` weight shapes `[3,196]`,
  `[2,3]`, `[10,2]`. The PIM `Gemm.cpp` will tile exactly these shapes.

## 9. Experiment — prune to the crossbar boundary

### 9.1 Setup

- Dense network: `[196,200,200,10]`, trained, seed 0.
- Dense test accuracy: **0.969**.
- Crossbar size: 128. Dense hardware: **10 crossbars, 5 cores**.
- Prune each hidden layer toward the 128 boundary, then compact.
- Compare three methods at each crossbar count:
  - manifold neuron pruning with noise Omega (the default `make_omega`, uniform
    pixels 0–255);
  - manifold neuron pruning with data Omega (real training images);
  - magnitude neuron pruning (mean of `|w|`, no Omega);
  - and a from-scratch network of the same shape as a control.

### 9.2 Results — accuracy at each crossbar count

| Crossbars | Method | Shape | Test accuracy |
|---|---|---|---|
| 10 | dense (reference) | `[196,200,200,10]` | 0.969 |
| 7 | manifold, data Omega | `[196,146,127,10]` | **0.969** |
| 7 | from-scratch control | `[196,136,127,10]` | 0.962 |
| 6 | magnitude | `[196,124,144,10]` | **0.966** |
| 6 | from-scratch control | `[196,124,144,10]` | 0.959 |
| 4 | manifold, data Omega | `[196,128,106,10]` | **0.966** |
| 4 | manifold, noise Omega | `[196,128,122,10]` | **0.679** |
| 4 | from-scratch control | `[196,128,128,10]` | 0.962 |

### 9.3 The Omega finding — this is the headline

The input sample Omega decides the result of structured pruning.

- With **noise** Omega the manifold method fails. Accuracy is already 0.878 at 10
  crossbars (width `[163,147]`). It falls to 0.679 at 4 crossbars.
- With **data** Omega the manifold method holds. Accuracy is 0.966 at 4 crossbars.
- The change from noise to data is a **+29 point** swing at the same hardware.

The cause is clear. The default Omega is uniform pixel noise. "The least change
of the function on noise" is not "the least loss of accuracy on digits". A neuron
that matters little for noise can matter a lot for real digits.

> [!important] Statement
> For structured pruning, the default `make_omega` (noise) is harmful. Data Omega
> (real images, the Lazarevich variant) is required, not optional.

Two more results:

- **Manifold with data Omega beats magnitude on hardware.** It reaches 4
  crossbars at 0.966. Magnitude stopped at 6 crossbars at 0.966. So the thesis
  criterion, with data Omega, finds a tighter hardware shape at equal accuracy.
- **Pruning does not beat a from-scratch network of the same shape.** A
  from-scratch `[196,128,128,10]` reaches 0.962 at 4 crossbars, near the dense
  0.969. So the real baseline is "train the crossbar-shaped network", not "the
  dense network". Manifold with data Omega **ties** this baseline, but it needs
  no retraining and no labels. This is the correct claim for a post-training
  method.

### 9.4 Hardware numbers

MVM dynamic energy uses 24490 pJ per 128x128 crossbar at 8-bit.

| Shape | Crossbars | Cores | MVMs | MVM energy (pJ) | Gain vs dense |
|---|---|---|---|---|---|
| `[196,200,200,10]` | 10 | 5 | 10 | 244900 | 1.00x |
| `[196,124,144,10]` | 6 | 4 | 6 | 146940 | 1.67x |
| `[196,128,106,10]` | 4 | 3 | 4 | 97960 | **2.50x** |

So structured pruning plus compaction gives a real cut: 10 → 4 crossbars, and a
2.5x cut of MVM dynamic energy. The latency gain is smaller, because the cores
run in parallel.

## 10. Ranked integration paths

The review made 12 ideas. It tested each idea against the code. The result was:
0 clear wins, 6 conditional wins, 2 marginal, 4 dead ends. The paths below are
the useful ones, in order.

### P1 — Neuron pruning to the crossbar boundary, plus compaction (do first)

- Effort: medium. Risk: low. Compiler change: none.
- Prune whole neurons until each layer width is a multiple of `crossbar_size`.
  Then compact. The smaller shape makes onnx-mlir ask for fewer crossbars.
- Proven this session: 10 → 4 crossbars, accuracy 0.966 (with data Omega).
- Condition: use data Omega. Noise Omega fails.

### P2 — Activation bit-width by the same search (largest free lever)

- Effort: medium. Risk: low.
- `PimCodeGen.cpp` hardcodes `mbiw = 8`. It never emits `setbw`. So every MVM
  runs at 8-bit.
- The MVM cost is linear in the bit-width. 8 → 4 bit gives about 2x. 8 → 2 bit
  gives about 4x. The ADC term (94% of the MVM energy) sits inside the multiplier.
- Use the same greedy `d_W` search to pick the bits per layer.
- This lever is independent of sparsity. It multiplies with P1.

### P3 — Harvest the dead input pixels (nearly free)

- Effort: small. Risk: low. Compiler change: none.
- The data-Omega runs already kill many input pixels. Drop the all-zero columns
  of `W[0]` and shrink the input tile.
- Note the limit. The gain exists because 196 is just above 128. At
  `crossbar_size = 256` the gain is zero.

### P4 — Block (SxS) pruning plus tile removal in the compiler

- Effort: large. Risk: medium.
- Make the search unit an `S x S` weight block. Teach `Gemm.cpp` to skip an
  all-zero tile. This removes interior holes that compaction cannot reach.
- Condition: block magnitude may score blocks as well as the manifold distance,
  at much lower cost. Run that control.

### P5 — ADC-stripe pruning (sharpest idea, needs a simulator change)

- Effort: large. Risk: high.
- The ADC works on 16-column groups. `adc_times = ceil(128/16) = 8` dominates the
  MVM cost. So the real quantum is a 16-column stripe, not a 128x128 tile.
- Prune aligned 16-column stripes. This needs a small change in the simulator to
  model active columns. Note also: N:M sparsity gives no gain here; 16-column
  stripes do.

## 11. Paths that do not work (dead ends)

Do not spend time on these. The review confirmed each one against the code.

- **Port the ARM annealer to core-to-mesh placement.** The energy table is a
  constant (46 pJ per flit). No placement can change the energy. The latency
  effect is below 0.4%.
- **Weight bit-width as a column-packing dimension.** The value saturates at 1.0x.
  One crossbar column is one output line, so a weight needs at least one cell.
- **Channel pruning on the benchmarked CNNs.** The `ceil(C/S)` step works against
  the `H*W` multiplier. The large-`H*W` layers already use one tile.
- **Sparsity as analog-noise protection.** The measured effect is **negative**.
  The `adjust()` step makes the kept weights larger. On a fixed-range analog cell
  this lowers the signal-to-noise ratio. At 55% sparsity and noise 0.10, accuracy
  was 0.449, against 0.568 for the dense network. This negative result is worth a
  paragraph in the thesis.

## 12. Defects found in onnx-mlir-priv

Report these to the fork owners. They are independent of the thesis.

1. **Wrong energy accounting.** `PimCodeGen.cpp:582` writes the group *index* where
   the simulator expects the crossbar *count*. So group 0 is billed zero energy.
   The MVM energy is distorted by a factor of `(W-1)/2`. Fix this before any
   energy claim. Any existing energy figure uses a distorted baseline.
2. **Wrong bit-width tag.** `mbiw` is hardcoded to 8, but the tiling uses
   `cell_precision = 2`. The instruction does not match the tiling.
3. **Truncated NoC table.** The table zeroes 27% to 98% of the transfers in the
   benchmark runs.

## 13. Recommendations and next steps

1. **Make data Omega the default for structured pruning.** This one change turns a
   failing result into a working result. It is the highest-value edit.
2. **Add the activation bit-width lever (P2).** It gives 2x to 4x and is
   independent of sparsity.
3. **Repeat the control with more seeds and more data sets.** The 29-point Omega
   gap is large, but confirm it.
4. **Fix the `array_group_map` energy defect** before any energy headline.
5. **Report the staircase figure** (accuracy against crossbars). This is the
   correct axis for a hardware thesis.

## 14. File pointers

**Thesis repo (`onnxmlir-pim-sparsify`):**

- `backend/compact.py` — new. Compaction, cost model, self-check.
- `sparsifier/neuron_sparsifier.py:78` — zeroes a neuron, keeps the shape.
- `sparsifier/sparsifier.py:46` — `make_omega` (noise Omega).
- `artifacts/control_width200/` — pruned and compacted networks from this session.

**PIM fork (`onnx-mlir-priv`):**

- `src/Accelerators/PIM/Conversion/ONNXToSpatial/Utils/AnnotateReplication.cpp:75` — core count from shape.
- `src/Accelerators/PIM/Conversion/ONNXToSpatial/Math/Gemm.cpp:115` — tile count from shape.
- `src/Accelerators/PIM/Compiler/PimCodeGen.cpp:205,582` — `mbiw=8`, energy defect.
- `src/Accelerators/PIM/Compiler/PimCompilerOptions.cpp:44` — `crossbar_size`.

## 15. Glossary

| Term | Meaning |
|---|---|
| PIM | Processing-In-Memory. Compute inside the memory array. |
| Crossbar | A fixed-size analog array. It does one matrix-vector multiply. |
| MVM | Matrix-Vector Multiply. |
| Tile | One crossbar-size block of a weight matrix. |
| ADC | Analog-to-Digital Converter. It reads the crossbar output. |
| Omega | The input sample set that the manifold distance uses. |
| Unstructured sparsity | Single zero weights at any position. |
| Structured sparsity | Whole zero neurons (a row and a column). |
| Compaction | Removal of dead neurons to make a smaller dense network. |

---

#thesis/pim #sparsification/hardware #status/draft
