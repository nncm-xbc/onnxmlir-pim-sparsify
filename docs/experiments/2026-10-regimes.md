# Experiment log — regimes: harder tasks (B), structured width-200 (A), width-50 weight-level (C)

Prepared 2026-10-01 on branch `fable-profiling`. Queue: `scripts/run/queue_regimes.txt` (91
commands). Progress checkpoints are appended to §6 automatically.

## 0. Launch (tomorrow morning)

```
cd ~/repos/onnxmlir-pim-sparsify
scripts/run/seedband_launch.sh scripts/run/queue_regimes.txt docs/experiments/2026-10-regimes.md
```
Idempotent: re-running after a crash resumes from checkpoints and skips finished runs. To make a
reboot self-heal, add to `crontab -e`:
```
@reboot sleep 60 && /home/simon/repos/onnxmlir-pim-sparsify/scripts/run/seedband_launch.sh scripts/run/queue_regimes.txt docs/experiments/2026-10-regimes.md
```
Status at any time: `tail -5 artifacts/seedband/queue_regimes.supervise.log`. The queue can be
stopped after any block (`kill $(cat artifacts/seedband/queue_regimes.pid)`; the current run
resumes later from its checkpoint). Estimated wall time: **B ≈ 11 h, A ≈ 28 h, C ≈ 75 h**
(serial, single GPU; per-step costs measured on existing runs: width-200 neuron step 16 s,
width-50 weight step 3.7 s + adjustment, baseline 500-step runs 20/35/56 min for
exhaustive/magnitude/OBD).

## 1. Motivation

The seed band (`2026-09-seedband.md`) showed that on the baseline `[196,10,10,10]` MNIST net
every adjusted selector reaches the same collapse (88 ± 2 %), and magnitude pruning is at least
as good as the exhaustive d_W search at every sparsity. Hypothesis: that network has too much
*room* — 225 of 2 160 weights have exactly zero removal cost at w⁽⁰⁾ and the plateau is long —
so any sensible removal order survives, and selectors cannot differ. Two ways to remove the
room: a harder task at fixed capacity (B), and large moves — whole neurons — where the
quadratic model behind cheap scores breaks down (A). Block C tests the counter-hypothesis that
width alone changes the picture (prediction: it does not; wider MNIST nets are *more* redundant).

Block A is also the hardware result: on crossbar accelerators cost is a function of layer
width (`ceil(w/128)` tiles), unstructured sparsity saves nothing, and neuron pruning to a tile
boundary followed by compaction (`backend/compact.py`) is the only sparsity that pays.

## 2. Hypotheses (pre-registered; written before any Block A/B/C run)

### Block B — harder tasks, baseline net, 500 steps, three selectors on one dense net per dataset
- **HB1 (room is measurable).** The zero-cost fraction at w⁽⁰⁾ (`scripts/zero_cost_fraction.py`,
  uniform Ω, B = 10⁴) is lower on Fashion-MNIST and KMNIST than on MNIST (10.4 % on seed 0),
  lowest on KMNIST (dense accuracy 0.70: the net is under-capacity).
  *Refuted if* KMNIST's fraction ≥ MNIST's.
- **HB2 (separation grows with less room).** The spread at step 500 among {exhaustive,
  magnitude, OBD} grows as the zero-cost fraction shrinks, and on KMNIST the exhaustive search
  beats magnitude by > 1 point. *Refuted if* the KMNIST gap is ≤ 1 point or has the wrong sign.
- **HB3 (data-Ω wins because of data structure).** On S1 (structureless Gaussian inputs,
  random-teacher labels) neuron pruning with noise-Ω and data-Ω tie (|Δacc| ≤ 2 points at every
  matched neuron sparsity, e31 vs e32); on MNIST (e33 vs e11, same dense net) data-Ω is better
  by > 5 points at 50 % neuron sparsity. *Refuted if* S1 shows a > 2-point gap, or MNIST ≤ 5.
- **HB4 (planted structure is found).** On S2 the input weights surviving at step 500
  concentrate on the 20 planted pixels: precision@176-removed > 5× chance.
- **HB5 (generalisation).** On Fashion/KMNIST/USPS the exhaustive search keeps ≥ dense − 3
  points at 23.1 % sparsity, as on MNIST.

### Block A — `[196,200,200,10]`, neuron pruning to exhaustion, Ω = 10⁴ training images
- **HA1 (large moves separate selectors).** At 4 crossbars (both hidden widths ≤ 128) the
  manifold search with data-Ω is ≥ 2 points above magnitude neuron pruning on average over 5
  seeds, and reaches ≥ 0.95 on ≥ 4 seeds (seed-0 control: 0.966 vs magnitude stuck at 6 xb).
  *Refuted if* mean gap < 1 point.
- **HA2 (noise-Ω fails for structured pruning).** Noise-Ω manifold is < 0.80 at 4 crossbars on
  both seeds (seed-0 control: 0.679). *Refuted if* either seed ≥ 0.90.
- **HA3 (ties the retrained baseline).** Data-Ω manifold at 4 xb is within 1 point of the
  from-scratch `[196,128,128,10]` net of the same seed; same at 6 and 7 xb vs e40 controls.
- **HA4 (second-order proxy degrades on large moves).** OBD neuron pruning trails the manifold
  search by ≥ 1 point at 4 crossbars (seeds 0–2). *Refuted if* tie — which would mean the
  quadratic model survives neuron-sized moves and the search's edge is Ω, not exactness.
- **HA5 (tile-aligned stopping).** The crossbar-count trajectory is monotone and the accuracy
  at each count is attained at the first step reaching it within 1 point of the best at it.

### Block C — `[196,50,50,10]`, full-budget weight-level, 5 seeds
- **HC1 (width alone does not help).** Collapse of exhaustive, magnitude (5 seeds) and OBD
  (2 seeds) within 2 points of one another; zero-cost fraction at width 50 ≥ baseline's.
  *Refuted if* the exhaustive search collapses > 2 points later than magnitude on ≥ 4 seeds.

### Block D — `[784,512,512,10]` on full-resolution MNIST, structured pruning to crossbar boundaries (added 2026-10-04)

**Motivation.** Block A's selector advantage (search ≫ magnitude at 4 crossbars) is the thesis's
one result where the method beats its cheap competitor. It was obtained at 80k weights on
14×14 inputs. Block D asks whether it survives a 10× larger net on the standard dataset:
670k weights, 1 024 hidden neurons, 48 crossbars dense at S = 128 (784 inputs = 7 tiles,
512 = 4), pruned to `[128,128]` = 9 crossbars. Weight-level selectors are not run at this size:
the exhaustive search is O(N²) and infeasible here (manuscript limitation); the structured
search is O(neurons) per step and is the variant the hardware chapter needs.

**Budget.** Measured on the seed-0 net with B = 10 000: 154 s prune + 91 s adjust per step
(GPU shared with Block B), i.e. ~55 h per 800-step run. Block D therefore uses **B = 2 000**
training images, validated by **D0**: e38 seed 0 rerun at B = 2 000 on the identical dense
net must reproduce the B = 10 000 accuracies at 7 and 4 crossbars within 1 point (the weight-
level B-ablation of the manuscript found insensitivity over 10²–10⁵; D0 checks the same for
structured data-Ω pruning). Estimated: D0 35 min, controls 15 × ~7 min, search ~11 h and
magnitude ~4 h per seed → ≈ 80 h for five seeds. Neuron runs now checkpoint every 10 steps
with resumable (W + b, atomic) checkpoints, so a crash costs ≤ 10 steps.

**Hypotheses (pre-registered).**
- **HD0 (budget).** |acc(B=2000) − acc(B=10000)| ≤ 1 point at 7 and 4 crossbars on e38 seed 0.
  *Refuted if* > 1 point at either count → Block D must use B = 10 000 and fewer seeds.
- **HD1 (advantage persists at scale).** At 9 crossbars the search is ≥ 2 points above magnitude
  on ≥ 4 of 5 seeds, and the mechanism is the same: magnitude reaches 9 crossbars by draining
  one layer below 128 while the search takes both layers to the boundary.
  *Refuted if* mean gap < 1 point.
- **HD2 (ties the retrained control at every count).** Search within 1 point of the
  from-scratch `[784,w,w,10]` net at 33, 20 and 9 crossbars (w = 384, 256, 128), paired by seed.
- **HD3 (dense accuracy held to 20 crossbars).** The search is within 1 point of dense
  (seed-0 dense 0.958) at 20 crossbars on every seed — a 2.4× tile reduction at no cost.
- **HD4 (magnitude is fine until a layer hits the boundary).** Magnitude ties the search at the
  first count it reaches with both layers ≥ 128 (33 or 20 crossbars), as at width 200.

**Conditions.** `data/mnist784/` (raw 28×28 pixels 0–255 from the Colab CSVs, no resize; same
20k/10k rows), SGD lr 0.01, 1 000 epochs, seed = run seed; dense seed-0 net 0.958 test
accuracy (train 1.000). Configs e42 (search + magnitude, shared net), e43 (controls), e44 (D0).
Queue `scripts/run/queue_regimesD.txt`, 31 commands; launch after Block C.

## 3. Conditions

Common: MNIST-style CSVs (14×14, first 10 000 train / 1 000 test rows), SGD lr 0.01, 1 000 epochs
(5 000 for S1/S2), seed = run seed for training and Ω, adjustment as Section 4.3, checkpoints
resumable, supervisor `RESTART_SLEEP=60 MAX_RESTARTS=10`. Code: this commit. Hardware: RTX 2070
SUPER; **+12 V rail reads 10.0–10.1 V and the host crashed 7× under load last week** — the
queue is checkpointed every 20 (A), 50 (B) or 200 (C) steps for that reason.

Shared-config design: one config = one dense net; each selector writes its own subdir
(`sparsified/`, `magnitude_sparsified/`, `obd_sparsified/`, `neuron_*_sparsified/`), so every
comparison within a config is paired by construction (no copying, no re-training).

| Block | Configs | Runs | Ω |
|---|---|---|---|
| B0 | e05, e25 s1–4, e03_w50 s0, e03_w200 s0 | zero-cost diagnostic only | as config (noise) |
| B1 | e26 Fashion, e27 KMNIST, e28 USPS, e29 S1, e30 S2 | train + diagnostic + exhaustive/magnitude/OBD 500 steps | noise |
| B2 | e31/e32 (S1 neuron noise/data), e33 (MNIST neuron data, e11 net) | 15 neuron steps | noise / test rows (B = 1 000) |
| A | e38 s0–4 (data-Ω), e39 s0–1 (noise-Ω), e40 controls ×15 | train; manifold ×5, magnitude ×5, noise ×2, OBD ×3 (400 steps); 15 trainings | training rows, B = 10⁴ |
| C | e41 s0–4 | train; magnitude ×5, exhaustive ×5, OBD ×2 (12 800 steps) | noise |

Metrics: Block B/C as `scripts/analyze_seedband.py` (extend `SEL` with the new prefixes);
Block A `scripts/analyze_crossbar.py` (accuracy at the first step reaching each crossbar count,
crossbars = 2⌈w₁/128⌉ + ⌈w₁/128⌉⌈w₂/128⌉ + ⌈w₂/128⌉, mirroring the PIM backend); room
`artifacts/<name>/zero_cost.json`.

## 4. Pre-launch validation (run 2026-10-02 08:00–08:35, before launch)

- Unit tests on the uncommitted runner / neuron-sparsifier changes: 56/56 pass (CPU).
- `zero_cost_fraction.py` on e05: **225 of 2 160** exactly-zero-cost weights (10.4 %), 1 dead +
  5 always-on units per hidden layer — reproduces Section 6.5 of the manuscript exactly.
- Smoke runs (2 steps, scratch copies, Ω = 10 000 training rows): neuron manifold ✓, neuron
  magnitude ✓ (width-200 net, both compact to `[196,200,198,10]`), weight-level magnitude with
  `omega_source: train` on the width-50 net ✓.
- **Neuron OBD ✗ — dropped from Block A.** On the width-200 net it did not complete step 0 within
  the 20-minute smoke timeout (GPU at 99 %): `_diag_hessian` costs one Hessian-vector product
  per weight, 41 210 here against 2 160 on the baseline, so 400 steps × 3 seeds is weeks. The
  estimate of 28 h for Block A did not account for it. **HA4 is therefore not testable in this
  queue**; it would need a cheaper Gauss–Newton diagonal (the batched per-output gradient route
  of the conclusion's future work) and is left open.
- Queue reordered to **B0 → A → B1/B2 → C** (A is the hardware result and the one the thesis
  needs first; each later block can be dropped). 88 commands; A ≈ 20 h, B ≈ 11 h, C ≈ 75 h.
- Host: +12 V rail 10.14 V at launch; `@reboot` hook installed.

## 5. Results

### 5.A Block A — structured pruning of `[196,200,200,10]` to crossbar boundaries (complete 2026-10-03 19:55)

12 neuron runs × 400 steps + 15 control trainings, all rc = 0 (one crash restart, §6). Ω = 10 000
training images (test split held out). Numbers = `scripts/analyze_crossbar.py` (accuracy at the
first step reaching each crossbar count, i.e. what compaction at that moment deploys) and the
final test accuracy of the e40 from-scratch nets.

| Seed | Dense (10 xb) | d_W search, 7 xb | d_W search, **4 xb** | Magnitude, 6 xb | Magnitude, **4 xb** | Noise-Ω search, 4 xb | From-scratch 4 / 6 / 7 xb |
|---|---|---|---|---|---|---|---|
| 0 | 0.969 | 0.970 `[146,128]` | **0.970** `[128,102]` | 0.967 `[128,144]` | 0.873 `[6,128]` | 0.670 `[128,118]` | 0.961 / 0.966 / 0.964 |
| 1 | 0.962 | 0.965 `[147,128]` | **0.959** `[128,106]` | 0.964 `[128,157]` | 0.659 `[3,128]` | 0.688 `[128,120]` | 0.966 / 0.957 / 0.967 |
| 2 | 0.962 | 0.962 `[159,128]` | **0.961** `[128,96]` | 0.962 `[128,144]` | 0.927 `[12,128]` | — | 0.959 / 0.955 / 0.959 |
| 3 | 0.962 | 0.962 `[158,128]` | **0.963** `[128,97]` | 0.962 `[128,155]` | 0.898 `[7,128]` | — | 0.961 / 0.958 / 0.961 |
| 4 | 0.961 | 0.959 `[158,128]` | **0.960** `[128,105]` | 0.959 `[128,148]` | 0.655 `[4,128]` | — | 0.960 / 0.962 / 0.960 |
| mean ± std | 0.963 ± 0.003 | 0.964 ± 0.004 | **0.963 ± 0.004** | 0.963 ± 0.003 | **0.802 ± 0.134** | 0.679 | 0.961 ± 0.003 / 0.960 ± 0.004 / 0.962 ± 0.003 |

(Shapes are the two hidden widths. Magnitude never passes through 7 crossbars: it trims layer 1
to 128 first, reaching 6 xb at steps 114–127, then keeps removing from layer 1 and reaches 4 xb
only by emptying it to 3–12 neurons at steps 259–268. The d_W search reaches 7 xb at steps
112–125 and 4 xb at steps 165–175 by taking layer 1 to 128 and layer 2 to 96–106.)

**Paired differences at 4 crossbars (search − other), seeds 0–4:**
- vs magnitude: +9.7, +30.0, +3.4, +6.5, +30.5 points — mean **+16.0 ± 13.0**, min +3.4.
- vs from-scratch `[196,128,128,10]`: +0.9, −0.7, +0.2, +0.2, 0.0 — mean **+0.1 ± 0.6**.
- vs dense: +0.1, −0.3, −0.1, +0.1, −0.1 — mean **0.0 ± 0.2**.
- Noise-Ω vs data-Ω (seeds 0–1): −30.0, −27.1.

Hardware reading (`backend.compact.crossbar_cost`, S = 128): 10 → 4 crossbars and 5 → 3 cores
on every seed, with the compacted `[196,128,~100,10]` net holding ~49 % of the dense weight count
and the dense accuracy to within 0.3 points.

### 5.A.1 Verdicts

- **HA1 — supported, by a wide margin.** The search is ≥ 2 points above magnitude at 4 crossbars
  on 5/5 seeds (min +3.4, mean +16) and ≥ 0.95 on 5/5 seeds. The mechanism is visible in the
  shapes: magnitude's per-neuron score carries no information about the tile boundary or about
  layer balance, so after layer 1 reaches 128 it keeps draining layer 1 (its smallest
  neurons) and arrives at 4 crossbars with a 3–12-neuron bottleneck; the global objective
  d_W prices a removal by its effect on the output and moves to layer 2 as soon as layer 1's
  neurons become load-bearing. Up to 6 crossbars magnitude is as good as the search (0.963 vs
  0.964): the gap is entirely about *where* the removals land once one layer is at the boundary.
- **HA2 — supported.** Noise-Ω search: 0.670 and 0.688 at 4 crossbars (both < 0.80), already
  0.72–0.76 at 7 crossbars, on the identical dense nets that reach 0.959–0.970 with
  training-image Ω. A 27–30-point swing from the Ω choice alone, replicated; the seed-0 control
  (0.679) was not an artefact.
- **HA3 — supported.** The search ties the from-scratch net of the same shape at 4 crossbars
  (+0.1 ± 0.6, |Δ| ≤ 0.9 on every seed) and at 7 (+0.1 ± 0.3); magnitude ties it at 6
  (+0.3 ± 0.4). Pruning does not beat retraining the crossbar-shaped network — it matches it
  **without labels, without retraining, and while also telling you which shape to use**.
- **HA4 — dropped** (neuron OBD infeasible at this width, §4).
- **HA5 — supported for the search, not for magnitude.** Search: accuracy at the first step
  reaching each count is within 0.4 points of the best at that count on every seed
  (crossbar trajectory monotone by construction), so "compact at the first step that reaches the
  target" is a sound stopping rule. Magnitude: on seed 4 the first 4-xb state (0.655) is 13
  points below the best at 4 xb (0.785), i.e. its trajectory keeps changing after the boundary.

### 5.A.2 What this changes for the thesis

This is the selector result the weight-level seed band did not provide. At the granularity the
hardware bills for, the exhaustive d_W search beats magnitude by 16 points on average and ties
a retrained network, on five seeds; the quadratic-model argument that explained the weight-level
tie does not apply to neuron-sized moves, and the data shows it. Two caveats to carry into the
text: (i) the result needs a data-informed Ω (unlabelled images), so the structured variant is
*label-free* but not *data-free*; (ii) magnitude's failure is a layer-balance failure, and a
tile-aware magnitude rule (stop draining a layer at the boundary) has not been tried — it is
the obvious cheap competitor and should be named as such. Manuscript update deferred until
Block B is in.


### 5.B Block B — harder tasks and Ω-gap tests (complete 2026-10-04 17:03)

All 32 runs rc = 0. Metrics as planned; step-500 accuracy = last logged `val_acc` (23.1 %
weight sparsity). Zero-cost fraction = share of the 2 160 prunable weights whose exact removal
cost on the config's Ω is 0 (`zero_cost.json`).

#### Redundancy diagnostic (HB1)

| Dense net | zero-cost | dead / always-on units per hidden layer | cost q25 / q50 |
|---|---|---|---|
| MNIST seed 0 (e05) | **10.4 %** (225) | 1,1 / 5,5 | 108 / 1 090 |
| MNIST seeds 1–4 (e25) | 1.8 / 0.0 / 0.9 / 1.9 % | 0–2 dead | 3–88 / 240–925 |
| MNIST width 50 (e03) | 5.6 % (716/12 800) | 2,5 / 7,21 | 0.75 / 35 |
| Fashion-MNIST | 0.05 % (1) | 0,0 / 2,6 | 178 / 1 180 |
| KMNIST | **0.0 %** (0) | 0,0 / 2,3 | **283 / 1 370** |
| USPS | 0.8 % (18) | 0,1 / 4,4 | 5.7 / 215 |
| S1 (teacher) | 0.05 % (1) | 0,0 / 0,0 | 41 / 3 020 |
| S2 (planted) | 0.2 % (4) | 0,0 / 0,0 | 0.026 / 164 |

#### Step-500 accuracy, three selectors on one dense net per dataset (HB2, HB5)

| Dataset | Dense | Exhaustive d_W | Magnitude | OBD with d_W |
|---|---|---|---|---|
| Fashion-MNIST | 0.815 | 0.807 | 0.806 | 0.807 |
| **KMNIST** | 0.703 | **0.532** | **0.709** | **0.536** |
| USPS | 0.918 | 0.917 | 0.918 | 0.918 |
| S1 (teacher labels) | 0.571 | 0.569 | 0.576 | 0.569 |
| S2 (planted) | 0.801 | 0.801 | 0.800 | 0.801 |

KMNIST trajectory: exhaustive and OBD fall from 0.70 to 0.58 between steps 25 and 50 (first
step < 0.65: 36 and 37) and to 0.53 by step 499; magnitude stays at 0.70 throughout. Over the
first ten exhaustive removals the measured change of the function on the uniform-noise Ω is
d_W ≤ 8·10⁻³ — the criterion sees almost no change while KMNIST accuracy is collapsing. Three
of the first eleven removals are output-layer weights. OBD, which scores on the same Ω,
follows the same path; magnitude, which uses Ω only in the adjustment, is unharmed.

#### Ω gap, neuron pruning at matched neuron sparsity (HB3)

| Neurons removed (of 20) | S1 noise-Ω | S1 data-Ω | MNIST noise-Ω (e11) | MNIST data-Ω (e33) |
|---|---|---|---|---|
| 5 (25 %) | 0.558 | 0.566 | 0.870 | 0.910 |
| 8 (40 %) | 0.482 | 0.516 | 0.794 | 0.902 |
| 9 (45 %) | 0.476 | 0.462 | 0.715 | 0.873 |
| 10 (50 %) | 0.442 | 0.430 | 0.683 | 0.697 |
| 11 (55 %) | 0.351 | 0.408 | 0.563 | 0.698 |
| 12 (60 %) | 0.321 | 0.390 | 0.600 | 0.698 |
| 14 (70 %) | 0.233 | 0.332 | 0.256 | 0.449 |

#### Planted structure (HB4), S2 after 500 steps

| Selector | input-layer removals | on the 20 planted pixels | surviving fan-out, planted / other |
|---|---|---|---|
| Exhaustive | 453 | 49 (10.8 %; chance 10.2 %) | 7.55 / 7.70 (ratio 0.98) |
| Magnitude | 467 | 38 (8.1 %) | 8.10 / 7.56 (1.07) |
| OBD | 451 | 48 (10.6 %) | 7.60 / 7.71 (0.99) |

#### Verdicts

- **HB1 — supported in letter, premise wrong.** KMNIST has the lowest zero-cost fraction (0 %)
  and the highest removal costs (q25 = 283 vs 108 on MNIST seed 0). But the "10.4 % room" that
  motivated the block is a **seed-0 artefact**: MNIST seeds 1–4 have 0–1.9 %, like every other
  dataset. Room is better measured by the cost quantiles; by those, KMNIST and Fashion are the
  least redundant, S2 and USPS the most.
- **HB2 — refuted, with the opposite sign.** On the one hard task the selectors do separate, by
  17.7 points — but it is *magnitude* that wins. The exhaustive search and OBD, the two
  selectors that score removals on the uniform-noise Ω, lose 17 points at 23 % sparsity on a
  network that magnitude prunes losslessly. On the four easier tasks all three tie within a point.
- **HB3 — refuted as stated; the S1 arm was mis-designed.** Data-Ω beats noise-Ω on MNIST by
  10–19 points at 40–45 % and 55–70 % neuron sparsity (but by only 1.4 at exactly 50 %, the
  pre-registered checkpoint), and *also* on S1 by 3–10 points past 40 %. The S1 null does not
  isolate "structure": its data-Ω rows are clipped Gaussians around 127.5 while noise-Ω is
  uniform on [0, 255], so the comparison still contains a distribution mismatch. A clean null
  needs noise-Ω drawn from S1's own marginal; left for later.
- **HB4 — refuted.** No selector concentrates removals away from the planted pixels: removals
  on planted pixels sit at chance (8–11 % vs 10.2 %) and surviving fan-out ratios are 0.98–1.07.
  At 23 % sparsity with all selectors still lossless, nothing has forced a choice yet; and a
  removal cost measured on uniform noise cannot distinguish informative from uninformative
  pixels, because on noise every pixel carries the same signal.
- **HB5 — refuted on KMNIST only.** Fashion −0.8 and USPS −0.1 points vs dense (within 3);
  KMNIST −17.1.

#### What Block B means

The block was designed to find a regime where the exhaustive search beats cheap selectors. It
found the opposite regime: **where the task is hard, scoring removals on uniform noise is
actively harmful**, and the two Ω-based selectors fail together while the Ω-free one is fine.
This is the weight-level twin of Block A's noise-Ω failure (HA2) and of the structured Ω
gap above. The consistent picture across A and B is that d_W is the right objective but
uniform noise is the wrong domain to evaluate it on once the network's behaviour on data
diverges from its behaviour on noise — i.e. on anything harder than MNIST. The thesis's
"data-free" claim therefore needs to be narrowed to "label-free": the method needs unlabelled
inputs from the task distribution, and the manuscript should say so.

**Pending (launched 18:55):** e27b — KMNIST exhaustive search with Ω = 10 000 training images
on the identical dense net. If it recovers magnitude's 0.70, the failure is attributable to Ω
alone; if not, to the criterion. Result appended below when done.

## 6. Progress checkpoints (auto-appended)

**2026-10-02 08:42** — (re)launch of `scripts/run/queue_regimes.txt`: 87 commands queued

**2026-10-02 08:42** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e03_width_200_seed_0/sparsified | 499 | 0.6 % | 0.969 | 0.000000e+00 |
| e03_width_50_seed_0/sparsified | 499 | 3.9 % | 0.930 | 0.000000e+00 |
| e25_stupidity_seed_1/sparsified | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_stupidity_seed_2/sparsified | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_stupidity_seed_3/sparsified | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_stupidity_seed_4/sparsified | 2159 | 100.0 % | 0.107 | 2.013833e+06 |

```
[supervise 08:42:20] launch [/home/simon/venv/general/bin/python3 scripts/zero_cost_fraction.py experiments/e25_stupidity_seed_1.json] (attempt 0/10)
```

**2026-10-02 09:12** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e03_width_200_seed_0/sparsified | 499 | 0.6 % | 0.969 | 0.000000e+00 |
| e03_width_50_seed_0/sparsified | 499 | 3.9 % | 0.930 | 0.000000e+00 |
| e25_stupidity_seed_1/sparsified | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_stupidity_seed_2/sparsified | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_stupidity_seed_3/sparsified | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_stupidity_seed_4/sparsified | 2159 | 100.0 % | 0.107 | 2.013833e+06 |

```
[supervise 08:55:44] launch [/home/simon/venv/general/bin/python3 scripts/zero_cost_fraction.py experiments/e03_width_200_seed_0.json] (attempt 1/10)
[supervise 09:05:44] STALL experiments/e03_width_200_seed_0.json (no progress for 600s) -- SIGKILL, will resume
[supervise 09:06:44] launch [/home/simon/venv/general/bin/python3 scripts/zero_cost_fraction.py experiments/e03_width_200_seed_0.json] (attempt 2/10)
```

**2026-10-02 09:42** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e03_width_200_seed_0/sparsified | 499 | 0.6 % | 0.969 | 0.000000e+00 |
| e03_width_50_seed_0/sparsified | 499 | 3.9 % | 0.930 | 0.000000e+00 |
| e25_stupidity_seed_1/sparsified | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_stupidity_seed_2/sparsified | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_stupidity_seed_3/sparsified | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_stupidity_seed_4/sparsified | 2159 | 100.0 % | 0.107 | 2.013833e+06 |

```
[supervise 09:28:45] launch [/home/simon/venv/general/bin/python3 scripts/zero_cost_fraction.py experiments/e03_width_200_seed_0.json] (attempt 4/10)
[supervise 09:38:45] STALL experiments/e03_width_200_seed_0.json (no progress for 600s) -- SIGKILL, will resume
[supervise 09:39:45] launch [/home/simon/venv/general/bin/python3 scripts/zero_cost_fraction.py experiments/e03_width_200_seed_0.json] (attempt 5/10)
```

**2026-10-02 10:15 — B0 width-200 diagnostic dropped.** `zero_cost_fraction.py` on
`e03_width_200_seed_0` sweeps ~41 000 weights and writes nothing before the end, so the
supervisor's 600 s stall watchdog killed it seven times in a row (08:45–10:11, 85 min lost).
Removed from the queue; the four e25 and the width-50 diagnostics completed. Relaunched; Block A
starts now. (HB1/HC1 use the e25 / width-50 numbers; the width-200 zero-cost fraction can be
computed later with `STALL_TIMEOUT=7200` outside the queue.)

**2026-10-02 10:11** — (re)launch of `scripts/run/queue_regimes.txt`: 81 commands queued

**2026-10-02 10:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|

```
[supervise 10:00:45] STALL experiments/e03_width_200_seed_0.json (no progress for 600s) -- SIGKILL, will resume
[supervise 10:01:45] launch [/home/simon/venv/general/bin/python3 scripts/zero_cost_fraction.py experiments/e03_width_200_seed_0.json] (attempt 7/10)
[supervise 10:11:49] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_0.json] (attempt 0/10)
```

**2026-10-02 10:41** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_sparsified | 86 | 21.5 % | 0.969 | 4.064299e+02 |

```
[supervise 10:11:49] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_0.json] (attempt 0/10)
[supervise 10:12:39] DONE experiments/e38_w200_neuron_data_seed_0.json rc=0
[supervise 10:12:39] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_0.json] (attempt 0/10)
```

**2026-10-02 11:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_sparsified | 183 | 45.8 % | 0.968 | 4.274984e+03 |

```
[supervise 10:11:49] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_0.json] (attempt 0/10)
[supervise 10:12:39] DONE experiments/e38_w200_neuron_data_seed_0.json rc=0
[supervise 10:12:39] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_0.json] (attempt 0/10)
```

**2026-10-02 11:41** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_sparsified | 293 | 73.2 % | 0.961 | 2.377236e+04 |

```
[supervise 10:11:49] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_0.json] (attempt 0/10)
[supervise 10:12:39] DONE experiments/e38_w200_neuron_data_seed_0.json rc=0
[supervise 10:12:39] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_0.json] (attempt 0/10)
```

**2026-10-02 12:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 22 | 5.5 % | 0.969 | 2.274804e+01 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |

```
[supervise 10:12:39] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_0.json] (attempt 0/10)
[supervise 12:06:22] DONE experiments/e38_w200_neuron_data_seed_0.json rc=0
[supervise 12:06:22] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_0.json] (attempt 0/10)
```

**2026-10-02 12:41** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 157 | 39.2 % | 0.967 | 3.923743e+03 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |

```
[supervise 10:12:39] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_0.json] (attempt 0/10)
[supervise 12:06:22] DONE experiments/e38_w200_neuron_data_seed_0.json rc=0
[supervise 12:06:22] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_0.json] (attempt 0/10)
```

**2026-10-02 13:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 293 | 73.2 % | 0.593 | 1.517177e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |

```
[supervise 10:12:39] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_0.json] (attempt 0/10)
[supervise 12:06:22] DONE experiments/e38_w200_neuron_data_seed_0.json rc=0
[supervise 12:06:22] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_0.json] (attempt 0/10)
```

**2026-10-02 13:41** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 49 | 12.2 % | 0.960 | 3.231976e+00 |

```
[supervise 13:27:52] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e39_w200_neuron_noise_seed_0.json] (attempt 0/10)
[supervise 13:28:45] DONE experiments/e39_w200_neuron_noise_seed_0.json rc=0
[supervise 13:28:45] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e39_w200_neuron_noise_seed_0.json] (attempt 0/10)
```

**2026-10-02 14:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 142 | 35.5 % | 0.692 | 4.614797e+03 |

```
[supervise 13:27:52] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e39_w200_neuron_noise_seed_0.json] (attempt 0/10)
[supervise 13:28:45] DONE experiments/e39_w200_neuron_noise_seed_0.json rc=0
[supervise 13:28:45] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e39_w200_neuron_noise_seed_0.json] (attempt 0/10)
```

**2026-10-02 14:41** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 246 | 61.5 % | 0.496 | 2.726007e+04 |

```
[supervise 13:27:52] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e39_w200_neuron_noise_seed_0.json] (attempt 0/10)
[supervise 13:28:45] DONE experiments/e39_w200_neuron_noise_seed_0.json rc=0
[supervise 13:28:45] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e39_w200_neuron_noise_seed_0.json] (attempt 0/10)
```

**2026-10-02 15:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 367 | 91.8 % | 0.333 | 7.633182e+04 |

```
[supervise 13:27:52] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e39_w200_neuron_noise_seed_0.json] (attempt 0/10)
[supervise 13:28:45] DONE experiments/e39_w200_neuron_noise_seed_0.json rc=0
[supervise 13:28:45] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e39_w200_neuron_noise_seed_0.json] (attempt 0/10)
```

**2026-10-02 15:41** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 63 | 15.8 % | 0.960 | 2.313379e+02 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |

```
[supervise 15:18:36] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_1.json] (attempt 0/10)
[supervise 15:19:28] DONE experiments/e38_w200_neuron_data_seed_1.json rc=0
[supervise 15:19:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_1.json] (attempt 0/10)
```

**2026-10-02 16:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 157 | 39.2 % | 0.964 | 3.208377e+03 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |

```
[supervise 15:18:36] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_1.json] (attempt 0/10)
[supervise 15:19:28] DONE experiments/e38_w200_neuron_data_seed_1.json rc=0
[supervise 15:19:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_1.json] (attempt 0/10)
```

**2026-10-02 16:41** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 263 | 65.8 % | 0.959 | 1.430141e+04 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |

```
[supervise 15:18:36] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_1.json] (attempt 0/10)
[supervise 15:19:28] DONE experiments/e38_w200_neuron_data_seed_1.json rc=0
[supervise 15:19:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_1.json] (attempt 0/10)
```

**2026-10-02 17:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 387 | 96.8 % | 0.821 | 8.460772e+05 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |

```
[supervise 15:18:36] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_1.json] (attempt 0/10)
[supervise 15:19:28] DONE experiments/e38_w200_neuron_data_seed_1.json rc=0
[supervise 15:19:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_1.json] (attempt 0/10)
```

**2026-10-02 17:41** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_magnitude_sparsified | 121 | 30.2 % | 0.964 | 2.379466e+03 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 4.945521e+06 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |

```
[supervise 15:19:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_1.json] (attempt 0/10)
[supervise 17:14:07] DONE experiments/e38_w200_neuron_data_seed_1.json rc=0
[supervise 17:14:07] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_1.json] (attempt 0/10)
```

**2026-10-02 18:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_magnitude_sparsified | 258 | 64.5 % | 0.887 | 4.523429e+05 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 4.945521e+06 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |

```
[supervise 15:19:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_1.json] (attempt 0/10)
[supervise 17:14:07] DONE experiments/e38_w200_neuron_data_seed_1.json rc=0
[supervise 17:14:07] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_1.json] (attempt 0/10)
```

**2026-10-02 18:41** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.777770e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 4.945521e+06 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |
| e39_w200_neuron_noise_seed_1/neuron_sparsified | 30 | 7.5 % | 0.950 | 2.704322e-01 |

```
[supervise 18:32:30] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e39_w200_neuron_noise_seed_1.json] (attempt 0/10)
[supervise 18:33:22] DONE experiments/e39_w200_neuron_noise_seed_1.json rc=0
[supervise 18:33:22] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e39_w200_neuron_noise_seed_1.json] (attempt 0/10)
```

**2026-10-02 19:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.777770e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 4.945521e+06 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |
| e39_w200_neuron_noise_seed_1/neuron_sparsified | 121 | 30.2 % | 0.768 | 3.186201e+03 |

```
[supervise 18:32:30] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e39_w200_neuron_noise_seed_1.json] (attempt 0/10)
[supervise 18:33:22] DONE experiments/e39_w200_neuron_noise_seed_1.json rc=0
[supervise 18:33:22] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e39_w200_neuron_noise_seed_1.json] (attempt 0/10)
```

**2026-10-02 19:41** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.777770e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 4.945521e+06 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |
| e39_w200_neuron_noise_seed_1/neuron_sparsified | 223 | 55.8 % | 0.592 | 1.900091e+04 |

```
[supervise 18:32:30] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e39_w200_neuron_noise_seed_1.json] (attempt 0/10)
[supervise 18:33:22] DONE experiments/e39_w200_neuron_noise_seed_1.json rc=0
[supervise 18:33:22] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e39_w200_neuron_noise_seed_1.json] (attempt 0/10)
```

**2026-10-02 20:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.777770e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 4.945521e+06 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |
| e39_w200_neuron_noise_seed_1/neuron_sparsified | 341 | 85.2 % | 0.466 | 4.507746e+04 |

```
[supervise 18:32:30] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e39_w200_neuron_noise_seed_1.json] (attempt 0/10)
[supervise 18:33:22] DONE experiments/e39_w200_neuron_noise_seed_1.json rc=0
[supervise 18:33:22] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e39_w200_neuron_noise_seed_1.json] (attempt 0/10)
```

**2026-10-02 20:41** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.777770e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 4.945521e+06 |
| e38_w200_neuron_data_seed_2/neuron_sparsified | 46 | 11.5 % | 0.962 | 2.219513e+01 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |
| e39_w200_neuron_noise_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.239544e+06 |

```
[supervise 20:24:34] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_2.json] (attempt 0/10)
[supervise 20:25:28] DONE experiments/e38_w200_neuron_data_seed_2.json rc=0
[supervise 20:25:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_2.json] (attempt 0/10)
```

**2026-10-02 21:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.777770e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 4.945521e+06 |
| e38_w200_neuron_data_seed_2/neuron_sparsified | 139 | 34.8 % | 0.961 | 1.505656e+03 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |
| e39_w200_neuron_noise_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.239544e+06 |

```
[supervise 20:24:34] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_2.json] (attempt 0/10)
[supervise 20:25:28] DONE experiments/e38_w200_neuron_data_seed_2.json rc=0
[supervise 20:25:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_2.json] (attempt 0/10)
```

**2026-10-02 21:41** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.777770e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 4.945521e+06 |
| e38_w200_neuron_data_seed_2/neuron_sparsified | 242 | 60.5 % | 0.958 | 1.060691e+04 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |
| e39_w200_neuron_noise_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.239544e+06 |

```
[supervise 20:24:34] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_2.json] (attempt 0/10)
[supervise 20:25:28] DONE experiments/e38_w200_neuron_data_seed_2.json rc=0
[supervise 20:25:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_2.json] (attempt 0/10)
```

**2026-10-02 22:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.777770e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 4.945521e+06 |
| e38_w200_neuron_data_seed_2/neuron_sparsified | 363 | 90.8 % | 0.923 | 2.427546e+05 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |
| e39_w200_neuron_noise_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.239544e+06 |

```
[supervise 20:24:34] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_2.json] (attempt 0/10)
[supervise 20:25:28] DONE experiments/e38_w200_neuron_data_seed_2.json rc=0
[supervise 20:25:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_2.json] (attempt 0/10)
```

**2026-10-02 22:41** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.777770e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 4.945521e+06 |
| e38_w200_neuron_data_seed_2/neuron_magnitude_sparsified | 89 | 22.2 % | 0.962 | 6.696542e+02 |
| e38_w200_neuron_data_seed_2/neuron_sparsified | 399 | 100.0 % | 0.116 | 4.791953e+06 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |
| e39_w200_neuron_noise_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.239544e+06 |

```
[supervise 20:25:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_2.json] (attempt 0/10)
[supervise 22:19:27] DONE experiments/e38_w200_neuron_data_seed_2.json rc=0
[supervise 22:19:31] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_2.json] (attempt 0/10)
```

**2026-10-02 23:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.777770e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 4.945521e+06 |
| e38_w200_neuron_data_seed_2/neuron_magnitude_sparsified | 224 | 56.0 % | 0.956 | 1.897282e+04 |
| e38_w200_neuron_data_seed_2/neuron_sparsified | 399 | 100.0 % | 0.116 | 4.791953e+06 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |
| e39_w200_neuron_noise_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.239544e+06 |

```
[supervise 20:25:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_2.json] (attempt 0/10)
[supervise 22:19:27] DONE experiments/e38_w200_neuron_data_seed_2.json rc=0
[supervise 22:19:31] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_2.json] (attempt 0/10)
```

**2026-10-02 23:41** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.777770e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 4.945521e+06 |
| e38_w200_neuron_data_seed_2/neuron_magnitude_sparsified | 362 | 90.5 % | 0.087 | 5.019970e+06 |
| e38_w200_neuron_data_seed_2/neuron_sparsified | 399 | 100.0 % | 0.116 | 4.791953e+06 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |
| e39_w200_neuron_noise_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.239544e+06 |

```
[supervise 20:25:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_2.json] (attempt 0/10)
[supervise 22:19:27] DONE experiments/e38_w200_neuron_data_seed_2.json rc=0
[supervise 22:19:31] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_2.json] (attempt 0/10)
```

**2026-10-03 00:11** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_0/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_0/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.535344e+06 |
| e38_w200_neuron_data_seed_1/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.777770e+06 |
| e38_w200_neuron_data_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 4.945521e+06 |
| e38_w200_neuron_data_seed_2/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.588431e+06 |
| e38_w200_neuron_data_seed_2/neuron_sparsified | 399 | 100.0 % | 0.116 | 4.791953e+06 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 79 | 19.8 % | 0.962 | 5.489563e+02 |
| e39_w200_neuron_noise_seed_0/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.159988e+06 |
| e39_w200_neuron_noise_seed_1/neuron_sparsified | 399 | 100.0 % | 0.087 | 1.239544e+06 |

```
[supervise 23:43:33] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_3.json] (attempt 0/10)
[supervise 23:44:28] DONE experiments/e38_w200_neuron_data_seed_3.json rc=0
[supervise 23:44:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_3.json] (attempt 0/10)
```

**2026-10-03 10:27** — (re)launch of `scripts/run/queue_regimes.txt`: 67 commands queued

**2026-10-03 10:27** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_sparsified | 149 | 37.2 % | 0.963 | 2.611117e+03 |

```
[supervise 23:44:28] DONE experiments/e38_w200_neuron_data_seed_3.json rc=0
[supervise 23:44:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_3.json] (attempt 0/10)
step   90 | acc=0.9630 | neurons_pruned=  90/ 400 | sparsity=0.2250 | d_m=8.0022e+02[supervise 10:27:26] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w2
```

**2026-10-03 10:35 — eighth host crash; neuron runs are not resumable.** Host down 00:34–10:26
(`last -x`; GPU telemetry normal to the end, +12 V 10.08 V). The `@reboot` hook relaunched the
queue at 10:27 (67 commands left). The interrupted run, e38 seed 3 d_W search (step 149 of 400,
checkpoint `step_0140`), **restarted from step 0**: the neuron sparsifiers write W-only
checkpoints (no biases, not atomic), which `_latest_checkpoint` deliberately ignores, so
resume only works for the weight-level runs. Cost here 40 min; left as is (a mid-run fix would
change code under a running queue; neuron runs are ≤ 2 h each). Completed: seeds 0–2 all
variants, noise-Ω seeds 0–1. Block A ETA ≈ 17:15 today.

**2026-10-03 10:57** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_sparsified | 89 | 22.2 % | 0.963 | 7.976916e+02 |

```
[supervise 23:44:28] DONE experiments/e38_w200_neuron_data_seed_3.json rc=0
[supervise 23:44:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_3.json] (attempt 0/10)
step   90 | acc=0.9630 | neurons_pruned=  90/ 400 | sparsity=0.2250 | d_m=8.0022e+02[supervise 10:27:26] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w2
```

**2026-10-03 11:27** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_sparsified | 179 | 44.8 % | 0.961 | 3.642059e+03 |

```
[supervise 23:44:28] DONE experiments/e38_w200_neuron_data_seed_3.json rc=0
[supervise 23:44:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_3.json] (attempt 0/10)
step   90 | acc=0.9630 | neurons_pruned=  90/ 400 | sparsity=0.2250 | d_m=8.0022e+02[supervise 10:27:26] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w2
```

**2026-10-03 11:57** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_sparsified | 288 | 72.0 % | 0.952 | 2.619122e+04 |

```
[supervise 23:44:28] DONE experiments/e38_w200_neuron_data_seed_3.json rc=0
[supervise 23:44:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_3.json] (attempt 0/10)
step   90 | acc=0.9630 | neurons_pruned=  90/ 400 | sparsity=0.2250 | d_m=8.0022e+02[supervise 10:27:26] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w2
```

**2026-10-03 12:27** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 18 | 4.5 % | 0.962 | 1.796537e+01 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |

```
step   90 | acc=0.9630 | neurons_pruned=  90/ 400 | sparsity=0.2250 | d_m=8.0022e+02[supervise 10:27:26] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w2
[supervise 12:22:58] DONE experiments/e38_w200_neuron_data_seed_3.json rc=0
[supervise 12:22:58] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_3.json] (attempt 0/10)
```

**2026-10-03 12:57** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 144 | 36.0 % | 0.962 | 4.010019e+03 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |

```
step   90 | acc=0.9630 | neurons_pruned=  90/ 400 | sparsity=0.2250 | d_m=8.0022e+02[supervise 10:27:26] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w2
[supervise 12:22:58] DONE experiments/e38_w200_neuron_data_seed_3.json rc=0
[supervise 12:22:58] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_3.json] (attempt 0/10)
```

**2026-10-03 13:27** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 272 | 68.0 % | 0.801 | 9.642633e+05 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |

```
step   90 | acc=0.9630 | neurons_pruned=  90/ 400 | sparsity=0.2250 | d_m=8.0022e+02[supervise 10:27:26] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w2
[supervise 12:22:58] DONE experiments/e38_w200_neuron_data_seed_3.json rc=0
[supervise 12:22:58] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_3.json] (attempt 0/10)
```

**2026-10-03 13:29** — queue PAUSED by user (supervisor stopped during e38 seed-3 magnitude at step 281; @reboot hook removed). Resume: `scripts/run/seedband_launch.sh scripts/run/queue_regimes.txt docs/experiments/2026-10-regimes.md` (skips the 20 finished runs; the interrupted neuron run restarts from step 0).

**2026-10-03 14:51** — (re)launch of `scripts/run/queue_regimes.txt`: 66 commands queued

**2026-10-03 14:51** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 280 | 70.0 % | 0.820 | 8.486810e+05 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |

```
[supervise 12:22:58] DONE experiments/e38_w200_neuron_data_seed_3.json rc=0
[supervise 12:22:58] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_3.json] (attempt 0/10)
step  189 | acc=0.9600 | neurons_pruned= 189/ 400 | sparsity=0.4725 | d_m=9.0408e+03[supervise 14:51:14] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experime
```

**2026-10-03 15:21** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 134 | 33.5 % | 0.963 | 3.403826e+03 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |

```
[supervise 12:22:58] DONE experiments/e38_w200_neuron_data_seed_3.json rc=0
[supervise 12:22:58] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_3.json] (attempt 0/10)
step  189 | acc=0.9600 | neurons_pruned= 189/ 400 | sparsity=0.4725 | d_m=9.0408e+03[supervise 14:51:14] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experime
```

**2026-10-03 15:51** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 270 | 67.5 % | 0.808 | 6.986616e+05 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |

```
[supervise 12:22:58] DONE experiments/e38_w200_neuron_data_seed_3.json rc=0
[supervise 12:22:58] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_3.json] (attempt 0/10)
step  189 | acc=0.9600 | neurons_pruned= 189/ 400 | sparsity=0.4725 | d_m=9.0408e+03[supervise 14:51:14] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experime
```

**2026-10-03 16:21** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 399 | 100.0 % | 0.116 | 4.837949e+06 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |
| e38_w200_neuron_data_seed_4/neuron_sparsified | 27 | 6.8 % | 0.961 | 2.725480e+00 |

```
[supervise 16:10:42] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_4.json] (attempt 0/10)
[supervise 16:11:32] DONE experiments/e38_w200_neuron_data_seed_4.json rc=0
[supervise 16:11:32] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_4.json] (attempt 0/10)
```

**2026-10-03 16:51** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 399 | 100.0 % | 0.116 | 4.837949e+06 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |
| e38_w200_neuron_data_seed_4/neuron_sparsified | 118 | 29.5 % | 0.958 | 1.426819e+03 |

```
[supervise 16:10:42] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_4.json] (attempt 0/10)
[supervise 16:11:32] DONE experiments/e38_w200_neuron_data_seed_4.json rc=0
[supervise 16:11:32] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_4.json] (attempt 0/10)
```

**2026-10-03 17:21** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 399 | 100.0 % | 0.116 | 4.837949e+06 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |
| e38_w200_neuron_data_seed_4/neuron_sparsified | 219 | 54.8 % | 0.959 | 7.125633e+03 |

```
[supervise 16:10:42] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_4.json] (attempt 0/10)
[supervise 16:11:32] DONE experiments/e38_w200_neuron_data_seed_4.json rc=0
[supervise 16:11:32] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_4.json] (attempt 0/10)
```

**2026-10-03 17:51** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 399 | 100.0 % | 0.116 | 4.837949e+06 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |
| e38_w200_neuron_data_seed_4/neuron_sparsified | 335 | 83.8 % | 0.950 | 6.384395e+04 |

```
[supervise 16:10:42] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e38_w200_neuron_data_seed_4.json] (attempt 0/10)
[supervise 16:11:32] DONE experiments/e38_w200_neuron_data_seed_4.json rc=0
[supervise 16:11:32] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_4.json] (attempt 0/10)
```

**2026-10-03 18:21** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 399 | 100.0 % | 0.116 | 4.837949e+06 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |
| e38_w200_neuron_data_seed_4/neuron_magnitude_sparsified | 70 | 17.5 % | 0.959 | 4.883469e+02 |
| e38_w200_neuron_data_seed_4/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |

```
[supervise 16:11:32] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_4.json] (attempt 0/10)
[supervise 18:05:19] DONE experiments/e38_w200_neuron_data_seed_4.json rc=0
[supervise 18:05:19] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_4.json] (attempt 0/10)
```

**2026-10-03 18:51** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 399 | 100.0 % | 0.116 | 4.837949e+06 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |
| e38_w200_neuron_data_seed_4/neuron_magnitude_sparsified | 207 | 51.7 % | 0.960 | 1.501685e+04 |
| e38_w200_neuron_data_seed_4/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |

```
[supervise 16:11:32] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_4.json] (attempt 0/10)
[supervise 18:05:19] DONE experiments/e38_w200_neuron_data_seed_4.json rc=0
[supervise 18:05:19] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_4.json] (attempt 0/10)
```

**2026-10-03 19:21** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 399 | 100.0 % | 0.116 | 4.837949e+06 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |
| e38_w200_neuron_data_seed_4/neuron_magnitude_sparsified | 345 | 86.2 % | 0.087 | 4.970190e+06 |
| e38_w200_neuron_data_seed_4/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |

```
[supervise 16:11:32] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_sparsifier experiments/e38_w200_neuron_data_seed_4.json] (attempt 0/10)
[supervise 18:05:19] DONE experiments/e38_w200_neuron_data_seed_4.json rc=0
[supervise 18:05:19] launch [/home/simon/venv/general/bin/python3 -m sparsifier.neuron_magnitude_sparsifier experiments/e38_w200_neuron_data_seed_4.json] (attempt 0/10)
```

**2026-10-03 19:51** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e26_fashion/sparsified | 249 | 11.5 % | 0.815 | 2.010561e+02 |
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 399 | 100.0 % | 0.116 | 4.837949e+06 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |
| e38_w200_neuron_data_seed_4/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |
| e38_w200_neuron_data_seed_4/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |

```
[supervise 19:35:27] launch [/home/simon/venv/general/bin/python3 scripts/zero_cost_fraction.py experiments/e26_fashion.json] (attempt 0/10)
[supervise 19:35:41] DONE experiments/e26_fashion.json rc=0
[supervise 19:35:41] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e26_fashion.json] (attempt 0/10)
```

**2026-10-03 20:21** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e26_fashion/magnitude_sparsified | 259 | 12.0 % | 0.814 | 1.918324e+02 |
| e26_fashion/sparsified | 499 | 23.1 % | 0.807 | 1.491830e+03 |
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 399 | 100.0 % | 0.116 | 4.837949e+06 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |
| e38_w200_neuron_data_seed_4/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |
| e38_w200_neuron_data_seed_4/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |

```
[supervise 19:35:41] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e26_fashion.json] (attempt 0/10)
[supervise 20:06:39] DONE experiments/e26_fashion.json rc=0
[supervise 20:06:39] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e26_fashion.json] (attempt 0/10)
```

**2026-10-03 20:51** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e26_fashion/magnitude_sparsified | 499 | 23.1 % | 0.806 | 9.102411e+02 |
| e26_fashion/obd_sparsified | 150 | 6.9 % | 0.816 | 4.544728e+01 |
| e26_fashion/sparsified | 499 | 23.1 % | 0.807 | 1.491830e+03 |
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 399 | 100.0 % | 0.116 | 4.837949e+06 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |
| e38_w200_neuron_data_seed_4/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |
| e38_w200_neuron_data_seed_4/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |

```
[supervise 20:06:39] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e26_fashion.json] (attempt 0/10)
[supervise 20:34:38] DONE experiments/e26_fashion.json rc=0
[supervise 20:34:38] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e26_fashion.json] (attempt 0/10)
```

**2026-10-03 21:21** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e26_fashion/magnitude_sparsified | 499 | 23.1 % | 0.806 | 9.102411e+02 |
| e26_fashion/obd_sparsified | 409 | 18.9 % | 0.812 | 7.791627e+02 |
| e26_fashion/sparsified | 499 | 23.1 % | 0.807 | 1.491830e+03 |
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 399 | 100.0 % | 0.116 | 4.837949e+06 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |
| e38_w200_neuron_data_seed_4/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |
| e38_w200_neuron_data_seed_4/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |

```
[supervise 20:06:39] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e26_fashion.json] (attempt 0/10)
[supervise 20:34:38] DONE experiments/e26_fashion.json rc=0
[supervise 20:34:38] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e26_fashion.json] (attempt 0/10)
```

**2026-10-03 21:51** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e26_fashion/magnitude_sparsified | 499 | 23.1 % | 0.806 | 9.102411e+02 |
| e26_fashion/obd_sparsified | 499 | 23.1 % | 0.807 | 1.427573e+03 |
| e26_fashion/sparsified | 499 | 23.1 % | 0.807 | 1.491830e+03 |
| e27_kmnist/sparsified | 309 | 14.3 % | 0.584 | 7.654589e+02 |
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 399 | 100.0 % | 0.116 | 4.837949e+06 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |
| e38_w200_neuron_data_seed_4/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |
| e38_w200_neuron_data_seed_4/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |

```
[supervise 21:31:42] launch [/home/simon/venv/general/bin/python3 scripts/zero_cost_fraction.py experiments/e27_kmnist.json] (attempt 0/10)
[supervise 21:31:55] DONE experiments/e27_kmnist.json rc=0
[supervise 21:31:55] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e27_kmnist.json] (attempt 0/10)
```

**2026-10-03 22:21** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e26_fashion/magnitude_sparsified | 499 | 23.1 % | 0.806 | 9.102411e+02 |
| e26_fashion/obd_sparsified | 499 | 23.1 % | 0.807 | 1.427573e+03 |
| e26_fashion/sparsified | 499 | 23.1 % | 0.807 | 1.491830e+03 |
| e27_kmnist/magnitude_sparsified | 291 | 13.5 % | 0.704 | 6.821083e+02 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e38_w200_neuron_data_seed_3/neuron_magnitude_sparsified | 399 | 100.0 % | 0.116 | 4.837949e+06 |
| e38_w200_neuron_data_seed_3/neuron_sparsified | 399 | 100.0 % | 0.099 | 5.256802e+06 |
| e38_w200_neuron_data_seed_4/neuron_magnitude_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |
| e38_w200_neuron_data_seed_4/neuron_sparsified | 399 | 100.0 % | 0.107 | 4.678919e+06 |

```
[supervise 21:31:55] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e27_kmnist.json] (attempt 0/10)
[supervise 22:03:49] DONE experiments/e27_kmnist.json rc=0
[supervise 22:03:49] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e27_kmnist.json] (attempt 0/10)
```

**2026-10-03 22:30** — queue INTERRUPTED by user for the night (Block A complete and analysed; Block B in progress, 47/87 commands done; stopped during: -m sparsifier.magnitude_sparsifier experiments/e27_kmnist.json; @reboot hook removed). Resume: `scripts/run/seedband_launch.sh scripts/run/queue_regimes.txt docs/experiments/2026-10-regimes.md` — skips finished runs; a weight-level run resumes from its checkpoint, a neuron run restarts.

**2026-10-04 09:52** — (re)launch of `scripts/run/queue_regimes.txt`: 39 commands queued

**2026-10-04 09:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 458 | 21.2 % | 0.705 | 2.566574e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |

```
[supervise 22:03:49] DONE experiments/e27_kmnist.json rc=0
[supervise 22:03:49] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e27_kmnist.json] (attempt 0/10)
step  349 | acc=0.7040 | NZ=  1811 | sparsity=0.1616 | d_m=1.1120e+03[supervise 09:52:29] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e27_kmnist.json] (
```

**2026-10-04 10:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 248 | 11.5 % | 0.581 | 4.064381e+02 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |

```
step  349 | acc=0.7040 | NZ=  1811 | sparsity=0.1616 | d_m=1.1120e+03[supervise 09:52:29] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e27_kmnist.json] (
[supervise 09:55:17] DONE experiments/e27_kmnist.json rc=0
[supervise 09:55:17] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e27_kmnist.json] (attempt 0/10)
```

**2026-10-04 10:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/sparsified | 45 | 2.1 % | 0.918 | 5.618712e-03 |

```
[supervise 10:50:28] launch [/home/simon/venv/general/bin/python3 scripts/zero_cost_fraction.py experiments/e28_usps.json] (attempt 0/10)
[supervise 10:50:41] DONE experiments/e28_usps.json rc=0
[supervise 10:50:41] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e28_usps.json] (attempt 0/10)
```

**2026-10-04 11:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 1 | 0.0 % | 0.918 | 5.511580e-10 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |

```
[supervise 10:50:41] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e28_usps.json] (attempt 0/10)
[supervise 11:22:18] DONE experiments/e28_usps.json rc=0
[supervise 11:22:18] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e28_usps.json] (attempt 0/10)
```

**2026-10-04 11:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 6 | 0.3 % | 0.918 | 8.317425e-04 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |

```
[supervise 11:22:18] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e28_usps.json] (attempt 0/10)
[supervise 11:51:40] DONE experiments/e28_usps.json rc=0
[supervise 11:51:40] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e28_usps.json] (attempt 0/10)
```

**2026-10-04 12:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 280 | 13.0 % | 0.917 | 8.138076e+00 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |

```
[supervise 11:22:18] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e28_usps.json] (attempt 0/10)
[supervise 11:51:40] DONE experiments/e28_usps.json rc=0
[supervise 11:51:40] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e28_usps.json] (attempt 0/10)
```

**2026-10-04 12:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/sparsified | 34 | 1.6 % | 0.571 | 3.769477e-04 |

```
[supervise 12:50:08] launch [/home/simon/venv/general/bin/python3 scripts/zero_cost_fraction.py experiments/e29_s1_teacher.json] (attempt 0/10)
[supervise 12:50:22] DONE experiments/e29_s1_teacher.json rc=0
[supervise 12:50:22] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e29_s1_teacher.json] (attempt 0/10)
```

**2026-10-04 13:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 31 | 1.4 % | 0.570 | 1.492440e+00 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |

```
[supervise 12:50:22] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e29_s1_teacher.json] (attempt 0/10)
[supervise 13:20:28] DONE experiments/e29_s1_teacher.json rc=0
[supervise 13:20:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e29_s1_teacher.json] (attempt 0/10)
```

**2026-10-04 13:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 33 | 1.5 % | 0.571 | 3.958497e-03 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |

```
[supervise 13:20:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e29_s1_teacher.json] (attempt 0/10)
[supervise 13:48:45] DONE experiments/e29_s1_teacher.json rc=0
[supervise 13:48:45] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e29_s1_teacher.json] (attempt 0/10)
```

**2026-10-04 14:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 307 | 14.2 % | 0.571 | 6.453080e-01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |

```
[supervise 13:20:28] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e29_s1_teacher.json] (attempt 0/10)
[supervise 13:48:45] DONE experiments/e29_s1_teacher.json rc=0
[supervise 13:48:45] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e29_s1_teacher.json] (attempt 0/10)
```

**2026-10-04 14:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 499 | 23.1 % | 0.569 | 3.805032e+01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |
| e30_s2_planted/sparsified | 87 | 4.0 % | 0.801 | 1.866930e-03 |

```
[supervise 14:47:07] launch [/home/simon/venv/general/bin/python3 scripts/zero_cost_fraction.py experiments/e30_s2_planted.json] (attempt 0/10)
[supervise 14:47:20] DONE experiments/e30_s2_planted.json rc=0
[supervise 14:47:20] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e30_s2_planted.json] (attempt 0/10)
```

**2026-10-04 15:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 499 | 23.1 % | 0.569 | 3.805032e+01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |
| e30_s2_planted/magnitude_sparsified | 93 | 4.3 % | 0.801 | 1.444518e+01 |
| e30_s2_planted/sparsified | 499 | 23.1 % | 0.801 | 2.247713e-01 |

```
[supervise 14:47:20] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e30_s2_planted.json] (attempt 0/10)
[supervise 15:17:11] DONE experiments/e30_s2_planted.json rc=0
[supervise 15:17:11] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e30_s2_planted.json] (attempt 0/10)
```

**2026-10-04 15:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 499 | 23.1 % | 0.569 | 3.805032e+01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |
| e30_s2_planted/magnitude_sparsified | 499 | 23.1 % | 0.800 | 1.058812e+03 |
| e30_s2_planted/obd_sparsified | 66 | 3.1 % | 0.801 | 4.208306e-03 |
| e30_s2_planted/sparsified | 499 | 23.1 % | 0.801 | 2.247713e-01 |

```
[supervise 15:17:11] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e30_s2_planted.json] (attempt 0/10)
[supervise 15:45:00] DONE experiments/e30_s2_planted.json rc=0
[supervise 15:45:00] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e30_s2_planted.json] (attempt 0/10)
```

**2026-10-04 16:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 499 | 23.1 % | 0.569 | 3.805032e+01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |
| e30_s2_planted/magnitude_sparsified | 499 | 23.1 % | 0.800 | 1.058812e+03 |
| e30_s2_planted/obd_sparsified | 252 | 11.7 % | 0.801 | 1.788708e-01 |
| e30_s2_planted/sparsified | 499 | 23.1 % | 0.801 | 2.247713e-01 |

```
[supervise 15:17:11] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e30_s2_planted.json] (attempt 0/10)
[supervise 15:45:00] DONE experiments/e30_s2_planted.json rc=0
[supervise 15:45:00] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e30_s2_planted.json] (attempt 0/10)
```

**2026-10-04 16:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 499 | 23.1 % | 0.569 | 3.805032e+01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |
| e30_s2_planted/magnitude_sparsified | 499 | 23.1 % | 0.800 | 1.058812e+03 |
| e30_s2_planted/obd_sparsified | 499 | 23.1 % | 0.801 | 1.035167e+00 |
| e30_s2_planted/sparsified | 499 | 23.1 % | 0.801 | 2.247713e-01 |

```
[supervise 15:45:00] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e30_s2_planted.json] (attempt 0/10)
[supervise 16:51:40] DONE experiments/e30_s2_planted.json rc=0
[supervise 16:51:40] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e31_s1_neuron_noise.json] (attempt 0/10)
```

**2026-10-04 17:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 499 | 23.1 % | 0.569 | 3.805032e+01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |
| e30_s2_planted/magnitude_sparsified | 499 | 23.1 % | 0.800 | 1.058812e+03 |
| e30_s2_planted/obd_sparsified | 499 | 23.1 % | 0.801 | 1.035167e+00 |
| e30_s2_planted/sparsified | 499 | 23.1 % | 0.801 | 2.247713e-01 |
| e31_s1_neuron_noise/neuron_sparsified | 14 | 70.0 % | 0.233 | 1.683401e+05 |
| e32_s1_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.332 | 6.916703e+04 |
| e33_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.449 | 1.421884e+05 |
| e41_w50_full_seed_0/magnitude_sparsified | 264 | 2.1 % | 0.945 | 2.820614e-01 |

```
[supervise 17:02:43] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e41_w50_full_seed_4.json] (attempt 0/10)
[supervise 17:03:24] DONE experiments/e41_w50_full_seed_4.json rc=0
[supervise 17:03:24] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e41_w50_full_seed_0.json] (attempt 0/10)
```

**2026-10-04 17:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 499 | 23.1 % | 0.569 | 3.805032e+01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |
| e30_s2_planted/magnitude_sparsified | 499 | 23.1 % | 0.800 | 1.058812e+03 |
| e30_s2_planted/obd_sparsified | 499 | 23.1 % | 0.801 | 1.035167e+00 |
| e30_s2_planted/sparsified | 499 | 23.1 % | 0.801 | 2.247713e-01 |
| e31_s1_neuron_noise/neuron_sparsified | 14 | 70.0 % | 0.233 | 1.683401e+05 |
| e32_s1_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.332 | 6.916703e+04 |
| e33_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.449 | 1.421884e+05 |
| e41_w50_full_seed_0/magnitude_sparsified | 680 | 5.3 % | 0.945 | 5.505893e+00 |

```
[supervise 17:02:43] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e41_w50_full_seed_4.json] (attempt 0/10)
[supervise 17:03:24] DONE experiments/e41_w50_full_seed_4.json rc=0
[supervise 17:03:24] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e41_w50_full_seed_0.json] (attempt 0/10)
```

**2026-10-04 18:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 499 | 23.1 % | 0.569 | 3.805032e+01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |
| e30_s2_planted/magnitude_sparsified | 499 | 23.1 % | 0.800 | 1.058812e+03 |
| e30_s2_planted/obd_sparsified | 499 | 23.1 % | 0.801 | 1.035167e+00 |
| e30_s2_planted/sparsified | 499 | 23.1 % | 0.801 | 2.247713e-01 |
| e31_s1_neuron_noise/neuron_sparsified | 14 | 70.0 % | 0.233 | 1.683401e+05 |
| e32_s1_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.332 | 6.916703e+04 |
| e33_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.449 | 1.421884e+05 |
| e41_w50_full_seed_0/magnitude_sparsified | 1098 | 8.6 % | 0.944 | 2.009831e+01 |

```
[supervise 17:02:43] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e41_w50_full_seed_4.json] (attempt 0/10)
[supervise 17:03:24] DONE experiments/e41_w50_full_seed_4.json rc=0
[supervise 17:03:24] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e41_w50_full_seed_0.json] (attempt 0/10)
```

**2026-10-04 18:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 499 | 23.1 % | 0.569 | 3.805032e+01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |
| e30_s2_planted/magnitude_sparsified | 499 | 23.1 % | 0.800 | 1.058812e+03 |
| e30_s2_planted/obd_sparsified | 499 | 23.1 % | 0.801 | 1.035167e+00 |
| e30_s2_planted/sparsified | 499 | 23.1 % | 0.801 | 2.247713e-01 |
| e31_s1_neuron_noise/neuron_sparsified | 14 | 70.0 % | 0.233 | 1.683401e+05 |
| e32_s1_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.332 | 6.916703e+04 |
| e33_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.449 | 1.421884e+05 |
| e41_w50_full_seed_0/magnitude_sparsified | 1477 | 11.5 % | 0.945 | 4.510614e+01 |

```
[supervise 17:02:43] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e41_w50_full_seed_4.json] (attempt 0/10)
[supervise 17:03:24] DONE experiments/e41_w50_full_seed_4.json rc=0
[supervise 17:03:24] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e41_w50_full_seed_0.json] (attempt 0/10)
```

**2026-10-04 19:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 499 | 23.1 % | 0.569 | 3.805032e+01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |
| e30_s2_planted/magnitude_sparsified | 499 | 23.1 % | 0.800 | 1.058812e+03 |
| e30_s2_planted/obd_sparsified | 499 | 23.1 % | 0.801 | 1.035167e+00 |
| e30_s2_planted/sparsified | 499 | 23.1 % | 0.801 | 2.247713e-01 |
| e31_s1_neuron_noise/neuron_sparsified | 14 | 70.0 % | 0.233 | 1.683401e+05 |
| e32_s1_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.332 | 6.916703e+04 |
| e33_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.449 | 1.421884e+05 |
| e41_w50_full_seed_0/magnitude_sparsified | 1748 | 13.7 % | 0.945 | 7.823100e+01 |

```
[supervise 17:02:43] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e41_w50_full_seed_4.json] (attempt 0/10)
[supervise 17:03:24] DONE experiments/e41_w50_full_seed_4.json rc=0
[supervise 17:03:24] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e41_w50_full_seed_0.json] (attempt 0/10)
```

**2026-10-04 19:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 499 | 23.1 % | 0.569 | 3.805032e+01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |
| e30_s2_planted/magnitude_sparsified | 499 | 23.1 % | 0.800 | 1.058812e+03 |
| e30_s2_planted/obd_sparsified | 499 | 23.1 % | 0.801 | 1.035167e+00 |
| e30_s2_planted/sparsified | 499 | 23.1 % | 0.801 | 2.247713e-01 |
| e31_s1_neuron_noise/neuron_sparsified | 14 | 70.0 % | 0.233 | 1.683401e+05 |
| e32_s1_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.332 | 6.916703e+04 |
| e33_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.449 | 1.421884e+05 |
| e41_w50_full_seed_0/magnitude_sparsified | 2100 | 16.4 % | 0.945 | 1.292444e+02 |

```
[supervise 17:02:43] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e41_w50_full_seed_4.json] (attempt 0/10)
[supervise 17:03:24] DONE experiments/e41_w50_full_seed_4.json rc=0
[supervise 17:03:24] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e41_w50_full_seed_0.json] (attempt 0/10)
```

**2026-10-04 20:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 499 | 23.1 % | 0.569 | 3.805032e+01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |
| e30_s2_planted/magnitude_sparsified | 499 | 23.1 % | 0.800 | 1.058812e+03 |
| e30_s2_planted/obd_sparsified | 499 | 23.1 % | 0.801 | 1.035167e+00 |
| e30_s2_planted/sparsified | 499 | 23.1 % | 0.801 | 2.247713e-01 |
| e31_s1_neuron_noise/neuron_sparsified | 14 | 70.0 % | 0.233 | 1.683401e+05 |
| e32_s1_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.332 | 6.916703e+04 |
| e33_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.449 | 1.421884e+05 |
| e41_w50_full_seed_0/magnitude_sparsified | 2501 | 19.5 % | 0.947 | 2.130182e+02 |

```
[supervise 17:02:43] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e41_w50_full_seed_4.json] (attempt 0/10)
[supervise 17:03:24] DONE experiments/e41_w50_full_seed_4.json rc=0
[supervise 17:03:24] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e41_w50_full_seed_0.json] (attempt 0/10)
```

**2026-10-04 20:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 499 | 23.1 % | 0.569 | 3.805032e+01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |
| e30_s2_planted/magnitude_sparsified | 499 | 23.1 % | 0.800 | 1.058812e+03 |
| e30_s2_planted/obd_sparsified | 499 | 23.1 % | 0.801 | 1.035167e+00 |
| e30_s2_planted/sparsified | 499 | 23.1 % | 0.801 | 2.247713e-01 |
| e31_s1_neuron_noise/neuron_sparsified | 14 | 70.0 % | 0.233 | 1.683401e+05 |
| e32_s1_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.332 | 6.916703e+04 |
| e33_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.449 | 1.421884e+05 |
| e41_w50_full_seed_0/magnitude_sparsified | 2919 | 22.8 % | 0.945 | 3.722889e+02 |

```
[supervise 17:02:43] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e41_w50_full_seed_4.json] (attempt 0/10)
[supervise 17:03:24] DONE experiments/e41_w50_full_seed_4.json rc=0
[supervise 17:03:24] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e41_w50_full_seed_0.json] (attempt 0/10)
```

**2026-10-04 21:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e27_kmnist/magnitude_sparsified | 499 | 23.1 % | 0.709 | 3.403265e+03 |
| e27_kmnist/obd_sparsified | 499 | 23.1 % | 0.536 | 3.280454e+03 |
| e27_kmnist/sparsified | 499 | 23.1 % | 0.532 | 3.732659e+03 |
| e28_usps/magnitude_sparsified | 499 | 23.1 % | 0.918 | 8.651375e+02 |
| e28_usps/obd_sparsified | 499 | 23.1 % | 0.918 | 5.271754e+01 |
| e28_usps/sparsified | 499 | 23.1 % | 0.917 | 3.622925e+01 |
| e29_s1_teacher/magnitude_sparsified | 499 | 23.1 % | 0.576 | 2.293963e+03 |
| e29_s1_teacher/obd_sparsified | 499 | 23.1 % | 0.569 | 3.805032e+01 |
| e29_s1_teacher/sparsified | 499 | 23.1 % | 0.569 | 3.804575e+01 |
| e30_s2_planted/magnitude_sparsified | 499 | 23.1 % | 0.800 | 1.058812e+03 |
| e30_s2_planted/obd_sparsified | 499 | 23.1 % | 0.801 | 1.035167e+00 |
| e30_s2_planted/sparsified | 499 | 23.1 % | 0.801 | 2.247713e-01 |
| e31_s1_neuron_noise/neuron_sparsified | 14 | 70.0 % | 0.233 | 1.683401e+05 |
| e32_s1_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.332 | 6.916703e+04 |
| e33_neuron_data/neuron_sparsified | 14 | 70.0 % | 0.449 | 1.421884e+05 |
| e41_w50_full_seed_0/magnitude_sparsified | 3337 | 26.1 % | 0.946 | 5.528312e+02 |

```
[supervise 17:02:43] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e41_w50_full_seed_4.json] (attempt 0/10)
[supervise 17:03:24] DONE experiments/e41_w50_full_seed_4.json rc=0
[supervise 17:03:24] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e41_w50_full_seed_0.json] (attempt 0/10)
```

**e27b result (2026-10-04 21:45) — the KMNIST failure is Ω alone.** Same dense net, same
exhaustive d_W search, Ω = 10 000 KMNIST training images instead of uniform noise: accuracy
0.703 at every logged step through 23.1 % sparsity (min 0.701), identical to magnitude and to
dense. With noise Ω the same search was at 0.583 by step 50 and 0.532 at step 499. The
criterion is sound; the uniform-noise domain is not. This closes HB2's "opposite sign": with a
task-distribution Ω the exhaustive search ties magnitude on KMNIST as it does everywhere else
at this sparsity. Manuscript consequence: replace "data-free" by "label-free" throughout, and
report the noise-Ω runs as the ablation that shows why.

**Block C timing (2026-10-04 21:45) — plan estimate was 4–5× too low.** Magnitude seed 0 runs at
~750 steps/h (adjustment-bound at 12 800 weights): 17 h per magnitude run, not 2. Projected:
magnitude 5 × 17 h, exhaustive 5 × ~25 h (3.7 s search + 4.8 s adjust per step, search shrinking
with N), OBD 2 × ~85 h → **≈ 370 h (15 days)** for the block as queued, against 75 h planned.
Decision referred to the author (options: full; seeds 0–1 only ≈ 85 h; magnitude + exhaustive
seeds 0–1 and no OBD ≈ 60 h; drop C and cite the existing 500-step width sweep).
