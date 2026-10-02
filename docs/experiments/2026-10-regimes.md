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

*(after each block completes)*

## 6. Progress checkpoints (auto-appended)
