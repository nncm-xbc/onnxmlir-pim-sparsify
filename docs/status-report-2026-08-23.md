---
title: Sparsification Thesis — Status Report
date: 2026-08-23
project: A-Posteriori NN Sparsification (Manifold Distance)
repos:
  - onnxmlir-pim-sparsify (branch fable-profiling)
  - ThesisTex (branch fable-draft)
status: experiments-complete-pending-decisions
phase: experiments → writing
tags:
  - thesis
  - sparsification
  - status-report
  - experiments
  - pim
last_campaign: 2026-07-20
sparsify_head: cbd8cbb
thesis_head: 6b6ff6c
---

# Sparsification Thesis — Status Report

> [!abstract] TL;DR
> The full experimental campaign is **done and health-checked**: all ablations, five
> method baselines, six full-budget stupidity runs, a 20-config multi-seed sweep,
> MC-variance, and neuron-level variants. Two findings were **corrected** this cycle:
> the E6 tanh/sigmoid "activations are worse" result (a training artifact — now fixed,
> both ≥ 0.90 dense) and the **width-200 collapse** (a mislabeled-checkpoint artifact —
> the real result is the *opposite*: wider nets prune more gracefully). Remaining work
> is **deferred experiments** (new datasets) and **two decisions** (`backend/compact.py`
> provenance; whether to seed-replicate the deep-sparsity regime).

Related: [[handover]] · [[pim-integration-report]]

---

## 1. Current status

> [!success] Complete and verified
> - **Infrastructure**: resumable runner + watchdog supervisor. 50-job campaign ran with
>   **0 stalls / 0 retries / 0 failures**; every wedge-class failure is now auto-killed and
>   resumed from checkpoint (≤ checkpoint_every steps lost).
> - **Experiments**: 54 sparsification logs on disk, all health-checked (0 NUL bytes,
>   contiguous steps, no NaN/inf). See [Data inventory](#5-data-inventory).
> - **Figures**: headline method comparison, seed-variance bands, MC-variance,
>   neuron-vs-weight, width sweep, stupidity trajectory — all regenerated from current logs.
> - **Thesis**: intro, related work, algorithm, theory, experiments, conclusion drafted on
>   `fable-draft`; builds clean (48 pp, no undefined refs). Width finding integrated
>   (Figure 6.5).

| Area | State | Evidence |
|---|---|---|
| Runner robustness | ✅ done | `sparsifier/runner.py`, `scripts/run/supervise.py` |
| Reference + ablations (e01–e06, e10) | ✅ done | 500-step logs, all healthy |
| Method baselines (magnitude/kwon/obd/obs/lazarevich) | ✅ done | e08/e09/e12/e13/e14 |
| Full-budget stupidity runs (6 selectors) | ✅ done | e05/e15/e16/e17/e18/e19 @ 2160 |
| Multi-seed sweep (e03/e04 × 5 seeds) | ✅ done | 20 configs |
| MC-variance | ✅ done | `artifacts/mc_variance/results.csv` (21 000 rows) |
| Neuron-level variants + width sweep | ✅ done | e11/e20/e21/e22/e23/e24 |
| E6 tanh/sigmoid | ✅ **fixed** | dense 0.912 / 0.905 |
| Width-200 collapse | ✅ **debunked + corrected** | e23/e24 vs e11 |
| New datasets | ⏳ deferred | — |
| `backend/compact.py` | ❓ unresolved | untracked, orphan |

---

## 2. Recent changes and their logic

> [!note] Chronological, with the *why* behind each
> Commits are on `onnxmlir-pim-sparsify@fable-profiling` unless noted.

### 2.1 Resumable runner + watchdog `e885018`
**What**: checkpoints now save `W` **and** `b` (adjust mutates `b`; mask = `W!=0` on load);
auto-resume from the newest checkpoint and truncate the CSV to match; Ω is seeded.
`supervise.py` runs a queue serially, SIGKILLs on a stall (no artifact progress for
`STALL_TIMEOUT`), relaunches, gives up after `MAX_RESTARTS`.
**Why**: the `e16_kwon_full` "3-day process" was a **wedged JAX/CUDA call**, not a code
loop — no internal iteration cap can catch that, only an external progress watchdog. The
single 8 GB GPU forces **serial** execution, so parallelism lives in *authoring*, not runs.

> [!tip] Gotcha worth remembering
> Use `JAX_PLATFORMS=cuda`, **not** `gpu` — this build errors on `gpu` ("Backend 'rocm'
> not known"). The watchdog correctly fast-failed that typo instead of hanging.

### 2.2 Batch campaign `8becc03` → results in `564933a`
**What**: parallel authoring agents produced 20 multi-seed configs, `mc_variance.py`, three
neuron-level sparsifiers (e20/21/22), and full-budget configs (e17/18/19); one serial
watchdog queue ran all 50 jobs.
**Why**: agents have **disjoint file scopes** (skill: dispatching-parallel-agents), so
authoring is safely concurrent; GPU execution is serialized centrally. CSV byte-compatibility
with the old per-file `main()`s was verified so completed baselines stay valid (no re-runs).

### 2.3 E6 tanh/sigmoid fix `753b35d`, `454d296`
**What**: activation-aware Xavier init **plus** per-config `input_scale=255` and
`learning_rate` (tanh 0.5, sigmoid 2.0). Defaults are no-ops → **ReLU byte-identical**
(verified).
**Why**: two coupled root causes, isolated systematically —
> [!warning] Init alone was not enough
> 1. **Input saturation** — inputs are unnormalised (range ≈ [-30, 281]); tanh/sigmoid
>    saturate, ReLU is scale-tolerant. → normalise.
> 2. **JIT-baked LR** — `@jit update` baked `step_size=0.01`, tuned for raw-scale ReLU
>    gradients and far too small once inputs shrink. This is why the init-only fix
>    plateaued at 0.62. → per-config LR set before the first trace.
>
> Result: tanh **0.13 → 0.62 → 0.912**, sigmoid **0.13 → 0.30 → 0.905**. Conclusion for the
> thesis: the dense failure was a **training artifact**, not the activations.

### 2.4 Width-200 debunk + corrected finding `1d5e547`, `c6ea37e`, `cbd8cbb` (+ thesis `6b6ff6c`)
**What**: proved the old `e03_width_200` dense net is **byte-identical to the
`[196,10,10,10]` baseline** (config-topology bug) → the "0.662 collapse" was a mislabeled
baseline on old code. Ran neuron-pruning of the **correct** wide nets (e23/e24 vs e11);
added Figure 6.5 + prose + provenance footnote to the thesis.
**Why**: [[systematic-debugging]] Phase 1 — verify the premise reproduces before "fixing".
Weight-pruning the correct wide net to 23.1 % is **infeasible** (~50 s/step × 18 757 steps
≈ 263 h); neuron pruning (400 neurons, 390 steps, ~1 h) is the feasible probe.

> [!success] Corrected result (opposite of the old claim)
> There is **no width-200 collapse**. At matched neuron-sparsity, wider nets degrade *more*
> gracefully: width-200 holds **0.28 at 97 %** neuron-sparsity, while the narrow baseline
> cliffs to chance at ~70 %. Mild crossover: narrow is marginally better in the 25–55 % band.

---

## 3. Open issues / missing info & data

> [!bug] `backend/compact.py` — unknown provenance
> Appeared in the tree 2026-07-16; **no agent was scoped to `backend/`** and no session
> authored it. Coherent, on-topic (physically removes dead neurons from neuron-pruned nets
> so the PIM crossbar compiler actually benefits — a real gap: zeroed weights don't shrink
> crossbar dims). **Imported by nothing** (orphan). Left **untracked** pending a decision.
> Its companion `docs/pim-integration-report.md` is also untracked.
> **Decision needed**: recognise & wire-in (+test), or delete. → see [Decision D1](#decisions-to-make).

> [!warning] Single-run deep-sparsity regime
> The full-budget stupidity runs (e05/e15/e16/e17/e18/e19) and the neuron/width comparisons
> are **single runs per config**. The seed sweep (§exp-seeds) shows run-to-run spread is
> large (depth_2layer 0.800 ± 0.132). The thesis "Threats to Validity" already flags this;
> the two-regime interpretation **deserves seed replication** before being treated as
> established. → [Decision D2](#decisions-to-make).

> [!missing] Data / experiments not yet collected
> - **New datasets**: full-res 784 MNIST (e07 config exists, needs `data/X_train_full.csv`),
>   Fashion-MNIST (loader), synthetic Gaussian (generator), CIFAR-10 (loader). All need a
>   `dataeng` change first — none are one-command runs.
> - **MC-variance hypothesis untested**: whether MC noise in `d̂_B` *causes* the seed spread
>   (needs common-random-numbers or B-scaled study).
> - **Neuron-level width for other selectors**: only manifold has the width sweep; magnitude/
>   OBD/OBS neuron variants exist but only at baseline width (e20/21/22, 20 neurons).

> [!note] Minor / housekeeping
> - Superseded partial dirs on disk (`e05_..._partial_1221`, `e12_..._partial_430`,
>   `e11_euron_baseline_partial_9`) — harmless, each has a complete sibling.
> - Thesis branch is `fable-draft` (memory once referenced `thesis-complete-draft`).

---

## 4. Future steps & decisions

### Decisions to make

> [!question] D1 — `backend/compact.py`
> **Options**: (a) wire it into the neuron-pruning pipeline + add a test + commit; (b) delete
> it and its report; (c) leave untracked.
> **Logic**: it fills a *real* gap (structured sparsity → smaller crossbars is the actual PIM
> payoff, per [[pim-integration-report]] / [[thesis_pim_integration]]). But un-provenanced,
> untested, unimported code shouldn't be silently committed. **Recommend (a)** *if* you
> confirm it's wanted — it connects the sparsifier to the hardware-contribution chapter.

> [!question] D2 — Replicate the deep-sparsity regime over seeds?
> **Options**: (a) re-run the 6 full-budget stupidity runs × 3–5 seeds (~expensive: OBD-full
> is ~4 h each → days); (b) keep single-run + the existing Threats-to-Validity caveat.
> **Logic**: the *headline* stupidity-point claim is the thesis's core theoretical payoff, so
> a seed band strengthens it — but it is the single most expensive item on the board.
> **Recommend**: replicate only the **manifold** full-budget run over seeds (bounds the claim
> that matters most), leave baselines single-run.

> [!question] D3 — Which new dataset (if any) first?
> **Logic**: the "data-free / structure-agnostic" claim is best tested by the **synthetic
> Gaussian** (cheapest, most diagnostic — a structureless input should give a uniformly random
> sparsity pattern). Fashion-MNIST is the cheapest *real* generalisation check. Full-res 784
> and CIFAR are higher-effort. **Recommend Gaussian → Fashion-MNIST** if pursuing datasets.

### Suggested sequencing

> [!todo] If continuing
> 1. **Resolve D1** (`compact.py`) — small, unblocks the hardware chapter narrative.
> 2. **D2 manifold-only seed band** for the stupidity point — protects the core claim.
> 3. **D3 Gaussian** experiment — directly tests the data-free thesis, cheap.
> 4. Fashion-MNIST for generalisation; full-res / CIFAR only if a reviewer asks.
> 5. Thesis writing pass: fold seed bands + any new datasets into Ch. 6; finalise figures.

---

## 5. Data inventory

> [!info] All logs health-checked (0 NUL, contiguous, finite). `artifacts/` is gitignored.

**Reference & ablations (500 steps)**: `baseline`, `e01_no_adjust`, `e02_omega_{100,1000,100000}`,
`e03_width_{50,200}`, `e04_depth_{2,4}layer`, `e06_act_{tanh,sigmoid}` (retrained), `e10_seed_{1..4}`.

**Method baselines (500 steps)**: `e08_omega_data` (lazarevich), `e09_magnitude_search`,
`e12_kwon`, `e13_obd`, `e14_obs`.

**Full-budget stupidity (2160 steps)**: `e05_stupidity_point` (manifold), `e15_magnitude_full`,
`e16_kwon_full`, `e17_obd_full`, `e18_obs_full`, `e19_lazarevich_full`.

**Multi-seed sweep (500 steps × 5)**: `e03_width_50_seed_{0..4}`, `e03_width_200_seed_{0..4}`,
`e04_depth_2layer_seed_{0..4}`, `e04_depth_4layer_seed_{0..4}`.

**Neuron-level**: `e11_neuron_baseline` (20 n, 15 steps), `e20_neuron_magnitude`,
`e21_neuron_obd`, `e22_neuron_obs` (baseline width, 15 steps),
`e23_neuron_width_200` (400 n, 390 steps), `e24_neuron_width_50` (100 n, 95 steps).

**Derived**: `artifacts/mc_variance/results.csv` (21 000 rows); figures under `images/`
(comparison, seeds, mc_variance, neuron_comparison, width_neuron, runs, …).

### Key numbers

| Result | Value |
|---|---|
| Baseline manifold, 23.1 % sparsity | acc 0.917 (dense 0.916) |
| Selector comparison @ 23.1 % | all baselines within 0.5 pt; exhaustive search buys nothing at moderate sparsity |
| Stupidity point (all 6 selectors) | collapse to ~0.087 at 99.95 % weight sparsity |
| Multi-seed final acc (500 steps) | width_50 0.926±0.020 · width_200 0.962±0.004 · depth_2 0.800±0.132 · depth_4 0.129±0.006 |
| E6 dense (retrained) | tanh 0.912 · sigmoid 0.905 |
| Width neuron-pruning @ 70 % neuron-sparsity | w10 0.256 (cliff) · w50 0.396 · w200 0.459 |
| MC-variance | top-1 stability ≥ 90 % needs B ≥ 10 000 |

---

## 6. Environment / reproduction notes

> [!tip] For the next session
> - GPU: single RTX 2070 SUPER (8 GB) → **serial** GPU jobs. `JAX_PLATFORMS=cuda`.
> - Run a queue: `python scripts/run/supervise.py <queue.txt>` (full command lines; derives
>   the monitored artifact dir from the `.json` arg).
> - Regenerate thesis figures: `python visualize/thesis_figures.py` → writes PDFs into
>   `ThesisTex/main_doc/Images/`.
> - Build thesis: `cd ThesisTex/main_doc/src && latexmk -pdf Thesis.tex` → `../build/Thesis.pdf`.
> - Heads: sparsify `cbd8cbb` · thesis `6b6ff6c`.
