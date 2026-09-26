# Experiment log — D2: seed band for the stupidity point

Started 2026-09-24. Branch `fable-profiling`. Queue: `scripts/run/queue_seedband.txt`.
Analysis: `scripts/analyze_seedband.py` (definitions frozen before launch, see §4).
Raw progress lines are appended automatically to §6 by `scripts/run/seedband_logger.sh`.

## 1. Motivation

The manuscript's deep-sparsity claims (Table "collapse statistics", Section "The deep-sparsity
regime and the stupidity point") rest on **one run per selector** on the seed-0 network, while
the measured seed spread at 23.1 % sparsity is ±6 points. Threats to Validity explicitly asks
for replication over seeds. Two claims are at stake:

- C1 — the exhaustive d_W search collapses at 90.6 % sparsity (step 1 956), with accuracy
  ≥ 0.85 held to 78.3 %.
- C2 — OBD with d_W as loss (second-order selector) "edges" the exhaustive search by one
  point (collapse 91.6 % vs 90.6 %), read as a tie.
- C3 — the displacement blow-up rule (25-step rolling median of d_W > 10× its steps-0–499
  baseline) is a usable early stopping signal (fires at step 1 769 / 81.9 %, 187 steps before
  collapse).

## 2. Hypotheses (pre-registered, written before any new run)

- **H1 (C1 is typical).** Across seeds 0–4 the exhaustive-search collapse sparsity has
  std ≤ 2 points, and seed 0 (90.6 %) lies within the seed range.
  *Refuted if* std > 2 points or seed 0 is an extreme.
- **H2 (C2 is a tie).** On the paired seeds 0–2, |collapse(OBD) − collapse(exhaustive)| ≤ the
  exhaustive seed std, with no consistent sign.
  *Refuted if* OBD collapses later on all three seeds by more than the exhaustive seed std
  (→ "OBD is better", manuscript must say so), or earlier on all three.
- **H3 (C3 generalises).** The stop rule fires before collapse on every run, with lead
  ≥ 50 steps.
  *Refuted if* it fires after collapse, or not at all, on any run.
- **H4 (plateau spread).** Accuracy at step 500 (23.1 %) reproduces a spread of several points
  across seeds (manuscript: 0.865 ± 0.061 on the older e10 nets). Note: e25 nets are **not** the
  e10 nets (training batch sampling was seeded after e10 was trained), so this is a fresh
  sample, not a replication of those four numbers.

## 3. Initial conditions and parameters

| Item | Value |
|---|---|
| Dataset | MNIST 14×14 (Colab 20k/10k sample, PIL bicubic), `data/X_*_small.csv`; train on first 10 000 rows, accuracy on first 1 000 test rows |
| Network | `[196, 10, 10, 10]` ReLU MLP, 2 160 prunable weights, 30 biases (adjusted, never pruned) |
| Training | SGD lr 0.01, 1 000 epochs × 10 batches of 128, seed = run seed (JAX PRNGKey + numpy batch RNG) |
| Ω | uniform on [0,255]^196, B = 10 000 samples, drawn once, `omega_seed` = run seed |
| Budget | 2 160 steps (every prunable weight) |
| Adjustment | as Section 4.3: α₀ = 1e-11, ×1.2 on accept, ÷2 on reject, stop at α ≤ 1e-14 / rel. improvement < 1e-9 / 500 inner iters |
| Checkpoints | every 50 steps (`<run>/<sub>/checkpoints/step_XXXX`), resumable |
| Supervisor | `scripts/run/supervise.py`, STALL_TIMEOUT 600 s, MAX_RESTARTS 5 |
| Hardware | Ryzen 5 3600, RTX 2070 SUPER (driver 580.178.04), Linux 7.0.0-31 |
| Software | JAX 0.6.0 (float64, CUDA), NumPy 2.2.5, C++ search kernel `prune_ext` |
| Code | `onnxmlir-pim-sparsify@e800200` + this log/analysis commit |

Runs (in queue order):

| Run | Selector | Seed (train = Ω) | Paired with | Est. time |
|---|---|---|---|---|
| e25_stupidity_seed_1 | exhaustive d_W | 1 | e34 seed 1 | ~2 h |
| e25_stupidity_seed_2 | exhaustive d_W | 2 | e34 seed 2 | ~2 h |
| e25_stupidity_seed_3 | exhaustive d_W | 3 | — | ~2 h |
| e25_stupidity_seed_4 | exhaustive d_W | 4 | — | ~2 h |
| e34_obd_full_seed_1 | OBD with d_W | 1 | e25 seed 1 | ~4 h |
| e34_obd_full_seed_2 | OBD with d_W | 2 | e25 seed 2 | ~4 h |
| *(existing)* e05_stupidity_point | exhaustive d_W | 0 | e17 | — |
| *(existing)* e17_obd_full | OBD with d_W | 0 | e05 | — |

Pairing check (after the run): `W_0.npy` of e25 seed k and e34 seed k must be byte-identical
(same seeded training on the same device).

## 4. Metric definitions (frozen)

Implemented in `scripts/analyze_seedband.py`, identical to the manuscript:

- **sp≥0.85 / sp≥0.5** — last sparsity at which `val_acc` reaches the threshold.
- **collapse** — first step with acc < 0.5 whose remaining trajectory averages < 0.5.
- **stop rule** — 25-step centred rolling median of per-step displacement `d_W` first exceeds
  10× its baseline (median of that rolling median over steps 0–499).
- **acc@80 / acc@90** — 25-step centred rolling-median accuracy at the step nearest 80 / 90 %.

Validation against the published seed-0 numbers (run before launch):

| Metric | Manuscript | Script |
|---|---|---|
| exhaustive sp≥0.85 / sp≥0.5 / collapse | 78.3 / 90.5 / 90.6 % (step 1 956) | 78.29 / 90.51 / 90.56 % (step 1 956) ✓ |
| OBD sp≥0.85 / sp≥0.5 / collapse | 79.9 / 91.8 / 91.6 % (step 1 979) | 79.86 / 91.81 / 91.62 % (step 1 979) ✓ |
| exhaustive stop rule | step 1 769 (81.9 %) | step 1 769 (81.90 %) ✓ |
| exhaustive acc@80 / acc@90 | 0.831 / 0.565 | 0.829 / 0.549 ✗ |
| OBD acc@80 / acc@90 | 0.845 / 0.584 | 0.842 / 0.577 ✗ |

The manuscript's "rolling accuracies at matched sparsity" could not be reproduced by any
standard window (tried centred/trailing mean/median, 25/51/101 steps, ±0.5/1/2-point sparsity
bands; differences ≤ 0.016). The seed band uses the script definition for **all** seeds including
seed 0, so it is internally consistent; the manuscript's single-run values are left untouched
unless the seed-band update replaces them.

## 5. Results

*(filled in after the queue completes)*

## 6. Progress checkpoints (auto-appended)


**2026-09-24 23:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|

```
[supervise 23:22:44] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_1.json] (attempt 0/5)
```

**2026-09-24 23:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 511 | 23.7 % | 0.786 | 2.979668e+02 |

```
[supervise 23:22:44] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_1.json] (attempt 0/5)
[supervise 23:23:21] DONE experiments/e25_stupidity_seed_1.json rc=0
[supervise 23:23:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json] (attempt 0/5)
```

**2026-09-25 00:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 1008 | 46.7 % | 0.765 | 5.894597e+03 |

```
[supervise 23:22:44] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_1.json] (attempt 0/5)
[supervise 23:23:21] DONE experiments/e25_stupidity_seed_1.json rc=0
[supervise 23:23:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json] (attempt 0/5)
```

**2026-09-25 00:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 1519 | 70.3 % | 0.722 | 7.141308e+04 |

```
[supervise 23:22:44] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_1.json] (attempt 0/5)
[supervise 23:23:21] DONE experiments/e25_stupidity_seed_1.json rc=0
[supervise 23:23:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json] (attempt 0/5)
```

**2026-09-26 — interruption.** The workstation went down at ~00:52 on 2026-09-25 (last log
write; host up again 07:56) with e25 seed 1 at step 1 521 / 2 160 (latest checkpoint
`step_1500`). Supervisor and logger did not survive the reboot. Relaunched with
`scripts/run/seedband_launch.sh`, which skips training for nets that already exist (retraining
would replace the reference network mid-run) and resumes from the checkpoint: the runner
truncates the CSV to step 1 500 and continues from 1 501 with the dense seed-1 net and the same
seeded Ω. A `@reboot` crontab entry now calls the launcher so a further outage self-heals.

**2026-09-26 18:52** — (re)launch: 11 commands queued

**2026-09-26 18:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 1500 | 69.4 % | 0.722 | 6.411619e+04 |

```
[supervise 23:22:44] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_1.json] (attempt 0/5)
[supervise 23:23:21] DONE experiments/e25_stupidity_seed_1.json rc=0
[supervise 23:23:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json] (attempt 0/5)
step 1411 | acc=0.7410 | NZ=   749 | sparsity=0.6532 | d_m=4.5264e+04[supervise 18:52:01] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json] (attempt 0/5)
```
