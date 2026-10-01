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

Queue completed 2026-09-27 09:43 (all 11 commands rc=0; one host outage, see §6). Pairing check:
`W_0.npy` of e25 seed k and e34 seed k are byte-identical for k = 1, 2 ✓. Seed 0 = existing
e05 / e17. Numbers from `python scripts/analyze_seedband.py`.

### 5.1 Per run

| Selector | Seed | Dense | Acc @ step 500 (23.1 %) | Last sp. acc ≥ 0.85 | Last sp. acc ≥ 0.5 | Collapse (step) | Stop rule fires (acc then) | Lead [steps] |
|---|---|---|---|---|---|---|---|---|
| exhaustive | 0 | 0.916 | 0.915 | 78.3 % | 90.5 % | 90.6 % (1 956) | 1 769 (0.772) | 187 |
| exhaustive | 1 | 0.909 | 0.785 | 18.0 % | 88.2 % | 88.2 % (1 906) | 415 (0.809) | 1 491 |
| exhaustive | 2 | 0.911 | 0.909 | 55.8 % | 87.5 % | 87.6 % (1 892) | 1 996 (0.175) | −104 |
| exhaustive | 3 | 0.916 | 0.893 | 32.4 % | 86.9 % | 86.8 % (1 875) | 361 (0.897) | 1 514 |
| exhaustive | 4 | 0.920 | 0.920 | 79.4 % | 88.5 % | 88.6 % (1 913) | 1 629 (0.849) | 284 |
| OBD | 0 | 0.916 | 0.917 | 79.9 % | 91.8 % | 91.6 % (1 979) | 1 636 (0.859) | 343 |
| OBD | 1 | 0.909 | 0.861 | 23.8 % | 88.4 % | 88.5 % (1 911) | 1 877 (0.541) | 34 |
| OBD | 2 | 0.911 | 0.911 | 56.9 % | 88.2 % | 88.3 % (1 907) | 2 006 (0.183) | −99 |

### 5.2 Aggregates (mean ± std [min, max])

| Metric | Exhaustive (5 seeds) | OBD (3 seeds) |
|---|---|---|
| Dense accuracy | 0.914 ± 0.004 | 0.912 ± 0.004 |
| Acc @ 23.1 % | 0.884 ± 0.057 [0.785, 0.920] | 0.896 ± 0.031 [0.861, 0.917] |
| Last sparsity acc ≥ 0.85 | 52.8 ± 27.4 % [18.0, 79.4] | 53.5 ± 28.2 % [23.8, 79.9] |
| Last sparsity acc ≥ 0.5 | 88.3 ± 1.4 % [86.9, 90.5] | 89.5 ± 2.0 % [88.2, 91.8] |
| **Collapse sparsity** | **88.4 ± 1.4 %** [86.8, 90.6] | **89.5 ± 1.9 %** [88.3, 91.6] |
| Acc @ 80 % (rolling median) | 0.713 ± 0.112 | 0.721 ± 0.105 |
| Acc @ 90 % (rolling median) | 0.400 ± 0.103 | 0.453 ± 0.109 |

Paired OBD − exhaustive collapse sparsity: seed 0 **+1.06**, seed 1 **+0.23**, seed 2 **+0.69**
points (mean +0.66).

### 5.3 Hypothesis verdicts

- **H1 — partially refuted.** The collapse point is tight (std 1.4 ≤ 2 points ✓), but seed 0 is
  the **maximum** of the five seeds (90.6 % vs 86.8–88.6 % for seeds 1–4), which the
  pre-registered criterion counts as a refutation. The published single-run number overstates
  the typical collapse by ~2 points: the band is **88.4 ± 1.4 %**. The "acc ≥ 0.85 held to
  78.3 %" figure is far from typical: across seeds it ranges 18–79 % (mean 53 %). Seeds 1 and 3
  lose several points already in the plateau (0.785 and 0.893 at 23.1 %), so the *knee* is
  highly seed-dependent while the *collapse* is not.
- **H2 — supported in magnitude, with a consistent sign.** OBD collapses later on all three
  paired seeds, by 0.2–1.1 points — every difference below the exhaustive seed std (1.4), so the
  refutation threshold is not met and "tied within noise" stands. But the direction is
  consistent (3/3; probability 1/4 under a symmetric null, so not significant at n = 3). Honest
  reading: *OBD matches the exhaustive search at collapse and is, if anything, marginally ahead.*
- **H3 — refuted.** The displacement blow-up rule is not a reliable stopping signal. It fired
  after collapse on 2 of 8 runs (exhaustive seed 2, OBD seed 2), with a lead of only 34 steps
  on a third (OBD seed 1), and spuriously early on 2 runs (exhaustive seeds 1 and 3, at 16–19 %
  sparsity, while accuracy was 0.81–0.90). Only 3 of 8 runs show the seed-0 behaviour (fires
  shortly before collapse with accuracy still ≥ 0.77). The manuscript's "usable stopping signal,
  not a sharp one" must be withdrawn or restricted to seed 0.
- **H4 — supported.** Accuracy at 23.1 % sparsity is 0.884 ± 0.057 (range 0.785–0.920),
  matching the spread previously reported on the e10 nets (0.865 ± 0.061).

### 5.4 Consequences for the manuscript

1. Table "collapse statistics": replace the single-run exhaustive and OBD rows by seed means ± std
   (5 and 3 seeds); keep magnitude and Kwon as single seed-0 runs, marked as such.
2. The comparison with **magnitude** (single run, collapse 88.9 %) is no longer supported: it lies
   inside the exhaustive seed range [86.8, 90.6]. The "four points over magnitude" claim must go
   until magnitude is replicated over seeds. The gap to the **Kwon** first-order score (77.6 %,
   single run) is ~9 points below the lowest exhaustive seed and survives.
3. OBD vs exhaustive: "tied within noise, OBD marginally later on every paired seed".
4. Stopping rule: withdraw as a general signal; report it as seed-0 only and refuted across seeds.
5. The "knee" (acc ≥ 0.85) is strongly seed-dependent and should be reported as a range.

Suggested follow-up (not run): magnitude and Kwon full-budget runs on seeds 1–4 (~35 + 28 min
each, ~4 h total) to restore the selector ordering claim on equal footing.

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

**2026-09-26 19:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2028 | 93.9 % | 0.242 | 1.178659e+06 |

```
[supervise 23:22:44] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_1.json] (attempt 0/5)
[supervise 23:23:21] DONE experiments/e25_stupidity_seed_1.json rc=0
[supervise 23:23:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json] (attempt 0/5)
step 1411 | acc=0.7410 | NZ=   749 | sparsity=0.6532 | d_m=4.5264e+04[supervise 18:52:01] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json] (attempt 0/5)
```

**2026-09-26 19:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 363 | 16.8 % | 0.911 | 7.721959e+00 |

```
[supervise 19:29:20] DONE experiments/e25_stupidity_seed_1.json rc=0
[supervise 19:29:20] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json] (attempt 0/5)
[supervise 19:29:57] DONE experiments/e25_stupidity_seed_2.json rc=0
[supervise 19:29:57] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json] (attempt 0/5)
```

**2026-09-26 20:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 864 | 40.0 % | 0.906 | 7.136466e+02 |

```
[supervise 19:29:20] DONE experiments/e25_stupidity_seed_1.json rc=0
[supervise 19:29:20] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json] (attempt 0/5)
[supervise 19:29:57] DONE experiments/e25_stupidity_seed_2.json rc=0
[supervise 19:29:57] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json] (attempt 0/5)
```

**2026-09-26 20:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 1373 | 63.6 % | 0.786 | 2.405757e+04 |

```
[supervise 19:29:20] DONE experiments/e25_stupidity_seed_1.json rc=0
[supervise 19:29:20] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json] (attempt 0/5)
[supervise 19:29:57] DONE experiments/e25_stupidity_seed_2.json rc=0
[supervise 19:29:57] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json] (attempt 0/5)
```

**2026-09-26 21:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 1895 | 87.7 % | 0.484 | 3.419168e+05 |

```
[supervise 19:29:20] DONE experiments/e25_stupidity_seed_1.json rc=0
[supervise 19:29:20] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json] (attempt 0/5)
[supervise 19:29:57] DONE experiments/e25_stupidity_seed_2.json rc=0
[supervise 19:29:57] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json] (attempt 0/5)
```

**2026-09-26 21:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 253 | 11.7 % | 0.919 | 9.195321e+01 |

```
[supervise 21:36:52] DONE experiments/e25_stupidity_seed_2.json rc=0
[supervise 21:36:52] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json] (attempt 0/5)
[supervise 21:37:29] DONE experiments/e25_stupidity_seed_3.json rc=0
[supervise 21:37:29] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json] (attempt 0/5)
```

**2026-09-26 22:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 747 | 34.6 % | 0.840 | 2.356692e+03 |

```
[supervise 21:36:52] DONE experiments/e25_stupidity_seed_2.json rc=0
[supervise 21:36:52] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json] (attempt 0/5)
[supervise 21:37:29] DONE experiments/e25_stupidity_seed_3.json rc=0
[supervise 21:37:29] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json] (attempt 0/5)
```

**2026-09-26 22:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 1251 | 57.9 % | 0.649 | 1.807190e+04 |

```
[supervise 21:36:52] DONE experiments/e25_stupidity_seed_2.json rc=0
[supervise 21:36:52] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json] (attempt 0/5)
[supervise 21:37:29] DONE experiments/e25_stupidity_seed_3.json rc=0
[supervise 21:37:29] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json] (attempt 0/5)
```

**2026-09-26 23:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 1767 | 81.8 % | 0.580 | 1.878833e+05 |

```
[supervise 21:36:52] DONE experiments/e25_stupidity_seed_2.json rc=0
[supervise 21:36:52] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json] (attempt 0/5)
[supervise 21:37:29] DONE experiments/e25_stupidity_seed_3.json rc=0
[supervise 21:37:29] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json] (attempt 0/5)
```

**2026-09-26 23:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 152 | 7.0 % | 0.920 | 1.088360e-02 |

```
[supervise 23:44:09] DONE experiments/e25_stupidity_seed_3.json rc=0
[supervise 23:44:09] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json] (attempt 0/5)
[supervise 23:44:46] DONE experiments/e25_stupidity_seed_4.json rc=0
[supervise 23:44:46] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json] (attempt 0/5)
```

**2026-09-27 00:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 644 | 29.8 % | 0.916 | 1.661336e+03 |

```
[supervise 23:44:09] DONE experiments/e25_stupidity_seed_3.json rc=0
[supervise 23:44:09] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json] (attempt 0/5)
[supervise 23:44:46] DONE experiments/e25_stupidity_seed_4.json rc=0
[supervise 23:44:46] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json] (attempt 0/5)
```

**2026-09-27 00:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 1148 | 53.1 % | 0.909 | 2.572736e+04 |

```
[supervise 23:44:09] DONE experiments/e25_stupidity_seed_3.json rc=0
[supervise 23:44:09] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json] (attempt 0/5)
[supervise 23:44:46] DONE experiments/e25_stupidity_seed_4.json rc=0
[supervise 23:44:46] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json] (attempt 0/5)
```

**2026-09-27 01:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 1662 | 76.9 % | 0.835 | 1.837600e+05 |

```
[supervise 23:44:09] DONE experiments/e25_stupidity_seed_3.json rc=0
[supervise 23:44:09] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json] (attempt 0/5)
[supervise 23:44:46] DONE experiments/e25_stupidity_seed_4.json rc=0
[supervise 23:44:46] launch [/home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json] (attempt 0/5)
```

**2026-09-27 01:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 11 | 0.5 % | 0.909 | 7.248228e-05 |

```
[supervise 01:50:08] DONE experiments/e25_stupidity_seed_4.json rc=0
[supervise 01:50:08] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json] (attempt 0/5)
[supervise 01:50:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 01:50:44] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json] (attempt 0/5)
```

**2026-09-27 02:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 287 | 13.3 % | 0.905 | 5.268352e+01 |

```
[supervise 01:50:08] DONE experiments/e25_stupidity_seed_4.json rc=0
[supervise 01:50:08] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json] (attempt 0/5)
[supervise 01:50:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 01:50:44] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json] (attempt 0/5)
```

**2026-09-27 02:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 562 | 26.0 % | 0.820 | 5.762594e+02 |

```
[supervise 01:50:08] DONE experiments/e25_stupidity_seed_4.json rc=0
[supervise 01:50:08] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json] (attempt 0/5)
[supervise 01:50:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 01:50:44] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json] (attempt 0/5)
```

**2026-09-27 03:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 837 | 38.8 % | 0.774 | 2.381645e+03 |

```
[supervise 01:50:08] DONE experiments/e25_stupidity_seed_4.json rc=0
[supervise 01:50:08] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json] (attempt 0/5)
[supervise 01:50:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 01:50:44] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json] (attempt 0/5)
```

**2026-09-27 03:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 1112 | 51.5 % | 0.762 | 9.437218e+03 |

```
[supervise 01:50:08] DONE experiments/e25_stupidity_seed_4.json rc=0
[supervise 01:50:08] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json] (attempt 0/5)
[supervise 01:50:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 01:50:44] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json] (attempt 0/5)
```

**2026-09-27 04:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 1386 | 64.2 % | 0.740 | 4.009818e+04 |

```
[supervise 01:50:08] DONE experiments/e25_stupidity_seed_4.json rc=0
[supervise 01:50:08] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json] (attempt 0/5)
[supervise 01:50:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 01:50:44] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json] (attempt 0/5)
```

**2026-09-27 04:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 1660 | 76.9 % | 0.699 | 1.373448e+05 |

```
[supervise 01:50:08] DONE experiments/e25_stupidity_seed_4.json rc=0
[supervise 01:50:08] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json] (attempt 0/5)
[supervise 01:50:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 01:50:44] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json] (attempt 0/5)
```

**2026-09-27 05:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 1933 | 89.5 % | 0.417 | 5.577757e+05 |

```
[supervise 01:50:08] DONE experiments/e25_stupidity_seed_4.json rc=0
[supervise 01:50:08] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json] (attempt 0/5)
[supervise 01:50:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 01:50:44] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json] (attempt 0/5)
```

**2026-09-27 05:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 42 | 1.9 % | 0.911 | 7.960641e-02 |

```
[supervise 05:46:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 05:46:44] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
```

**2026-09-27 06:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 317 | 14.7 % | 0.911 | 8.766810e+00 |

```
[supervise 05:46:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 05:46:44] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
```

**2026-09-27 06:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 592 | 27.4 % | 0.911 | 7.491715e+01 |

```
[supervise 05:46:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 05:46:44] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
```

**2026-09-27 07:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 867 | 40.1 % | 0.906 | 7.757505e+02 |

```
[supervise 05:46:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 05:46:44] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
```

**2026-09-27 07:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 1141 | 52.8 % | 0.899 | 6.055070e+03 |

```
[supervise 05:46:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 05:46:44] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
```

**2026-09-27 08:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 1415 | 65.5 % | 0.792 | 2.884885e+04 |

```
[supervise 05:46:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 05:46:44] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
```

**2026-09-27 08:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 1689 | 78.2 % | 0.635 | 1.135762e+05 |

```
[supervise 05:46:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 05:46:44] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
```

**2026-09-27 09:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 1963 | 90.9 % | 0.295 | 5.472491e+05 |

```
[supervise 05:46:44] DONE experiments/e34_obd_full_seed_1.json rc=0
[supervise 05:46:44] launch [/home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
```

**2026-09-27 09:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 10:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 10:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 11:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 11:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 12:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 12:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 13:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 13:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 14:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 14:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 15:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 15:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 16:22** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 16:52** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e25_seed_1 | 2159 | 100.0 % | 0.107 | 2.431626e+06 |
| e25_seed_2 | 2159 | 100.0 % | 0.107 | 1.567784e+06 |
| e25_seed_3 | 2159 | 100.0 % | 0.087 | 2.302118e+06 |
| e25_seed_4 | 2159 | 100.0 % | 0.107 | 2.013833e+06 |
| e34_seed_1 | 2159 | 100.0 % | 0.107 | 2.439091e+06 |
| e34_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |

```
[supervise 05:47:21] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 05:47:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json] (attempt 0/5)
[supervise 09:43:05] DONE experiments/e34_obd_full_seed_2.json rc=0
[supervise 09:43:05] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_2.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_3.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.sparsifier experiments/e25_stupidity_seed_4.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_1.json=ok, /home/simon/venv/general/bin/python3 scripts/train.py experiments/e34_obd_full_seed_2.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_2.json=ok
```

**2026-09-27 17:10** — queue complete (09:43). The completion watcher and the logger failed to
exit because their own command lines matched the supervisor `pgrep` pattern (bug in the
watcher, not in the runs); both were stopped manually and the `@reboot` crontab entry removed.

---

# Extension (queued 2026-09-27): baseline selectors over seeds

## E.1 Motivation

§5.4 left two manuscript claims resting on single seed-0 runs: the position of **magnitude**
(collapse 88.9 %, inside the exhaustive seed range) and of the **first-order Kwon score**
(77.6 %). OBD covers only seeds 0–2, and **OBS** (layer-wise, no adjustment; 71.5 %) is a
single run. This extension gives every selector five seeds on the same five dense networks.

## E.2 Hypotheses (pre-registered)

- **H5 (magnitude).** Paired over seeds 0–4, collapse(exhaustive) − collapse(magnitude) has
  mean within ±1 point of zero (tie), i.e. the seed-0 gap of 1.7 points does not persist.
  *Refuted if* the exhaustive search collapses later on ≥ 4 of 5 seeds with mean gap > 1 point
  (→ the manuscript may restore an advantage over magnitude).
- **H6 (first-order).** Kwon collapses ≥ 5 points earlier than the exhaustive search on every
  seed. *Refuted if* any seed shows a gap < 5 points.
- **H7 (OBD).** With five paired seeds, mean collapse(OBD) − collapse(exhaustive) stays within
  one exhaustive std (1.4 points). *Refuted if* it exceeds it (→ "OBD better").
- **H8 (OBS).** Layer-wise OBS without adjustment collapses before every adjusted selector except
  Kwon on every seed. *Refuted if* any seed has OBS collapsing after the exhaustive search.

## E.3 Conditions

Identical to §3, except: no training — each run's dense `W_*/b_*` are byte-identical copies of
`e25_stupidity_seed_k` (checked with `cmp` before launch, 14/14 ✓), and `omega_seed` = k. Seed 0
of every selector is the existing run (e05, e15, e16, e17, e18).

| Runs | Module | Seeds | Est. per run (from seed 0) |
|---|---|---|---|
| e35_magnitude_full_seed_k | magnitude_sparsifier | 1–4 | ~2.0 h |
| e36_kwon_full_seed_k | kwon_sparsifier | 1–4 | ~2.0 h |
| e34_obd_full_seed_k | obd_sparsifier | 3–4 | ~3.9 h |
| e37_obs_full_seed_k | obs_sparsifier (no adjustment) | 1–4 | ~1.6 h |

Total ≈ 30 h, queue `scripts/run/queue_seedband2.txt`, launched with
`scripts/run/seedband_launch.sh` (now pidfile-based, skips finished runs; `@reboot` hook
re-installed). Metrics: `scripts/analyze_seedband.py` (extended to five selectors; seed-0
magnitude/Kwon/OBS rows reproduce the manuscript table exactly).

## E.4 Progress checkpoints (auto-appended)

**2026-09-27 18:20** — (re)launch of `scripts/run/queue_seedband2.txt`: 14 commands queued

**2026-09-27 18:20** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|

```
[supervise 18:20:55] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_1.json] (attempt 0/5)
```

**2026-09-27 18:50** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 494 | 22.9 % | 0.905 | 1.117056e+03 |

```
[supervise 18:20:55] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_1.json] (attempt 0/5)
```

**2026-09-27 20:01** — (re)launch of `scripts/run/queue_seedband2.txt`: 14 commands queued

**2026-09-27 20:01** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|

```
[supervise 18:20:55] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_1.json] (attempt 0/5)
step  821 | acc=0.9030 | NZ=  1339 | sparsity=0.3801 | d_m=5.6552e+03[supervise 20:01:36] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full
```

**2026-09-28 07:50** — (re)launch of `scripts/run/queue_seedband2.txt`: 14 commands queued

**2026-09-28 07:50** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_2 | 240 | 11.1 % | 0.911 | 1.168168e+02 |

```
[supervise 20:01:54] EXIT experiments/e35_magnitude_full_seed_1.json rc=1 -- will resume from checkpoint
[supervise 20:01:54] GAVE UP on experiments/e35_magnitude_full_seed_1.json after 5 restarts
[supervise 20:01:54] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_2.json] (attempt 0/5)
```

**2026-09-28 08:30 — two host crashes + corrupt checkpoint.** Timeline from `last -x` and
`journalctl`: the host went down abruptly at ~19:15 on 09-27 (e35 magnitude seed 1 at step
823), rebooted at 20:00 (kernel 7.0.0-31 → -34), and went down again at ~20:16; it was back at
07:49 on 09-28. Neither boot's journal records a kernel, GPU (NVRM/Xid), OOM or thermal event
before it ends — consistent with a hard freeze or power loss, as on 09-25 at 00:52. All three
stops happened under sustained GPU load. Both relaunches (20:01, 07:51) failed on seed 1 with
`EOFError`: checkpoint `step_0850` existed but its six `.npy` files were zero-length (page
cache lost), and the CSV tail was NUL bytes; the supervisor gave up on seed 1 and moved to seed 2.
Fix in `sparsifier/runner.py`: checkpoint discovery skips checkpoints that fail to load
(seed 1 now resumes from `step_0800`; the CSV is truncated to it), and checkpoints are written
atomically (temp dir + fsync + rename; the CSV is fsynced at each checkpoint). Tests 56/56.
The launcher now also records GPU telemetry every 10 s (`artifacts/seedband/gpu_telemetry.csv`:
temperature, power, utilisation, SM clock, throttle reasons) to diagnose further crashes. Data up
to each run's last valid checkpoint is unaffected; lost steps are recomputed deterministically.

**2026-09-28 07:56** — (re)launch of `scripts/run/queue_seedband2.txt`: 14 commands queued

**2026-09-28 07:56** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_2 | 256 | 11.9 % | 0.911 | 1.310947e+02 |

```
[supervise 07:51:21] GAVE UP on experiments/e35_magnitude_full_seed_1.json after 5 restarts
[supervise 07:51:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_2.json] (attempt 0/5)
[supervise 07:56:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_1.json] (attempt 0/5)
```

**2026-09-28 08:26** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 1344 | 62.2 % | 0.868 | 4.993094e+04 |
| e35_magnitude_full_seed_2 | 256 | 11.9 % | 0.911 | 1.310947e+02 |

```
[supervise 07:51:21] GAVE UP on experiments/e35_magnitude_full_seed_1.json after 5 restarts
[supervise 07:51:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_2.json] (attempt 0/5)
[supervise 07:56:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_1.json] (attempt 0/5)
```

**2026-09-28 08:56** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 1889 | 87.5 % | 0.511 | 6.125446e+05 |
| e35_magnitude_full_seed_2 | 256 | 11.9 % | 0.911 | 1.310947e+02 |

```
[supervise 07:51:21] GAVE UP on experiments/e35_magnitude_full_seed_1.json after 5 restarts
[supervise 07:51:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_2.json] (attempt 0/5)
[supervise 07:56:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_1.json] (attempt 0/5)
```

**2026-09-28 09:26** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 532 | 24.6 % | 0.910 | 1.244374e+03 |

```
[supervise 07:56:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_1.json] (attempt 0/5)
[supervise 09:10:43] DONE experiments/e35_magnitude_full_seed_1.json rc=0
[supervise 09:10:43] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_2.json] (attempt 0/5)
```

**2026-09-28 09:56** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 1076 | 49.8 % | 0.902 | 1.269161e+04 |

```
[supervise 07:56:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_1.json] (attempt 0/5)
[supervise 09:10:43] DONE experiments/e35_magnitude_full_seed_1.json rc=0
[supervise 09:10:43] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_2.json] (attempt 0/5)
```

**2026-09-28 10:26** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 1620 | 75.0 % | 0.772 | 1.346684e+05 |

```
[supervise 07:56:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_1.json] (attempt 0/5)
[supervise 09:10:43] DONE experiments/e35_magnitude_full_seed_1.json rc=0
[supervise 09:10:43] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_2.json] (attempt 0/5)
```

**2026-09-28 10:56** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 23 | 1.1 % | 0.916 | 1.824837e-01 |

```
[supervise 09:10:43] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_2.json] (attempt 0/5)
[supervise 10:54:55] DONE experiments/e35_magnitude_full_seed_2.json rc=0
[supervise 10:54:55] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_3.json] (attempt 0/5)
```

**2026-09-28 11:26** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 566 | 26.2 % | 0.916 | 1.377734e+03 |

```
[supervise 09:10:43] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_2.json] (attempt 0/5)
[supervise 10:54:55] DONE experiments/e35_magnitude_full_seed_2.json rc=0
[supervise 10:54:55] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_3.json] (attempt 0/5)
```

**2026-09-28 11:56** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 1109 | 51.3 % | 0.914 | 2.067358e+04 |

```
[supervise 09:10:43] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_2.json] (attempt 0/5)
[supervise 10:54:55] DONE experiments/e35_magnitude_full_seed_2.json rc=0
[supervise 10:54:55] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_3.json] (attempt 0/5)
```

**2026-09-28 12:26** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 1652 | 76.5 % | 0.858 | 1.645801e+05 |

```
[supervise 09:10:43] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_2.json] (attempt 0/5)
[supervise 10:54:55] DONE experiments/e35_magnitude_full_seed_2.json rc=0
[supervise 10:54:55] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_3.json] (attempt 0/5)
```

**2026-09-28 12:56** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 42 | 1.9 % | 0.920 | 1.612114e+00 |

```
[supervise 10:54:55] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_3.json] (attempt 0/5)
[supervise 12:53:53] DONE experiments/e35_magnitude_full_seed_3.json rc=0
[supervise 12:53:53] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_4.json] (attempt 0/5)
```

**2026-09-28 13:26** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 584 | 27.0 % | 0.918 | 3.093140e+03 |

```
[supervise 10:54:55] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_3.json] (attempt 0/5)
[supervise 12:53:53] DONE experiments/e35_magnitude_full_seed_3.json rc=0
[supervise 12:53:53] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_4.json] (attempt 0/5)
```

**2026-09-28 13:56** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 1127 | 52.2 % | 0.916 | 3.040703e+04 |

```
[supervise 10:54:55] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_3.json] (attempt 0/5)
[supervise 12:53:53] DONE experiments/e35_magnitude_full_seed_3.json rc=0
[supervise 12:53:53] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_4.json] (attempt 0/5)
```

**2026-09-28 14:26** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 1668 | 77.2 % | 0.831 | 2.412745e+05 |

```
[supervise 10:54:55] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_3.json] (attempt 0/5)
[supervise 12:53:53] DONE experiments/e35_magnitude_full_seed_3.json rc=0
[supervise 12:53:53] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_4.json] (attempt 0/5)
```

**2026-09-28 14:56** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013176e+06 |
| e36_kwon_full_seed_1 | 62 | 2.9 % | 0.906 | 2.449136e+03 |

```
[supervise 12:53:53] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_4.json] (attempt 0/5)
[supervise 14:52:47] DONE experiments/e35_magnitude_full_seed_4.json rc=0
[supervise 14:52:47] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_1.json] (attempt 0/5)
```

**2026-09-28 15:26** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013176e+06 |
| e36_kwon_full_seed_1 | 604 | 28.0 % | 0.888 | 8.399718e+03 |

```
[supervise 12:53:53] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_4.json] (attempt 0/5)
[supervise 14:52:47] DONE experiments/e35_magnitude_full_seed_4.json rc=0
[supervise 14:52:47] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_1.json] (attempt 0/5)
```

**2026-09-28 15:56** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013176e+06 |
| e36_kwon_full_seed_1 | 1144 | 53.0 % | 0.661 | 5.661930e+04 |

```
[supervise 12:53:53] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_4.json] (attempt 0/5)
[supervise 14:52:47] DONE experiments/e35_magnitude_full_seed_4.json rc=0
[supervise 14:52:47] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_1.json] (attempt 0/5)
```

**2026-09-28 16:26** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013176e+06 |
| e36_kwon_full_seed_1 | 1685 | 78.0 % | 0.456 | 3.411505e+05 |

```
[supervise 12:53:53] launch [/home/simon/venv/general/bin/python3 -m sparsifier.magnitude_sparsifier experiments/e35_magnitude_full_seed_4.json] (attempt 0/5)
[supervise 14:52:47] DONE experiments/e35_magnitude_full_seed_4.json rc=0
[supervise 14:52:47] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_1.json] (attempt 0/5)
```

**2026-09-28 16:56** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013176e+06 |
| e36_kwon_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e36_kwon_full_seed_2 | 70 | 3.2 % | 0.909 | 2.507232e+02 |

```
[supervise 14:52:47] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_1.json] (attempt 0/5)
[supervise 16:52:19] DONE experiments/e36_kwon_full_seed_1.json rc=0
[supervise 16:52:19] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_2.json] (attempt 0/5)
```

**2026-09-28 17:26** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013176e+06 |
| e36_kwon_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e36_kwon_full_seed_2 | 610 | 28.2 % | 0.874 | 8.282588e+03 |

```
[supervise 14:52:47] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_1.json] (attempt 0/5)
[supervise 16:52:19] DONE experiments/e36_kwon_full_seed_1.json rc=0
[supervise 16:52:19] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_2.json] (attempt 0/5)
```

**2026-09-28 17:56** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013176e+06 |
| e36_kwon_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e36_kwon_full_seed_2 | 1152 | 53.3 % | 0.677 | 4.951363e+04 |

```
[supervise 14:52:47] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_1.json] (attempt 0/5)
[supervise 16:52:19] DONE experiments/e36_kwon_full_seed_1.json rc=0
[supervise 16:52:19] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_2.json] (attempt 0/5)
```

**2026-09-28 18:26** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013176e+06 |
| e36_kwon_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e36_kwon_full_seed_2 | 1696 | 78.5 % | 0.533 | 2.752357e+05 |

```
[supervise 14:52:47] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_1.json] (attempt 0/5)
[supervise 16:52:19] DONE experiments/e36_kwon_full_seed_1.json rc=0
[supervise 16:52:19] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_2.json] (attempt 0/5)
```

**2026-09-28 18:56** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013176e+06 |
| e36_kwon_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e36_kwon_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567789e+06 |
| e36_kwon_full_seed_3 | 79 | 3.7 % | 0.914 | 1.010218e+02 |

```
[supervise 16:52:19] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_2.json] (attempt 0/5)
[supervise 18:51:51] DONE experiments/e36_kwon_full_seed_2.json rc=0
[supervise 18:51:51] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_3.json] (attempt 0/5)
```

**2026-09-28 19:26** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013176e+06 |
| e36_kwon_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e36_kwon_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567789e+06 |
| e36_kwon_full_seed_3 | 621 | 28.7 % | 0.889 | 1.004159e+04 |

```
[supervise 16:52:19] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_2.json] (attempt 0/5)
[supervise 18:51:51] DONE experiments/e36_kwon_full_seed_2.json rc=0
[supervise 18:51:51] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_3.json] (attempt 0/5)
```

**2026-09-28 19:56** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013176e+06 |
| e36_kwon_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e36_kwon_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567789e+06 |
| e36_kwon_full_seed_3 | 1162 | 53.8 % | 0.646 | 3.715829e+04 |

```
[supervise 16:52:19] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_2.json] (attempt 0/5)
[supervise 18:51:51] DONE experiments/e36_kwon_full_seed_2.json rc=0
[supervise 18:51:51] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_3.json] (attempt 0/5)
```

**2026-09-28 20:26** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e35_magnitude_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e35_magnitude_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567781e+06 |
| e35_magnitude_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279166e+06 |
| e35_magnitude_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013176e+06 |
| e36_kwon_full_seed_1 | 2159 | 100.0 % | 0.107 | 2.431433e+06 |
| e36_kwon_full_seed_2 | 2159 | 100.0 % | 0.107 | 1.567789e+06 |
| e36_kwon_full_seed_3 | 1694 | 78.4 % | 0.579 | 3.567972e+05 |

```
[supervise 16:52:19] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_2.json] (attempt 0/5)
[supervise 18:51:51] DONE experiments/e36_kwon_full_seed_2.json rc=0
[supervise 18:51:51] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_3.json] (attempt 0/5)
```

**2026-09-28 22:10** — (re)launch of `scripts/run/queue_seedband2.txt`: 8 commands queued

**2026-09-28 22:10** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e36_kwon_full_seed_3 | 1897 | 87.8 % | 0.304 | 8.862545e+05 |

```
[supervise 18:51:51] DONE experiments/e36_kwon_full_seed_2.json rc=0
[supervise 18:51:51] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_3.json] (attempt 0/5)
step 1883 | acc=0.3170 | NZ=   277 | sparsity=0.8718 | d_m=8.5335e+05[supervise 22:10:53] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_3.js
```

**2026-09-28 22:15 — third crash; hardware lead.** The host went down again at ~20:38
(last GPU telemetry line 20:38:43; reboot 22:09). Magnitude seeds 1–4 and Kwon seeds 1–2
had completed (9/14 runs); Kwon seed 3 was at step 1 898. The `@reboot` hook relaunched the
queue unaided and Kwon seed 3 resumed from its last checkpoint (CSV clean, no NUL bytes) —
the checkpoint hardening worked. GPU telemetry over 12.5 h of full load was unremarkable:
74–76 °C, ≤ 101 W, no throttle reasons, normal readings up to the last line. Motherboard
sensors (ASUS WMI) however read the **+12 V rail at 10.1–10.2 V** under this load — about 16 %
below nominal, outside the ATX ±5 % window (11.4–12.6 V). Together with four abrupt stops
under sustained load and no kernel/GPU/thermal log entry, this points at the power supply
(or the board's 12 V sensing) rather than the GPU or the software. A board-telemetry logger
(`scripts/run/board_telemetry.sh` → `artifacts/seedband/board_telemetry.csv`: +12 V, +5 V,
Vcore, CPU/VRM/chipset temperature, VRM current every 10 s, synced to disk) now runs alongside
the queue, so the rail voltage just before the next crash will be on record.

**2026-09-28 22:40** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e36_kwon_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.279165e+06 |
| e36_kwon_full_seed_4 | 208 | 9.6 % | 0.919 | 1.323762e+03 |

```
step 1883 | acc=0.3170 | NZ=   277 | sparsity=0.8718 | d_m=8.5335e+05[supervise 22:10:53] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_3.js
[supervise 22:28:34] DONE experiments/e36_kwon_full_seed_3.json rc=0
[supervise 22:28:34] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.json] (attempt 0/5)
```

**2026-09-29 19:48** — (re)launch of `scripts/run/queue_seedband2.txt`: 7 commands queued

**2026-09-29 19:48** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e36_kwon_full_seed_4 | 513 | 23.8 % | 0.915 | 1.123685e+04 |

```
step 1883 | acc=0.3170 | NZ=   277 | sparsity=0.8718 | d_m=8.5335e+05[supervise 22:10:53] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_3.js
[supervise 22:28:34] DONE experiments/e36_kwon_full_seed_3.json rc=0
[supervise 22:28:34] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.json] (attempt 0/5)
```

**2026-09-29 20:18** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e36_kwon_full_seed_4 | 1028 | 47.6 % | 0.888 | 6.405740e+04 |

```
[supervise 22:28:34] DONE experiments/e36_kwon_full_seed_3.json rc=0
[supervise 22:28:34] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.json] (attempt 0/5)
step  467 | acc=0.9170 | NZ=  1693 | sparsity=0.2162 | d_m=9.7293e+03[supervise 19:48:40] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.js
```

**2026-09-29 20:48** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e36_kwon_full_seed_4 | 1554 | 71.9 % | 0.659 | 3.006074e+05 |

```
[supervise 22:28:34] DONE experiments/e36_kwon_full_seed_3.json rc=0
[supervise 22:28:34] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.json] (attempt 0/5)
step  467 | acc=0.9170 | NZ=  1693 | sparsity=0.2162 | d_m=9.7293e+03[supervise 19:48:40] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.js
```

**2026-09-29 21:18** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e36_kwon_full_seed_4 | 2095 | 97.0 % | 0.105 | 1.961407e+06 |

```
[supervise 22:28:34] DONE experiments/e36_kwon_full_seed_3.json rc=0
[supervise 22:28:34] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.json] (attempt 0/5)
step  467 | acc=0.9170 | NZ=  1693 | sparsity=0.2162 | d_m=9.7293e+03[supervise 19:48:40] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.js
```

**2026-09-29 21:48** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e34_obd_full_seed_3 | 239 | 11.1 % | 0.917 | 8.070590e+01 |
| e36_kwon_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013480e+06 |

```
step  467 | acc=0.9170 | NZ=  1693 | sparsity=0.2162 | d_m=9.7293e+03[supervise 19:48:40] launch [/home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.js
[supervise 21:22:15] DONE experiments/e36_kwon_full_seed_4.json rc=0
[supervise 21:22:15] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_3.json] (attempt 0/5)
```

**2026-09-29 22:18** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e34_obd_full_seed_3 | 400 | 18.5 % | 0.907 | 3.057396e+02 |
| e36_kwon_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013480e+06 |

```
[supervise 22:18:40] EXIT experiments/e34_obd_full_seed_3.json rc=1 -- will resume from checkpoint
[supervise 22:18:40] GAVE UP on experiments/e34_obd_full_seed_3.json after 5 restarts
[supervise 22:18:40] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_4.json] (attempt 0/5)
```

**2026-09-29 22:48** — supervisor exited

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e34_obd_full_seed_3 | 400 | 18.5 % | 0.907 | 3.057396e+02 |
| e36_kwon_full_seed_4 | 2159 | 100.0 % | 0.107 | 2.013480e+06 |

```
[supervise 22:20:23] EXIT experiments/e37_obs_full_seed_4.json rc=1 -- will resume from checkpoint
[supervise 22:20:23] GAVE UP on experiments/e37_obs_full_seed_4.json after 5 restarts
[supervise 22:20:23] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd
```

**2026-09-30 19:33** — (re)launch of `scripts/run/queue_seedband2.txt`: 6 commands queued

**2026-09-30 19:33** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e34_obd_full_seed_3 | 400 | 18.5 % | 0.907 | 3.057396e+02 |

```
[supervise 22:20:23] GAVE UP on experiments/e37_obs_full_seed_4.json after 5 restarts
[supervise 22:20:23] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd
[supervise 19:33:41] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_3.json] (attempt 0/5)
```

**2026-09-30 20:03** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e34_obd_full_seed_3 | 677 | 31.3 % | 0.869 | 1.546829e+03 |

```
[supervise 22:20:23] GAVE UP on experiments/e37_obs_full_seed_4.json after 5 restarts
[supervise 22:20:23] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd
[supervise 19:33:41] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_3.json] (attempt 0/5)
```

**2026-09-30 20:33** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e34_obd_full_seed_3 | 954 | 44.2 % | 0.720 | 6.548888e+03 |

```
[supervise 22:20:23] GAVE UP on experiments/e37_obs_full_seed_4.json after 5 restarts
[supervise 22:20:23] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd
[supervise 19:33:41] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_3.json] (attempt 0/5)
```

**2026-09-30 21:03** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e34_obd_full_seed_3 | 1230 | 56.9 % | 0.672 | 1.739265e+04 |

```
[supervise 22:20:23] GAVE UP on experiments/e37_obs_full_seed_4.json after 5 restarts
[supervise 22:20:23] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd
[supervise 19:33:41] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_3.json] (attempt 0/5)
```

**2026-09-30 21:33** — (re)launch of `scripts/run/queue_seedband2.txt`: 6 commands queued

**2026-09-30 21:33** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e34_obd_full_seed_3 | 1481 | 68.6 % | 0.609 | 4.526384e+04 |

```
[supervise 22:20:23] QUEUE COMPLETE: /home/simon/venv/general/bin/python3 -m sparsifier.kwon_sparsifier experiments/e36_kwon_full_seed_4.json=ok, /home/simon/venv/general/bin/python3 -m sparsifier.obd
[supervise 19:33:41] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_3.json] (attempt 0/5)
step 1455 | acc=0.6010 | NZ=   705 | sparsity=0.6736 | d_m=4.3003e+04[supervise 21:33:21] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_3.json
```

**2026-09-30 22:03** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e34_obd_full_seed_3 | 1685 | 78.0 % | 0.599 | 1.263849e+05 |

```
[supervise 21:37:15] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_3.json] (attempt 1/5)
[supervise 21:37:32] EXIT experiments/e34_obd_full_seed_3.json rc=1 -- will resume from checkpoint
[supervise 21:37:32] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_3.json] (attempt 2/5)
```

**2026-09-30 22:33** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e34_obd_full_seed_3 | 1959 | 90.7 % | 0.385 | 6.120487e+05 |

```
[supervise 21:37:15] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_3.json] (attempt 1/5)
[supervise 21:37:32] EXIT experiments/e34_obd_full_seed_3.json rc=1 -- will resume from checkpoint
[supervise 21:37:32] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_3.json] (attempt 2/5)
```

**2026-09-30 23:03** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e34_obd_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.301754e+06 |
| e34_obd_full_seed_4 | 80 | 3.7 % | 0.920 | 7.407263e-03 |

```
[supervise 21:37:32] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_3.json] (attempt 2/5)
[supervise 22:55:08] DONE experiments/e34_obd_full_seed_3.json rc=0
[supervise 22:55:08] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_4.json] (attempt 0/5)
```

**2026-09-30 23:33** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e34_obd_full_seed_3 | 2159 | 100.0 % | 0.087 | 2.301754e+06 |
| e34_obd_full_seed_4 | 352 | 16.3 % | 0.920 | 3.337226e+01 |

```
[supervise 21:37:32] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_3.json] (attempt 2/5)
[supervise 22:55:08] DONE experiments/e34_obd_full_seed_3.json rc=0
[supervise 22:55:08] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_4.json] (attempt 0/5)
```

**2026-10-01 08:01** — (re)launch of `scripts/run/queue_seedband2.txt`: 5 commands queued

**2026-10-01 08:01** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e34_obd_full_seed_4 | 441 | 20.4 % | 0.920 | 1.891506e+02 |

```
[supervise 22:55:08] DONE experiments/e34_obd_full_seed_3.json rc=0
[supervise 22:55:08] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_4.json] (attempt 0/5)
step  349 | acc=0.9200 | NZ=  1811 | sparsity=0.1616 | d_m=3.1264e+01[supervise 08:01:39] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_4.json
```

**2026-10-01 08:05** — (re)launch of `scripts/run/queue_seedband2.txt`: 5 commands queued

**2026-10-01 08:05** — supervisor running

| run | step | sparsity | acc | d_manifold |
|---|---|---|---|---|
| e34_obd_full_seed_4 | 433 | 20.0 % | 0.919 | 1.625131e+02 |

```
[supervise 22:55:08] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_4.json] (attempt 0/5)
step  349 | acc=0.9200 | NZ=  1811 | sparsity=0.1616 | d_m=3.1264e+01[supervise 08:01:39] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_4.json
[supervise 08:05:48] launch [/home/simon/venv/general/bin/python3 -m sparsifier.obd_sparsifier experiments/e34_obd_full_seed_4.json] (attempt 0/10)
```

**2026-10-01 08:15 — supervisor retry bug fixed; queue restarted.** Reboot history (`last -x`):
hosts went down 09-27 19:15, 09-27 20:16, 09-28 20:38, 09-29 23:01, 09-30 21:31, 09-30 23:43 —
six abrupt stops in four days, all under GPU load; `+12V` reads 10.08 V at 08:05 today. On
09-29 the supervisor marked OBD seeds 3–4 and OBS seeds 1–4 `GAVE UP` within ~2 min: after a
non-zero exit it relaunched immediately (the 15 s pause applied only after a stall-kill), so
five retries fired before the dead child's GPU memory was released (`RESOURCE_EXHAUSTED`,
`No BLAS support for stream`). Fix: `supervise.py` now sleeps before every relaunch;
`seedband_launch.sh` sets `RESTART_SLEEP=60 MAX_RESTARTS=10`. The running supervisor (old
code) was stopped at OBD seed 4 step 433 and relaunched; the run resumed from checkpoint
`step_0400` (verified: no duplicate steps). Remaining: OBD seed 4 (~3.3 h) + OBS seeds 1–4
(~1.6 h each) ≈ 10 h. Data so far: magnitude 1–4, Kwon 1–4, OBD 3 complete (11/14 runs).
