# Handover: current state (2026-07-11)

Experiment-running was made robust and the experiment backlog was pushed forward.
The 2026-06 section below is retained for the perf/correctness technical record
(fp-contract, two-network `d()`, verification methodology) — still valid.

## Runner robustness (why e16 hung, and the fix)
`e16_kwon_full` "ran for days" in June because a single JAX/CUDA call **wedged on
the 8 GB GPU** (froze ~step 1900/2160, no crash, blocked the whole driver chain).
It was **not** a code loop — `adjust()` is bounded. Fix = external watchdog +
resumable checkpoints, now in the tree:
- **`scripts/run/supervise.py`** — runs a queue (full command lines; derives the
  monitored dir from the `.json` arg) serially, SIGKILLs the process group after
  `STALL_TIMEOUT` (default 600 s; campaign uses 900) of no artifact progress,
  relaunches (runner auto-resumes), gives up after `MAX_RESTARTS`. **Use
  `JAX_PLATFORMS=cuda`, not `gpu`** (this build errors on `gpu`).
- **`sparsifier/runner.py`** — checkpoints now save **W and b** (adjust mutates
  b; mask = W!=0 on load); auto-resumes from the latest checkpoint carrying
  `b_*.npy` and truncates the CSV to match; Ω is **seeded** (`omega_seed` /
  `train.seed`). `RESUME=0` forces fresh. Old W-only checkpoints are ignored →
  re-runs of old-code experiments start fresh (safe).
Verified: kill-mid-run resumes to a contiguous log; watchdog stall→kill→give-up
tested. CSV output verified byte-compatible with the old per-file `main()`s.

## Experiments — done since June
- Re-ran the three incomplete/corrupted ones clean under the watchdog:
  **e16_kwon_full** 2160/2160, **baseline** 500/500 (23.1% sparsity, acc 0.917;
  old log was NUL-corrupted), **e11_neuron** 15/15.
- All 25 pre-existing logs health-checked: no corruption, contiguous, sane.
- **Headline method-comparison figure** rendered (`images/comparison/method_comparison.png`
  via `visualize/plot_methods.py`; assemble `artifacts/comparison/<method>.csv`
  from the 500-step logs). Manifold is competitive with the best baseline (OBD) on
  accuracy-vs-sparsity while much cheaper on the compute-vs-sparsity Pareto.

## Experiments — batch DONE (2026-07-13, all 50 jobs rc=0, 0 stalls/retries/failures)
Queue `scripts/run/queue_batch.txt`; code committed `8becc03`. All 28 new logs
health-checked (0 NUL, contiguous, sane). Results:
- **Full-budget stupidity runs** (2160 steps) e17_obd_full / e18_obs_full /
  e19_lazarevich_full — all collapse to acc ≈ 0.087–0.089 at 99.95% sparsity,
  matching e05/e15/e16. The stupidity-point set now covers all 6 selectors.
- **Neuron-level variants** (15 steps, vs e11 manifold-neuron acc 0.256@70%):
  OBD-neuron **0.257@70%** (≈ manifold, best), OBS-neuron 0.091@70% (collapses),
  magnitude-neuron 0.087@100% (collapses). → neuron-level pruning needs a
  second-order/manifold criterion; magnitude is not enough structurally.
- **Multi-seed sweep** (final val_acc, mean±std over seeds 0–4):
  width_50 0.926±0.020 (3.9% sp); width_200 0.962±0.004 (0.6% sp, ~undent);
  **depth_2layer 0.800±0.132** (24% sp — very high cross-seed variance, range
  0.555–0.922, a real fragility finding); depth_4layer 0.129±0.006 (collapsed
  every seed). Bands ready for the seed-variance figure (`plot_seeds.py`).
- **MC-variance**: `artifacts/mc_variance/results.csv` (21000 rows = 7 B ×
  100 samples × 30 candidates) for `plot_mc_variance.py`.
- **E6 Xavier retrain — PARTIAL, acceptance NOT met.** Activation-aware Xavier
  init lifted tanh dense 0.30→**0.62** and sigmoid 0.13→**0.30**, but both are
  still < 0.85. Root cause is deeper than init: **inputs are unnormalised**
  (range ≈ [-30, 281], mean 33) so tanh/sigmoid first-layer pre-activations
  saturate (ReLU is scale-tolerant, they are not). tanh's curve is still slowly
  climbing at epoch 999 (plateau ~0.62). Real fix = normalise inputs (e.g. /255
  or standardise) for the non-ReLU configs, then retrain; likely also needs the
  Ω sampler range adjusted to match. NOT yet done.

## Still pending
- **E6 follow-up (new)**: input normalisation for tanh/sigmoid, then retrain +
  re-sparsify (see above). Init fix is in; scaling fix is not.
- **Open investigation** (user-deferred): width-200 collapse (0.916→0.662); the
  multi-seed width_200 band above is at ~0.6% sparsity (500 steps), so it does
  not yet probe the collapse — needs a higher step budget.
- **New datasets** (user-deferred): e07 full-res 784 MNIST, Fashion-MNIST,
  synthetic Gaussian, CIFAR-10 (need loaders / dataeng changes first).

---

# Handover: `fable-profiling` branch (2026-06-12/13)

Performance + correctness overhaul of the sparsification stack. All work is
committed on `fable-profiling` (branched from `bugfix-precision-and-experiments`):

- `0bba051` — bit-exact activation caching in prune_ext, deterministic ties, OBS OOM fix
- `cce1bc1` — revert of a self-introduced fallback regression (see below)
- `f206ec3` — O(1) incremental candidate evaluation; seeded train batches

**All 56 tests pass** (`JAX_PLATFORMS=cpu python3 -m pytest tests/ -q`).

## What changed and why

### C++ kernel `prune_ext` (the manifold search)
1. **Activation caching**: forward the unmodified net once per call (per-sample
   `z`/`a` per layer + baseline SSE); per candidate only the affected unit and
   downstream layers are evaluated. Exact shortcuts return baseline SSE when the
   candidate's input is 0.0 or its unit activation is unchanged (dead ReLU).
2. **O(1) delta updates** (`f206ec3`): `z' = z − W[i,j]·src[j]`, first downstream
   layer via masked column update. Algebraically identical, **not** bit-identical
   to a full re-forward (last-ulp rounding).
3. **Deterministic tie-breaking**: lexicographic (distance, candidate index).
   The old OpenMP reduction returned a different winner every run for exact
   ties — and exact ties are common (dead units). Old experiment trajectories
   were therefore never reproducible (evidence: `benchmark/profiles/before.json`
   vs `before2.json`, same code, different 30-step sequences).
4. **`-ffp-contract=off`** in `prune_ext/CMakeLists.txt`: FMA contraction must
   not differ between the cache-fill and candidate-eval inlining sites, or the
   shortcut equality checks break. Do not remove this flag.

Timings (baseline [196,10,10,10], |Ω|=10000): 6.05 s/step (original) →
0.97 (cached exact) → **0.46 s/step** (delta). 13× total; more on wider nets.

### Python
- `adjust()`: reference outputs forwarded once per call (`_d_val_grad_to_outputs`)
  — verified **bit-identical** outputs, ~1.2× CPU.
- **Do not** apply the same trick to the pure-Python fallback search in
  `prune()` — tried in `0bba051`, reverted in `cce1bc1`: a separate JIT program
  changes XLA fusion so zero-effect candidates no longer give *exactly* 0.0,
  breaking the `min_dist == 0` early exit and shifting selection. The comment
  in `sparsifier.py` explains.
- Kwon/magnitude/OBD scoring: vectorised argmin (kwon paid 2 GPU syncs per
  weight: 0.50 → 0.006 s/step in vivo). Selection bit-identical (np.argmin
  first-occurrence = scan order).
- OBD `_diag_hessian`: chunked vmapped HVPs (chunk=64). Bit-equal H_diag on
  CPU; in vivo GPU 7.8 → 3.5 s/step.
- OBS: **was hard-crashing with GPU OOM at the documented baseline size**
  (`jax.hessian` pushes all N=2160 basis vectors at once → 1.7 GB alloc).
  Fixed with chunked HVPs (chunk=240 in `_hessian_and_inv`, this hunk was
  applied by a concurrent edit, kept); python index/scoring/update loops
  vectorised. 2.9× on CPU and unbroken on GPU.
- Off-by-one fixed: `[act]*(len(topology)-2) + ['linear']` (was -1; harmless
  only because zip truncated and is_last overrode).
- `scripts/train.py`: batch sampling now seeded from the config (was unseeded).

## Verification methodology (reproducible)
Old implementations were rebuilt as parallel modules and compared in-process:
- `git show <rev>:<file> > /tmp/old_mods/old_<name>.py` for Python;
- for C++: `sed 's/PYBIND11_MODULE(prune_ext, m)/PYBIND11_MODULE(<newname>, m)/'`
  then `g++ -O3 -march=native -ffp-contract=off [-fopenmp -DUSE_OPENMP] -std=c++17
  -shared -fPIC -I$(python3 -c 'import pybind11;print(pybind11.get_include())')
  $(python3-config --includes) x.cpp -o <newname>$(python3-config --extension-suffix)`.
  (/tmp copies are gone after reboot; rebuild from git as above. The cached-exact
  kernel for A/B is the `prune_ext/` tree at commit `cce1bc1`.)

Results: cached-exact kernel bit-identical to original over 110 chained steps
(relu/tanh/sigmoid, incl. pre-pruned masks) when compiled with the same
fp-contract setting. Delta kernel vs cached-exact: 60/60 selection agreement
per activation on dense nets with nonzero distances (winning distances within
3e-14 relative), 120/120 with bit-equal distances on real baseline trajectories.

## Open items
1. **GPU after-profile** (only unfinished task): when the GPU is free, run
   `python benchmark/profile_sparsifier.py --label after --compare benchmark/profiles/before.json`.
   Expected: large speedups; fingerprints for `ext_search`/`prune_adjust` WILL
   differ from `before.json` — old ties were non-deterministic and
   `-ffp-contract` changed the noise floor on exact-zero candidates. That diff
   is explained, not a regression. A background watcher for this was armed in
   the previous session and dies with the machine.
2. **Interrupted experiment queue** (driven by another session via
   `scripts/run_method_baselines.sh`): at interruption time `e13_obd` was
   mid-run (~step 270/500, partial log in `artifacts/e13_obd/obd_sparsified/`)
   and `e14_obs` likely not started. Both should be (re)run with the new code —
   e12_kwon completed on the new code already.
3. Optional (deliberately not done mid-campaign): factor the six near-identical
   `main()` functions in `sparsifier/*` into a shared runner (~600 dup lines).
   CSV schema must remain byte-compatible.
4. Note for thesis text: pre-fix manifold-search runs were non-reproducible
   (tie non-determinism). Post-fix runs are deterministic; distances also sit
   on a slightly different (lower) rounding-noise floor for zero-effect
   candidates. Worth a footnote if old and new run logs are ever compared.
