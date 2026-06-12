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
