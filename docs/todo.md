# TODO — Open Work & Experiment Backlog

Snapshot: 2026-04-27 (code-review findings 2026-05-25). Tracks code, experiments,
visualizations, theory, and writing for the manifold-distance sparsification thesis.

Legend: `[x]` done · `[~]` partial / running · `[ ]` not started.

Conventions:
- Configs live in `experiments/<name>.json`; use `experiments/baseline.json` as the
  template and vary one axis at a time. Outputs go to `artifacts/<name>/` (training)
  and `artifacts/<name>/sparsified/` (log + checkpoints).
- CSV schema: `sparsifier/sparsifier.py:265–278`; every variant reuses it.
- After a run: `python visualize/plot_run.py artifacts/<name>/sparsified/sparsification_log.csv images/runs/<name>`
  and update the P1 table below with final dense/sparse accuracy and sparsity.

---

## 1. Experiments to run

### Reference & ablations — MNIST 14×14, `[196,10,10,10]`, ReLU

| ID  | Config                       | Status | Steps    | dense→final acc | Final sparsity |
| --- | ---------------------------- | ------ | -------- | --------------- | -------------- |
|  —  | `baseline.json`              | `[~]`  | 240/500  | 0.916 → 0.917   | 11.1%          |
| E1  | `e01_no_adjust.json`         | `[x]`  | 500      | 0.912 → 0.913   | 23.1%          |
| E2a | `e02_omega_100.json`         | `[x]`  | 500      | 0.916 → 0.917   | 23.1%          |
| E2b | `e02_omega_1000.json`        | `[x]`  | 500      | 0.919 → 0.916   | 23.1%          |
| E2c | `e02_omega_100000.json`      | `[x]`  | 500      | 0.920 → 0.921   | 23.1%          |
| E3a | `e03_width_50.json`          | `[x]`  | 500      | 0.916 → 0.917   | 23.1%          |
| E3b | `e03_width_200.json`         | `[x]`  | 500      | 0.916 → **0.662** | 23.1%        |
| E4a | `e04_depth_2layer.json`      | `[x]`  | 500      | 0.916 → 0.784   | 23.1%          |
| E4b | `e04_depth_4layer.json`      | `[x]`  | 500      | 0.916 → 0.654   | 23.1%          |
| E5  | `e05_stupidity_point.json`   | `[~]`  | 1221/2160| 0.916 → 0.739   | 56.5%          |
| E6a | `e06_act_tanh.json`          | `[x]`* | 500      | **0.302** → 0.306 | 23.1%        |
| E6b | `e06_act_sigmoid.json`       | `[x]`* | 500      | **0.126** → 0.126 | 23.1%        |

\* E6a/E6b dense baselines are untrained-quality (see §2 CR-A init bug); re-train with
activation-appropriate init before drawing any conclusion.

### Finish in-flight runs (no new code)
- [ ] **Baseline → 500 steps** (only 240 logged; the rest of the ablation series is
  incomparable until step counts match). Delete `artifacts/baseline/sparsified/` and
  re-run, or extend the `range(sp['steps'])` loop. Acceptance: 500 rows, summary PNG regenerated.
- [ ] **E5 stupidity point → 2160 steps** (≈940 more; needed for the empirical
  stupidity-point figure). Acceptance: log reaches 2160 capturing the collapse region;
  weight heatmaps at steps `[0,500,1000,1500,2000]`.
- [ ] **E10 seed sweep — finish seed_3 (~460/500) and seed_4 (~112/500)** to 500, then
  build the multi-seed aggregate figure (`plot_seeds.py`).
- [ ] **E11 neuron baseline → 15 steps** (only 9/15 logged). Rename the misspelled
  artifact dir `e11_euron_baseline` → `e11_neuron_baseline`, then resume. Acceptance:
  15 steps, weight heatmap at `[0,5,10,14]`. (Config also has a `name` typo + missing
  `train` block — see §2.)

### Method-comparison baselines (configs exist, runs pending)
Highest-leverage missing experiments — convert the manifold method's competitiveness
claim from theoretical to empirical. Command: `python -m sparsifier.<module> experiments/<config>.json`.
- [ ] **Lazarevich data-Ω** — `e08_omega_data.json` / `lazarevich_sparsifier` (real
  calibration images; tests whether the "data-free" claim matters).
- [ ] **Magnitude search** — `e09_magnitude_search.json` / `magnitude_sparsifier`
  (`argmin|w|` + manifold adjust; if comparable, the search step is the cheap part to drop).
- [ ] **Kwon / Fisher** — `e12_kwon.json` / `kwon_sparsifier` (`(∂d_W/∂w)²·w²` global argmin). *(done on new code)*
- [ ] **OBD** — `e13_obd.json` / `obd_sparsifier` (diagonal-Hessian `½·H_ii·w²`). *(interrupted ~step 270/500; re-run on new code)*
- [ ] **OBS** — `e14_obs.json` / `obs_sparsifier` (Hessian-inverse + closed-form `δw`, no gradient adjust). *(re-run on new code)*
- [ ] **Headline comparison figure (B6)** — once all five + baseline are at matched step
  count, drop each `sparsification_log.csv` into `artifacts/comparison/` (renamed to
  strategy) and run `visualize/plot_comparison.py`. Panels: accuracy-vs-sparsity,
  search-time/step, total-time/step. Extend `_COLORS` if incomplete.

### Fix broken / undersized runs
- [ ] **Re-train E6 tanh/sigmoid with Xavier init** then re-sparsify; determine whether
  dense failure is init or genuinely worse activations. Acceptance: dense val_acc ≥ 0.85.
- [ ] **Investigate width-200 collapse** (0.916→0.662 at 23.1%, opposite of lottery-ticket
  prediction). No new config: (1) plot `d_manifold`/`d_W` trajectories; (2) repeat at
  B=100k to test MC noise in 81k-weight space; (3) repeat with larger initial `alfa`
  (currently `1e-11`); document in experiments chapter.
- [ ] **Larger architecture step budget** — `mnist_784_256_128_10.json` (~235k weights);
  500 steps ≈ 0.25% sparsity is uninformative. Prefer switching to `neuron_sparsifier`
  (≈384 prunable neurons) over raising steps to 10k (50× runtime). Note: config first
  layer is `[784,…]` but data is 196-dim → crashes; fix topology or supply 784-dim data.

### New datasets & pipelines
- [ ] **Full-resolution MNIST (e07)** — `e07_fullres_784.json` exists but needs
  `data/X_train_full.csv` etc. Add a `--full-res` flag to `scripts/dataeng.py` to skip
  the 14×14 downsample and write 784-column CSVs (preserve existing behaviour), then run.
- [ ] **Fashion-MNIST** — add loader to `dataeng.py` (raw format identical to MNIST),
  generate `*_fmnist_small.csv`, create `fmnist_baseline.json` + mirror configs of the
  key ablations (baseline, e01_no_adjust, e03_width_50, e04_depth_4layer, e05_stupidity_point).
- [ ] **Synthetic Gaussian (structureless)** — direct test of the "data-free,
  structure-agnostic" claim. New generator `scripts/dataeng_gaussian.py` (don't pollute
  `dataeng.py`): K=10 Gaussian blobs in d=196, centres on a regular simplex, N=10k train
  + 1k test, σ tuned so Bayes-optimal ≈95%. Configs `gauss_{baseline,no_adjust,stupidity_point}.json`.
  Hypothesis: sparsity pattern should be uniformly random (no edge-detector weights);
  compare via `plot_weights.py` heatmaps + `plot_pixel_saliency.py`.
- [ ] **CIFAR-10 small MLP (known-failure case, low priority)** — flatten 3072→768 (2×2
  avg-pool) or 1024 grayscale, topology `[…,256,128,10]`; expect ~50% dense. One config,
  one run, for the failure-mode discussion.
- [ ] **Per-dataset comparative table** — dense acc, achievable sparsity at <2 pp drop,
  FLOPs reduction, runtime.

### Structured pruning (E11 follow-ups)
- [ ] Direct overlay of weight vs neuron pruning at matched parameter-removal counts.
- [ ] Neuron-level variants of magnitude / OBD / OBS for parity with weight-level variants.

### Statistical / measurement experiments
- [ ] **Multi-seed sweep of every key ablation** — only the baseline arch has multi-seed
  coverage (E10, ±0.1 observed cross-seed spread). Add `seed: 0..4` to each ablation
  (`e03_width_50_seed_<n>.json` etc.; prefix lets `plot_seeds.py` find matches). Prioritise
  widths (E3) and depths (E4) — they contradict the headline result most. Aggregate into
  mean±std accuracy/sparsity bands.
- [ ] **MC-estimator characterisation** — new one-off `scripts/mc_variance.py`: load
  `artifacts/baseline/`, for `B ∈ {100,300,1k,3k,10k,30k,100k}` draw 100 independent Ω
  samples; record (i) var(d) for a fixed candidate, (ii) rank-1 candidate, (iii) Spearman
  correlation between two independent-Ω rankings. Output `artifacts/mc_variance/results.csv`
  (`B,candidate_idx,sample_idx,d,rank`); consumed by `plot_mc_variance.py`. Also plot
  pruning-decision agreement rate between two independent Ω samples per B.
- [ ] **Adjust-step efficiency logging** — in `prune()` capture `d(probe_net, og_net, omega)`
  after zeroing and *before* `adjust()`; return it in `PruneMeta` and add a
  `d_manifold_pre_adjust` column to every variant's CSV. Consumed by `plot_adjust_efficiency.py`.

**Sequencing** — *1 day*: finish baseline+E5, run the five method baselines, then B6.
*1 week*: add Xavier re-train (E6), multi-seed sweep, Gaussian. *1 month*: add full-res,
Fashion-MNIST, MC-variance characterisation, longer follow-up runs.

---

## 2. Code / infrastructure

### Implemented
- [x] `Layer(W,b,mask)` NamedTuple refactor across `mlp/`, `sparsifier/`, `scripts/`
- [x] `prune()` copy-once + zero + evaluate + restore (no per-candidate deepcopy)
- [x] Activation-aware forward pass (`hidden_activation`: relu/tanh/sigmoid)
- [x] C++/OpenMP parallel candidate search (`prune_ext`)
- [x] JSON-driven experiment configs consumed by `scripts/train.py`
- [x] Per-step CSV logging (per-layer NZ, timing, candidate `(l,i,j)`, `d_manifold`, `d_W`) + periodic weight checkpoints
- [x] Sparsifier variants: `sparsifier.py` (reference manifold prune+adjust),
  `neuron_sparsifier.py`, `magnitude_sparsifier.py`, `lazarevich_sparsifier.py`
  (`omega_source='data'`), `kwon_sparsifier.py`, `obd_sparsifier.py`, `obs_sparsifier.py`

### Open code work
- [ ] Fix activation-aware weight init in `scripts/train.py` (sigmoid/tanh baselines collapse — blocks E6)
- [ ] `dataeng.py` mode to emit full-res 784-input MNIST splits (for e07)
- [ ] Loaders for Fashion-MNIST and CIFAR-10 (raw → CSV in `data/`)
- [ ] Synthetic Gaussian generator (controlled-structure baseline)
- [ ] Orchestrator CLI flag / config field to dispatch per-config to the right sparsifier
  module (each variant currently has its own `__main__`)
- [ ] Standardise log column names across variants so `plot_run.py` works without branching
- [ ] Decide + document: standalone module vs onnx-mlir integration (consult Prof. Agosta)

### Code-review findings (2026-05-25, 6-agent read-only review)
Tags: **[BUG]** correctness · **[INCONSIST]** · **[PERF]** · **[CPLX]** dead/complex · **[TEST]**.
File:line refs are from that snapshot and may drift.

**Critical — affects experiment / result validity**
- [ ] **[BUG]** `mlp/mlp.py:18-20` — `random_layer_params` fixed `scale=1e-2` + random biases; tanh/sigmoid never train. Use He/Xavier, zero biases. *(same as init bug above; blocks E6)*
- [ ] **[BUG]** `scripts/train.py:64` — minibatches via unseeded global `np.random.choice`; `train.seed` only seeds JAX init ⇒ E10 seed sweep isn't seed-controlled. Seed NumPy / use a JAX PRNG.
- [ ] **[BUG/math]** `sparsifier/obd_sparsifier.py:12-14` — "OBD ≡ Kwon on ReLU" is false (verified). OBD=`Σ(∂rₖ/∂w)²·w²`, Kwon=`(Σrₖ∂rₖ/∂w)²·w²`. Fix doc/thesis; treat as distinct criteria.
- [ ] **[BUG]** `backend/compiler.py:748` — SA acceptance `uniform()>exp(-T)` ignores `dE`. Use `exp(-dE/T)`.
- [ ] **[BUG]** `sparsifier/kwon_sparsifier.py:57,69` — step 1 `net==og_net` ⇒ `d=0`, grad 0 everywhere ⇒ argmin picks (0,0). Warm-start or fall back to magnitude on first step.
- [ ] **[BUG]** `sparsifier/obs_sparsifier.py:153-176` — closed-form `δw` writes into `H_inv` rows of already-pruned coords then masks, instead of inverting the active-set sub-matrix ⇒ diverges once partially pruned. (Also the one thing that can break the `W==0⟺mask==0` invariant.)
- [ ] **[BUG]** `backend/compiler.py:1292,1295,1323,1326` — ReLU block emits `vmsr fpexc,r3` with `r3` uninitialized. Remove.
- [ ] **[BUG]** `backend/compiler.py:1165` — `MOV r7,<imm>` with arbitrary decimal offset; use `MOVW`/`MOVT` or `LDR =imm`.
- [ ] **[BUG/metrics]** `sparsifier/sparsifier.py:59` — `_d_impl` omits `1/B`, so logged `d_manifold` ≈10⁴× the defined `d_W`. Harmless for selection (argmin scale-invariant) but the Lipschitz/stupidity-point validation must divide by B, and `alfa`/`tol` (`:98`) are tuned to the inflated scale — re-tune if normalized.

**Other correctness / consistency**
- [ ] **[BUG]** `run_experiments.sh:36,75-77` — skip-on-`W_0.npy` freezes stale baseline weights for e03_width_50/200, e04_depth_2/4layer (never retrain). Delete those artifact dirs + uncomment.
- [ ] **[BUG]** `run_experiments.sh:14,40,48` — `set -euo pipefail` + `| tee` ⇒ one crashed run aborts the batch. Wrap calls with `|| log FAILED`.
- [ ] **[BUG]** `experiments/mnist_784_256_128_10.json:3` — `[784,…]` vs 196-dim data ⇒ crash; still in the run list.
- [ ] **[BUG/latent]** `sparsifier/sparsifier.py:236` & `lazarevich_sparsifier.py:46` — `[act]*(len(topology)-1)+['linear']` makes N+1 entries; `zip` drops `'linear'`, mis-tags output layer. Masked only because C++ forces log-softmax last. *(off-by-one already fixed to -2 on fable-profiling — verify)*
- [ ] **[INCONSIST]** `prune_ext/forward.hpp:62-65` vs `mlp/mlp.py:57-64` & `sparsifier.py:144-187` — JAX applies one global `hidden_activation`; C++ honors per-layer; Python `d()` ignores its `activations` arg. Make JAX per-layer or assert homogeneous.
- [ ] **[BUG/latent]** `backend/compiler.py:203` + `network.py` — IR/`Program.run` use bare `W` (no mask), skip on `W!=0`; only correct while `W==0⟺mask==0`. Also `compiler.py:1286`/`network.py:34-36` hardcode ReLU ⇒ non-ReLU models compile wrong.
- [ ] **[BUG]** `backend/compiler.py:751` — `int(anneal_iterations/10)` div-by-zero for small layers; guard `max(1,…)`.
- [ ] **[BUG/portability]** `prune_ext` build `-march=native` bakes host ISA into the committed `.so` ⇒ SIGILL elsewhere. Use `-march=x86-64-v2`.
- [ ] **[INCONSIST]** `backend/compiler.py:546-576` — input/output mapping omits unallocated temps ⇒ `executable` never loads some inputs. Iterate full `topology[0]`, assert each has a location.
- [ ] **[INCONSIST]** `backend/compiler.py:637-657` — `compute_signals` ticks COMMENT nodes, shifting temporal distances. Skip COMMENTs.
- [ ] **[INCONSIST]** `backend/compiler.py:1188-1209` — statement dispatch keys on positional `flatten()` indices / `len==13`; dispatch on node structure instead.
- [ ] **[INCONSIST]** `sparsifier/obs_sparsifier.py:38-44,231-240` — log omits `adjust_time_s` (11 vs 12 cols). Keep the column as `0.0`.
- [ ] **[INCONSIST]** `sparsifier/kwon_sparsifier.py:83`, `obd_sparsifier.py:126` — missing the `min_score>0` guard before `adjust()`.
- [ ] **[INCONSIST]** `sparsifier/neuron_sparsifier.py:179-180,207-209` — mixes pre-step `neuron_sparsity` with post-step `neurons_pruned` in one row (off-by-one). Log both consistently.
- [ ] **[BUG]** `experiments/e11_neuron_baseline.json` — `name` typo `e11_euron_baseline`; no `train` block ⇒ KeyError in `train.py`.
- [ ] **[INCONSIST]** `run_experiments.sh:12` — header references deleted `experiments/mnist_neuron.json`; drop it.

**Performance (runtime only)**
- [ ] **[PERF]** `prune_ext/prune_ext.cpp:103-137` — recomputes full multi-layer forward per candidate though only the candidate's layer + downstream change. Cache prefix activations. *(largely addressed on fable-profiling — verify)*
- [ ] **[PERF]** `sparsifier/sparsifier.py:96-116` — `adjust` runs a full forward over all 10⁴ Ω per line-search trial. Mini-batch Ω or closed-form step on the changed column.
- [ ] **[PERF]** `sparsifier/sparsifier.py:172` — Python fallback recomputes `batched_predict(og_net,omega)` per candidate (only bites when `_USE_EXT` off).
- [ ] **[PERF]** `sparsifier/obd_sparsifier.py:75` — materializes full N×N Hessian for the diagonal only (OOMs wide nets). Use HVPs. *(addressed on fable-profiling — verify)*
- [ ] **[PERF]** `sparsifier/obs_sparsifier.py:96-117` — rebuilds flat-index map in Python triple-loops each step + O(N³) inverse. Precompute the map once.
- [ ] **[PERF]** `sparsifier/{magnitude,kwon,obd}_sparsifier.py` — vectorize selection: `np.where(mask==0,inf,score)` + `argmin`. *(done on fable-profiling — verify)*
- [ ] **[PERF]** `prune_ext/forward.hpp:58` — multiplies `mask*w*src` for masked zeros. Skip `mask==0`.
- [ ] **[PERF]** `backend/compiler.py:721,743,752` — `cost()` 3×/SA-iter, `build_D_reconstructed` rebuilds O(n²) per call. Cache + incremental ΔE over the two swapped rows/cols.
- [ ] **[PERF]** `backend/compiler.py:659-681` — `compute_signals_distance_matrix` recomputes signal vectors in the inner loop (O(n²·L)); precompute per-temp + exploit symmetry.
- [ ] **[PERF]** `visualize/plot_pruning_order.py:71` — `df.iterrows()` loop; vectorize via `groupby([...]).step.min()`.

**Complexity, dead code & tests**
- [ ] **[CPLX]** Variant `main()`s are ~95% verbatim copies of `sparsifier.py:213-339`. Extract one `run_sparsification_loop(prune_fn, meta_cols, out_subdir, omega_fn)` (~120 dup lines/file; CSV schema must stay byte-compatible).
- [ ] **[CPLX]** ~15 `visualize/*.py` duplicate CSV loading, color maps, argparse `--out`/save footer, `W_*.npy` loaders. Factor `visualize/_common.py`.
- [ ] **[CPLX]** Dead code: `sparsifier.py:77-82` `_zero_weight`; `compiler.py:818-826` no-op `np.max(len(...))` + unused `sparsify`/`compile_time_data` params; `:867-875` unused `phi`/`phi_inv` lambdas.
- [ ] **[CPLX/risk]** `compiler.py:760-808` `density_optimizer_memory_subset_for_output` — OOB-indexing risk on non-contiguous constrained addresses. Decompose + add bounds asserts.
- [ ] **[BUG/CPLX]** `mlp/dataset.py` & `mlp/topology.py` aren't imported by `train.py`/`dataeng.py` (dead), yet `tests/test_{dataset,topology}.py` only exercise them ⇒ false test confidence. Route the pipeline through them or test the real loaders.
- [ ] **[TEST]** `tests/test_benchmark.py:79-98` writes its own synthetic CSV instead of invoking `sparsifier.main()`; assert against a real produced CSV / the source header list.
- [ ] **[TEST]** Variant tests only smoke-test; none assert the selection criterion (would have caught OBD≡Kwon). Add known-net cases where criteria disagree.
- [ ] **[TEST]** `tests/test_prune_ext.py:41,57` — reference float64, C++ float32, no tolerance. Compute reference in float32 or assert only when top-2 distances are well separated.
- [ ] **[BUG]** `visualize/plot_adjust_efficiency.py:42` — reads `d_manifold_pre_adjust` (not in schema) ⇒ always raises. Add the column (§1 adjust-efficiency) or shelve.
- [ ] **[BUG]** `visualize/plot_bias_drift.py:44` — needs per-layer bias columns (not in schema). Add bias logging (§3 viz) or shelve.
- [ ] **[BUG]** `visualize/plot_mc_variance.py:71-81` — top-1 uses global `rank.min()` across the whole B-group instead of per `sample_idx`; biases the stability fraction. (Data doesn't exist yet — §1 MC.)
- [ ] **[INCONSIST]** `benchmark/compare_strategies.py:44` — header omits `total_W`/`d_W`/`candidate_*`/`layer_*_NZ`; emit full schema or document.
- [ ] **[INCONSIST]** `experiments/*.json` schema drift (missing `seed`, missing `train`, varying `_experiment`, `omega_source` only on e08). Add a required-keys validator at the top of `train.py`.
- [ ] **[INCONSIST/minor]** Misc: `dataeng.py:39-43` reshape hard-coded `14*14`; `plot_pixel_saliency.py:102` default `--shape 14 14`; `plot_adjust_efficiency.py:73-78` comment/code mismatch; `correctness_check.py:74` compares search-only (ignores adjust); `compare_strategies.py:30` doesn't thread `activations`; `plot_seeds.py:74`/`compare_runs_table.py:121` silent `'unk'`; `sparsifier.py:181` dead `min_dist==0` early-exit.

**Verified non-issues (don't re-chase)** — Ω range `[0,255]` matches raw MNIST + uniform-noise
Ω is the intended data-free design (the `[0,1]` in `mlp/dataset.py` is the dead module);
`prune()` in-place restore is safe; `stop_gradient(mask)` prevents adjust reviving pruned
weights; loss form correct for one-hot; C++ OpenMP region race-free, pybind keeps buffers
alive, log-softmax matches JAX; `W==0⟺mask==0` currently holds (revisit if OBS lands).

---

## 3. Visualization

### Implemented
- [x] `plot_run.py` (6-panel per-run summary) · `plot_comparison.py` · `plot_weights.py`
  (magnitude+mask heatmaps) · `live_view.py` (live polling dashboard)
- [x] `images/runs/<exp>/run_summary.png` for every completed run (2026-04-27);
  weight heatmaps for baseline + e05

### Tools to add (ranked by thesis value)
- [ ] **Method comparison overlay (`plot_methods.py`)** — `plot_run.py` panels overlaying manifold/magnitude/OBD/OBS/Lazarevich/Kwon. The accuracy-vs-sparsity figure is the thesis headline.
- [ ] **Seed-variance bands (`plot_seeds.py`)** — accuracy ± std and `d_manifold` ± std over seeds 0–4. Critical for credibility (results are single-seed).
- [ ] **Per-layer pruning stacked area (`plot_layer_dynamics.py`)** — stacked NZ per layer across steps; reveals input-vs-hidden preference.
- [ ] **Pruning order / hit map** — step each weight died, as a heatmap matching W shape; side-by-side exposes whether methods target the same weights.
- [ ] **Sparsity-pattern overlap matrix** — Jaccard between final masks, method×method. Tests "is manifold pruning disguised magnitude pruning?".
- [ ] **Critical-weight / lottery-ticket plot** — intersect final masks across seeds 0–4; fraction surviving vs intersection size.
- [ ] **Adjust-step efficiency** — `d_manifold` before vs after the adjust call (needs the §1 logging change); quantifies search vs adjust contribution.
- [ ] **Bias drift** — `||b−b₀||` per layer over steps (needs bias logging in CSV).
- [ ] **Spectral evolution** — singular values of `W[l]` over steps; detects rank collapse before accuracy collapse (likely the stupidity-point signature).
- [ ] **Per-step Pareto** — accuracy preserved vs cumulative wall-clock, one curve per method.
- [ ] **Stupidity-point landscape (E5)** — twin-axis `log d_manifold` + accuracy vs step, markers at "last weight in L0/L1/L2 removed".
- [ ] **MC variance plot** — scatter of d̂ across 100 independent Ω draws at B∈{100,1k,10k,100k}; justifies the choice of B.
- [ ] **Confusion-matrix evolution** — class-wise accuracy across sparsity steps.
- [ ] **Pixel-saliency from L0 mask** — sum surviving weights per input column, plot as 14×14; compare to an input-saliency baseline.

---

## 4. Theory & Lean proofs

- [x] Sparsification stability (Lipschitz bound) — formalised
- [x] Stupidity-point existence via Heine-Borel — formalised
- [x] Weight-vs-neuron pruning support monotonicity — formalised
- [ ] Validate the stability bound numerically against the empirical `d_manifold` trajectory (E5)
- [ ] Connect the formal stupidity-point statement to the empirical "step at which `d_manifold` becomes monotonically increasing" criterion (currently informal)
- [ ] Counter-examples for edge cases (degenerate architectures, all-zero biases)
- [ ] Optional: rate of approach to the stupidity point (currently existence only)
- [ ] Prove greedy optimality at each step; derive convergence rate if possible
- [ ] Explicit Lipschitz constants in terms of architecture; explicit gradient formulas

---

## 5. Compiler & hardware

- [x] Tree-IR generation from sparse MLP
- [x] Two-stage register + memory allocation with simulated annealing
- [x] ARM assembly emitter validated with Unicorn emulator
- [x] Sparsity → FLOPs reduction validated (23.1% sparsity → 23.1% FLOPs)
- [ ] Decide standalone module vs onnx-mlir integration (consult advisor)
- [ ] PIM cost model: weight vs neuron pruning, formal comparison; memory-access & energy analysis
- [ ] Identify + document compiler bottlenecks for ARM targets
- [ ] PIM-specific code-generation passes
- [ ] Latency measurements on real ARM hardware (Cortex-A) — current "FLOPs reduction" is a proxy

---

## 6. Writing / thesis

### Introduction
- [ ] Literature review: magnitude pruning (Han 2015), SNIP (Lee 2019), lottery ticket
  (Frankle & Carbin), OBD/OBS, Kwon, Lazarevich; structured vs unstructured; compare to manifold method
- [ ] Motivation (post-training sparsification, PIM bottleneck) · contribution summary · outline

### Algorithm chapter
- [ ] Formalize `𝒲`/`ℱ`, define `d_W` with justification, neighborhood/sparsity graph
- [ ] Pseudocode + complexity analysis · adjust-step justification
- [ ] Lipschitz / stupidity-point proofs (cross-reference Lean) · illustrative diagrams

### Experiments chapter
- [ ] §setup, §results, §analysis · per-architecture tables
- [ ] Aggregate comparison figure (needs §1 method baselines)
- [ ] Seed-variance figure (needs §1 seed runs) · stupidity-point figure (needs E5 finished)
- [ ] Failure-mode discussion (width-200 collapse, sigmoid case)

### Hardware chapter
- [ ] PIM background · pruning-cost model · compiler section · software-engineering section

### Conclusion & defense
- [ ] Contributions, limitations, future work
- [ ] Defense slides (30–45 min) + demo · practice · Q&A prep
- [ ] Final review: citations, equations/proofs, notation consistency, figures, university formatting

---

## Open questions (advisor / future work)

1. Why does width-200 collapse at 23.1% while width-50 survives? MC noise vs vanishing adjust gradient vs architectural fragility.
2. Why is seed-2 (dense 0.928) more fragile than seed-0 (dense 0.916)? Does dense accuracy correlate with adjust-step capacity?
3. How does `d_W`-search sparsity compare to magnitude pruning at matched sparsity? (Jaccard plot, §3)
4. Is the manifold sparsity pattern reproducible across seeds? (Lottery-ticket plot, §3)
5. Does `omega_source='data'` (Lazarevich) differ substantially from uniform Ω? — direct test of the data-free claim.
6. Submodularity of the sparsification objective?
7. Precise relationship between critical parameters and lottery tickets?
