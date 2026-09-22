# Dataset research — generalisation, data-free claim, literature defensibility

Date: 2026-09-22. Scope: web research + reading only (nothing downloaded or run).
Constraint recap: ≤784 inputs (ideally 196), ~10 classes, CSV `X` (features) / `Y` (one-hot),
Ω-noise is uniform 0–255 so **inputs must be on a 0–255 scale** for noise-Ω vs data-Ω to be comparable.
Current pipeline: `scripts/dataeng.py` (PIL `Image.resize((14,14))`, default bicubic on float "F" mode — explains
small negative pixel values in `X_train_small.csv`), MNIST 14×14 [196,10,10,10] dense acc 0.916.

Legend: **V** = URL verified to resolve (HTTP 200 / API hit on 2026-09-22). "est." = my estimate, not from a source.

---

## 1. Candidate datasets

### 1a. Image datasets (drop-in for `dataeng.py`)

| Dataset | Official source | Mirror(s) | License | Samples (train/test) | Native | Classes | DL size | Map to 196/10 | Dense small-MLP acc | What it tests |
|---|---|---|---|---|---|---|---|---|---|---|
| **Fashion-MNIST** | github.com/zalandoresearch/fashion-mnist **V** | torchvision `FashionMNIST` **V**; OpenML 40996 **V**; HF `zalando-datasets/fashion_mnist` **V**; TFDS `fashion_mnist` **V** | MIT | 60k/10k | 28×28 gray, 0–255 | 10 | ~30 MB (26 MB + 4.3 MB imgs) | identical to MNIST path: resize 28→14 (use same PIL resize as MNIST for consistency; `BOX`/area is cleaner), flatten | repo lists MLP 256-128-100 = 88.3 %; [196,10,10,10] ≈ 82–85 % (est.) | Same format, harder, *less spatially sparse* (clothing fills the frame → fewer always-zero border pixels). Tests whether the MNIST sparsity map (border pruned first) is a data artefact. |
| **KMNIST** (Kuzushiji-MNIST) | github.com/rois-codh/kmnist **V** (files at codh.rois.ac.jp/kmnist/dataset/ — 403 to curl, use mirrors) | torchvision `KMNIST` **V**; OpenML 41982 **V**; HF `tanganke/kmnist` **V**; TFDS `kmnist` **V** | CC BY-SA 4.0 (cite Clanuwat et al. 2018) | 60k/10k | 28×28 gray | 10 | ~20 MB | same as MNIST | 4-NN 92.1 % (vs 97.1 % MNIST) at 28×28 (repo); small MLP ≈ 75–82 % (est.) | Harder handwritten, more strokes; 14×14 loses detail — accuracy ceiling test. |
| **USPS** | LIBSVM multiclass page csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html **V** | torchvision `USPS` **V** (returns 0–255 uint8); OpenML 41082 **V** (CC0); HF `flwrlabs/usps` **V** (2.57 MB) | CC0 on OpenML; original license unstated | 7,291/2,007 | **16×16 native** | 10 (imbalanced, majority 16.7 %) | ~2.6 MB | either keep 256 inputs ([256,…]) or resize 16→14; values [-1,1] in LIBSVM → rescale to 0–255 | ~93–95 % (est.) | Same task as MNIST, different acquisition (the OBD dataset family) — very cheap, and 256 inputs = 2×128 crossbar tiles exactly. |
| **EMNIST Digits / Letters** | nist.gov/itl/products-and-services/emnist-dataset **V**; zip biometrics.nist.gov/cs_links/EMNIST/gzip.zip **V** | torchvision `EMNIST(split=…)` **V**; OpenML 41039 (Balanced) **V**; HF `tanganke/emnist_letters` **V** | NIST public data (cite Cohen et al. 2017) | Digits 240k/40k; Letters 124.8k/20.8k | 28×28 (stored **transposed**) | 10 / 26 | ~560 MB zip (all splits) | transpose, resize 14; Letters → pick 10 letters or keep 26 | Digits ≈ MNIST; Letters small-MLP ≈ 70 % (est.) | Digits: nothing new vs MNIST. Letters: class-count scaling. Low priority; big download. |
| notMNIST (small) | yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html **V** | none canonical (Kaggle copies) | unstated | ~18.7k (small) | 28×28 | 10 (A–J) | ~8 MB | resize 14 | ~85–88 % (est.) | Font glyphs; noisy labels, no official split — **skip** (weak provenance for committee). |
| SVHN gray | ufldl.stanford.edu/housenumbers **V** | torchvision `SVHN`; HF `ufldl-stanford/svhn` **V** | non-commercial research | 73k/26k | 32×32 RGB | 10 | ~180 MB | gray → 14×14 | small MLP ≈ 40–60 % (est.) | Natural images; MLP too weak at 196-in → accuracy drop dominates. **Skip.** |
| CIFAR-10 gray | cs.toronto.edu/~kriz/cifar.html **V** | torchvision `CIFAR10`; OpenML 40927 **V**; HF `uoft-cs/cifar10` **V** | none stated (cite Krizhevsky 2009) | 50k/10k | 32×32 RGB | 10 | ~163 MB | gray → 14×14 | ≈ 30–35 % (est.) | Same as SVHN. **Skip.** |

### 1b. Tabular datasets (MLP-native; pipeline infers topology from CSV so no need to force 196)

| Dataset | Source (UCI) | Mirror | License | Samples | Features | Classes | DL size | Mapping | Dense MLP acc | What it tests |
|---|---|---|---|---|---|---|---|---|---|---|
| **optdigits** | archive.ics.uci.edu/dataset/80 **V** | OpenML 28 **V**; `sklearn.datasets.load_digits` (test part only, 1,797) | CC BY 4.0 | 3,823/1,797 | 64 (8×8, ints 0–16) | 10 | 0.58 MB | ×16 → 0–255; topology [64,…] | ~96–97 % (est.; SVM/kNN literature ~98 %) | Tiny, 8×8 = still an image → can show spatial map at a third resolution. Runs in minutes. |
| **pendigits** | archive.ics.uci.edu/dataset/81 **V** | OpenML 32 **V** | CC BY 4.0 | 7,494/3,498 (writer-disjoint) | 16 (pen trajectory, 0–100) | 10 | <1 MB | ×2.55 → 0–255; [16,…] | ~97 % (est.) | Non-image, non-spatial features, 10 classes. Very cheap. Tests "structure-agnostic" on real data with no pixel geometry. |
| HAR (smartphones) | archive.ics.uci.edu/dataset/240 **V** | OpenML 1478 **V** | CC BY 4.0 | 7,352/2,947 (subject-disjoint) | 561 (pre-normalised [-1,1]) | 6 | 58 MB | rescale to 0–255; [561,…] | ~93–95 % (est.) | Many correlated/redundant engineered features → does pruning pick a small subset? Larger input, costlier. |
| letter | UCI 59 | OpenML 6 **V** | CC BY 4.0 | 20,000 | 16 | 26 | <1 MB | [16,…,26] | ~90 %+ at width ≥50 (est.) | 26 classes; off-spec. Optional. |
| covertype | UCI 31 | OpenML 150 **V** | CC BY 4.0 | 581,012 | 54 | 7 (majority 48.8 %) | 11 MB | subsample 20k | ~70–80 % small MLP (est.) | Imbalanced, mixed binary/continuous. **Skip.** |
| Adult | UCI 2 | OpenML 1590 **V** | CC BY 4.0 | 48,842 | 14 (categorical → ~100 one-hot) | 2 | 4 MB | one-hot + scale | ~85 % | Binary; far from spec. **Skip.** |

OpenML counts verified via `https://www.openml.org/api/v1/json/data/qualities/<id>` (NumberOfFeatures includes target).

---

## 2. What comparable pruning papers use (defensibility)

| Paper | Setting | Datasets / models | Calibration / sample set | Relevance |
|---|---|---|---|---|
| LeCun, Denker, Solla 1989 — OBD (NeurIPS) | post-training saliency (diag. Hessian) + retrain | handwritten digit (US zip-code) recognition net | full train set | Our method's ancestor; used **USPS-type 16×16 digits** → USPS is a historically faithful second dataset. (Exact param count: check paper before quoting.) |
| Hassibi & Stork 1993 — OBS (NeurIPS) | full inverse Hessian | XOR, **MONK's problems**, **NETtalk** (18k→1.56k weights) | full train set | Toy/tabular benchmarks were accepted for small-net pruning → tabular + synthetic are defensible. Not in `Thesis_bibliography.bib` yet. |
| Frankle & Carbin 2019 — LTH (ICLR) | train-prune-rewind | **LeNet-300-100 on MNIST** (266K weights, ~98 % dense), Conv-2/4/6 + VGG/ResNet on CIFAR-10 | — | Canonical "MLP-on-MNIST" pruning benchmark; our [196,…] is a downscaled cousin. Not in bib. |
| Singh & Alistarh 2020 — WoodFisher (NeurIPS) | one-shot OBS-style, no retrain variant | **MLPNet 784→40→20→10 on MNIST** (Fig. 5b, App. S2), ResNet-20 CIFAR-10, ResNet-50/MobileNetV1 ImageNet | 50,000 samples for Fisher (MLPNet) | Closest published analogue to our tiny MLP. Not in bib. |
| Lazarevich, Kozlov, Malinin 2021 (arXiv 2104.15023) | post-training, layer-wise calibration | ResNet-50 etc. on ImageNet | small calibration set; **data-free variant uses synthetic fractal images** | Direct precedent for "noise/synthetic Ω" — they needed *structured* synthetic data (fractals), consistent with our finding that pure noise-Ω is poor for neuron pruning. In bib. |
| Kwon et al. 2022 (NeurIPS, arXiv 2204.09656) | retraining-free structured (heads/filters) | BERT-base, DistilBERT on GLUE, SQuAD | **2K training examples** (“typically 1–2K”) | Structured pruning with a small label-free-ish sample set. In bib. |
| Frantar, Singh, Alistarh 2022 — OBC (NeurIPS, arXiv 2208.11580) | post-training layer-wise OBS | ResNets/ImageNet, YOLOv5/COCO, BERT/SQuAD | **1,024 random training samples** | Layer-wise reconstruction ≈ our d_W restricted per layer. Not in bib. |
| Frantar & Alistarh 2023 — SparseGPT (ICML) | one-shot, LLMs | OPT, BLOOM | 128 × 2048-token C4 segments | Scale extreme; cite for calibration-set size only. In bib. |

Takeaway for the committee: **MNIST-family MLPs are the standard small-network pruning benchmark** (OBD, LTH, WoodFisher); modern post-training papers use 128–2K calibration samples, and the only data-free precedent (Lazarevich) uses structured synthetic images, not white noise. None of these papers use Fashion-MNIST, but it is the standard drop-in replacement (Xiao et al. 2017) and requires zero pipeline change.

---

## 3. Synthetic benchmark for the "data-free / structure-agnostic" claim

### Literature patterns
| Design | Source | Property |
|---|---|---|
| i.i.d. Gaussian inputs + random teacher net, student learns teacher | Gardner & Derrida 1989; Saad & Solla 1995 (PRL); Goldt et al. 2019 (NeurIPS, "Dynamics of SGD for two-layer NNs in the teacher-student setup") | Isotropic: no input is privileged → null model |
| Hidden manifold model (low-dim latent → high-dim inputs) | Goldt et al. 2020 (PRX, "Modelling the influence of data structure…") | Shows results on i.i.d. Gaussian inputs differ qualitatively from structured data |
| Gaussian vs non-Gaussian inputs → localised receptive fields only with non-Gaussian structure | Ingrosso & Goldt 2022 (PNAS, "Data-driven emergence of convolutional structure") | Directly predicts: Gaussian inputs ⇒ no spatial pattern in learned (and hence pruned) weights |
| `make_classification`: Gaussian clusters on hypercube vertices, n_informative / n_redundant / useless features | scikit-learn docs **V**; Guyon 2003 (NIPS feature-selection challenge, MADELON) | Planted ground truth of which inputs matter |

### Recommended spec (two datasets, both 196 × 10, same CSV format)

**S1 — null (structureless) teacher-student**
| Item | Value |
|---|---|
| Inputs | x ~ N(0, I_196), then `x_px = clip(127.5 + 42.5·x, 0, 255)` (±3σ → 0–255, so noise-Ω uniform[0,255] has the same range) |
| Teacher | [196,10,10,10] ReLU, He-normal weights, zero bias, seed 1 (same topology as student ⇒ realisable) |
| Labels | z = teacher(x); standardise each logit column over the train set (z−μ)/σ (balances classes; it's an affine map foldable into the last layer, so still realisable); y = argmax; one-hot |
| n_train / n_test | 20,000 / 10,000 (same as current `X_train_small`/`X_test_small`) |
| Seeds | data seed 0, teacher seed 1, student training seed 0 |
| Also save | teacher weights (`.npz`) → allows measuring student-teacher alignment after pruning |
| Predictions | (a) input-layer survival per pixel ≈ uniform: χ² vs Binomial, Moran's I on 14×14 grid ≈ 0 (MNIST ≫ 0); (b) data-Ω ≈ Gaussian-noise-Ω for neuron pruning (gap closes) — the clean test that data-Ω wins *because of* data structure |

**S2 — planted structure (`make_classification`)**
| Item | Value |
|---|---|
| Call | `make_classification(n_samples=30000, n_features=196, n_informative=20, n_redundant=0, n_repeated=0, n_classes=10, n_clusters_per_class=1, class_sep=1.5, flip_y=0.0, shuffle=False, random_state=0)` |
| Feature placement | informative = first 20 columns (shuffle=False); apply a saved permutation `perm = default_rng(0).permutation(196)` so they sit at random pixels |
| Scaling | per-feature standardise, then same `127.5 + 42.5·x` clip map |
| Split | first 20,000 train / last 10,000 test |
| Predictions | input weights on the 176 useless features pruned first; precision@k of surviving inputs vs the 20 planted ones; neuron pruning should keep units wired to informative inputs |

S1 tests "pattern is uniform when data has no structure"; S2 tests "the method finds structure when it exists but isn't spatial". Both cost the same as an MNIST run (same shape); generation is ~20 lines numpy/sklearn (sklearn 1.6.1 already installed).

---

## 4. Ranked recommendation (run order)

| # | Dataset | Why | Relative cost (MNIST run = 1×) |
|---|---|---|---|
| 1 | **Fashion-MNIST 14×14**, [196,10,10,10] + one wide net ([196,200,200,10]) | Zero code change (swap CSV in `dataeng.py`), MIT, universally recognised; tests whether results + border-first sparsity map are MNIST artefacts | 1× per config |
| 2 | **Synthetic S1 + S2** (spec above) | Answers Q2 directly; same shape, no download; S1 gives the null model for the "data-Ω beats noise-Ω" claim | 1× each (2×) |
| 3 | **USPS 16×16 → 256 inputs** (or 14×14) | Historical OBD dataset; native low-res (no resize artefact); 256 = 2 × crossbar-128 tiles — natural PIM tile-boundary structured-pruning demo; only 7.3k train | ~1.3× at 256-in (weights scale with input), fewer samples to scan |
| 4 | **pendigits** (or optdigits) | Real, non-spatial, 10-class tabular; [16,…] is tiny ⇒ can run full budgets and many seeds in minutes | ≪0.1× |

Deliberately skipped: SVHN/CIFAR gray (MLP accuracy too low at 196-in to say anything about pruning), notMNIST (provenance), EMNIST (560 MB for nothing beyond MNIST unless class-count scaling is wanted), covertype/Adult (off-spec classes/imbalance). KMNIST is the fallback if a second image set is wanted after Fashion-MNIST.

Bib additions needed for §2: Hassibi & Stork 1993, Frankle & Carbin 2019, Singh & Alistarh 2020, Frantar et al. 2022 (OBC), Xiao et al. 2017 (Fashion-MNIST), Goldt et al. 2019/2020, Ingrosso & Goldt 2022, Clanuwat et al. 2018 (KMNIST) if used.

---

## 5. Sources

- Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist ; Xiao, Rasul, Vollgraf, arXiv 1708.07747
- KMNIST: https://github.com/rois-codh/kmnist ; Clanuwat et al., arXiv 1812.01718
- EMNIST: https://www.nist.gov/itl/products-and-services/emnist-dataset ; Cohen et al., arXiv 1702.05373
- USPS: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html ; https://huggingface.co/datasets/flwrlabs/usps ; https://www.openml.org/d/41082
- UCI: https://archive.ics.uci.edu/dataset/80 (optdigits), /81 (pendigits), /240 (HAR)
- OpenML API: https://www.openml.org/api/v1/json/data/<id> (IDs 40996, 41982, 41082, 28, 32, 6, 1478, 150, 1590, 41039, 40927)
- torchvision datasets: https://pytorch.org/vision/stable/datasets.html ; TFDS: https://www.tensorflow.org/datasets/catalog/fashion_mnist
- notMNIST: https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html ; SVHN: http://ufldl.stanford.edu/housenumbers/ ; CIFAR: https://www.cs.toronto.edu/~kriz/cifar.html
- sklearn make_classification: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_classification.html
- LeCun, Denker, Solla, "Optimal Brain Damage", NeurIPS 1989
- Hassibi & Stork, "Second order derivatives for network pruning: Optimal Brain Surgeon", NeurIPS 1993
- Frankle & Carbin, arXiv 1803.03635 (ICLR 2019) — LeNet-300-100 table read from PDF
- Singh & Alistarh, arXiv 2004.14340 (NeurIPS 2020) — MLPNet 784-40-20-10, 50k Fisher samples read from PDF
- Lazarevich, Kozlov, Malinin, arXiv 2104.15023
- Kwon et al., arXiv 2204.09656 (NeurIPS 2022) — "2K examples" read from PDF
- Frantar, Singh, Alistarh, arXiv 2208.11580 (NeurIPS 2022) — "1024 random training samples" read from PDF
- Frantar & Alistarh, SparseGPT, ICML 2023
- Saad & Solla, PRL 74 (1995); Goldt et al., NeurIPS 2019 (arXiv 1906.08632); Goldt et al., PRX 10 (2020) (arXiv 1909.11500); Ingrosso & Goldt, PNAS 119 (2022) (arXiv 2202.00565)

Caveats: accuracies marked "est." are unverified priors; OBD/OBS dataset details are from memory of the papers (PDFs not fetched) — check before quoting numbers.

---

## 7. Implementation notes (2026-09-22)

Built by `scripts/prep_dataset.py {fashion,kmnist,usps,s1_teacher,s2_planted}` into `data/<name>/`
(regenerated deterministically — not committed). The MNIST CSVs were reverse-engineered first:
Colab's 20k/10k MNIST sample, PIL mode-'F' bicubic 28→14 (hence values ≈ −30..280); new image
datasets use the identical pipeline and a fixed 20k/10k subsample.

Deviations from §3 spec, found while building:
- **S1**: per-column logit standardisation alone left classes at 3–19 %; added fitted per-class
  logit offsets (still an affine last-layer map ⇒ realisable). Offsets saved in `teacher.npz`.
- **S2**: `shuffle=False` also sorts *rows* by class (train split had only classes 0–6); rows are
  now shuffled with seed 1, feature order preserved. Planted pixels in `informative_pixels.npy`.

Measured dense [196,10,10,10] test accuracy (CPU, seed 0; MNIST baseline ≈ 0.85–0.91):

| Dataset | epochs | acc |
|---|---|---|
| Fashion-MNIST | 1000 | 0.815 |
| KMNIST | 1000 | 0.703 |
| USPS | 1000 | 0.918 |
| S1 teacher | 5000 | 0.571 (plateaus ~0.59, overfits; tiny label margins) |
| S2 planted | 5000 | 0.801 |

S1/S2 need 5000 epochs: inputs centred at 127.5 (vs mostly-zero MNIST) slow SGD at lr 0.01.
S1's low ceiling is acceptable — its tests are the spatial survival pattern and the Ω gap, not accuracy.

Experiments: `experiments/e26`–`e33`, queue `scripts/run/queue_datasets.txt`.
`neuron_sparsifier` now honours `sparsify.omega_source` (default noise, unchanged).
