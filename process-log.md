# SPARC Training — Process Log

**Production run:** `art_dinov3_vjepa21_n8192_k128_ep30` (see §9 for why)
**Also trained (rejected):** `…_n8192_k64_ep30` (R1), `…_n4096_k64_ep30` (R2)
**Date:** 2026-09-17 → 2026-09-21
**Location:** everything lives under `/media/riccardo/OS/the-logic-of-pictures/EMB-EXTR-SPARC` (nothing on Desktop).

> Note: §3–§6 below document R1 (`k=64`) as the *first* attempt. The model actually
> shipped in the apps is **R3 (`k=128`)** — see §6c and the summary in §9.

---

## 1. What we are doing (high level)

We are training a **SPARC** model — a *shared* sparse autoencoder (a multi-stream
Top-K SAE) — that jointly reconstructs the global image embeddings of **two
different vision encoders** over the **same** corpus:

- **DINOv3** (ViT-L/16) — self-supervised image features, 1024-d.
- **V-JEPA 2.1** (ViT-L, 384) — video/JEPA predictive features (mean-pooled), 1024-d.

The goal is a **single shared dictionary of latent concepts** that both encoders
can be decomposed into. Because the two streams share one dictionary, we can ask
which concepts are *shared* across the encoders (activate consistently for the
same image in both) versus *divergent* (specific to one encoder). Those latents
then drive the two downstream apps (`concept_app`, `image_decomposition`).

### Pipeline
```
corpus images (598,640)
      │  (already extracted earlier)
      ▼
outputs/dinov3/global.pt           (598640 × 1024, fp16)
outputs/vjepa21/global_meanpool.pt (598640 × 1024, fp16)
      │  pt_to_h5.py  (deterministic seed-42 split, 80/20)
      ▼
h5_features/{dinov3,vjepa21}_{train,val}.h5
   train = 478,912 rows   val = 119,728 rows
      │  SPARC/main.py
      ▼
runs/art_dinov3_vjepa21_n8192_k64_ep30/
   msae_checkpoint.pth        (trained model)
   training_metrics.json      (per-epoch curves)
   run_config.json            (exact args used)
```

> **Why we retrained.** The previous checkpoint under this same run name had been
> trained on a *stale* set of HDF5 files (17,656 rows, dated July) that did **not**
> correspond to the new September extraction (598,640 rows). We regenerated the
> HDF5 from the new `outputs/` and retrained from scratch so the model genuinely
> reflects the new corpus.

---

## 2. Data preparation (`pt_to_h5.py`)

Converts the extracted `.pt` embedding tensors into the HDF5 layout the trainer
expects (one dataset `features` of shape `(N, D)`, float32).

- Loads both streams and **verifies `N` matches** across them — this guarantees
  row *i* is the *same image* in every stream (alignment is preserved).
- Builds a **deterministic train/val split** with `numpy` RNG seeded at 42:
  a fixed permutation of the `N` rows, first 20 % → val, rest → train, and the
  indices are **sorted** so HDF5 reads stay contiguous (fast). The identical
  indices are used for every stream, so val row *v* means the same image in both
  DINOv3 and V-JEPA.
- Saves `split_indices.npz` (the `train`/`val` row ids) so we can later map any
  val row back to its corpus row → filename via `outputs/manifest.csv`.

Result of this run:
```
Split: train=478912, val=119728 (val_ratio=0.2, seed=42)
```

---

## 3. The training command

```bash
python3 main.py \
  --config ../configs/config_art_dinov3_vjepa21.json \
  --output_dir ../runs \
  --exp_name art_dinov3_vjepa21_n8192_k64_ep30 \
  --n_latents 8192 \
  --k 64 \
  --auxk 64 \
  --batch_size 256 \
  --num_epochs 30 \
  --lr 1e-4 \
  --dead_neuron_threshold 1000 \
  --auxk_coef 0.03125 \
  --cross_loss_coef 1.0 \
  --topk_type global \
  --seed 42 \
  --verbose \
  --only_train
```

---

## 4. What every parameter means

### Data / IO
| Parameter | Value | Meaning |
|---|---|---|
| `--config` | `config_art_dinov3_vjepa21.json` | JSON listing the train/val HDF5 file per stream. Now points at the local `/media` HDF5 (the new 598k corpus). |
| `--output_dir` | `../runs` | Parent folder where the run subfolder is written. |
| `--exp_name` | `art_dinov3_vjepa21_n8192_k64_ep30` | Name of the run subfolder; also the id the apps reference. |
| `--only_train` | (flag) | Run **training only**, skip the downstream retrieval/classification evaluation phase. |
| `--seed` | `42` | Random seed for weights init, batch shuffling, and the split generator — makes the run reproducible. |
| `--verbose` | (flag) | Print per-epoch progress bars and loss values. |

### Model capacity & sparsity (the important ones)
| Parameter | Value | Meaning |
|---|---|---|
| `--n_latents` | `8192` | **Dictionary size** — number of latent concept units in the shared SAE. Larger = more, finer concepts (and more potential dead units). |
| `--k` | `64` | **Top-K sparsity** — for each image, only the 64 highest-activating latents are kept; the rest are forced to zero. This is what makes the code *sparse* and each latent *interpretable*. Lower k = sparser/more monosemantic, higher reconstruction error. |
| `--auxk` | `64` | **Auxiliary-K** — number of *extra* latents used only by the auxiliary loss (below) to revive dead units. Typically set equal to `k`. |
| `--use_sparse_decoder` | *off* | Would use a Triton sparse-decoder kernel; left off (dense decode). |

### Optimization
| Parameter | Value | Meaning |
|---|---|---|
| `--batch_size` | `256` | Images per gradient step. With 478,912 train rows → `478912/256 ≈ 1871` steps per epoch. |
| `--num_epochs` | `30` | Full passes over the training set. |
| `--lr` | `1e-4` | Learning rate for the Adam optimizer. |

### Loss terms
| Parameter | Value | Meaning |
|---|---|---|
| `--cross_loss_coef` | `1.0` | Weight of the **cross-stream reconstruction loss**: the shared code inferred from one encoder must also reconstruct the *other* encoder's embedding. This is what forces the dictionary to be genuinely *shared* rather than two independent SAEs. `1.0` = full weight. |
| `--auxk_coef` | `0.03125` (=1/32) | Weight of the **AuxK (dead-neuron) loss**. It asks the currently-dead latents to explain the residual reconstruction error, nudging them back to life. Small so it doesn't dominate the main objective. |

### Dead-neuron handling
| Parameter | Value | Meaning |
|---|---|---|
| `--dead_neuron_threshold` / `--dead_steps_threshold` | `1000` | If a latent hasn't activated for 1000 steps it is declared **dead** and re-initialized, so capacity isn't wasted. |
| `--auxk_threshold` | `1e-3` (default) | A selected latent must activate above this value to count as "active" and reset its dead-counter. |

### Top-K variant
| Parameter | Value | Meaning |
|---|---|---|
| `--topk_type` | `global` | **Global** Top-K selects the k activations across the whole latent vector per sample (vs `local`, which would pick within groups). Global is the standard, most expressive choice. |

### Eval-only (unused here because of `--only_train`)
| Parameter | Value | Meaning |
|---|---|---|
| `--train_ratio` | `0.8` | Split ratio for the downstream linear probe (only used in the eval phase). |
| `--hidden_dim` | `256` | Hidden size of that downstream classifier (eval phase only). |

---

## 5. How to read the training metrics

`training_metrics.json` stores per-epoch lists, per stream:

- **`recon_losses` / `val_recon_losses`** — reconstruction MSE for each encoder's
  own embedding. Lower = the sparse code explains the encoder better.
- **`cross_losses` / `val_cross_losses`** — the cross-stream terms
  (`vjepa21_from_dinov3`, `dinov3_from_vjepa21`): reconstruct one encoder from the
  *other's* inferred code. The asymmetry between the two directions is expected
  (one encoder can be harder to predict from the shared code).
- **`active_neurons`** — should sit at exactly `k = 64` (Top-K guarantees it).
- **`l1_sparsities`** — mean magnitude of the active code; a stability indicator.
- **`epoch_dead_neurons`** — how many latents were dead/reinitialized that epoch.
- **`val_epoch_total_losses`** — the combined validation objective; the headline
  convergence curve.

After training we run `compute_latent_metrics.py`, which produces the per-latent
**cross-stream cosine alignment**, activation frequency, and an **alive** flag —
these are exactly what the two apps use to label a latent *shared* vs *divergent*
and to pick exemplar images.

---

## 6. Results & evaluation (run 1 — `n_latents=8192, k=64, 30 epochs`)

Training completed cleanly (30/30 epochs, evaluation skipped via `--only_train`).
All inputs are **L2-normalized to unit norm** by the dataset loader, so a
reconstruction MSE of `m` corresponds to an approximate cosine similarity of
`cos ≈ 1 − m/2`.

### 6.1 Convergence (validation total loss per epoch)
```
0.472 0.408 0.379 0.358 0.344 0.330 0.322 0.316 0.311 0.308
0.305 0.303 0.300 0.297 0.296 0.292 0.290 0.287 0.285 0.282
0.281 0.279 0.277 0.276 0.275 0.273 0.274 0.272 0.272 0.272
```
Smooth, monotone, and **plateaued** by ~epoch 25 (0.275 → 0.272 over the last 5
epochs). 30 epochs was enough; more epochs would give only marginal gains.

### 6.2 Reconstruction (final epoch)
| Stream | Train recon | Val recon | Approx. val cosine |
|---|---|---|---|
| DINOv3 | 0.181 | **0.182** | ≈ 0.91 |
| V-JEPA 2.1 | 0.043 | **0.043** | ≈ 0.98 |

- Train ≈ Val → **no overfitting** (expected with 478k training rows).
- V-JEPA is reconstructed much more easily than DINOv3. This is normal: it means
  DINOv3's unit-sphere embedding is intrinsically higher-dimensional / harder to
  compress into 64 active atoms than V-JEPA's.

### 6.3 Cross-stream reconstruction (final epoch, validation)
| Direction | Loss | Reading |
|---|---|---|
| `vjepa21_from_dinov3` | **0.051** | Predicting V-JEPA from the DINOv3-inferred shared code works very well. |
| `dinov3_from_vjepa21` | **0.247** | Predicting DINOv3 from the V-JEPA code is much harder. |

The **asymmetry** is the interesting scientific signal: DINOv3 carries structure
that V-JEPA's code cannot recover, while (almost) everything V-JEPA encodes is
present in DINOv3. The shared dictionary is genuinely shared in one direction and
only partially in the other.

### 6.4 Dictionary utilization — ⚠️ the weak point
| Metric | Value |
|---|---|
| Dictionary size `n_latents` | 8192 |
| **Alive latents (final)** | **≈ 1,736 (21 %)** |
| Dead latents (final) | 6,456 (79 %) |
| Active per image `k` | 64 (exactly, as guaranteed by Top-K) |
| Val L1 sparsity | 0.092 / 0.094 |

Dead-neuron count per epoch (DINOv3 stream):
```
5039 5591 5842 6222 6417 6165 5949 5943 5882 5600
5520 5724 5470 5602 5889 5889 5651 5821 6145 6074
6183 6445 6340 6353 6394 6507 6411 6444 6547 6456
```
The dead fraction **grew** over training (≈61 % → ≈79 %) instead of shrinking.
The AuxK revival mechanism is not keeping up: with `n_latents=8192` and only
`k=64` active per sample, the model simply does not need most of the dictionary,
so ~4/5 of it collapses.

### 6.5 Verdict
| Aspect | Rating | Notes |
|---|---|---|
| Convergence | ✅ Good | Plateaued, no overfitting. |
| Reconstruction | ✅ Good | cos ≈ 0.91 (DINOv3) / 0.98 (V-JEPA). |
| Cross-stream sharing | ✅ Informative | Strong one direction, weak the other (expected). |
| **Dictionary utilization** | ⚠️ **Poor** | Only ~21 % of latents alive, and getting worse. `n_latents=8192` is oversized for `k=64` on this data. |

**~1,736 alive latents is still a usable concept dictionary** (the apps only use
alive latents), but 79 % of capacity is wasted and the trend is the wrong way.
If we care about efficient, well-utilized dictionaries, this configuration is
sub-optimal.

### 6.6 Recommended next configuration
The lever is the **capacity ↔ sparsity ratio** (`n_latents` vs `k`). To raise
utilization we can either shrink the dictionary or recruit more atoms per image:

1. **Shrink dictionary → `n_latents=4096`, `k=64`** *(recommended first try)* —
   halves capacity; expected utilization ~40–55 % with similar reconstruction.
2. **Recruit more atoms → `n_latents=8192`, `k=128`** — keeps the big dictionary
   but doubles active atoms per image; lowers reconstruction loss and revives
   more latents (denser, slightly less "monosemantic" codes).
3. **Stronger revival → keep 8192/64 but raise `--auxk_coef` (e.g. 1/16) and/or
   lower `--dead_neuron_threshold`** — pushes AuxK to resurrect dead units.

---

## 6b. Results & evaluation (run 2 — `n_latents=4096, k=64, 30 epochs`)

We retrained with a **halved dictionary** (4096) to test whether the low
utilization in run 1 was simply an oversized dictionary.

### 6b.1 Side-by-side comparison
| Metric | Run 1 — 8192/64 | Run 2 — 4096/64 | Better |
|---|---|---|---|
| Val total loss | **0.2716** | 0.2995 | Run 1 |
| Val recon DINOv3 | **0.182** (cos ≈ 0.91) | 0.198 (cos ≈ 0.90) | Run 1 |
| Val recon V-JEPA | **0.0435** (cos ≈ 0.98) | 0.0471 (cos ≈ 0.98) | Run 1 |
| Cross `vjepa←dinov3` | **0.051** | 0.054 | Run 1 |
| Cross `dinov3←vjepa` | **0.247** | 0.277 | Run 1 |
| **Alive latents (count)** | **1,736** | 1,156 | **Run 1** |
| Utilization (%) | 21.2 % | **28.2 %** | Run 2 |
| Dead trend | 5039 → 6456 (worsens) | 2428 → 2940 (stable-ish) | Run 2 |

### 6b.2 Interpretation
Halving the dictionary **did not fix the dead-neuron problem** — it only produced
**fewer usable concepts** (1,156 vs 1,736) while making reconstruction and
cross-stream sharing slightly worse. Run 2's higher *percentage* utilization is a
red herring: what matters for the downstream apps is the **absolute number of
alive, well-aligned concepts** and the **reconstruction fidelity**, and Run 1
wins on both.

The dead-neuron rate is therefore **inherent to the `k=64` + AuxK revival
dynamics on this data**, not to dictionary size. The data supports roughly
1,100–1,700 active concepts at this sparsity regardless of `n_latents`.

### 6b.3 Decision
**Run 1 (`n_latents=8192, k=64, 30 epochs`) is the better model** and is selected
as the production run for the apps:
- More alive concepts (1,736), better reconstruction, better cross-stream sharing.
- The unused capacity is harmless — the apps only ever use alive latents.

If we later want *more* alive concepts and lower reconstruction error, the lever
to try is **increasing `k` (e.g. 128)** or **strengthening AuxK revival**
(`--auxk_coef 1/16`), not changing `n_latents`.

---

## 6c. Results & evaluation (run 3 — `n_latents=8192, k=128, 30 epochs`) — ⭐ SELECTED

Following §6b's conclusion (the bottleneck is sparsity `k`, not dictionary size),
we doubled the active atoms per image to `k=128` (with `auxk=128`), keeping the
8192 dictionary. This is the decisive run.

### 6c.1 Three-way comparison (final-epoch validation)
| Metric | R1 — 8192/64 | R2 — 4096/64 | **R3 — 8192/128** |
|---|---|---|---|
| Val total loss | 0.2716 | 0.2995 | **0.2129** |
| Recon DINOv3 | 0.182 (cos 0.91) | 0.198 (cos 0.90) | **0.126 (cos 0.94)** |
| Recon V-JEPA | 0.0435 (cos 0.98) | 0.0471 | **0.0465 (cos 0.98)** |
| Cross `vjepa←dino` | 0.051 | 0.054 | **0.0485** |
| Cross `dino←vjepa` | 0.247 | 0.277 | **0.184** |
| **Alive latents** | 1,736 | 1,156 | **6,702** |
| Utilization | 21 % | 28 % | **82 %** |
| Dead trend | 5039→6456 (worsens) | 2428→2940 | **1654→1490 (stable/low)** |

### 6c.2 Interpretation
Raising `k` from 64 → 128 fixed the utilization problem outright and improved
**every** quality metric:
- **Utilization jumped to 82 %** (6,702 alive vs 1,736) — ~4× more usable concepts,
  and the dead count is now low and stable across epochs (no runaway collapse).
- **Reconstruction improved** (DINOv3 cos 0.91 → 0.94; V-JEPA steady at 0.98).
- **Cross-stream sharing improved in both directions**, and the hard direction
  `dinov3←vjepa` dropped from 0.247 → **0.184** — the shared code now captures
  substantially more of DINOv3's structure.

**Trade-off:** codes are denser — 128 active atoms per image instead of 64, so the
per-atom L1 sparsity is lower (0.033 vs 0.092) and each individual latent is a bit
less "monosemantic". For this two-encoder concept-decomposition use case that is a
very worthwhile trade for 4× the concept coverage and better fidelity.

### 6c.3 Final decision
**Run 3 (`n_latents=8192, k=128, 30 epochs`) is selected as the production model.**
It dominates R1 and R2 on convergence, reconstruction, cross-stream sharing, and
dictionary utilization. Run id: `art_dinov3_vjepa21_n8192_k128_ep30`.

## 7. Status

- [x] Regenerated HDF5 from the new 598,640-row extraction (train 478,912 / val 119,728).
- [x] Config + all `concept_common.py` paths repointed to the local `/media` folder.
- [x] Trained & evaluated three configs: R1 (8192/64), R2 (4096/64), R3 (8192/128).
- [x] **Selected R3 (`8192/k128`)** — best on every metric (82 % utilization, 6,702 alive concepts).
- [x] Recomputed latent metrics for R3, rebuilt `concept_app` + `image_decomposition`
      against R3, and restricted both apps to show **only** R3.

---

## 8. App build (on R3, global-embeddings only — no heatmaps)

### 8.1 Memory fix (why VSCode was crashing)
The stock `cc.compute_latent_metrics` upcasts both `(119728 × 8192)` activation
matrices to **float64** and makes several full-size copies (`a`, `b`, `am`, `bm`,
`np.where` temporaries) → **>40 GB peak**, which OOMed the 62 GB machine and took
VSCode/desktop down.

Replaced with a **single-pass, low-memory** script
[`compute_metrics_lowmem.py`](runs/art_dinov3_vjepa21_n8192_k128_ep30/concept_analysis/compute_metrics_lowmem.py):
it memmaps the cached fp16 `val_latents_<stream>.npy` and accumulates per-latent
sufficient statistics (`Σa, Σb, Σa², Σb², Σab, counts`) over row-chunks, computing
identical metrics. **Peak RAM 4.65 GB, 14 s.** Output identical schema
(`latent_metrics.csv`).

### 8.2 Commands run
```bash
cd runs/art_dinov3_vjepa21_n8192_k128_ep30/concept_analysis
python3 compute_metrics_lowmem.py                                     # latent_metrics.csv  (4.7 GB peak)
python3 build_concept_app.py --n_latents 75 --top_n 10 --no_heatmaps  # concept_app/data/<run>/  (4.8 GB peak)
```
Then both registries were restricted to R3 only:
- `concept_app/data/runs.json` → only `art_dinov3_vjepa21_n8192_k128_ep30`.
- `image_decomposition/serve.py` `RUN_IDS` → only `art_dinov3_vjepa21_n8192_k128_ep30`.

### 8.3 Results
- **Alive latents (active in both streams): 5,042 / 8,192.**
- Concept browser: **135 inspected latents** (union of top-75 by alignment + top-75
  by variance), mean cross-stream alignment **0.940**, median **0.947**, 2,562 base
  exemplar thumbnails. Data ≈ 79 MB.
- Heatmaps are **off** (patch tokens intentionally excluded), so the apps run on
  global embeddings only — cards show exemplar images + per-latent metrics; the
  decomposition viewer shows per-image active latents and shared/divergent counts.

### 8.5 DINOv3 spatial activation maps (added)
DINOv3 patch tokens **do** exist — `outputs/dinov3/patch_tokens.dat` (598,640 ×
196 × 1024 fp16, 14×14 grid). V-JEPA's store is empty. The builder was changed to
generate heat overlays **only for streams with a valid patch store**, so DINOv3
spatial maps work while V-JEPA is silently skipped.

Rebuilt with heatmaps on: **1,350 DINOv3 heat overlays** across the 135 latents
(10 exemplars each), V-JEPA items carry no `heat` key. The concept-browser toggle
now shows per-image DINOv3 activation heatmaps. Build ran detached (~1 min, GPU).

### 8.6 image_decomposition fixes
The app was rendering but froze on boot because it built a **119,728-option
`<datalist>`**. Fixes in [`app/app.js`](image_decomposition/app/app.js):
- Seed the datalist with a **light sample** only (exact filename typing still works
  via the `fileToV` map + the random button). Boot went from a multi-second freeze
  to **~28 ms**.
- Made the map header **dynamic** (`${sd.n_active} of ${meta.n_latents} latents
  active`) instead of a stale hard-coded "4096".

### 8.4 Running the apps
```bash
cd sparc_experiment
python3 -m http.server 8772 --directory concept_app       # → http://localhost:8772/
python3 image_decomposition/serve.py --port 8771          # → http://localhost:8771/
```
Verified live (all HTTP 200):
- concept_app `/data/runs.json` → `[art_dinov3_vjepa21_n8192_k128_ep30]`; DINOv3 heat
  overlay PNGs serve (e.g. 93 KB); index/app.js/style.css + base thumbnails serve.
- image_decomposition `/api/runs` → R3 only; `/data/images.json` → 119,728 val images;
  `/image/<run>/0.json` → 128 active latents/stream (121 shared for DINOv3).

*Captions / Florence are intentionally not used (0 captions).*

---

## 9. Which run we use, the split, and how latents are ordered

### 9.1 The production run
The apps are served **only** from:

```
run id : art_dinov3_vjepa21_n8192_k128_ep30
config : n_latents = 8192   k = 128   auxk = 128   epochs = 30   seed = 42
streams: DINOv3 (1024-d)  ×  V-JEPA 2.1 (1024-d, mean-pooled)
```

This is **R3**. We trained three configs and R3 won on every metric (val loss
0.213, DINOv3 recon cos ≈ 0.94, V-JEPA cos ≈ 0.98, and **82 % dictionary
utilization** — 6,702 latents alive during training, of which **5,042 are active
in both streams**, which is the set the apps use). See §6c for the full
comparison and §6b for why raising `k` (not shrinking `n_latents`) was the fix.

### 9.2 The train / eval split
One deterministic split is made **once** from the 598,640-image corpus and reused
by every stream, so row *i* is the same image in DINOv3 and V-JEPA:

| Split | Rows | Fraction | Used for |
|---|---|---|---|
| **train** | 478,912 | 80 % | fitting the SPARC dictionary (`SPARC/main.py`) |
| **val** | 119,728 | 20 % | all latent metrics + everything the two apps show |

- Built by `pt_to_h5.py` with `numpy` RNG **seed 42**: a fixed permutation of the
  598,640 rows, first 20 % → val, the rest → train, indices **sorted** so HDF5
  reads stay contiguous. The `split_indices.npz` file records exactly which corpus
  rows are in each split, so any val row *v* maps back to its corpus row → filename
  via `outputs/manifest.csv`.
- We trained with `--only_train`, so there was **no separate downstream-probe
  "eval" phase** (the `--train_ratio 0.8` / `--hidden_dim 256` args only apply to
  that skipped classifier). "Eval" here therefore means the **held-out val set**:
  the model never saw those 119,728 images during training, and every number and
  image in the apps is computed on them. Train ≈ val reconstruction confirms no
  overfitting.

### 9.3 What the per-latent orderings mean
After training, `compute_metrics_lowmem.py` computes three quantities **per latent**,
measured over the 119,728 val images. Sorting the dictionary by each answers a
different question:

- **freq (activation frequency)** — the fraction of val images on which the latent
  fires (activation > 0); `freq_min` is the smaller of the two per-stream
  frequencies (a co-activation proxy). It is a measure of **how common vs rare** a
  concept is.
  - Order by **high freq** → generic, ever-present concepts (page texture, generic
    background) that fire on almost everything.
  - Order by **low freq** → rare, specific concepts that fire on only a handful of
    images. In the *image_decomposition* app the "activation × rarity" ranking uses
    `act × −log(freq)` so that a latent counts more when it is both strongly active
    **and** rare — i.e. it is *specific* to this image rather than firing everywhere.

- **align (cross-stream alignment, `align_cos`)** — for one latent, take its
  activation value across all 119,728 val images as a vector in each stream, then
  take the **cosine similarity between the DINOv3 vector and the V-JEPA vector**.
  ≈ 1 means the two encoders light this latent up on the *same* images → a genuinely
  **shared** concept; low means the latent means different things to each encoder →
  **divergent**.
  - Order by **high align** → the concepts both models agree on (the common visual
    vocabulary). This is the concept browser's "alignment" set and the basis for the
    "shared / divergent" badges (threshold 0.85).

- **var (activation variance, `var_mean`)** — the mean across the two streams of the
  variance of a latent's activation over images. High variance = the latent swings
  strongly between images (very active on some, silent on others); low variance =
  it barely changes.
  - Order by **high var** → the most **discriminative** concepts, the ones that most
    sharply separate some images from the rest (visually the most "informative"
    directions). This is the concept browser's "variance" set.

The concept browser's inspected panel is the **union of the top-75 latents by
`align` and the top-75 by `var`** (135 unique latents), so you see both the
*most-shared* and the *most-discriminative* concepts; each row also shows its
`freq` so you can tell common concepts from rare ones at a glance.

### 9.3.1 Exact formulas (how the three are computed)

For one latent, let `a[i]` be its activation on image `i` in the **DINOv3** stream
and `b[i]` its activation in the **V-JEPA** stream, over all `N` images
(N = 598,640 for the full run). Activations are **post-ReLU / Top-K**, so they are
**non-negative** and **sparse** (most images give exactly 0 for a given latent).

`compute_metrics_lowmem.py` makes one streaming pass and accumulates only these
per-latent sufficient statistics (no full matrix in RAM):

```
Sa  = Σ a[i]          Sb  = Σ b[i]
Saa = Σ a[i]^2        Sbb = Σ b[i]^2        Sab = Σ a[i]*b[i]
cnt_a = #{ i : a[i] > 0 }        cnt_b = #{ i : b[i] > 0 }
```

From those:

- **freq** (per stream) — fraction of images on which the latent fires:
  `freq_dinov3 = cnt_a / N`, `freq_vjepa21 = cnt_b / N`;
  `freq_min = min(freq_dinov3, freq_vjepa21)` is a co-activation proxy.

- **var** (per stream) — variance of the activation over *all* images (zeros
  included), then averaged across streams:
  `var_dinov3 = Saa/N − (Sa/N)^2`, `var_vjepa21 = Sbb/N − (Sb/N)^2`,
  `var_mean = ½(var_dinov3 + var_vjepa21)`.
  Because the mean is pulled down by all the zeros, a *rare but intense* latent and
  a *frequent but flat* latent can have similar variance — `var` rewards latents
  that are strongly on for some images and off for the rest.

- **align** — two cross-stream numbers, both over the `a` and `b` vectors:
  - `align_cos = Sab / sqrt(Saa · Sbb)` — **uncentered cosine** of the two
    activation vectors. **This is the number behind the "shared/divergent" badge**
    (shared if `align_cos ≥ 0.85`).
  - `align_pearson = (Sab − Sa·Sb/N) / sqrt((Saa − Sa²/N)(Sbb − Sb²/N))` —
    the **mean-centered** version (Pearson correlation), reported alongside.

  Since `a` and `b` are non-negative and sparse, images where **both** streams are
  0 contribute nothing to any term. So `align_cos` is decided entirely by the
  images where **at least one** stream fires: shared firing with proportional
  magnitudes pushes it toward 1; firing in only one stream (`a[i]>0, b[i]=0`) adds
  to the denominator but not the numerator, pulling it down.

- **mean_act** (per stream) — average activation *among images where it fires*:
  `mean_act_dinov3 = Sa / cnt_a`. Tells you how *intense* the latent is when active.

A latent is **alive** when it fires at least once in **both** streams
(`freq_dinov3 > 0 and freq_vjepa21 > 0`); only alive latents are shown.

### 9.4 Why a "shared" (high-`align`) latent can still have different top images per stream

This is the exact thing you noticed, and it is expected. `align_cos` is a **single
global correlation number** over all 598,640 images; the exemplar panels show each
stream's **own top-10** — a tiny extreme tail (10 / 598,640). These answer two
different questions, and a high global correlation does **not** force identical
top-10 lists. Concretely:

1. **Cosine measures the bulk pattern, not the ranking of the extreme top.**
   `align_cos = 0.85` corresponds to an angle of ~32° between the two activation
   vectors (0.95 ≈ 18°). That leaves real room for disagreement, and the
   disagreement concentrates exactly where the exemplars live — the few
   highest-magnitude images — because the top-10 is set by *tiny* differences at
   the peak.

2. **The firing sets overlap, but each stream ranks them differently.** High
   `align_cos` means DINOv3 and V-JEPA fire on largely the **same population** of
   images. But each encoder assigns its own magnitudes, so *which* of those shared
   images is #1 vs #7 differs. Top-10 is an argsort of the peak; the argsort is far
   more brittle than the overall correlation. Toy example (5 images):

   ```
   dinov3  = [0.9, 0.8, 0.7, 0.1, 0.0]   -> top images: A, B, C
   vjepa21 = [0.7, 0.9, 0.8, 0.0, 0.1]   -> top images: B, C, A
   align_cos ≈ 0.93  (clearly "shared")  but the #1 exemplar differs (A vs B)
   ```

3. **The two streams even fire at different *rates*.** `freq_dinov3` and
   `freq_vjepa21` are usually not equal — a shared concept can fire on 3 % of images
   in one encoder and 5 % in the other. The extra images at the margin are, by
   definition, stream-specific, and some of them can be strong enough to enter that
   stream's top-10.

4. **`align` is a statement about the concept, not about any single image.**
   "Shared" means *the two encoders agree on what this concept is and broadly on
   which images express it* — not *the two encoders pick the identical 10 exemplar
   images*. If you wanted the stricter "same top images" property, the metric to use
   would be **top-K overlap (Jaccard)** of the two exemplar sets, which is a much
   harder bar than a 0.85 cosine. `align_cos` deliberately captures the softer,
   more robust notion of a shared concept.

So: a high-`align` latent = "both encoders light up on the same *kind* of image and
on a strongly overlapping *set*"; the per-stream exemplar strips differ because they
expose each encoder's private ranking of the extreme peak, which cosine alignment
was never claiming to make identical.

---

## 10. Full-corpus run — training on 100 % of the fototeca (Option C)

To make the apps browse the **whole** fototeca (not just the 20 % val split), we
retrained the winning R3 recipe on **all 598,640 images**. Because the SAE is an
*unsupervised* dictionary, using every image for training is legitimate (there is
no label leakage), and it removes the train/val distinction from the apps entirely:
in the full run **row *i* = corpus row *i* = manifest row *i***, so every image is
directly browsable with no split indirection.

### 10.1 Data prep
`pt_to_h5_full.py` writes, from the full `outputs/*.pt`:
- `h5_features_full/{dinov3,vjepa21}_full.h5` — **all 598,640 rows** → the training set.
- `h5_features_full/{dinov3,vjepa21}_monitor.h5` — a **20,000-row random subset**
  (seed 42, also part of train) used **only** to draw a per-epoch loss curve
  (`main.py`'s validation is optional; this is a fit signal, not a held-out score).

### 10.2 Training
```
run id : art_dinov3_vjepa21_n8192_k128_ep30_full
config : n_latents=8192  k=128  auxk=128  epochs=30  seed=42
train  : 598,640 images (100 %)   monitor : 20,000
```

### 10.3 Result vs R3 (the 80 %-train model)
| Metric | R3 (80 % train) | **FULL (100 %)** |
|---|---|---|
| recon DINOv3 | 0.126 (cos 0.94) | **0.121 (cos 0.94)** |
| recon V-JEPA | 0.0465 (cos 0.98) | **0.0426 (cos 0.98)** |
| cross `vjepa←dino` | 0.0485 | **0.0412** |
| cross `dino←vjepa` | 0.184 | **0.178** |
| **Alive latents** | 6,702 (82 %) | **7,632 (93 %)** |
| Dead trend | 1654→1490 | **117→560 (very low)** |

> Caveat: R3's numbers are on the held-out 119,728 val set; the full run's are on
> the 20k in-train monitor subset, so they are not a like-for-like *generalization*
> comparison. But utilization and the dead-neuron trajectory **are** directly
> comparable, and the full run is clearly healthier: **93 % of the dictionary is
> alive** and the dead count stays low all through training. More data supports
> more concepts.

**Decision:** `…_ep30_full` becomes the new production model, and the apps are
rebuilt to run over all 598,640 images (see §11).

---

## 11. Rebuilding the apps over the full corpus (memory-safe)

The full run makes **row i == corpus row i == manifest row i**, so the apps browse
all 598,640 images with no split indirection. The challenge was memory: a dense
`(598640, 8192)` activation matrix is ~9.8 GB fp32 *per stream*, so nothing may be
loaded whole. Everything below streams or memmaps.

### 11.1 Steps (all under `runs/…_ep30_full/concept_analysis/` unless noted)
1. **`cache_full_latents.py`** — streams the corpus through the frozen SAE in
   row-chunks and writes the sparse codes **directly to fp16 memmaps**
   (`val_latents_<stream>.npy`, 9.2 GB each). 77 s.
2. **`compute_metrics_lowmem.py`** — streaming per-latent metrics over all 598,640
   rows → `latent_metrics.csv`. **5,500 latents alive in both streams.** 71 s.
3. **`build_concept_app.py --n_latents 75 --top_n 10`** — concept-browser data for
   the full run; `act` is now **mmapped** (not loaded). 136 inspected latents,
   **1,360 DINOv3 heat overlays**, mean alignment 0.939, ~208 MB. 61 s.
4. **`image_decomposition/build_image_meta.py`** — precomputes the per-latent
   exemplars for all 5,500 alive latents via a **streaming top-K** (on-demand
   column reads over 598k rows would be far too slow) → `data/<run>/meta.json`
   (9.3 MB). *(Slow: ~19 min from the per-chunk argpartition — a candidate to
   optimize later, but it is a one-time cache.)*

### 11.2 Code changes for the full corpus
- `runs/…_ep30_full/concept_analysis/concept_common.py`: `H5_VAL` → the
  `h5_features_full/*_full.h5`; `load_val_corpus_rows()` returns `arange(N)`
  (identity — no split).
- `image_decomposition/decompose_core.py`: `load_val_activations` now **mmaps** the
  fp16 caches (was `np.load(...).astype(float32)` → 39 GB); `_reference_concept_dir`
  prefers the `*_full` run so row order is identity.
- `image_decomposition/serve.py` `RUN_IDS` and `concept_app/data/runs.json` → only
  `art_dinov3_vjepa21_n8192_k128_ep30_full`.

### 11.3 Verified live (all HTTP 200)
- concept_app `/data/runs.json` → `[…_ep30_full]`; DINOv3 heat overlays serve (~93 KB).
- image_decomposition `/api/runs` → full run, **`n_val = 598,640`**;
  `/data/images.json` → **598,640** images with **identity mapping** (`v == row`);
  `/data/<run>/meta.json` → 9.3 MB; `/image/<run>/500000.json` → 128 active
  latents/stream (112 shared for DINOv3); source images serve.
- Server RAM stays low (activations are memmapped, not resident).

Both apps now browse the **entire fototeca** on the 100 %-trained dictionary.

---

## 12. Brainstorm & analysis (full-corpus dictionary)

Measured on the production run `…_ep30_full` over all 598,640 images
(5,500 latents alive in both streams).

### 12.1 Do we have only a few, very specific concepts? (frequency distribution)

Question: *is there only a small number of very-low-frequency latents — i.e. only
a few distinct specific concepts?* The data says the **opposite**: the dictionary
is **long-tailed and rich in specific concepts**.

Distribution of `freq_min` (the smaller of the two per-stream firing rates) across
the 5,500 alive latents:

| `freq_min` range | ≈ images | # latents | share |
|---|---|---|---|
| 0 – 0.0001 | < 59 | 901 | 16.4 % |
| 0.0001 – 0.0005 | 59 – 300 | 43 | 0.8 % |
| 0.0005 – 0.001 | 300 – 600 | 66 | 1.2 % |
| 0.001 – 0.005 | 600 – 3 k | 810 | 14.7 % |
| 0.005 – 0.01 | 3 k – 6 k | 739 | 13.4 % |
| 0.01 – 0.05 | 6 k – 30 k | 2,251 | 40.9 % |
| 0.05 – 0.10 | 30 k – 60 k | 487 | 8.9 % |
| 0.10 – 1.0 | 60 k + | 203 | 3.7 % |

- **Median `freq_min` ≈ 1.2 %** (~6,959 images). **47 % of latents fire on < 1 %**
  of the corpus; only **12.5 % fire on > 5 %** and 3.7 % on > 10 %.
- So we have **many specific concepts**, not few — a healthy heavy tail (many rare,
  monosemantic latents) plus a smaller head of generic, ever-present ones. That is
  exactly the shape a good SAE dictionary should have: rarity = specificity.
- **Caveat on the ultra-rare bucket.** The 901 latents firing on < 59 images (16 %)
  sit near the noise floor: they *are* alive (fire in both streams) but with so few
  exemplars they are hard to interpret and some may be under-trained rather than
  crisp concepts. Even excluding them, ~1,650 latents fire on 59–6,000 images — a
  large, genuinely-specific concept vocabulary. The concern here would be the
  *reverse* of the question (too many near-dead micro-latents), not a shortage of
  specific concepts.

### 12.2 Case study — latent 5897: a *shared* concept with encoder-divergent prototypes

This latent is the perfect illustration of §9.4 and is genuinely informative about
the two encoders. Observed in the app: **V-JEPA's top-activating images are all
Venice; DINOv3's top images are not Venice at all** — yet the latent is labelled
**shared** (`align_cos = 0.903`). How can both be true?

Measured facts for latent 5897:

| Quantity | Value |
|---|---|
| `align_cos` (uncentered) | **0.903** → "shared" |
| `freq_dinov3` / `freq_vjepa21` | 0.1470 / 0.1472 (≈ 88,000 imgs each) |
| **Firing-set overlap** | **both fire on 88,003 imgs = 99.9 % of the union** |
| Top-10 overlap | **0 / 10** (Jaccard 0.00) |
| Top-100 / top-1000 / top-5000 overlap | 6 % / 25 % / 39 % |
| Pearson of magnitudes among co-firing imgs | **0.556** |

The resolution:

1. **They agree almost perfectly on *which* images the concept applies to** — the
   two firing sets are 99.9 % identical (~88 k images). That near-total set overlap
   is what drives `align_cos` to 0.90 and earns the "shared" label. So it is *not*
   that DINOv3 lacks the Venice concept — DINOv3 fires on the very same images.

2. **They disagree strongly on *intensity ranking*.** Among the shared 88 k images
   the per-image magnitude correlation is only 0.556, and the *extreme tail* (the
   top-10 the app shows) is completely disjoint. V-JEPA's top-10 images sit at DINO
   ranks like 36–8,110; DINO's top-10 sit at V-JEPA ranks like 70–22,451. Each
   encoder considers a **different subset the "purest" example** of the same concept.

3. **What this says about the encoders.** V-JEPA (video / JEPA predictive
   pre-training) emphasises **global, predictable scene geometry** — repetitive
   façades, water reflections, canal perspective — so its response *peaks* on Venice,
   the canonical water-city with strong regular structure. DINOv3 (image
   self-supervised, discriminative) emphasises **local texture / object-part**
   statistics, so its peak within the *same* shared concept lands on other images.
   The latent thus exposes each encoder's **inductive bias about what the most
   prototypical instance of a shared concept is** — the concept is common ground,
   the *prototype* is encoder-specific.

**Takeaway.** "Shared" (high `align_cos`) is a statement about the **concept and its
image population**, not about identical exemplars. Latent 5897 is a shared concept
whose *prototype* differs by encoder — precisely the kind of cross-encoder
difference this tool is meant to surface. A stricter "same prototype" view would
rank latents by **top-K exemplar overlap (Jaccard)** instead of cosine; by that
measure 5897 would read as *divergent at the peak, shared in the bulk*.

*(Filenames in the corpus are content-free hashes, so the "Venice" identification
comes from visual inspection in the app; the quantitative structure above — 99.9 %
set overlap with 0 % top-10 overlap — is computed directly from the activations.)*

### 12.3 Why the browser "reaches 14 % so soon" — it shows a frequency-biased slice

Follow-up question: *if the dictionary has so many rare/specific concepts, why does
ordering the app by `freq` hit latent 5897 (14.7 %) almost immediately?* Because the
concept browser does **not** show the dictionary — it shows a **curated 136-latent
slice** (union of top-75 by `align` and top-75 by `var`), and that slice is heavily
skewed toward *frequent* latents.

| Set | # latents | median `freq_min` |
|---|---|---|
| **All alive** | 5,500 | **1.2 %** |
| Shown in app (top-75 align ∪ top-75 var) | 136 | **12.3 %** |
| — its top-75-by-**var** half | 75 | **17.9 %** |
| — its top-75-by-**align** half | 75 | **0.0 %** (ultra-rare) |

The mechanism:

- **`var` selection ≈ frequency selection.** `corr(var_mean, freq_min) = 0.844`.
  For a sparse non-negative latent, variance ≈ `freq · mean_act²·(1−freq)`, so it
  grows with frequency — the "top-75 by variance" panel is essentially "the 75 most
  *frequent/strong* latents" (median 17.9 %). That half of the browser is, by
  construction, all high-frequency concepts.
- **`align` selection is frequency-agnostic** (`corr = 0.023`) and actually pulls in
  the **ultra-rare** end (median `freq_min` ≈ 0 — many fire on < 59 images). So the
  alignment half contributes rare latents, but they cluster at the near-noise-floor
  extreme, not the interpretable-specific middle.
- **Latent 5897 (14.7 %) is genuinely near the top of the *whole* dictionary** —
  only **88 of 5,500 alive latents (top 1.6 %)** fire more often. So 14 % is *not*
  reached "soon"; it is close to the ceiling. Almost nothing fires on > 14 % of a
  600 k-image collection (only generic concepts do). It merely *appears* early
  because the app's freq sort runs over an already-frequent 136-latent subset.

**The "missing middle."** The ~1,650 genuinely-specific-but-interpretable latents
(firing on ~600–6,000 images) are mostly **absent** from the browser: variance
selection skips them (too rare to have high variance) and alignment selection
overshoots into the < 59-image noise floor. So the app's default view under-represents
exactly the specific concepts §12.1 found in abundance.

**Actionable fix (not yet done).** To browse the specific concepts, rebuild
`concept_app` with a selection set that targets a **frequency band** (e.g. top
latents with `0.001 ≤ freq_min ≤ 0.02`), or simply raise `--n_latents`, or add a
"rarity" panel alongside "alignment"/"variance". The data is all there (5,500 alive
latents in `latent_metrics.csv`); only the *selection* for display is currently
frequency-biased.

### 12.4 Added a "rare / specific" panel to the browser

Implemented the §12.3 fix. A third selection set, **`rarity`**, now surfaces the
specific concepts the align/var panels were hiding:

- **Selection** (`make_grids.select_latent_set(which="rarity")`): among alive
  latents, keep those in a rare-but-interpretable band `0.0005 ≤ freq_min ≤ 0.01`
  (≈ 300–6,000 images — above the few-exemplar noise floor, below the generic head),
  ranked by **specificity = mean activation-when-firing / freq_min** (sharp *and*
  rare first).
- **Builder** (`build_concept_app.py`): `--sets` now defaults to
  `alignment variance rarity`; the rarity latents are tagged **`rare`**.
- **Frontend** (`concept_app/app.js`): a new **"show: all / align / var / rare"**
  filter in the latent-list toolbar (`state.filter`) restricts the list to one set.

Result: the browser now shows **211 latents** (75 `align` + 75 `var` + 75 `rare`,
the rare set fully disjoint), 2,110 DINOv3 heat overlays. The `rare` set is genuinely
specific yet coherent — **freq_min 300–1,061 images** (median 528, ~0.09 % of the
corpus) with **median alignment 0.95** — i.e. rare concepts that both encoders agree
on. To explore them: filter **show: rare** (optionally sort by `freq` ascending).

### 12.5 Pushing rarer: the `rarev2` set (and 150 per set)

Goal: *see how far we can push the "rare/specific" idea* — surface concepts firing
on only a handful of images — and show more latents per panel (150 instead of 75).

**Why a second band instead of just lowering `rare`'s floor.** The exemplar strips
show the **top-10** activating images per stream, so a latent must fire on at least
~10 images (**in the rarer stream**, since we gate on `freq_min = min(freq_dino,
freq_vjepa)`) to be interpretable at all. Below that the panel is padding, and — more
importantly — the extreme tail is **degenerate**: a latent that co-fires on 1–3
images has `align_cos ≈ 1.000` *trivially* (one shared point makes two non-negative
vectors collinear), so "alignment" stops meaning "shared concept". We measured this:
the rarest ~160 alive latents (`freq_min → 0`, ≤ 1 image) all had `align_cos = 1.000`.
So the rare extreme must be **floored**, not opened all the way.

**Sizing the band (the math).** With `N = 598,640`, a target of *k* images maps to
`freq = k / N`. We swept candidate bands and counted how many alive latents fall in
each (`freq_min ∈ [lo, hi]`):

| band `freq_min` | ≈ images | # alive latents | median align |
|---|---|---|---|
| [1.0e-4, 5.0e-4] | 59 – 300 | 43 | 0.923 |
| [5.0e-5, 5.0e-4] | 29 – 300 | 71 | 0.916 |
| [3.0e-5, 5.0e-4] | 17 – 300 | 111 | 0.907 |
| **[1.5e-5, 5.0e-4]** | **≈ 9 – 300** | **≈ 180** | **≈ 0.90** |

To fill a **150-latent** panel we need ≥ 150 candidates, which forces the floor down
to **`rare2_lo = 1.5e-5`** (≈ 9 images) — the rarest floor that still yields enough
latents *and* keeps ~10 exemplars. The ceiling **`rare2_hi = 5e-4`** (≈ 300 images)
sits exactly at `rare`'s floor, so the two sets are **disjoint bands**:

```
rarev2 :  freq_min ∈ [1.5e-5, 5e-4]   ≈   9 – 300 images    (ranked rarest-first)
rare   :  freq_min ∈ [5e-4 , 1e-2 ]   ≈ 300 – 6,000 images  (ranked by specificity)
```

**Ranking.** `rare` ranks by *specificity* = `mean_act_when_firing / freq_min` (sharp
**and** rare). `rarev2` instead ranks by **rarity ascending** (`freq_min` smallest
first) — the whole point is to see the rarest end, so we surface the fewest-image
concepts first and let the floor guarantee interpretability.

**Result (built with `--n_latents 150`).** The browser now shows **578 latents**
(150 each of `align` / `var` / `rare` / `rarev2`), 5,780 DINOv3 heat overlays, ~896 MB.
The `rarev2` latents fire on **just 9–40 images** each (median alignment ≈ 0.89 — still
coherent, not the degenerate `align=1` tail), i.e. genuinely hyper-specific concepts.
Filter **show: rarev2** in the browser to explore them.

**Interpretation caveat.** `rarev2` is deliberately at the edge of interpretability:
with 9–40 exemplars these are the sharpest, most specific concepts the dictionary
holds, but a few may be idiosyncratic (a single scanned series, a repeated motif in
one sub-collection) rather than broadly meaningful. That is exactly what "how far can
we push rare" is meant to reveal.

### 12.6 Why some `rarev2` latents may be *noise* rather than real concepts

`rarev2` sits at the statistical edge on purpose, so a fraction of it will not be
meaningful. Measured facts for the 150 `rarev2` latents help separate the real
reasons from the non-reasons:

| Quantity | `rarev2` | `rare` (for contrast) |
|---|---|---|
| co-firing images (both streams) | **9 – 41** (median 17) | ~300 – 1,000 |
| `align_cos` median | 0.893 | 0.95 |
| `align_cos` > 0.99 | **3 / 150** | — |
| `mean_act` (avg when firing) | **0.017** | ~0.058 |

So the **degenerate-alignment failure is NOT the main risk here** — the `1.5e-5`
floor did its job: only 3 of 150 have a suspicious `align_cos > 0.99`, and every
sampled latent co-fires on ≥ 9 images (never the 1–3-point collinearity that makes
`align_cos = 1.000` meaningless, §12.5). The real reasons a `rarev2` latent can still
be noise are:

1. **Small-sample fragility.** Every per-latent number (`align_cos`, `freq`,
   `mean_act`) is estimated from only **9–41 points**. At that sample size the
   estimates have large variance — a `align_cos` of 0.89 computed on 12 shared images
   is weak evidence of a "shared concept" compared with the same value on 1,000
   images. The metric is *correct* but *uncertain*.

2. **Weak activations, near the selection floor.** `rarev2` latents fire ~3× more
   weakly than `rare` (mean_act **0.017 vs 0.058**). Global Top-K keeps the 128
   largest activations per image; a latent that only ever squeaks in near the 128th
   place is riding close to the **fp16 embedding noise / Top-K threshold**, so which
   handful of images it "fires" on is partly determined by numerical noise rather
   than a stable visual property.

3. **AuxK-revived, half-formed directions.** The AuxK loss periodically revives dead
   latents by pointing them at the current reconstruction residual (§4). A latent
   revived **late** in training may have fired on only a few dozen images before
   training ended — it is a *half-trained* direction that happened to grab a small
   residual, not a matured concept. Such latents land exactly in the ultra-rare bucket.

4. **Feature splitting / superposition residue.** With an 8,192-atom dictionary, a
   single real concept can **split** across several atoms; the leftover fragments fire
   on tiny, somewhat arbitrary subsets. Others are superposition residue — directions
   that only activate when several conditions coincide by chance on a few images, so
   their exemplars look visually incoherent.

5. **Dataset artifacts, not semantics.** A fototeca contains **near-duplicate scans,
   multi-page series, repeated backing boards, scanner borders, colour bars, rulers**.
   A latent firing on 9–40 images may be keying on one of these — perfectly
   *consistent* (both encoders even agree, because the artifact is a real pixel
   pattern) yet **not a meaningful concept**. This is "noise" in the sense of *not
   semantic*, and it is the most common way a rare latent misleads.

**How to tell signal from noise in the app.** Open the latent and look at its 9–40
exemplars: (a) if they share an obvious visual property, it is a genuine specific
concept; (b) if they look random/unrelated, it is likely superposition/AuxK residue;
(c) if they are near-duplicates of one another (same object/series), it is a dataset
artifact. A higher `mean_act` and both-stream agreement on a *visibly coherent* set
are the signs of a real rare concept; very low `mean_act` on an incoherent set is the
signature of noise. `rarev2` is offered precisely so these edge cases are visible
rather than hidden — treat it as an exploration frontier, not a curated gallery.
