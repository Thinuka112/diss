# Read-only codebase audit

Scope: static read of the repo + computations over existing CSVs/logs and the source
HPA XML. **No code was modified.** Every number below is either read from a cited
`file:line` or computed by a read-only script over existing data; computed CSVs are in
`audit/`. Where something does not exist, it is marked **NOT FOUND**.

## Project status (read this first — it frames everything below)

**The Stage-1 slice-quality classifier has not been built yet.** This is a *pre-build*
audit of the data, pipeline and labelling tooling — not a post-mortem of a finished model.
Concretely, what exists in the repo today is:

- **Data pipeline** — download → 8×8 tiling → `tissue_fraction` auto-seeding → HPA
  metadata join. Complete.
- **Labelling** — the web reviewer, 500 expert whole-slide labels (428/72), and the
  intra-rater κ = 0.735. Complete.
- **Classifier** — only **heuristic-label pilots** (`train_cnn.py`, `train_cnn_bulk.py`,
  `train_classifier.py`, `analyze_model.py`), each of which is *explicitly* labelled in its
  own docstring as a `tissue_fraction` blank/tissue detector, **not** a quality model
  trained on the human labels. The real model is still to be written.

So findings that mention "the model" are describing those pilots. The value of this audit
is that findings **B–D and the resolution issue (A) are data/pipeline problems that should
be fixed *before* the model is built** — which is exactly the right time to catch them.

## Five most serious findings (in order)

1. **The real slice-quality model does not exist yet — only heuristic-label pilots.** The
   expert judged whole 3000×3000 slides, but every classifier currently in the repo
   (`train_cnn.py`, `train_cnn_bulk.py`, `train_classifier.py`) trains and evaluates on
   96×96 tiles with **`tissue_fraction` heuristic labels**, not the 500 human labels. This
   is not a bug in a finished model — it is that the headline model the write-up describes
   **is still to be built**. Everything below is guidance for building it correctly.
2. **Cancer-donor contamination is real and unfiltered.** No morphology filter exists
   (D6 NOT FOUND). 209 / 3,100 downloaded images and **36 / 500 labelled images** carry an
   adenocarcinoma/carcinoid SNOMED morphology; **12 of those adenocarcinoma cores were
   labelled "usable."** 116 / 500 labelled images (23%) come from cancer-carrying donors.
3. **Donor leakage + almost no donors.** The split groups by *slide*, not *donor*; the
   entire 500-image set is only **22 donors** (one donor = 92 images), and the actual
   split code puts **8–9 donors in both train and test**. "Unseen slide" ≠ unseen donor.
4. **The model sees ~15× fewer pixels than the human did.** Native slide 3000², native
   tile 375², model input 96². Fine detail (folds, focal blur, oblique sectioning) is
   visible at native and largely gone at 96px (`audit/A6_resolution_comparison.png`).
5. **Thin evaluation.** Single fixed split, no cross-validation, no CIs/bootstrap (G1,
   G3); the CNN reports only accuracy + macro-F1 (no AUC-ROC, no per-class P/R) (G2);
   backbone is **ResNet-18, not the intended ResNet-50/timm** (G6); and there is **no
   train/val/test split over the human labels at all** (C3, C4).

---

## A. Input resolution and the label/input mismatch

**A1 — Native pixel size.** **3000 × 3000 px** (measured on disk). Samples:
`HPA_small_intestine/Duodenum_A1BG_Male_50_1904.jpg` → 3000×3000;
`review_tool/webapp/review_images/Duodenum_A1CF_Female_67_3479.jpg` → 3000×3000.
`cut_images.py:54` reads `image.size` but never records it.

**A2 — Transform pipeline (disk → tensor).** Three different paths exist:
- **`train_cnn.py` (primary CNN):** ① open tile, `.convert("RGB").resize((96,96))`
  (`train_cnn.py:97`, resample **not specified** → PIL default); ② in `TileDS.__getitem__`
  (`train_cnn.py:111-120`) train-only augmentation: horizontal flip, vertical flip,
  `rot90` (lines 114-118); ③ `float()/255.0` (line 119); ④ `Normalize(IMAGENET_MEAN,
  IMAGENET_STD)` (line 120; constants `train_cnn.py:56-57`). **No crop.**
- **`train_cnn_bulk.py` (at scale):** source slide cached at **512²** (`:97`), tile
  cropped in-RAM (`:145`), `resize((96,96))` (`:146`), same aug (`:147-152`), `/255` +
  `Normalize` (`:153-154`).
- **`train_classifier.py` (RF baseline):** `.convert("RGB").resize((32,32))`
  (`:65`), `/255.0` (`:66`). **No normalization.**
Interpolation/resample is **never set explicitly** in any path (PIL default).

**A3 — Final input resolution.** **96×96×3** for both CNNs (`train_cnn.py:51`,
`train_cnn_bulk.py:45`); **32×32×3** for the RF baseline (`train_classifier.py:51`).

**A4 — Tiling / patching / multi-scale.** Tiling: **YES** — fixed 8×8 grid = 64
tiles/slide (`cut_images.py:12-13,57-75`), and on-the-fly in `train_cnn_bulk.py:120-130,
145`. Multi-scale: **NOT FOUND.** Random/centre crop beyond the fixed grid: **NOT FOUND.**
The model's field of view is one 1/64 tile; it never sees the whole slide.

**A5 — Did the expert label at full resolution?** **YES.** The server hands out the
**original JPEG** unmodified via `FileResponse(it["image_path"], media_type="image/jpeg")`
(`review_tool/webapp/server.py:205-210`). The frontend is **OpenSeadragon** deep-zoom
(`static/app.js:48-54`) opened as `viewer.open({type:"image", url:n.image_url})`
(`app.js:82`) with `maxZoomPixelRatio: 2.5` and scroll-to-zoom. So the judgement is on
the full 3000² slide, zoomable — a **whole-slide** label.

**A6 — Is fine detail present at 96px?** See `audit/A6_resolution_comparison.png` (one
usable: `Duodenum_A1CF_Female_67_3479.jpg`; one unusable: `Duodenum_ABCG8_Female_60_3548.jpg`).
- **Native 3000² slide / 375² tile:** crypt architecture, individual epithelial nuclei,
  membrane vs cytoplasmic stain localisation, and thin tissue folds are all clearly
  resolvable — this is what the expert saw.
- **96² model input:** gross tissue-vs-background and coarse stain intensity survive, but
  individual nuclei, thin folds, focal blur and oblique-sectioning cues are largely lost
  (per-tile 375→96 = 3.9× linear, **15× fewer pixels**; vs the whole slide, ~1000× fewer).
The features a human uses to call "unusable" (folds/blur/oblique) live largely in the
detail that the 96px input discards.

## B. Data split and leakage

**B1 — Where the split is generated.** `train_cnn.py:123-128` (`split_60_20_20`);
`train_cnn_bulk.py:182-185`; `train_classifier.py:95-96`; `analyze_model.py:91-92`. All
use `sklearn.model_selection.GroupShuffleSplit`.

**B2 — Grouped/stratified by (in code).** **Grouped by `source_image`** (the slide
filename) — `groups`/`grp`/`gl` passed as the `groups` arg (`train_cnn.py:152,125-127`;
`train_cnn_bulk.py:178,182`; `train_classifier.py:84,96`). **No stratification** (`stratify=`
/ `StratifiedGroupKFold` NOT FOUND). NB: `dissertation/chapters/03_methods.tex:6` claims a
"gene-stratified split" — the code does not stratify.

**B3 — Donor/patient identifier.** `patientId`, the **last filename token**
(`{tissue}_{gene}_{sex}_{age}_{patientId}.jpg`), written by the downloader
(`download_small_intestine.py:105-106,118`) and surfaced as the **`patient_id`** column in
`image_metadata.csv` (`image_metadata.py:69,206`). It is **never used by any split.**

**B4 — Donors spanning splits (computed).** The split runs on the tile manifest
(100-slide set), grouped by slide. Reproducing the exact split code (seed 42):
- `train_classifier.py` 80/20 → **8 donors in both train and test**: 1904, 1920, 1961,
  2329, 2562, 3219, 3479, 3614.
- `train_cnn.py` 60/20/20 → **9 donors in >1 split**: 1647, 1904, 1920, 1961, 2329, 2562,
  3219, 3479, 3614.
CSV: `audit/donors_spanning_splits_100set.csv`. For the **500 human-labelled set there is
no split in code at all** (nothing trains on it), so donor-spanning there is N/A — but see
B5.

**B5 — Donor multiplicity.** **YES, heavily.** The 500 labelled images are from only **22
donors**; images-per-donor histogram (imgs:donors) =
`{1:3, 2:5, 3:2, 6:1, 9:1, 11:1, 24:1, 27:1, 31:1, 32:1, 50:1, 56:1, 67:1, 76:1, 92:1}`;
top donors: 1904→92, 1920→76, 2562→67, 3479→56, 3219→50 images.
CSV: `audit/donor_distribution_labelled500.csv`. (The full 3,100 pool is similarly narrow:
a random seed-42 sample of 500 hit only 22 donors — `build_review_set.py:63-64`.)

**B6 — Seed fixed and recorded.** **YES.** `SEED = 42` (`train_cnn.py:55`,
`train_cnn_bulk.py:49`, `train_classifier.py:55`, `analyze_model.py:41`), passed as
`random_state` to every split and to `set_seed` (`train_cnn.py:61-62,149`). Recorded in
output as the `"split": "60/20/20 grouped by slide, seed 42"` string in the metrics JSON
(`train_cnn.py:206`).

## C. Dataset and class balance

**C1 — Downloaded vs labelled.** Downloaded = **3,100** (100 in `HPA_small_intestine/` +
3,000 in `data_bulk/images/`). Labelled = **500** (`review_tool/webapp/review_images/`; the
export `review_tool/results/randy_original500_20260727.csv` has 500 rows, all reviewer=Randy).

**C2 — Class counts (labelled).** **usable = 428, unusable = 72** (computed from the export
CSV). Only these two label values are present.

**C3 — Per-split class counts.** **NOT FOUND / not applicable** — there is no train/val/test
split over the 500 human labels anywhere in the code. The only splits (§B1) run over
heuristic *tile* labels, a different target.

**C4 — Not-usable in the test split.** **NOT FOUND** — no test split over the human labels
exists, so there is no test-set unusable count for the expert-labelled task. (For the
heuristic tile task the test-set counts live in `models/*metrics.json`, but those are
`tissue_fraction` labels, not human quality labels.)

**C5 — Non-binary labels in the store.** Only **usable/unusable** appear as judgements
(export CSV). The schema also supports **`unset`** (undo → removes the current judgment;
`review_store.py:110-111,159-161`); **skip** is *not* recorded (`app.js:108` just advances);
re-judgements are tracked as `n_changes`/flips on read (`review_store.py:163-176`) but not
exported. No "uncertain" class exists. Local test sessions confirm the mechanism
(`session_..._Tin_...csv`: 3 `label` + 3 `unset`).

## D. Cancer donor contamination

**D1 — XML source.** File (not a live endpoint): **`C:\Users\thinu\Downloads\normal_expression_Small.xml`**
(`download_small_intestine.py:10`; `download_bulk.py:21`), a **3.1 GB** local HPA
"normal_expression" export. Parsed with `xml.etree.ElementTree.iterparse`
(`download_small_intestine.py:78-151`; `download_bulk.py:41-79`). It is **not in the repo**
(present only in Downloads on this machine).

**D2 — SNOMED fields read.** Only `<snomed>` elements' **`tissueDescription`** attribute
(`download_small_intestine.py:107-111`; `download_bulk.py:60-63`) — used for the
small-intestine filter and to form the `tissue` token of the filename (so tissueDescription
is retained via the filename). The **`snomedCode` attribute (incl. morphology M-codes) is
never read → discarded.** `image_metadata.py:20-30` deliberately adds no normal/cancer flag.

**D3 — Morphology audit (computed over all 3,100).** SNOMED morphology is encoded as
`snomedCode` (e.g. `M-00100` Normal, `M-81403` Adenocarcinoma, `M-82403` Carcinoid).
Streaming the XML and matching to on-disk filenames: **209 downloaded images carry a
neoplasm morphology (M-8xxxx / M-9xxxx), across 3 donors: 1879, 1904, 3548.** Full list:
`audit/cancer_donor_images.csv`. Caveat: **80** downloaded images had no morphology match in
this XML export (undetermined — e.g. the `Small_intestine,_NOS_..._3004` set and genes
absent from this export).

**D4 — Labelled images from those donors.** **116 / 500** labelled images come from the 3
neoplasm-carrying donors; of these, **36** are images whose *own* sample is tagged
adenocarcinoma (M-81403) — all from donors 3548 and 1904.

**D5 — Expert labels on the overlap (the important number).**
- The **36 own-sample adenocarcinoma cores: 24 unusable / 12 usable.** → 12 adenocarcinoma
  duodenum cores sit in the "usable" set of a *normal-tissue* quality dataset.
- The **116 donor-level images: 85 usable / 31 unusable.**
Caveat: these samples are tagged with **both** `M-00100` (Normal) and `M-81403`
(Adenocarcinoma) in the XML, so I cannot assert each individual core is tumour vs
tumour-adjacent normal — but the adenocarcinoma morphology is present on the sample, which
is the contamination signal, and it matches the earlier `Duodenum_AHNAK_Female_60_3548`
finding (that image is in the list, labelled unusable).

**D6 — Morphology filtering implemented?** **NOT FOUND.** Selection is by
`tissueDescription` small-intestine matching only (`download_small_intestine.py:107-111`;
`download_bulk.py:60-63`). Nothing excludes neoplasm morphology.

## E. Baseline

**E1 — Non-deep-learning baseline.** **YES**, several: (a) the `tissue_fraction` heuristic
itself — a luminance+saturation tissue-coverage measure (`filter_blank_tiles.py:47-59`),
which is both the weak label *and* the reference baseline; (b) `RandomForestClassifier` on
32×32 pixels (`train_classifier.py:100-102`); (c) a feature ablation RF over
tissue_fraction / gene-metadata / pixels / combined (`analyze_model.py:95-120`); (d) a
majority-class baseline reported by the CNNs (`train_cnn.py:191-192`). Dedicated **focus /
blur / edge-density** measures: **NOT FOUND.**

**E2 — Sam Legg heuristic.** **NOT FOUND** in code (grep: no matches for `legg`/`Sam Legg`
in any `*.py`). It appears only in prose stubs (`dissertation/chapters/03_methods.tex`,
`04_results.tex`). No reimplementation exists.

**E3 — Baseline numbers recorded.** **YES**, in files: `models/metrics.json` (RF baseline),
`models/feature_ablation.json`, `models/cnn_usable_metrics.json`,
`models/cnn_bulk_usable_metrics.json`, `models/cnn_expression_metrics.json`,
`models/README.md`, `models/predictions.csv`; plus `research/findings/*.md` and
`research/LOG.md`. **W&B: NOT FOUND** (no `wandb` import or directory).

## F. Annotation store and inter-rater feasibility

**F1 — Schema.** `review_store.py:28-37`, `SESSION_FIELDS` =
`[event_ts, reviewer, param, image_id, image_path, action, judgment, key]` — one append-only
row per keypress; `action ∈ {label, unset}`.

**F2 — Records annotator / timestamp / image id?** **YES** — `reviewer`, `event_ts`
(microsecond ISO-8601), `image_id` (+ `image_path`) written per judgement
(`review_store.py:90-100`).

**F3 — Second pass without overwriting the first?** **YES.** Each session is a **new file**
with a microsecond timestamp (`review_store.py:61-68` `session_path`), opened append-only
(`SessionLog.__init__ :84`, mode `"a"`); raw rows are never rewritten. Resolution is
**newest-wins on read** (`resolve_judgments :150,164-166`), so a second *same-param* pass
changes the *resolved* label but preserves all raw rows. To keep a re-label as a distinct
dataset, a **separate `param`** is used — the intra-rater pass used `slice_quality_recode`
(`server.py:68-71,189`), so pass 1 resolves untouched.

**F4 — Blind shuffled re-label subset?** **YES (blind + fixed subset).**
`profiles/slice_quality_recode.json` has `"blind": true`; `webapp/recode_set.txt` is the
fixed subset; `build_recode_set.py` picks it (seeded random); `server.py:68-71,189`
restricts `/api/next` to that subset. Note the queue orders by coverage then stable manifest
order (`server.py:199`) — the subset is a **seeded random sample**, but there is **no
per-session reshuffle** (per-reviewer shuffling NOT FOUND).

**F5 — Images labelled more than once (count).** **50** — the intra-rater set: pass-1
`slice_quality` ∩ pass-2 `slice_quality_recode` = 50 shared images
(`review_tool/results/intra_rater_agreement_20260727.json`, `n_shared: 50`). The exported
500 are otherwise one judgement each.

## G. Evaluation protocol

**G1 — Single split or CV?** **Single fixed split** (`GroupShuffleSplit(n_splits=1, …)`):
`train_cnn.py:125-127`; `train_cnn_bulk.py:182-183`; `train_classifier.py:95-96`.
Cross-validation (`KFold`/`StratifiedKFold`/`cross_val`): **NOT FOUND.**

**G2 — Metrics present.** CNN (`train_cnn.py:190-198`, `train_cnn_bulk.py:214-221`):
accuracy, macro-F1, majority baseline, confusion matrix. **AUC-ROC: NOT in the CNN.**
**Per-class precision/recall: NOT in the CNN.** RF baseline (`train_classifier.py:108-116`):
accuracy, precision/recall/F1 for the **positive class only** (`average="binary"`), ROC-AUC,
confusion. `review_store.per_class_prf` (`:279-294`) exists but is used only by
`compare_to_classifier.py`, not by any trainer. → **macro-F1** yes (CNN); **AUC-ROC** only
in the RF baseline; a full **per-class precision/recall table: NOT FOUND.**

**G3 — CIs / bootstrap / repeated runs.** **NOT FOUND.**

**G4 — Class weighting in the loss.** **YES.** CNN: inverse-frequency weights →
`CrossEntropyLoss(weight=w)` (`train_cnn.py:173-176`; `train_cnn_bulk.py:197-200`). RF:
`class_weight="balanced"` (`train_classifier.py:101`; `analyze_model.py:103`).

**G5 — Decision threshold.** Fixed at **argmax (≡0.5)** — `out.argmax(1)`
(`train_cnn.py:142`; `train_cnn_bulk.py:167`); RF predicts at 0.5 (`train_classifier.py:124`).
Threshold tuning: **NOT FOUND.**

**G6 — Backbones actually in code.** **ResNet-18 (ImageNet-pretrained) only** —
`models.resnet18(weights=…IMAGENET1K_V1)` (`train_cnn.py:169`; `train_cnn_bulk.py:194`);
plus classical RandomForest (`train_classifier.py`, `analyze_model.py`). **ResNet-50 / timm /
CTransPath: NOT FOUND.**

## H. Reproducibility

**H1 — Seeds for torch / numpy / dataloader.** `np.random.seed`, `torch.manual_seed`,
`torch.cuda.manual_seed_all` (`train_cnn.py:61-62` via `set_seed(SEED)` at `:149`;
`train_cnn_bulk.py:173`). Augmentation randomness uses the seeded global `np.random`;
DataLoader `num_workers=0` (`train_cnn.py:166`; `train_cnn_bulk.py:191`) so no per-worker
seeding is needed. `torch.backends.cudnn.deterministic` / `use_deterministic_algorithms`:
**NOT FOUND** (CUDA nondeterminism not fully pinned). RF `random_state=42`.

**H2 — Per-run config persisted.** **Partial.** The metrics JSON records arch, the split
description (seed as text), tile/slide counts, classes and metrics (`train_cnn.py:203-213`).
Hyperparameters (LR, EPOCHS, BATCH, INPUT_SIZE) and the transform pipeline are **module
constants, not dumped per run**; there is no run-config file and no W&B.

**H3 — Single command to regenerate the numbers.** **Partial NO.** `python train_cnn.py
usable` reproduces the CNN numbers deterministically *given* `manifest_labels.csv` +
`image_metadata.csv` already built (which need `cut_images.py` then `filter_blank_tiles.py`
first). The headline human-label numbers (500 labels, κ = 0.735) come from the web
app/exported CSV, **not** from any script. No one command regenerates the thesis' numbers.

**H4 — Reconstructable from XML end-to-end?** **NO.** Both downloaders hardcode
`C:\Users\thinu\Downloads\normal_expression_Small.xml` (`download_small_intestine.py:10`;
`download_bulk.py:21`) — a 3.1 GB file that exists **only on this machine** and is not in
the repo; and `image_metadata.py` fetches gene metadata **live from proteinatlas.org**.
`data_bulk/images/` (3,000) and `review_images/` (500) are gitignored. A fresh clone cannot
rebuild the dataset without that external XML + network access.

---

## Things I found that you did not ask about, that look wrong

1. **`image_metadata.py:20-26` is factually contradicted by the data.** Its comment asserts
   the images "come from HPA's normal-tissue atlas and are normal by construction," but the
   XML shows 209 downloaded images (incl. 36 labelled) carry adenocarcinoma/carcinoid
   morphology. The stated assumption is wrong.
2. **The hard cases are excluded from every evaluation.** The `usable` weak label drops the
   ambiguous mid-coverage band (`0.05 ≤ tissue_fraction < 0.5`) from train *and* test
   (`train_cnn.py:73-74`; `train_classifier.py:69-74`). The reported accuracies are only on
   the easy extremes (near-empty vs clearly-tissue); folds/blur/artefact tiles are never
   scored.
3. **The at-scale model sees even less detail than the small one.** `train_cnn_bulk.py:97`
   downsizes each slide to 512² *before* tiling, so each 96px tile derives from a **64px**
   source region (512/8) upscaled to 96 — vs 375px in `train_cnn.py`. The "100% at scale"
   and "98.8% at 100 imgs" results are therefore not at the same input resolution.
4. **The labelled-set pipeline never records the donor.** `review_set_manifest.csv` stores
   `gene, source_dir` but not `patient_id`; the donor is only recoverable by re-parsing the
   filename — so a leakage-safe split of the human set has nothing to group on out of the box.
5. **`image_metadata.csv` covers only the 100-image sample** (`image_metadata.py` points at
   `HPA_small_intestine/`); there is no in-repo per-image metadata for the 3,000 bulk images
   beyond `data_bulk/bulk_metadata.csv` (expression only).
6. **Dead code:** `train_cnn_bulk.py:189` builds `it = np.array(items, dtype=object)` which
   is never used.
7. **Default secrets in `server.py:38-40`** (`PASSCODE="review123"`, `ADMIN_TOKEN="admin123"`,
   `SECRET="dev-secret-change-me"`) — fine only if overridden by env in prod; a deployment
   note, not a research issue.

## Computed artefacts (in `audit/`)
- `cancer_donor_images.csv` — the 209 neoplasm-morphology downloaded images (+ labelled flag & expert label).
- `donor_distribution_labelled500.csv` — images-per-donor over the 500 labelled slides.
- `donors_spanning_splits_100set.csv` — donors leaking across the actual split code.
- `A6_resolution_comparison.png` — native slide/tile vs 96px model input, usable & unusable.
