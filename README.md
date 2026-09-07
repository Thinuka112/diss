# Slice-quality classification for HPA small-intestine histology

Code and evidence for an MSc dissertation (University of Bath, 2026). The project builds a two-stage slice-quality pipeline for Human Protein Atlas (HPA) immunohistochemistry images of small intestine. Stage 1 grades whole tissue sections as usable or unusable. Stage 2 localises the readable tissue within a section at tile level using multiple-instance learning (MIL).

This repository is the dissertation's supporting artefact. It replaces the print appendices for configuration, code, results, validation experiments and the data audit. Start at `docs/`.

## Quick start

Install Python 3.11+ and run `pip install -r requirements.txt` once. Then double-click:

1. **`RUN_1_DEMO.bat`** reproduces the dissertation's Stage-1 headline. The gold-only ResNet-50 at macro-F1 0.8041 argmax and 0.8199 tuned. It recomputes both from the shipped per-image predictions and checks every fold against the stored metrics. Standard library only. Ends with a MATCH line.
2. **`RUN_2_ANNOTATION_TOOL.bat`** opens the expert annotation tool on 12 bundled sample slides. Six usable and six unusable. Sign in with any name. Arrow keys judge. Esc quits. Judgments append to local session logs exactly as in the study.
3. **`RUN_3_STAGE2_EVAL.bat`** recomputes the Stage-2 localisation evaluation against the expert's tile gold standard from the shipped scores.

The executed experiment record is in `notebooks/` (01-06, readable without running anything). The statistical tests run from the root with `python tie_test.py`, `python mcnemar_test.py` and `python overlap_audit.py`. No GPU for any of this.

## The pipeline

The full pipeline is reconstructable from the raw HPA XML export:

```
download -> tile (8x8) -> auto-seed labels -> VLM labelling cascade -> expert review -> train -> evaluate
```

Every stage reads one CSV and writes another. They join on the image filename (`{tissue}_{gene}_{sex}_{age}_{patientId}.jpg`). The pipeline never deletes or moves an image. Unusable is a label. Not a deletion.

## Repository layout

| Path | What it is |
|---|---|
| `docs/` | The appendix-replacement documents. Configuration and seeds. The results map. Validation experiments. The data audit |
| `download_*.py`, `fetch_bulk_meta.py`, `cut_images.py`, `image_metadata.py` | Data acquisition from HPA plus tiling and the metadata join |
| `filter_blank_tiles.py`, `tile_label.py`, `label_bulk*.py`, `label_ensemble.py`, `few_shot_label.py` | Auto-seeding and the VLM labelling cascade |
| `tune_prompt.py`, `prompt_loop.py`, `score_sweep.py`, `compare_label_sets.py`, `ensemble_arms_eval.py`, `emit_threshold_variant.py`, `mine_unusable_candidates.py`, `make_*_gallery.py` | Labelling experiments and audits |
| `overlap_audit.py`, `tie_test.py`, `mcnemar_test.py` | The audits and statistical tests behind Chapter 5. Run from the root on shipped files |
| `probe/` | Standalone VLM probes. Can a vision-language model apply the expert's rubric? |
| `review_tool/` | The annotation tool the domain expert used. Desktop and deployable web app. Own README and DEPLOY.md. Image payloads are not shipped. Rebuild them with `build_review_set.py` / `mil_expert_set.py` |
| `train_*.py`, `analyze_model.py`, `assemble_*.py`, `yolo_cvF.py`, `resnet_cvF.py` | Stage-1 training, evaluation and dataset assembly. The cvF scripts are the final cross-validation engines |
| `mil_*.py`, `make_mil_heatmaps.py` | Stage-2 MIL localisation. Features, model, training, gold-standard evaluation, heatmaps |
| `notebooks/` | The experiment record in order. 01 dataset. 02 labelling. 03 Stage-1 classifier. 04 Stage-2 localisation. 05/06 final cross-validation |
| `models/` | The evidence chain. Metrics, predictions and caches per run. See `models/README.md` |
| `audit/`, `AUDIT.md` | The data audit. Cancer-donor exclusions, donor concentration, split leakage |
| `hex/` | Helpers for the university GPU node (garlick, Bath Hex cloud) |
| `Slice_Usability_Rubric_v1_0.docx` | The labelling rubric (v1.0, frozen 2026-08-12). Applied verbatim by the expert and the VLM panel |
| `sample_images/` | 12 balanced sample slides so the tool and demos work without downloading the dataset |
| `RUN_*.bat` | One-click launchers for the demo, the annotation tool and the Stage-2 evaluation |

## Installation

```
python -m venv .venv
.venv\Scripts\activate          # Windows;  source .venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Python 3.11+ is expected. `train_quality_cnn.py` requires a CUDA build of PyTorch and refuses to train on CPU. A full run is roughly 4-6 h on an RTX 2060. The dissertation's runs used garlick, a node of the University of Bath's Hex GPU cloud with 8 GB RTX 2080 cards.

## Data

Raw images are not shipped. To rebuild the dataset follow `docs/configuration.md`. In short. Download the HPA small-intestine XML export, set `XML_FILE` in each downloader's config block, then run download, `cut_images.py` and `image_metadata.py`.

The label CSVs behind every reported result ARE shipped in `review_tool/results/`. The expert-labelled datasets (`final_clean_dataset_*.csv`, `expert_*.csv`). The combined human+machine training set (`combined_human_confident_20260827.csv`). The Stage-2 tile gold standard (`loc_stretch_export_20260827.csv`). The machine-label panel output is `models/vlm_labels/ensemble_cascade_labels.csv`. The `reviewer` column is pseudonymised. `review_tool/results/README.txt` explains each file.

## Models and results

`models/` holds one folder per run. What ships is the evidence chain. Metrics JSONs, per-image predictions and confusion matrices. The CNN checkpoints (~100 MB per fold) are excluded for size. Every dissertation number maps to its artefact in `docs/results.md`, and the production model's per-image predictions ship in full, so the headline recomputes exactly (`python demo_stage1.py`).

The rejected PLIP baseline keeps its own demo. `python demo_plip_baseline.py` retrains its head from the shipped embedding cache and reproduces macro-F1 0.7010.

## Reproducibility notes

- Seeds are fixed (42) throughout. `train_quality_cnn.py` enables `cudnn.benchmark`, so CNN runs are seed-stable but not bit-identical across GPUs.
- Several scripts carry hard assertions on exact row counts (847 human labels, 1,280 gold tiles). These are reconciliation guards. They pin the shipped CSVs to the dissertation's numbers and abort loudly if the inputs drift.
- The labelling cascade scripts call paid VLM APIs and need keys via environment variables (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, or Google ADC for Vertex). Nothing in the training or evaluation path needs an API key.

## Licence and attribution

The HPA images this project analyses are from the Human Protein Atlas (Uhlén et al. 2015, proteinatlas.org) and are licensed CC BY-SA 3.0. The bundled `sample_images/` are HPA images and carry that licence and attribution. The code is © Thinuka Madapatha 2026, submitted for the degree of MSc at the University of Bath. See `LICENSE.md`.
