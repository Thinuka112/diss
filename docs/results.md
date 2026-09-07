# Results map

This file replaces dissertation Appendix D. Every number in the dissertation traces to a shipped artefact. This is the map.

## The final Stage-1 grid

Five-fold donor-grouped cross-validation on the 847 expert labels. Shared fold partition. Seed 42. Macro-F1 reported argmax / tuned.

| Model | Argmax | Tuned | Artefact |
|---|---|---|---|
| ResNet-50, gold only (production model) | 0.8041 ± 0.0207 | 0.8199 ± 0.0506 | `models/quality_union_humanonly/quality_metrics.json` |
| ResNet-50, combined 640px | 0.7775 ± 0.0448 | 0.7878 ± 0.0599 | `models/quality_resnet_cvF640/` |
| YOLOv8s-cls, gold only | 0.7758 ± 0.0615 | 0.8100 ± 0.0669 | `models/quality_yolo_cv847/cv_metrics.json` |
| YOLOv8s-cls, combined | 0.8152 ± 0.0377 | 0.8159 ± 0.0570 | `models/quality_yolo_cvF/cv_metrics.json` |

The production model also reports AUC 0.9159 and unusable recall 0.8007 at argmax. Its per-image predictions ship as `fold*_preds.csv` in the same folder. `python demo_stage1.py` recomputes the headline from them and checks every fold against the stored metrics.

## Combined-training treatments (ResNet, machine labels added)

| Treatment | Argmax | Tuned | Artefact |
|---|---|---|---|
| Naive pooling | 0.6985 | 0.7503 | `models/quality_combined_hard/` |
| Noise-aware recipe | 0.7583 | 0.7626 | `models/quality_combined_soft2stage/` |
| Confident subset | 0.7514 | 0.7609 | `models/quality_combined_confident/` |
| Trust-filtered 448px reproduction | 0.7461 | - | `models/quality_resnet_cvF/` |

No combined treatment beat gold-only for the ResNet. The same labels moved YOLO up by about 0.04. That asymmetry is the chapter's central finding.

## Earlier milestones and baselines

| Result | Value | Artefact |
|---|---|---|
| First 597-label baseline, 448px | 0.676 | `models/quality/quality_metrics_597baseline.json` |
| Enriched 847 labels, 448px | 0.727 | `models/quality/quality_metrics.json` |
| 640px on garlick | 0.762 | `models/quality_garlick640/` |
| Frozen PLIP head (rejected backbone) | 0.7010 | `models/quality_plip/`, `demo_plip_baseline.py` |
| Usable predictor baseline | 0.4462 | `baselines` key in any `quality_metrics.json` |
| Tissue-fraction heuristic baseline | 0.5656 | same |

## Statistical support

| Test | Result | Script |
|---|---|---|
| Paired t-test on the finalists' fold deltas | p = 0.386 argmax, p = 0.818 tuned | `tie_test.py` |
| Exact McNemar on the 847 paired predictions | p = 0.271 argmax, p = 0.461 tuned | `mcnemar_test.py` |
| Gold/silver label overlap audit | 759 overlap images. Usable strata 83.3-97.8% pure. Unusable strata 33.8-72.1% | `overlap_audit.py` |

All three run from the repo root with the shipped files. No GPU.

## Labelling reliability

The expert re-graded 50 slides blind. Agreement 47/50 = 94%. Cohen's kappa 0.735. Notebook `02_labelling.ipynb` recomputes both from the raw session CSVs in `review_tool/results/`.

## Stage 2 localisation

Scored against the expert's 1,280-tile gold standard over 20 slides. Pointing accuracy is the share of slides whose top-ranked tile is an expert yes-tile. Random baseline 0.327.

| Training data | Pointing | Top-3 | Top-5 | Artefact |
|---|---|---|---|---|
| Gold labels only | 0.80 | 0.95 | 0.95 | `models/mil/mil_gold_eval.json` |
| Combined data | 0.90 | 1.00 | 1.00 | `models/mil_combined/mil_gold_eval.json` |

`RUN_3_STAGE2_EVAL.bat` recomputes this evaluation from the shipped tile scores.

## The experiment record

Notebooks 01 to 06 are the executed record in order. Dataset, labelling, Stage-1 classifier, Stage-2 localisation, then the two final cross-validation runs. Notebook 05 self-verifies 12 of 12 grid values and notebook 06 verifies 8 of 8.
