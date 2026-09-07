# models/ — the evidence chain

One folder per model run. What ships is the evidence, not the weights. Metrics JSONs, per-image predictions and confusion matrices are small and carry every number the dissertation reports. The CNN checkpoints are around 100 MB per fold and are excluded. `docs/results.md` maps each dissertation number to its folder here.

## Stage 1 — the final grid

| Folder | Model |
|---|---|
| `quality_union_humanonly/` | Gold-only ResNet-50. The production model. Metrics plus per-image fold predictions |
| `quality_resnet_cvF640/` | ResNet-50 on combined data at 640px |
| `quality_resnet_cvF/` | Same at 448px. The cross-machine reproduction |
| `quality_yolo_cv847/` | YOLOv8s-cls on gold labels |
| `quality_yolo_cvF/` | YOLOv8s-cls on combined data |

## Stage 1 — combined-training treatments and milestones

| Folder | Run |
|---|---|
| `quality_combined_hard/` | Naive pooling of machine labels |
| `quality_combined_soft2stage/` | Noise-aware recipe |
| `quality_combined_confident/` | Confident machine subset |
| `quality/` | 448px gold runs. Holds the 597-label baseline JSON too |
| `quality_garlick640/` | First 640px run on garlick |
| `quality_plip/` | Frozen PLIP head. The rejected backbone. Includes the embedding cache the demo retrains from |

## Stage 2 — localisation

| Folder | Run |
|---|---|
| `mil/` | MIL trained on gold labels. Tile scores, slide predictions, gold-standard evaluation |
| `mil_combined/` | Same trained on combined data |

## Machine labels

`vlm_labels/ensemble_cascade_labels.csv` holds the seven-judge panel's verdict and vote for all 5,900 machine-labelled images. `overlap_audit.py` at the repo root audits it against the expert labels.

## Root-level files

The loose JSONs and PNGs here are the early pipeline-validation artefacts. Weak-label baselines and the expression-axis negative result. `docs/validation.md` explains each one.
