# Pipeline-validation experiments

This file replaces dissertation Appendix E. Three early experiments validated the pipeline before the main study. Two are deliberate sanity checks. One is a negative result. None of them is a headline claim.

## 1. The weak-label degenerate baseline

A RandomForest was trained on tile labels derived from tissue coverage. Unusable below 5% coverage. Usable above 50%. It scored perfect accuracy on held-out slides (`models/metrics.json`, test_accuracy 1.0). That result is degenerate by construction. The labels come from a pixel-coverage rule and the model sees the pixels. The experiment proves the tiling, splitting and evaluation plumbing works end to end. It says nothing about slice quality. It is why the project moved to expert labels. Artefacts: `models/metrics.json`. Script: `train_classifier.py` with labels from `filter_blank_tiles.py`.

## 2. The at-scale automated tissue filter

The same weak-label task rerun as a CNN at two scales. The tile CNN on the 100-slide set reached 0.9878 (`models/cnn_usable_metrics.json`). The bulk run over the full downloaded pool reached 1.0 (`models/cnn_bulk_usable_metrics.json`). Same caveat as above. These models learn the coverage rule, and they exist to confirm the training loop scales before the expensive expert-label experiments. Scripts: `train_cnn.py`, `train_cnn_bulk.py`.

## 3. The expression-axis negative result

An attempt to predict the atlas's gene-expression annotation from tile pixels. It failed. The tile CNN reached 0.4406 across the expression classes (`models/cnn_expression_metrics.json`) and the bulk run 0.5561 (`models/cnn_bulk_expression_metrics.json`). The feature study (`models/expression_metrics.json`, `models/feature_ablation.json`) points the same way. Expression level is a gene-level annotation. It is not reliably readable from a single tile. The axis was dropped and the project kept slice quality as the target. Scripts: `train_expression.py`, `train_cnn_bulk.py`.
