"""
train_classifier.py - baseline slice-quality (usable vs unusable) tile classifier.

HONEST FRAMING - read this before trusting the model:
  There are no human quality labels yet. The only signal available is tile tissue
  coverage (from filter_blank_tiles.py), so this trains a WEAK-LABEL baseline:

      unusable (0) : tissue_fraction <  UNUSABLE_MAX  (near-empty background)
      usable   (1) : tissue_fraction >= USABLE_MIN     (clearly tissue)
      (the ambiguous middle band is held out of train/eval, but still scored at
       inference so reviewers see the model's call on the uncertain tiles)

  So this model learns "does this tile contain tissue?" from raw pixels - a
  learned replacement for the fixed tissue_fraction threshold that generalises to
  new tiles. It is NOT yet a true artefact/fold/blur quality model; that needs
  hand-labelled tiles. Swap the weak labels for a `label` column filled by a human
  and rerun to get the real thing.

METHOD (leakage-safe):
  - Features: each tile downsampled to 32x32 RGB, flattened, scaled to [0,1].
  - Split: GroupShuffleSplit BY SOURCE SLIDE (tiles from one slide never straddle
    train and test), 80/20.
  - Model: RandomForest (class_weight balanced), fixed seed.
  - Reports held-out test accuracy/precision/recall/F1/AUC + confusion matrix.

OUTPUTS (models/):
  slice_quality_model.joblib, metrics.json, predictions.csv (all 6400 tiles),
  README.md (model card).
"""

import os
import csv
import json

import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, roc_auc_score)
import joblib

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TILES_DIR = os.path.join(SCRIPT_DIR, "HPA_small_intestine_tiles")
MANIFEST = os.path.join(TILES_DIR, "manifest_labels.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

IMG_SIZE = 32               # downsample tiles to IMG_SIZE x IMG_SIZE RGB
UNUSABLE_MAX = 0.05         # tissue_fraction below this -> weak label "unusable"
USABLE_MIN = 0.50           # tissue_fraction at/above this -> weak label "usable"
TEST_SIZE = 0.20            # fraction of SLIDES held out for test
RANDOM_STATE = 42
# ------------------------------------------------------------------

LABEL_NAMES = {0: "unusable", 1: "usable"}


def load_features(rel_path):
    """Load one tile (path relative to repo root) as a scaled 32x32 RGB vector."""
    full = rel_path if os.path.isabs(rel_path) else os.path.join(SCRIPT_DIR, rel_path)
    with Image.open(full) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        return np.asarray(im, dtype=np.float32).ravel() / 255.0


def weak_label(frac):
    if frac < UNUSABLE_MAX:
        return 0
    if frac >= USABLE_MIN:
        return 1
    return -1               # ambiguous - excluded from train/eval


def main():
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loading {len(rows)} tiles as {IMG_SIZE}x{IMG_SIZE} features...")

    X = np.stack([load_features(r["tile_path"]) for r in rows])
    fracs = np.array([float(r["tissue_fraction"]) for r in rows])
    groups = np.array([r["source_image"] for r in rows])
    wlabels = np.array([weak_label(f) for f in fracs])

    labeled = wlabels != -1
    Xl, yl, gl = X[labeled], wlabels[labeled], groups[labeled]
    print(f"Weak-labeled tiles: {labeled.sum()} "
          f"(usable={int((yl == 1).sum())}, unusable={int((yl == 0).sum())}); "
          f"ambiguous held out: {int((~labeled).sum())}")
    print(f"Slides: {len(set(groups))}")

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr, te = next(gss.split(Xl, yl, gl))
    print(f"Train tiles: {len(tr)} from {len(set(gl[tr]))} slides | "
          f"Test tiles: {len(te)} from {len(set(gl[te]))} slides")

    clf = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(Xl[tr], yl[tr])

    pred = clf.predict(Xl[te])
    prob = clf.predict_proba(Xl[te])[:, 1]
    acc = accuracy_score(yl[te], pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        yl[te], pred, average="binary", pos_label=1, zero_division=0)
    auc = roc_auc_score(yl[te], prob) if len(set(yl[te])) > 1 else float("nan")
    cm = confusion_matrix(yl[te], pred, labels=[0, 1])

    print("\n--- Held-out test (unseen slides) ---")
    print(f"accuracy : {acc:.3f}")
    print(f"precision: {prec:.3f}   recall: {rec:.3f}   f1: {f1:.3f}   auc: {auc:.3f}")
    print("confusion matrix (rows=true [unusable,usable], cols=pred):")
    print(f"            pred_unusable  pred_usable")
    print(f"  unusable  {cm[0,0]:>12}  {cm[0,1]:>11}")
    print(f"  usable    {cm[1,0]:>12}  {cm[1,1]:>11}")

    prob_all = clf.predict_proba(X)[:, 1]
    pred_all = (prob_all >= 0.5).astype(int)
    split = np.full(len(rows), "n/a", dtype=object)
    labeled_idx = np.where(labeled)[0]
    split[labeled_idx[tr]] = "train"
    split[labeled_idx[te]] = "test"

    os.makedirs(MODEL_DIR, exist_ok=True)
    pred_csv = os.path.join(MODEL_DIR, "predictions.csv")
    with open(pred_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tile_path", "source_image", "gene", "tissue_fraction",
                    "weak_label", "split", "pred_label", "prob_usable"])
        for i, r in enumerate(rows):
            wl = wlabels[i]
            w.writerow([r["tile_path"], r["source_image"], r["gene"],
                        r["tissue_fraction"],
                        LABEL_NAMES.get(wl, "ambiguous"), split[i],
                        LABEL_NAMES[pred_all[i]], round(float(prob_all[i]), 4)])

    model_path = os.path.join(MODEL_DIR, "slice_quality_model.joblib")
    joblib.dump({"model": clf, "img_size": IMG_SIZE,
                 "labels": LABEL_NAMES,
                 "weak_label_thresholds": {"unusable_max": UNUSABLE_MAX,
                                           "usable_min": USABLE_MIN}}, model_path)

    metrics = {
        "task": "slice-quality baseline: usable(has tissue) vs unusable(blank)",
        "label_source": "weak (tissue_fraction thresholds), not human-annotated",
        "n_tiles_total": len(rows),
        "n_weak_labeled": int(labeled.sum()),
        "n_usable": int((yl == 1).sum()),
        "n_unusable": int((yl == 0).sum()),
        "n_ambiguous_excluded": int((~labeled).sum()),
        "n_slides": len(set(groups)),
        "split": "GroupShuffleSplit by source slide, test_size=0.2",
        "img_size": IMG_SIZE,
        "model": "RandomForestClassifier(n_estimators=200, class_weight=balanced)",
        "test_accuracy": round(float(acc), 4),
        "test_precision_usable": round(float(prec), 4),
        "test_recall_usable": round(float(rec), 4),
        "test_f1_usable": round(float(f1), 4),
        "test_roc_auc": round(float(auc), 4),
        "confusion_matrix_labels": ["unusable", "usable"],
        "confusion_matrix": cm.tolist(),
        "predictions_all_tiles": {
            "usable": int((pred_all == 1).sum()),
            "unusable": int((pred_all == 0).sum()),
        },
    }
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    with open(os.path.join(MODEL_DIR, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(_model_card(metrics))

    print(f"\nSaved: {model_path}")
    print(f"       {pred_csv}")
    print(f"       {os.path.join(MODEL_DIR, 'metrics.json')}")
    print(f"       {os.path.join(MODEL_DIR, 'README.md')}")


def _model_card(m):
    return f"""# Slice-quality tile classifier - baseline (weak labels)

**Status: baseline / proof-of-pipeline.** Trained on weak labels derived from
tile tissue coverage, NOT human quality annotations. It learns "tile contains
tissue (usable) vs blank (unusable)" from raw pixels - a learned stand-in for the
fixed `tissue_fraction` threshold that generalises to unseen tiles.

## Data
- {m['n_tiles_total']} tiles from {m['n_slides']} slides (8x8 tiling of 100 HPA
  small-intestine images).
- Weak-labeled: {m['n_weak_labeled']} (usable {m['n_usable']}, unusable
  {m['n_unusable']}). Ambiguous middle band excluded from train/eval:
  {m['n_ambiguous_excluded']}.
- Labels from tissue_fraction: unusable < {UNUSABLE_MAX}, usable >= {USABLE_MIN}.

## Model
- {m['model']}, features = 32x32 RGB pixels scaled to [0,1].
- Split: {m['split']} (leakage-safe - no slide in both train and test).

## Held-out test metrics (unseen slides)
- accuracy {m['test_accuracy']}, precision {m['test_precision_usable']},
  recall {m['test_recall_usable']}, F1 {m['test_f1_usable']},
  ROC-AUC {m['test_roc_auc']}.
- confusion matrix [rows true unusable/usable, cols pred]: {m['confusion_matrix']}

## Files
- `slice_quality_model.joblib` - the fitted model + metadata.
- `predictions.csv` - prediction for every tile (with weak_label + split).
- `metrics.json` - the numbers above.

## To make this a real quality model
Fill the `label` column in `HPA_small_intestine_tiles/manifest_labels.csv` with
human usable/unusable calls (esp. on the ambiguous mid-coverage tiles: folds,
blur, artefacts), then retrain using those labels instead of the weak ones.
"""


if __name__ == "__main__":
    main()
