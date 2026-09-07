"""Offline demo: reproduce the rejected PLIP baseline from the shipped artefacts.

PLIP was one of the candidate backbones. Chapter 5 rejects it. Frozen PLIP plus a
logistic-regression head scored macro-F1 0.7010 against the fine-tuned ResNet-50's
0.8041. This demo retrains that head from the shipped embedding cache and the
expert label CSV under the dissertation protocol. Donor-grouped 5-fold CV. No raw
images and no GPU needed. It prints the per-fold and mean macro-F1 next to the
stored values from models/quality_plip/quality_plip_metrics.json.

Run:  python demo_plip_baseline.py     (needs numpy + scikit-learn only)

The production model's demo is demo_stage1.py. To score NEW images install torch
plus transformers and use train_quality_plip.py with the images present.
"""
import csv
import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "review_tool", "results", "final_clean_dataset_20260802.csv")
EMB_CACHE = os.path.join(SCRIPT_DIR, "models", "quality_plip", "plip_embeddings.npz")
METRICS = os.path.join(SCRIPT_DIR, "models", "quality_plip", "quality_plip_metrics.json")
CLASSES = ["unusable", "usable"]
SEED = 42


def main():
    np.random.seed(SEED)
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["label"] in CLASSES]
    cache = np.load(EMB_CACHE, allow_pickle=True)
    emb_of = {i: e for i, e in zip(cache["ids"], cache["emb"])}
    missing = [r["image_id"] for r in rows if r["image_id"] not in emb_of]
    if missing:
        raise SystemExit(f"{len(missing)} CSV ids missing from the embedding cache "
                         f"(first: {missing[:3]}) - cache and CSV out of sync")
    ids = [r["image_id"] for r in rows]
    y = np.array([CLASSES.index(r["label"]) for r in rows])
    groups = np.array([r["patient_id"] for r in rows])
    emb = np.stack([emb_of[i] for i in ids])
    print(f"slides={len(ids)}  donors={len(set(groups.tolist()))}  "
          f"labels: unusable={int((y == 0).sum())} usable={int((y == 1).sum())}")

    per_fold = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(emb, y, groups)):
        assert not (set(groups[tr]) & set(groups[te])), f"donor leak in fold {k}"
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))
        clf.fit(emb[tr], y[tr])
        f1 = f1_score(y[te], clf.predict(emb[te]), average="macro")
        per_fold.append(f1)
        print(f"  FOLD {k + 1}: macro-F1 {f1:.4f}")

    mean, std = float(np.mean(per_fold)), float(np.std(per_fold))
    print(f"\nreproduced : macro-F1 {mean:.4f} +/- {std:.4f}")
    with open(METRICS, encoding="utf-8") as fh:
        stored = json.load(fh)["plip_aggregate"]
    print(f"stored     : macro-F1 {stored['macro_f1_mean']:.4f} +/- {stored['macro_f1_std']:.4f} "
          f"(models/quality_plip/quality_plip_metrics.json)")
    ok = abs(mean - stored["macro_f1_mean"]) < 5e-4
    print("MATCH - the shipped artefacts reproduce the rejected PLIP baseline."
          if ok else "MISMATCH - see README reproducibility notes.")


if __name__ == "__main__":
    main()
