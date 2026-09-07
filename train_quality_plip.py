"""
train_quality_plip.py - Stage-1 slice quality with a FROZEN histology
vision-language backbone (PLIP), for an honest head-to-head vs the ResNet-50/ImageNet
model (train_quality_cnn.py).

Scope: research/scoping/03a-vlm-backbone.md. Exploratory - PLIP is NOT in the lit
review and NOT validated for quality control; this is the fair comparison, baselines
kept. A win is an original contribution; a null is still a valid finding.

    python train_quality_plip.py            # extract frozen PLIP features + 5-fold CV
    python train_quality_plip.py smoke      # 1 fold, small subset (sanity only)

Design (mirrors train_quality_cnn.py so the numbers are directly comparable):
  - Backbone: PLIP image encoder (CLIP ViT-B), FROZEN. Whole slide -> embedding.
    PLIP uses its own 224 px preprocessing (its processor) - a caveat vs the CNN's
    448 px, but it is the standard way to use the model.
  - Head: StandardScaler -> LogisticRegression(class_weight="balanced") on embeddings.
  - Split: donor-grouped GroupKFold(5) by patient_id - the SAME folds as the CNN run
    (GroupKFold is deterministic on the same row order), so test donors match per fold.
  - Metrics: macro-F1, AUC-ROC, per-class precision/recall, confusion; mean +/- std.
  - Same 847-label dataset. Embeddings are cached (extract once, reuse).

Weights: PLIP is fetched from Hugging Face ('vinid/plip') the first time it loads.
If the node has no internet, download once with hex/fetch_plip_weights.py, scp the
folder up, and set PLIP_MODEL to that path.
"""

import os
import sys
import csv
import json
import collections

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_auc_score, precision_recall_fscore_support)

# ------------------------------------------------------------------
# CONFIG - edit these only
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Same default dataset as train_quality_cnn.py so the comparison is on identical data.
CSV_PATH   = os.environ.get("QUALITY_CSV", os.path.join(SCRIPT_DIR, "review_tool", "results",
                          "final_clean_dataset_20260802.csv"))
IMG_DIR    = os.path.join(SCRIPT_DIR, "review_tool", "webapp", "review_images")
OUT_DIR    = os.path.join(SCRIPT_DIR, "models", "quality_plip")
EMB_CACHE  = os.path.join(OUT_DIR, "plip_embeddings.npz")

# PLIP weights: a HF id (auto-download) OR a local folder (offline; scp'd up).
PLIP_MODEL = os.environ.get("PLIP_MODEL", "vinid/plip")

N_FOLDS    = 5
SEED       = 42
EXTRACT_BATCH = 32                     # feature-extraction batch (frozen, no grad)
CLASSES = ["unusable", "usable"]       # index 1 = usable = positive class for AUC
# ------------------------------------------------------------------


def load_rows():
    """Read the label CSV -> (image_ids, y, groups). Load by exact image_id;
    filenames can contain commas, so never split the CSV on comma."""
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    idx = {c: i for i, c in enumerate(CLASSES)}
    ids, ys, groups, missing = [], [], [], []
    for r in rows:
        img_id = r["image_id"]
        if not os.path.isfile(os.path.join(IMG_DIR, img_id)):
            missing.append(img_id)
            continue
        ids.append(img_id)
        ys.append(idx[r["label"]])
        groups.append(r["patient_id"])
    if missing:
        print(f"WARNING: {len(missing)} image_ids in CSV not found on disk "
              f"(first few: {missing[:5]})")
    return ids, np.array(ys), np.array(groups)


def load_plip(device):
    """Load the frozen PLIP image encoder + its processor (CLIP architecture)."""
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError as e:
        raise SystemExit("transformers not installed - `pip install transformers` "
                          "(it is in hex/requirements.txt).") from e
    print(f"loading PLIP from {PLIP_MODEL} ...")
    model = CLIPModel.from_pretrained(PLIP_MODEL).to(device).eval()
    proc = CLIPProcessor.from_pretrained(PLIP_MODEL)
    return model, proc


@torch.no_grad()
def extract_features(ids, device):
    """Frozen PLIP image embedding per slide (L2-normalised). Cached to EMB_CACHE so
    the GPU pass runs once; re-runs reuse it."""
    if os.path.isfile(EMB_CACHE):
        d = np.load(EMB_CACHE, allow_pickle=True)
        if list(d["ids"]) == list(ids):
            print(f"using cached embeddings: {EMB_CACHE}")
            return d["emb"]
        print("cache ids differ from current dataset - re-extracting")
    model, proc = load_plip(device)
    embs = []
    for s in range(0, len(ids), EXTRACT_BATCH):
        batch_ids = ids[s:s + EXTRACT_BATCH]
        imgs = [Image.open(os.path.join(IMG_DIR, i)).convert("RGB") for i in batch_ids]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        # pooled [CLS] vision features (transformers-version robust; get_image_features
        # changed return type in transformers 5.x).
        feat = model.vision_model(pixel_values=px).pooler_output.float()
        feat = torch.nn.functional.normalize(feat, dim=1)      # L2-normalise (CLIP convention)
        embs.append(feat.cpu().numpy())
        if (s // EXTRACT_BATCH) % 5 == 0:
            print(f"  embedded {min(s + EXTRACT_BATCH, len(ids))}/{len(ids)}")
    emb = np.concatenate(embs, axis=0)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(EMB_CACHE, ids=np.array(ids, dtype=object), emb=emb)
    print(f"saved embeddings {emb.shape} -> {EMB_CACHE}")
    return emb


def fold_metrics(gts, preds, probs):
    """Per-fold metrics dict - same schema as train_quality_cnn.py."""
    macro_f1 = float(f1_score(gts, preds, average="macro"))
    acc = float(accuracy_score(gts, preds))
    p, r, f, _ = precision_recall_fscore_support(gts, preds, labels=[0, 1], zero_division=0)
    auc = float(roc_auc_score(gts, probs)) if len(set(gts.tolist())) == 2 else float("nan")
    cm = confusion_matrix(gts, preds, labels=[0, 1])
    return {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "auc": None if np.isnan(auc) else round(auc, 4),
        "precision": {"unusable": round(float(p[0]), 4), "usable": round(float(p[1]), 4)},
        "recall":    {"unusable": round(float(r[0]), 4), "usable": round(float(r[1]), 4)},
        "f1":        {"unusable": round(float(f[0]), 4), "usable": round(float(f[1]), 4)},
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(gts)),
        "test_class_counts": {CLASSES[k]: int(v)
                              for k, v in collections.Counter(gts.tolist()).items()},
    }


def mean_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not xs:
        return None, None
    return round(float(np.mean(xs)), 4), round(float(np.std(xs)), 4)


def majority_baseline(y, folds):
    f1s = []
    for tr, te in folds:
        maj = collections.Counter(y[tr].tolist()).most_common(1)[0][0]
        f1s.append(f1_score(y[te], np.full(len(te), maj), labels=[0, 1], average="macro"))
    return f1s


def save_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Greens")
    ax.set_xticks(range(len(CLASSES)), CLASSES, fontsize=9)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=9)
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "#222", fontsize=11)
    ax.set_title("PLIP (frozen) + logistic head\nslice quality, 5 donor-held-out folds", fontsize=10)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    np.random.seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else " (CPU)"))

    ids, y, groups = load_rows()
    print(f"slides={len(ids)}  donors={len(set(groups.tolist()))}  "
          f"class counts={ {CLASSES[k]: int(v) for k, v in collections.Counter(y.tolist()).items()} }")

    emb = extract_features(ids, dev)

    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(emb, y, groups))
    for k, (tr, te) in enumerate(folds):
        assert not (set(groups[tr]) & set(groups[te])), f"donor leak in fold {k}"

    run_folds = folds[:1] if smoke else folds
    os.makedirs(OUT_DIR, exist_ok=True)

    per_fold, all_gt, all_pred = [], [], []
    cm_sum = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    for k, (tr, te) in enumerate(run_folds):
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(class_weight="balanced", max_iter=2000, C=1.0))
        clf.fit(emb[tr], y[tr])
        pred = clf.predict(emb[te])
        prob = clf.predict_proba(emb[te])[:, CLASSES.index("usable")]
        fm = fold_metrics(y[te], pred, prob)
        fm["fold"] = k + 1
        fm["test_donors"] = sorted(set(groups[te].tolist()))
        per_fold.append(fm)
        cm_sum += np.array(fm["confusion_matrix"])
        print(f"  FOLD {k+1}: macro-F1 {fm['macro_f1']:.4f}  AUC {fm['auc']}  "
              f"unusable recall {fm['recall']['unusable']:.3f}")

    f1_m, f1_s = mean_std([f["macro_f1"] for f in per_fold])
    auc_m, auc_s = mean_std([f["auc"] for f in per_fold])
    rn_m, rn_s = mean_std([f["recall"]["unusable"] for f in per_fold])
    ru_m, ru_s = mean_std([f["recall"]["usable"] for f in per_fold])
    pn_m, pn_s = mean_std([f["precision"]["unusable"] for f in per_fold])
    pu_m, pu_s = mean_std([f["precision"]["usable"] for f in per_fold])
    maj = majority_baseline(y, run_folds)
    maj_m, maj_s = mean_std(maj)

    save_confusion(cm_sum, os.path.join(OUT_DIR, "quality_plip_confusion.png"))
    summary = {
        "task": "slice_quality_whole_slide_usable_vs_unusable",
        "model": "plip_frozen_vitb + StandardScaler + LogisticRegression(balanced)",
        "smoke": smoke,
        "config": {"backbone": PLIP_MODEL, "frozen": True, "head": "logreg (balanced)",
                   "n_folds": len(run_folds), "seed": SEED,
                   "split": "donor-grouped GroupKFold by patient_id",
                   "device": (torch.cuda.get_device_name(0) if dev.type == "cuda" else "cpu"),
                   "note": "PLIP native 224px preprocessing (vs CNN 448px)"},
        "data": {"csv": os.path.relpath(CSV_PATH, SCRIPT_DIR), "n_slides": len(ids),
                 "n_donors": len(set(groups.tolist())),
                 "class_counts": {CLASSES[k]: int(v)
                                  for k, v in collections.Counter(y.tolist()).items()}},
        "plip_aggregate": {
            "macro_f1_mean": f1_m, "macro_f1_std": f1_s,
            "auc_mean": auc_m, "auc_std": auc_s,
            "precision_usable_mean": pu_m, "precision_usable_std": pu_s,
            "precision_unusable_mean": pn_m, "precision_unusable_std": pn_s,
            "recall_usable_mean": ru_m, "recall_usable_std": ru_s,
            "recall_unusable_mean": rn_m, "recall_unusable_std": rn_s,
            "confusion_matrix_summed": cm_sum.tolist(),
            "confusion_rows_true_cols_pred_order": CLASSES,
        },
        "baselines": {"majority_class": {"per_fold_macro_f1": [round(x, 4) for x in maj],
                                         "macro_f1_mean": maj_m, "macro_f1_std": maj_s}},
        "per_fold": per_fold,
    }
    with open(os.path.join(OUT_DIR, "quality_plip_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n============ PLIP (frozen) donor-held-out CV ============")
    print(f"macro-F1 {f1_m} +/- {f1_s}   AUC {auc_m} +/- {auc_s}")
    print(f"recall  unusable {rn_m}+/-{rn_s}   usable {ru_m}+/-{ru_s}")
    print(f"majority baseline macro-F1 {maj_m} +/- {maj_s}")
    print(f"summed confusion [unusable,usable]:\n{cm_sum}")
    print(f"\nwrote {OUT_DIR}\\quality_plip_metrics.json + quality_plip_confusion.png")
    print("Compare against models/quality/quality_metrics.json (ResNet-50) on the SAME data.")


if __name__ == "__main__":
    main()
