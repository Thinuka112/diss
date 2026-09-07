"""
train_expression.py - predict HPA expression level (metadata) from tile pixels.

This is the SEPARATE, metadata-facing task (distinct from usable/unusable): predict
HPA's small-intestine IHC expression call (Not detected / Low / Medium / High) - the
pathologist's read of brown DAB staining - from tile pixels.

RESULT (classical RandomForest baseline, 100-image sample): it does NOT work. On
unseen slides it scored 0.527 (4-class) / 0.526 (binary), BELOW the 0.550 majority
baseline (per-image majority vote just predicts "not detected" for all 20 test
images). Expression is only weakly encoded per tile (whole-image label, many blank
tiles) and 50 genes is too few to generalise. Kept as the classical baseline the
CNN (train_cnn.py) + more data must beat -- NOT a positive result.

- ENTIRE dataset: all 6400 tiles, each labelled by its source image's expression
  level (from image_metadata.csv, joined on source_image).
- Split 80:20, BY SLIDE (GroupShuffleSplit) so test tiles come from images the
  model never saw - that is what makes "unseen" real (a plain tile-level split
  would leak: 64 tiles of one image would straddle train and test). Seed 42.
- Proof it works = beating the majority-class baseline on those unseen slides,
  with a confusion matrix that isn't a single column.

Reports 4-class and binary (detected vs not) at tile level, plus per-image
majority-vote accuracy. Writes models/expression_overview.png + a JSON. Logged.
"""

import os
import csv
import json
import collections

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TILES_DIR = os.path.join(SCRIPT_DIR, "HPA_small_intestine_tiles")
MANIFEST = os.path.join(TILES_DIR, "manifest.csv")
META_CSV = os.path.join(SCRIPT_DIR, "image_metadata.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

IMG_SIZE = 32
TEST_SIZE, SEED = 0.20, 42
LEVELS = ["not detected", "low", "medium", "high"]
LVL_IDX = {l: i for i, l in enumerate(LEVELS)}


def load_pixels(rel_path):
    full = rel_path if os.path.isabs(rel_path) else os.path.join(SCRIPT_DIR, rel_path)
    with Image.open(full) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        return np.asarray(im, dtype=np.float32).ravel() / 255.0


def main():
    meta = {}
    with open(META_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[r["source_image"]] = r["si_expression_level"]
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    rows = [r for r in rows if meta.get(r["source_image"], "") in LVL_IDX]
    print(f"{len(rows)} tiles with an expression label; loading pixels...")

    X = np.stack([load_pixels(r["tile_path"]) for r in rows])
    y4 = np.array([LVL_IDX[meta[r["source_image"]]] for r in rows])   # 4-class
    yb = (y4 > 0).astype(int)                                          # detected vs not
    groups = np.array([r["source_image"] for r in rows])

    tile_dist = collections.Counter(LEVELS[i] for i in y4)
    img_lvl = {r["source_image"]: meta[r["source_image"]] for r in rows}
    img_dist = collections.Counter(img_lvl.values())
    print("Per-tile class counts:", dict(tile_dist))
    print("Per-image class counts:", dict(img_dist))

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr, te = next(gss.split(X, y4, groups))
    print(f"Train {len(tr)} tiles / {len(set(groups[tr]))} slides | "
          f"Test {len(te)} tiles / {len(set(groups[te]))} unseen slides")

    def evaluate(y, title, classes):
        clf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        acc = accuracy_score(y[te], pred)
        f1 = f1_score(y[te], pred, average="macro")
        maj = collections.Counter(y[tr]).most_common(1)[0][0]
        maj_acc = accuracy_score(y[te], np.full(len(te), maj))
        cm = confusion_matrix(y[te], pred, labels=list(range(len(classes))))
        print(f"\n[{title}] unseen-slide accuracy={acc:.3f} macroF1={f1:.3f}  "
              f"(majority baseline {maj_acc:.3f})  -> lift {acc - maj_acc:+.3f}")
        return {"clf": clf, "pred": pred, "acc": acc, "f1": f1,
                "maj_acc": maj_acc, "cm": cm, "classes": classes}

    r4 = evaluate(y4, "4-class expression", LEVELS)
    rb = evaluate(yb, "binary detected/not", ["not detected", "detected"])

    # per-image aggregation: majority vote of a test image's tile predictions
    def per_image(pred, y):
        img_pred, img_true = {}, {}
        acc_rows = collections.defaultdict(list)
        for k, i in enumerate(te):
            acc_rows[groups[i]].append((pred[k], y[i]))
        correct = 0
        for src, pairs in acc_rows.items():
            vote = collections.Counter(p for p, _ in pairs).most_common(1)[0][0]
            truth = pairs[0][1]
            correct += (vote == truth)
        return correct / len(acc_rows)
    img_acc4 = per_image(r4["pred"], y4)
    img_accb = per_image(rb["pred"], yb)
    print(f"\nPer-image (majority vote over its tiles) 4-class acc={img_acc4:.3f}  "
          f"binary acc={img_accb:.3f}  ({len(set(groups[te]))} unseen images)")

    _plot(r4, rb, tile_dist)

    joblib.dump({"model": r4["clf"], "img_size": IMG_SIZE, "levels": LEVELS,
                 "task": "expression level from tile pixels"},
                os.path.join(MODEL_DIR, "expression_model.joblib"))
    out = {
        "task": "predict HPA small-intestine expression level from tile pixels",
        "n_tiles": len(rows), "n_slides": len(set(groups)),
        "split": "GroupShuffleSplit by slide 80:20, seed 42, UNSEEN test slides",
        "train_tiles": len(tr), "test_tiles": len(te),
        "test_slides": len(set(groups[te])),
        "per_tile_class_counts": dict(tile_dist),
        "four_class": {"unseen_accuracy": round(r4["acc"], 4),
                       "macro_f1": round(r4["f1"], 4),
                       "majority_baseline": round(r4["maj_acc"], 4),
                       "confusion_matrix": r4["cm"].tolist(),
                       "confusion_labels": LEVELS},
        "binary_detected": {"unseen_accuracy": round(rb["acc"], 4),
                            "macro_f1": round(rb["f1"], 4),
                            "majority_baseline": round(rb["maj_acc"], 4),
                            "confusion_matrix": rb["cm"].tolist(),
                            "confusion_labels": ["not detected", "detected"]},
        "per_image_majority_vote": {"four_class_acc": round(img_acc4, 4),
                                    "binary_acc": round(img_accb, 4)},
    }
    with open(os.path.join(MODEL_DIR, "expression_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {os.path.join(MODEL_DIR, 'expression_overview.png')}")
    print(f"       {os.path.join(MODEL_DIR, 'expression_metrics.json')}")
    print(f"       {os.path.join(MODEL_DIR, 'expression_model.joblib')}")


def _cm_panel(ax, cm, classes, title):
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > thr else "#222")
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title(title, fontsize=10)


def _plot(r4, rb, tile_dist):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("Expression level from tile pixels — evaluated on UNSEEN slides "
                 "(GroupShuffleSplit 80:20, seed 42)", fontsize=12, weight="bold")
    _cm_panel(axes[0], r4["cm"], r4["classes"],
              f"4-class  acc={r4['acc']:.2f} (base {r4['maj_acc']:.2f})")
    _cm_panel(axes[1], rb["cm"], rb["classes"],
              f"binary detected  acc={rb['acc']:.2f} (base {rb['maj_acc']:.2f})")
    ax = axes[2]
    names = ["4-class", "binary"]
    ax.bar(np.arange(2) - 0.2, [r4["acc"], rb["acc"]], 0.38, label="model", color="#2563eb")
    ax.bar(np.arange(2) + 0.2, [r4["maj_acc"], rb["maj_acc"]], 0.38,
           label="majority baseline", color="#cbd5e1")
    ax.set_xticks(range(2), names); ax.set_ylim(0, 1.05)
    ax.set_title("Model vs majority baseline (unseen)", fontsize=10)
    ax.legend(fontsize=8)
    for i, (m, b) in enumerate([(r4["acc"], r4["maj_acc"]), (rb["acc"], rb["maj_acc"])]):
        ax.text(i - 0.2, m + 0.01, f"{m:.2f}", ha="center", fontsize=8)
        ax.text(i + 0.2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(os.path.join(MODEL_DIR, "expression_overview.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
