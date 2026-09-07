"""
analyze_model.py - feature ablation + a diagram of the current slice-quality model.

Answers two questions:
  1. "How does accuracy look for the OTHER features?" The deployed model uses only
     32x32 pixels. Here we train the SAME RandomForest on each alternative feature
     set on the SAME slide-grouped split (seed 42) and compare:
        - tissue_fraction (1 feature)  = the baseline heuristic
        - gene metadata (expression level, reliability, sex, age)
        - pixels 32x32                 = the current model
        - all combined
  2. A picture of the current model -> models/model_overview.png
     (architecture block diagram + confusion matrix + feature-set accuracy).

Non-destructive: reads tiles + manifests + image_metadata.csv, writes PNGs + a
JSON under models/. Reuses the training config so numbers match train_classifier.py.
"""

import os
import csv
import json

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TILES_DIR = os.path.join(SCRIPT_DIR, "HPA_small_intestine_tiles")
MANIFEST = os.path.join(TILES_DIR, "manifest_labels.csv")
META_CSV = os.path.join(SCRIPT_DIR, "image_metadata.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

IMG_SIZE = 32
UNUSABLE_MAX, USABLE_MIN = 0.05, 0.50
TEST_SIZE, SEED = 0.20, 42

EXPR = {"not detected": 0, "low": 1, "medium": 2, "high": 3}
RELI = {"uncertain": 0, "approved": 1, "supported": 2, "enhanced": 3}
SEX = {"Male": 0, "Female": 1, "Unknown": 2}


def load_pixels(rel_path):
    full = rel_path if os.path.isabs(rel_path) else os.path.join(SCRIPT_DIR, rel_path)
    with Image.open(full) as im:
        im = im.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        return np.asarray(im, dtype=np.float32).ravel() / 255.0


def weak_label(frac):
    if frac < UNUSABLE_MAX:
        return 0
    if frac >= USABLE_MIN:
        return 1
    return -1


def main():
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    meta = {}
    with open(META_CSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            meta[r["source_image"]] = r
    print(f"{len(rows)} tiles; building feature sets...")

    X_pix = np.stack([load_pixels(r["tile_path"]) for r in rows])
    tf = np.array([float(r["tissue_fraction"]) for r in rows])
    groups = np.array([r["source_image"] for r in rows])
    y = np.array([weak_label(f) for f in tf])

    def meta_vec(src):
        m = meta.get(src, {})
        try:
            age = float(m.get("age", "nan"))
        except ValueError:
            age = -1.0
        return [EXPR.get(m.get("si_expression_level", ""), -1),
                RELI.get(m.get("reliability", ""), -1),
                SEX.get(m.get("sex", ""), 2), age]
    X_meta = np.array([meta_vec(r["source_image"]) for r in rows], dtype=np.float32)
    X_tf = tf.reshape(-1, 1)

    lab = y != -1
    idx = np.where(lab)[0]
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    tr_l, te_l = next(gss.split(idx, y[idx], groups[idx]))
    tr, te = idx[tr_l], idx[te_l]

    feature_sets = {
        "tissue_fraction\n(baseline, 1 feat)": X_tf,
        "gene metadata\n(expr, reliab, sex, age)": X_meta,
        "pixels 32x32\n(current model)": X_pix,
        "all combined": np.hstack([X_pix, X_tf, X_meta]),
    }
    results = {}
    for name, X in feature_sets.items():
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        results[name] = {"accuracy": accuracy_score(y[te], pred),
                         "macro_f1": f1_score(y[te], pred, average="macro")}

    maj = int(round(y[tr].mean()))
    maj_acc = accuracy_score(y[te], np.full(len(te), maj))
    results["majority class\n(reference)"] = {"accuracy": maj_acc, "macro_f1":
                                              f1_score(y[te], np.full(len(te), maj),
                                                       average="macro")}

    clf_pix = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=SEED, n_jobs=-1).fit(X_pix[tr], y[tr])
    cm = confusion_matrix(y[te], clf_pix.predict(X_pix[te]), labels=[0, 1])

    print("\n--- Accuracy by feature set (held-out slides) ---")
    for name, r in results.items():
        print(f"  {name.replace(chr(10),' '):<42} acc={r['accuracy']:.3f}  macroF1={r['macro_f1']:.3f}")

    _plot(results, cm, len(tr), len(te), len(set(groups[tr])), len(set(groups[te])))

    with open(os.path.join(MODEL_DIR, "feature_ablation.json"), "w", encoding="utf-8") as fh:
        json.dump({k.replace("\n", " "): v for k, v in results.items()}, fh, indent=2)
    print(f"\nSaved: {os.path.join(MODEL_DIR, 'model_overview.png')}")
    print(f"       {os.path.join(MODEL_DIR, 'feature_ablation.json')}")


def _box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                                linewidth=1.5, edgecolor="#333", facecolor=fc))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=14, color="#333", linewidth=1.3))


def _plot(results, cm, n_tr, n_te, s_tr, s_te):
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.32, wspace=0.25)

    ax = fig.add_subplot(gs[0, :]); ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
    ax.set_title("Current slice-quality model  —  RandomForest on tile pixels (weak-label baseline)",
                 fontsize=12, weight="bold")
    boxes = [(0.1, "input tile\n(1/64 of a slide)", "#eef2ff"),
             (2.0, "resize\n32x32 RGB", "#e0f2fe"),
             (3.9, "flatten\n3072 features\nscaled [0,1]", "#e0f2fe"),
             (5.9, "RandomForest\n200 trees\nclass_weight=balanced", "#dcfce7"),
             (8.2, "usable /\nunusable", "#fef9c3")]
    for x, t, c in boxes:
        _box(ax, x, 1.2, 1.6, 1.1, t, c)
    for i in range(len(boxes) - 1):
        _arrow(ax, boxes[i][0] + 1.6, 1.75, boxes[i + 1][0], 1.75)
    ax.text(5.0, 0.35,
            "Weak labels from tissue_fraction: unusable <0.05, usable >=0.50 "
            "(ambiguous 0.05-0.50 held out).  Split: GroupShuffleSplit by slide, seed 42 "
            f"(train {n_tr} tiles/{s_tr} slides, test {n_te}/{s_te}).",
            ha="center", va="center", fontsize=8.5, style="italic", color="#444")

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["pred unusable", "pred usable"])
    ax.set_yticks([0, 1], ["true unusable", "true usable"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "#222")
    ax.set_title("Confusion matrix — current model (held-out slides)", fontsize=10.5)

    ax = fig.add_subplot(gs[1, 1])
    names = list(results.keys())
    accs = [results[n]["accuracy"] for n in names]
    f1s = [results[n]["macro_f1"] for n in names]
    yb = np.arange(len(names))
    ax.barh(yb - 0.2, accs, height=0.38, label="accuracy", color="#2563eb")
    ax.barh(yb + 0.2, f1s, height=0.38, label="macro-F1", color="#93c5fd")
    ax.set_yticks(yb, names, fontsize=8)
    ax.set_xlim(0, 1.05); ax.invert_yaxis(); ax.legend(fontsize=8, loc="lower right")
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(a + 0.01, i - 0.2, f"{a:.2f}", va="center", fontsize=7.5)
        ax.text(f + 0.01, i + 0.2, f"{f:.2f}", va="center", fontsize=7.5)
    ax.set_title("Accuracy by feature set (same split)", fontsize=10.5)

    fig.savefig(os.path.join(MODEL_DIR, "model_overview.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
