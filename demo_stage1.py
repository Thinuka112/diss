"""demo_stage1.py - reproduce the dissertation's Stage-1 headline from the shipped artefacts.

The production model is the gold-only ResNet-50. Chapter 5 reports macro-F1 0.8041
argmax and 0.8199 tuned over five donor-grouped folds. This demo recomputes both
numbers from the shipped per-image predictions in models/quality_union_humanonly.
It checks every fold against the stored quality_metrics.json and prints MATCH or
MISMATCH. The predictions were regenerated from the fold checkpoints under a hard
gate. Each fold had to match the stored metrics to 4 dp before the file was written.

Run:  python demo_stage1.py
Standard library only. No GPU. Finishes in under a second.

demo_plip_baseline.py does the same for the rejected PLIP baseline (0.7010).
"""
import csv
import json
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARMB = os.path.join(SCRIPT_DIR, "models", "quality_union_humanonly")


def macro_f1(rows, thr):
    """rows: (true_label, p_usable) pairs. thr: decision threshold on p_usable."""
    tp = {"usable": 0, "unusable": 0}
    fp = {"usable": 0, "unusable": 0}
    fn = {"usable": 0, "unusable": 0}
    for t, p in rows:
        pred = "usable" if p >= thr else "unusable"
        if pred == t:
            tp[t] += 1
        else:
            fp[pred] += 1
            fn[t] += 1
    f1s = []
    for c in ("usable", "unusable"):
        prec = tp[c] / (tp[c] + fp[c]) if tp[c] + fp[c] else 0.0
        rec = tp[c] / (tp[c] + fn[c]) if tp[c] + fn[c] else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / 2


meta = json.load(open(os.path.join(ARMB, "quality_metrics.json")))
stored = [f["macro_f1"] for f in meta["per_fold"]]
thresholds = meta["tuned_aggregate"]["per_fold_threshold"]

arg_scores, tuned_scores = [], []
n_total, ok = 0, True
print("Gold-only ResNet-50. The dissertation's production Stage-1 model.\n")
for k in range(5):
    rows = []
    with open(os.path.join(ARMB, f"fold{k + 1}_preds.csv"), newline="") as f:
        for r in csv.DictReader(f):
            rows.append((r["true"], float(r["p_usable"])))
    n_total += len(rows)
    a = macro_f1(rows, 0.5)
    t = macro_f1(rows, thresholds[k])
    arg_scores.append(a)
    tuned_scores.append(t)
    match = abs(a - stored[k]) < 5e-4
    ok = ok and match
    print(f"fold {k + 1}: n={len(rows):3d}  argmax macro-F1 {a:.4f}  "
          f"stored {stored[k]:.4f}  {'MATCH' if match else 'MISMATCH'}")

mean = lambda xs: sum(xs) / len(xs)
print(f"\nimages scored: {n_total}")
print(f"argmax mean {mean(arg_scores):.4f}   (dissertation: "
      f"{meta['cnn_aggregate']['macro_f1_mean']:.4f})")
print(f"tuned  mean {mean(tuned_scores):.4f}   (dissertation: "
      f"{meta['tuned_aggregate']['macro_f1_mean']:.4f})")
print("\nRESULT: MATCH" if ok else "\nRESULT: MISMATCH")
raise SystemExit(0 if ok else 1)
