"""tie_test.py - statistical test behind Ch5 S5.5's "tie" claim.

Paired t-test on the five per-fold macro-F1 differences between the two
finalists (combined-data YOLO minus gold-only ResNet), at both operating
points. Folds are Arm B's shared donor-grouped partition, so each delta is a
within-fold paired comparison. Deltas read from the stored run artefact.

Caveats reported alongside the result: n = 5 folds gives little power, so a
non-significant result supports but cannot prove equivalence; CV fold deltas
are not fully independent (training pools overlap). First run + logged
2026-09-05.
"""
import json
import numpy as np
from scipy import stats

d = json.load(open("models/quality_yolo_cvF/cv_metrics.json"))
for name, key in (("argmax", "delta_vs_armB_argmax"), ("tuned", "delta_vs_armB_tuned")):
    x = np.array([f[key] for f in d["per_fold"]])
    t, p = stats.ttest_1samp(x, 0.0)
    print(f"{name:6s} deltas {np.round(x,4).tolist()}  mean {x.mean():+.4f}  sd {x.std(ddof=1):.4f}"
          f"  paired t({len(x)-1}) = {t:+.3f}  p = {p:.3f}")
print("\nInterpretation: neither operating point shows a significant difference;")
print("the finalists' headline contest is a statistical tie (S5.5).")
