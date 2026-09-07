"""Re-derive a label set at a different ensemble vote threshold — zero API cost.

label_ensemble.py stores every member's vote for each escalated slide, so the operating point is a
post-hoc dial rather than something baked into the run. Stage-1 slides were decided unanimously by
the cheap members and are unaffected by V; only escalated slides move.

  python emit_threshold_variant.py 4        # -> models/vlm_labels/ensemble_cascade_V4_labels.csv

Held-out accuracy by V, measured on probe/prompt_val.csv (LOG 2026-08-17):
  V=3 0.800 | V=4 0.830 | V=5 0.840 (the shipped default) | V=6 0.800
Lower V yields a larger unusable class at slightly lower accuracy — which trades better if the
downstream classifier is starved of minority examples.
"""
import collections
import csv
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(REPO, "models", "vlm_labels", "ensemble_cascade_labels.csv")
V = int(sys.argv[1]) if len(sys.argv) > 1 else 4
OUT = os.path.join(REPO, "models", "vlm_labels", f"ensemble_cascade_V{V}_labels.csv")

rows = list(csv.DictReader(open(SRC, newline="", encoding="utf-8")))
out, changed = [], 0
for r in rows:
    lab = r["label"]
    if r["stage"] == "2" and str(r["unusable_votes"]).isdigit():
        new = "unusable" if int(r["unusable_votes"]) >= V else "usable"
        changed += (new != lab)
        lab = new
    out.append({"source_image": r["source_image"], "label": lab, "donor": r["donor"],
                "stage": r["stage"], "unusable_votes": r["unusable_votes"]})

with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["source_image", "label", "donor", "stage", "unusable_votes"])
    w.writeheader()
    w.writerows(out)

c = collections.Counter(r["label"] for r in out)
n = c["usable"] + c["unusable"]
print(f"V={V}: {n} labels — usable {c['usable']}, unusable {c['unusable']} "
      f"({c['unusable'] / n * 100:.1f}%);  {changed} slides changed vs the shipped V=5 file")
print(f"unusable class is {c['unusable'] / 163:.1f}x the 163 human unusable labels")
print(f"wrote {OUT}")
