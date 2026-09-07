"""overlap_audit.py - reproduce the machine-label purity strata quoted in the
dissertation (Ch5 S5.4): agreement between the cascade ensemble's labels and
the expert's gold labels on the images both sources labelled.

Inputs (committed):
  models/vlm_labels/ensemble_cascade_labels.csv  (panel labels + votes + stage)
  review_tool/results/final_clean_dataset_20260802.csv  (expert gold labels)

Deterministic read-only join; no randomness. First logged: LOG 2026-08-28
(overlap audit); promoted to a committed script 2026-09-05 so the numbers
exist in the program, not only the log.
"""
import csv
from collections import defaultdict

GOLD = "review_tool/results/final_clean_dataset_20260802.csv"
MACHINE = "models/vlm_labels/ensemble_cascade_labels.csv"

gold = {}
with open(GOLD, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        gold[r["image_id"]] = r["label"]

rows = []
with open(MACHINE, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        if r["source_image"] in gold:
            rows.append((r["source_image"], r["label"], int(r["stage"]),
                         int(r["unusable_votes"]) if r["unusable_votes"] else None))

def agree(sub):
    n = len(sub); a = sum(1 for img, lab, *_ in sub if gold[img] == lab)
    return a, n, (100.0 * a / n if n else float("nan"))

print(f"overlap: {len(rows)} images labelled by both expert and panel")
a, n, p = agree(rows)
print(f"overall agreement: {a}/{n} = {p:.1f}%\n")

s1 = [r for r in rows if r[2] == 1]
s2 = [r for r in rows if r[2] == 2]
for lab in ("usable", "unusable"):
    sub = [r for r in s1 if r[1] == lab]
    a, n, p = agree(sub)
    print(f"stage-1 unanimous, machine says {lab:8s}: {a:3d}/{n:3d} = {p:.1f}%")
print()
buckets = defaultdict(list)
for r in s2:
    buckets[r[3]].append(r)
for v in sorted(buckets):
    a, n, p = agree(buckets[v])
    print(f"stage-2, unusable_votes={v} (machine says {buckets[v][0][1]:8s}): {a:3d}/{n:3d} = {p:.1f}%")

us = [r for r in rows if r[1] == "usable"]; uu = [r for r in rows if r[1] == "unusable"]
pa = [agree([r for r in s1 if r[1] == "usable"])[2]] + [agree(buckets[v])[2] for v in sorted(buckets) if buckets[v][0][1] == "usable"]
pu = [agree([r for r in s1 if r[1] == "unusable"])[2]] + [agree(buckets[v])[2] for v in sorted(buckets) if buckets[v][0][1] == "unusable"]
print(f"\nusable-call strata range:   {min(pa):.1f}-{max(pa):.1f}%")
print(f"unusable-call strata range: {min(pu):.1f}-{max(pu):.1f}%")
