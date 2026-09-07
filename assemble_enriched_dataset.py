"""
assemble_enriched_dataset.py - merge the active-learning top-up into the training set.

Combines the existing clean dataset (597 labels) with the expert's 250 model-mined
top-up labels (LOG 2026-08-02, 39% unusable) into one donor-tagged CSV ready for a
donor-grouped retrain. Applies the one expert-requested correction (slide 116).

Non-destructive: reads two CSVs, writes a new dated file. Never edits the inputs.

    python assemble_enriched_dataset.py

Inputs:
  review_tool/results/final_clean_dataset_20260728.csv  (597: image_id,label,patient_id,tissue,batch,reviewer,event_ts)
  review_tool/results/expert_topup_20260802.csv          (250: image_id,label,reviewer,event_ts)
Output:
  review_tool/results/final_clean_dataset_20260802.csv  (~847, same 7-column schema)
"""

import os
import csv
import collections

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(SCRIPT_DIR, "review_tool", "results")
BASE_CSV = os.path.join(RESULTS, "final_clean_dataset_20260728.csv")
TOPUP_CSV = os.path.join(RESULTS, "expert_topup_20260802.csv")
OUT_CSV = os.path.join(RESULTS, "final_clean_dataset_20260802.csv")

FIELDS = ["image_id", "label", "patient_id", "tissue", "batch", "reviewer", "event_ts"]
TOPUP_BATCH = "topup_mining"

# Expert-requested correction (LOG 2026-08-02): slide 116, in-app undo failed.
CORRECTIONS = {"Duodenum_BICRA_Male_35_3219.jpg": "unusable"}

# 16 cancer-carrying donors (AUDIT.md D3) - must never enter the training set.
CANCER_DONORS = {"1069", "1090", "1839", "1879", "1904", "2338", "2467", "2806",
                 "3049", "3548", "603", "604", "637", "700", "707", "997"}


def stem_tokens(image_id):
    return os.path.splitext(image_id)[0].split("_")


def patient_of(image_id):
    return stem_tokens(image_id)[-1]                      # {tissue}_{gene}_{sex}_{age}_{patientId}


def tissue_of(image_id):
    toks = stem_tokens(image_id)
    raw = "_".join(toks[:-4]) if len(toks) > 4 else "Unknown"    # tissue may be 'Small_intestine'
    # normalise to the base dataset's convention: spaces, no commas (e.g.
    # 'Small_intestine' -> 'Small intestine', 'Small_intestine,_NOS' -> 'Small intestine NOS')
    return " ".join(raw.replace("_", " ").replace(",", " ").split())


def main():
    with open(BASE_CSV, newline="", encoding="utf-8") as fh:
        base = list(csv.DictReader(fh))
    with open(TOPUP_CSV, newline="", encoding="utf-8") as fh:
        topup = list(csv.DictReader(fh))
    base_ids = {r["image_id"] for r in base}

    rows = list(base)                                     # keep the 597 verbatim
    added, dupes, cancer, corrected = 0, 0, 0, 0
    for r in topup:
        iid = r["image_id"]
        if iid in base_ids:                               # miner excluded labelled ids; guard anyway
            dupes += 1
            continue
        if patient_of(iid) in CANCER_DONORS:              # safety: no cancer donors
            cancer += 1
            continue
        label = r["label"]
        if iid in CORRECTIONS and label != CORRECTIONS[iid]:
            label = CORRECTIONS[iid]
            corrected += 1
        rows.append({
            "image_id": iid,
            "label": label,
            "patient_id": patient_of(iid),
            "tissue": tissue_of(iid),
            "batch": TOPUP_BATCH,
            "reviewer": r.get("reviewer", "expert"),
            "event_ts": r.get("event_ts", ""),
        })
        added += 1

    rows.sort(key=lambda r: r["image_id"])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    labels = collections.Counter(r["label"] for r in rows)
    batches = collections.Counter(r["batch"] for r in rows)
    tissues = collections.Counter(r["tissue"] for r in rows)
    donors = {r["patient_id"] for r in rows}
    unusable_donors = {r["patient_id"] for r in rows if r["label"] == "unusable"}
    cancer_in = donors & CANCER_DONORS

    print(f"base {len(base)} + topup added {added} (skipped {dupes} dup, {cancer} cancer) "
          f"-> {len(rows)} rows")
    print(f"corrections applied: {corrected} ({', '.join(CORRECTIONS)})")
    print(f"labels:   {dict(labels)}  (unusable {labels['unusable']}/{len(rows)} = "
          f"{labels['unusable']/len(rows):.1%})")
    print(f"batches:  {dict(batches)}")
    print(f"tissues:  {dict(tissues)}")
    print(f"donors:   {len(donors)} total; unusable class spans {len(unusable_donors)} donors")
    print(f"cancer-donor check: {'CLEAN' if not cancer_in else 'LEAK ' + str(cancer_in)}")
    print(f"wrote {os.path.relpath(OUT_CSV, SCRIPT_DIR)}")


if __name__ == "__main__":
    main()
