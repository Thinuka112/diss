"""Merge the expert's 847 human labels with the 5,900 machine cascade labels into ONE training CSV.

Mirrors assemble_enriched_dataset.py's shape (non-destructive: reads two CSVs, writes a third).
Scope: research/scoping/03c-combined-training.md.

Rules enforced here (the leakage guards live in the data, not in trust):
  - One row per image. On the 759 filename collisions the HUMAN label wins; the machine row is
    dropped, so no image can ever appear in two folds under donor-grouped CV.
  - Machine hard labels are binarised at V=4 (unusable if >=4 of 7 votes) for the naive-pool arm.
  - usable_frac carries the soft target for the noise-aware arm:
      human rows            -> 1.0 / 0.0 (hard)
      machine stage-1 rows  -> 1.0 / 0.0 (3/3 cheap members unanimous)
      machine stage-2 rows  -> 1 - unusable_votes/7  (the ensemble's actual uncertainty)
  - Cancer donors excluded as a safety no-op (both inputs are already clean).

Output: review_tool/results/combined_human_machine_20260818.csv
Columns: image_id,label,patient_id,tissue,batch,reviewer,event_ts,label_source,usable_frac
(the two extra columns are invisible to old consumers -- train_quality_cnn.py reads via DictReader)
"""
import collections
import csv
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HUMAN_CSV = os.path.join(SCRIPT_DIR, "review_tool", "results", "final_clean_dataset_20260802.csv")
MACHINE_CSV = os.path.join(SCRIPT_DIR, "models", "vlm_labels", "ensemble_cascade_labels.csv")
OUT_CSV = os.path.join(SCRIPT_DIR, "review_tool", "results", "combined_human_machine_20260818.csv")
V_HARD = 4                      # hard-label vote threshold for stage-2 machine rows
FIELDS = ["image_id", "label", "patient_id", "tissue", "batch", "reviewer", "event_ts",
          "label_source", "usable_frac"]
CANCER_DONORS = {"1069", "1090", "1839", "1879", "1904", "2338", "2467", "2806",
                 "3049", "3548", "603", "604", "637", "700", "707", "997"}


def stem_tokens(image_id):
    return os.path.splitext(image_id)[0].split("_")


def patient_of(image_id):
    return stem_tokens(image_id)[-1]


def tissue_of(image_id):
    toks = stem_tokens(image_id)
    raw = "_".join(toks[:-4]) if len(toks) > 4 else "Unknown"
    return " ".join(raw.replace("_", " ").replace(",", " ").split())


def main():
    human = list(csv.DictReader(open(HUMAN_CSV, newline="", encoding="utf-8")))
    machine = list(csv.DictReader(open(MACHINE_CSV, newline="", encoding="utf-8")))

    rows = []
    for r in human:
        rows.append({**{k: r[k] for k in FIELDS[:7]},
                     "label_source": "human",
                     "usable_frac": "1.0" if r["label"] == "usable" else "0.0"})
    human_ids = {r["image_id"] for r in rows}

    dupes = conflicts = cancer = bad = 0
    for r in machine:
        iid = r["source_image"]
        if r["label"] not in ("usable", "unusable"):
            bad += 1
            continue
        if iid in human_ids:
            dupes += 1
            hl = next(x["label"] for x in rows if x["image_id"] == iid)
            if hl != r["label"]:
                conflicts += 1
            continue
        if patient_of(iid) in CANCER_DONORS:
            cancer += 1
            continue
        assert r["donor"] == patient_of(iid), f"donor mismatch on {iid}"
        if r["stage"] == "2":
            v = int(r["unusable_votes"])
            hard = "unusable" if v >= V_HARD else "usable"
            frac = 1.0 - v / 7.0
        else:
            hard = r["label"]
            frac = 1.0 if hard == "usable" else 0.0
        rows.append({"image_id": iid, "label": hard, "patient_id": r["donor"],
                     "tissue": tissue_of(iid), "batch": "vlm_cascade",
                     "reviewer": "ensemble_cascade_7member", "event_ts": "",
                     "label_source": "machine", "usable_frac": f"{frac:.4f}"})

    rows.sort(key=lambda r: r["image_id"])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    # ---- asserted reconciliation report ----
    ids = [r["image_id"] for r in rows]
    donors = {r["patient_id"] for r in rows}
    n_h = sum(1 for r in rows if r["label_source"] == "human")
    n_m = len(rows) - n_h
    lab = collections.Counter(r["label"] for r in rows)
    assert len(ids) == len(set(ids)), "duplicate image_id in output"
    assert not (donors & CANCER_DONORS), "cancer donor leaked"
    assert n_h == len(human) == 847, f"human rows {n_h}"
    assert len(rows) == 5988, f"union size {len(rows)} != 5988"
    assert dupes == 759, f"dupes {dupes} != 759"
    print(f"combined: {len(rows)} rows = {n_h} human + {n_m} machine (V_HARD={V_HARD})")
    print(f"  dropped machine rows: {dupes} duplicates ({conflicts} label conflicts, human wins)"
          f", {cancer} cancer, {bad} invalid")
    print(f"  donors {len(donors)}   labels {dict(lab)}   "
          f"unusable {lab['unusable'] / len(rows) * 100:.1f}%")
    fr = [float(r["usable_frac"]) for r in rows if r["label_source"] == "machine"]
    soft = sum(1 for f in fr if 0.0 < f < 1.0)
    print(f"  machine soft targets: {soft} fractional (stage-2), {len(fr) - soft} hard (stage-1)")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
