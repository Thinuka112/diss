"""Arm E dataset: the expert's 847 + only the CONFIDENT machine labels (scope 03c extension, 2026-08-27).

Rationale (LOG 2026-08-22): "label accuracy binds, not volume" — so give the combined config its
best legitimate shot by keeping only machine rows whose labels are most likely correct:
  - stage-1 rows: the 3 cheap members were unanimous (measured most-reliable stratum), and
  - stage-2 rows: near-unanimous panel votes (unusable_votes <= 1 -> usable, >= 6 -> unusable);
    mid-vote rows (2..5 of 7) — where the noise concentrates (LOG 2026-08-17) — are DROPPED.
Everything else mirrors assemble_combined_dataset.py verbatim (imported, not copied): human wins
all filename collisions, cancer donors excluded, usable_frac soft targets preserved.

Output: review_tool/results/combined_human_confident_20260827.csv (same 9 columns).
"""
import collections
import csv
import os

import assemble_combined_dataset as base

OUT_CSV = os.path.join(base.SCRIPT_DIR, "review_tool", "results",
                       "combined_human_confident_20260827.csv")
V_LOW, V_HIGH = 1, 6            # stage-2 keep rule: v<=1 usable, v>=6 unusable


def main():
    human = list(csv.DictReader(open(base.HUMAN_CSV, newline="", encoding="utf-8")))
    machine = list(csv.DictReader(open(base.MACHINE_CSV, newline="", encoding="utf-8")))

    rows = []
    for r in human:
        rows.append({**{k: r[k] for k in base.FIELDS[:7]},
                     "label_source": "human",
                     "usable_frac": "1.0" if r["label"] == "usable" else "0.0"})
    human_ids = {r["image_id"] for r in rows}

    dupes = cancer = bad = dropped_mid = 0
    for r in machine:
        iid = r["source_image"]
        if r["label"] not in ("usable", "unusable"):
            bad += 1
            continue
        if iid in human_ids:
            dupes += 1
            continue
        if base.patient_of(iid) in base.CANCER_DONORS:
            cancer += 1
            continue
        assert r["donor"] == base.patient_of(iid), f"donor mismatch on {iid}"
        if r["stage"] == "2":
            v = int(r["unusable_votes"])
            if V_LOW < v < V_HIGH:          # uncertain middle: drop (the noise lives here)
                dropped_mid += 1
                continue
            hard = "unusable" if v >= V_HIGH else "usable"
            frac = 1.0 - v / 7.0
        else:
            hard = r["label"]
            frac = 1.0 if hard == "usable" else 0.0
        rows.append({"image_id": iid, "label": hard, "patient_id": r["donor"],
                     "tissue": base.tissue_of(iid), "batch": "vlm_cascade_confident",
                     "reviewer": "ensemble_cascade_7member", "event_ts": "",
                     "label_source": "machine", "usable_frac": f"{frac:.4f}"})

    rows.sort(key=lambda r: r["image_id"])
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=base.FIELDS)
        w.writeheader()
        w.writerows(rows)

    ids = [r["image_id"] for r in rows]
    donors = {r["patient_id"] for r in rows}
    n_h = sum(1 for r in rows if r["label_source"] == "human")
    n_m = len(rows) - n_h
    lab = collections.Counter(r["label"] for r in rows)
    m_unus = sum(1 for r in rows if r["label_source"] == "machine" and r["label"] == "unusable")
    assert len(ids) == len(set(ids)), "duplicate image_id in output"
    assert not (donors & base.CANCER_DONORS), "cancer donor leaked"
    assert n_h == len(human) == 847, f"human rows {n_h}"
    assert dupes == 759, f"dupes {dupes} != 759"
    assert n_m + dropped_mid + dupes + cancer + bad == len(machine), "machine rows unaccounted"
    print(f"confident combined: {len(rows)} rows = {n_h} human + {n_m} machine "
          f"(dropped {dropped_mid} mid-vote 2..5, {dupes} dupes, {cancer} cancer, {bad} invalid)")
    print(f"  donors {len(donors)}   labels {dict(lab)}   machine unusable {m_unus}")
    print(f"wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
