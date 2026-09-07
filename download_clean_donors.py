"""
download_clean_donors.py - top-up the image set with NEW, cancer-free donors.

Post-audit remediation (see AUDIT.md). The original downloaders capped at 2 images
PER GENE, which repeatedly grabbed the same handful of donors -> the labelled set
collapsed onto 22 donors, and no morphology filter meant cancer cores slipped in.

This script fixes both:
  - MORPHOLOGY FILTER: skip any sample whose SNOMED carries a neoplasm morphology
    code (M-8xxxx / M-9xxxx), and skip the 16 known cancer-carrying donors outright.
  - PER-DONOR CAP (not per-gene): take up to PER_DONOR images from each NEW donor,
    spreading the top-up across ~NEW_DONORS fresh donors for real donor variety.
  - NEW donors only: skip any donor already present on disk (sample/bulk/review set),
    so this genuinely adds people rather than more of the same.

Streams the same local HPA XML the other downloaders use, then fetches concurrently
into data_topup/images/ and writes data_topup/topup_manifest.csv. Non-destructive:
touches nothing that already exists. Re-run is idempotent (skips files on disk).
"""

import os
import csv
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# ------------------------------------------------------------------
# CONFIG - edit these only
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_FILE = r"C:\Users\<user>\Downloads\normal_expression_Small.xml"   # 3.1 GB local export
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "data_topup", "images")
MANIFEST = os.path.join(SCRIPT_DIR, "data_topup", "topup_manifest.csv")

NEW_DONORS = 40            # how many fresh clean donors to add (raised 20->40 for the mining diversity batch, LOG 2026-07-31)
PER_DONOR = 12             # max images per new donor (spreads across genes/stains)
WORKERS = 12

# donors already on disk are read dynamically (excluded so we add NEW people);
# these folders define "already have".
EXISTING_DIRS = [os.path.join(SCRIPT_DIR, "HPA_small_intestine"),
                 os.path.join(SCRIPT_DIR, "data_bulk", "images"),
                 os.path.join(SCRIPT_DIR, "review_tool", "webapp", "review_images")]
# ------------------------------------------------------------------

SMALL_INTESTINE_TERMS = ("small intestine", "duodenum", "jejunum", "ileum")
NEOPLASM = re.compile(r"^M-[89]")
# 16 cancer-carrying small-intestine donors identified from the XML (AUDIT.md D3).
CANCER_DONORS = {"1069", "1090", "1839", "1879", "1904", "2338", "2467", "2806",
                 "3049", "3548", "603", "604", "637", "700", "707", "997"}


def donor_of(fname):
    return os.path.splitext(fname)[0].split("_")[-1]


def existing_donors():
    donors = set()
    for d in EXISTING_DIRS:
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".jpg"):
                    donors.add(donor_of(f))
    return donors


def collect_targets(xml_file, skip_donors):
    """Stream the XML; return [(url, base, gene, tissue, pid)] for NEW clean donors,
    per-donor capped, across at most NEW_DONORS donors."""
    chosen = defaultdict(list)            # pid -> [(url, base, gene, tissue)]
    gene = "Unknown"
    sex = age = pid = "Unknown"
    tissue = "Small_intestine"
    is_si = False
    snomed_codes = []

    ctx = ET.iterparse(xml_file, events=("start", "end"))
    _, root = next(ctx)
    seen = 0
    for ev, el in ctx:
        if ev == "start":
            if el.tag == "entry":
                gene = "Unknown"
            elif el.tag == "patient":
                sex = age = pid = "Unknown"; tissue = "Small_intestine"; is_si = False
            elif el.tag == "sample":
                snomed_codes = []
            continue
        t = el.tag
        if t == "name" and gene == "Unknown":
            gene = el.text.replace(" ", "_") if el.text else "Unknown"
        elif t == "sex":
            sex = el.text or "Unknown"
        elif t == "age":
            age = el.text or "Unknown"
        elif t == "patientId":
            pid = el.text or "Unknown"
        elif t == "snomed":
            code = el.attrib.get("snomedCode", "")
            snomed_codes.append(code)
            td = el.attrib.get("tissueDescription", "")
            if any(s in td.lower() for s in SMALL_INTESTINE_TERMS):
                is_si = True; tissue = td.replace(" ", "_")
        elif t == "imageUrl" and el.text and is_si and "rna" not in el.text.lower():
            neoplasm = any(NEOPLASM.match(c) for c in snomed_codes)
            new_donor_ok = pid in chosen or len(chosen) < NEW_DONORS
            if (pid not in skip_donors and pid not in CANCER_DONORS
                    and not neoplasm and new_donor_ok
                    and len(chosen[pid]) < PER_DONOR):
                base = f"{tissue}_{gene}_{sex}_{age}_{pid}.jpg"
                if not any(b == base for _, b, _, _ in chosen[pid]):
                    chosen[pid].append((el.text, base, gene, tissue))
            # stop once every started donor is full
            if len(chosen) >= NEW_DONORS and all(len(v) >= PER_DONOR for v in chosen.values()):
                break
        elif t == "patient":
            is_si = False
        elif t == "entry":
            seen += 1
            if seen % 3000 == 0:
                root.clear()
        el.clear()

    targets = []
    for pid, items in chosen.items():
        for url, base, gene, tissue in items:
            targets.append((url, base, gene, tissue, pid))
    return targets


def fetch(url, path):
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except Exception:
        return False


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    skip = existing_donors()
    print(f"Excluding {len(skip)} donors already on disk; skipping {len(CANCER_DONORS)} cancer donors.")
    print("Streaming XML to collect new clean-donor targets...")
    targets = collect_targets(XML_FILE, skip)
    donors = sorted({t[4] for t in targets})
    print(f"Collected {len(targets)} images across {len(donors)} new donors: {donors}")

    todo = [(u, os.path.join(OUTPUT_FOLDER, b), b, g, ts, pid)
            for (u, b, g, ts, pid) in targets
            if not os.path.exists(os.path.join(OUTPUT_FOLDER, b))]
    print(f"{len(todo)} to download ({WORKERS} workers)...")

    ok, done_rows = 0, []
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, u, p): (b, g, ts, pid) for (u, p, b, g, ts, pid) in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            b, g, ts, pid = futs[fut]
            if fut.result():
                ok += 1
                done_rows.append((b, g, ts, pid))
            if i % 50 == 0:
                print(f"  {i}/{len(todo)} ({ok} ok)")

    # manifest of everything now in the folder (idempotent re-runs stay complete)
    all_rows = []
    on_disk = {f for f in os.listdir(OUTPUT_FOLDER) if f.lower().endswith(".jpg")}
    by_base = {b: (g, ts, pid) for (_, b, g, ts, pid) in targets}
    for f in sorted(on_disk):
        g, ts, pid = by_base.get(f, ("Unknown", "Unknown", donor_of(f)))
        all_rows.append((f, g, ts, pid))
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "gene", "tissue", "patient_id"])
        w.writerows(all_rows)

    donors_final = sorted({r[3] for r in all_rows})
    print(f"\nDownloaded {ok} new images. data_topup/images now has {len(on_disk)} "
          f"images across {len(donors_final)} donors.")
    print(f"Manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
