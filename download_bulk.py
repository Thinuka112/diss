"""
download_bulk.py - scale the image set past the 100-image sample.

Same selection logic as download_small_intestine.py (small-intestine sample IHC
images, non-'rna', named {tissue}_{gene}_{sex}_{age}_{patientId}.jpg) but:
  - collects the URL list by streaming the XML, then downloads CONCURRENTLY,
  - writes to data_bulk/images/ (gitignored - too big to commit),
  - skips any base filename already vendored in HPA_small_intestine/.

Config at top. This is I/O bound; WORKERS parallelism is the speedup.
"""

import os
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_FILE = r"C:\Users\<user>\Downloads\normal_expression_Small.xml"   # external 3 GB export
SAMPLE_DIR = os.path.join(SCRIPT_DIR, "HPA_small_intestine")          # the committed 100
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "data_bulk", "images")       # gitignored

MAX_IMAGES = 5000     # round-2 enrichment (2026-08-09): up to 5000 NEW images (existing ~3100 skipped) -> pool ~8000
MAX_PER_GENE = 8      # was 2 -> pull the 3rd-8th image/gene from existing donors (adds count/morphology, not donors)
WORKERS = 12

SMALL_INTESTINE_TERMS = ("small intestine", "duodenum", "jejunum", "ileum")


def collect_targets(xml_file, existing, limit, max_per_gene):
    """Stream the XML and return [(url, base_filename)] for SI sample images."""
    targets = []
    per_gene = {}
    gene = "Unknown"
    sex = age = pid = "Unknown"
    tissue = "Small_intestine"
    is_si = False

    ctx = ET.iterparse(xml_file, events=("start", "end"))
    _, root = next(ctx)
    seen = 0
    for ev, el in ctx:
        if ev == "start":
            if el.tag == "entry":
                gene = "Unknown"
            elif el.tag == "patient":
                sex = age = pid = "Unknown"; tissue = "Small_intestine"; is_si = False
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
        elif t == "snomed" and "tissueDescription" in el.attrib:
            td = el.attrib["tissueDescription"]
            if any(s in td.lower() for s in SMALL_INTESTINE_TERMS):
                is_si = True; tissue = td.replace(" ", "_")
        elif t == "imageUrl" and el.text and is_si and "rna" not in el.text.lower():
            if per_gene.get(gene, 0) < max_per_gene:
                base = f"{tissue}_{gene}_{sex}_{age}_{pid}.jpg"
                if base not in existing:
                    targets.append((el.text, base))
                    existing.add(base)
                per_gene[gene] = per_gene.get(gene, 0) + 1
                if len(targets) >= limit:
                    break
        elif t == "patient":
            is_si = False
        elif t == "entry":
            seen += 1
            if seen % 50 == 0:
                root.clear()
        el.clear()
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
    existing = set()
    if os.path.isdir(SAMPLE_DIR):
        existing |= {f for f in os.listdir(SAMPLE_DIR) if f.lower().endswith(".jpg")}
    existing |= {f for f in os.listdir(OUTPUT_FOLDER) if f.lower().endswith(".jpg")}
    already = len(os.listdir(OUTPUT_FOLDER))

    print(f"Streaming XML to collect up to {MAX_IMAGES} targets "
          f"(skipping {len(existing)} already-have)...")
    targets = collect_targets(XML_FILE, set(existing), MAX_IMAGES, MAX_PER_GENE)
    todo = [(u, os.path.join(OUTPUT_FOLDER, b)) for u, b in targets
            if not os.path.exists(os.path.join(OUTPUT_FOLDER, b))]
    print(f"Collected {len(targets)} targets, {len(todo)} to download, {WORKERS} workers.")

    ok = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, u, p): p for u, p in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            if fut.result():
                ok += 1
            if i % 200 == 0:
                print(f"  {i}/{len(todo)} done ({ok} ok)...")
    total = len([f for f in os.listdir(OUTPUT_FOLDER) if f.lower().endswith(".jpg")])
    print(f"\nDownloaded {ok} new (was {already}). data_bulk/images now has {total} images.")


if __name__ == "__main__":
    main()
