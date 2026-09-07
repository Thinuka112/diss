"""Fetch HPA expression level + reliability for every gene in the sample + bulk
image sets, writing data_bulk/bulk_metadata.csv. Reuses image_metadata.py's
per-gene web fetch. Slow (one pass per unique gene)."""

import os
import csv
import image_metadata as im

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DIRS = [os.path.join(SCRIPT_DIR, "HPA_small_intestine"),
        os.path.join(SCRIPT_DIR, "data_bulk", "images")]
OUT = os.path.join(SCRIPT_DIR, "data_bulk", "bulk_metadata.csv")

files = []
for d in DIRS:
    if os.path.isdir(d):
        files += [f for f in os.listdir(d) if f.lower().endswith(".jpg")]
files = sorted(set(files))
parsed = {f: im.parse_filename(f) for f in files}
genes = sorted({p[1] for p in parsed.values()})
print(f"{len(files)} images across {len(genes)} genes; fetching from HPA...")

cache = {}
for i, g in enumerate(genes, 1):
    im.gene_meta(g, cache)
    if i % 100 == 0:
        print(f"  {i}/{len(genes)} genes")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["source_image", "gene", "ensembl", "si_expression_level", "reliability"])
    for f in files:
        _, g, _, _, _ = parsed[f]
        m = cache[g]
        w.writerow([f, g, m["ensembl"], m["si_expression_level"], m["reliability"]])
n_lvl = sum(1 for f in files if cache[parsed[f][1]]["si_expression_level"])
print(f"wrote {OUT}: {len(files)} rows, {n_lvl} with a level, {len(genes)} genes")
