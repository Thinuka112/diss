"""
build_review_set.py - assemble a reproducible curated set of whole slides for the
web reviewer.

Draws N images (seeded, so it is reproducible) from the downloaded small-intestine
sets and copies them into review_tool/webapp/review_images/ so the web app is
self-contained (the source folders won't exist on a host later). Also writes a small,
committable manifest (image_id, gene, source) recording exactly what was selected.

Non-destructive: only reads the source folders and writes into review_images/ + the
manifest. Re-run with the same SEED to get the identical set.

  python build_review_set.py            # 500 images, seed 42
  python build_review_set.py --n 200 --seed 7
"""

import os
import csv
import random
import shutil
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SOURCE_DIRS = [os.path.join(REPO, "HPA_small_intestine"),
               os.path.join(REPO, "data_bulk", "images")]
OUT_DIR = os.path.join(HERE, "webapp", "review_images")
MANIFEST = os.path.join(HERE, "webapp", "review_set_manifest.csv")
SEX = ("Male", "Female", "Unknown")


def gene_of(fname):
    toks = os.path.splitext(fname)[0].split("_")
    for i, t in enumerate(toks):
        if t in SEX and i > 0:
            return toks[i - 1]
    return "Unknown"


def gather():
    """[(filename, source_path, source_dir)] across the source folders, de-duped by name."""
    seen, items = set(), []
    for d in SOURCE_DIRS:
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.lower().endswith(".jpg") and f not in seen:
                seen.add(f)
                items.append((f, os.path.join(d, f), os.path.basename(d)))
    return items


def main():
    ap = argparse.ArgumentParser(description="Build a curated review imageset.")
    ap.add_argument("--n", type=int, default=500, help="How many images to select.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed (reproducible).")
    args = ap.parse_args()

    pool = gather()
    if not pool:
        raise SystemExit(f"No source images found under: {SOURCE_DIRS}")
    n = min(args.n, len(pool))
    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(pool, n), key=lambda x: x[0])

    os.makedirs(OUT_DIR, exist_ok=True)
    copied = 0
    genes = set()
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "gene", "source_dir"])
        for fname, src, sdir in chosen:
            dst = os.path.join(OUT_DIR, fname)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
            copied += 1
            g = gene_of(fname)
            genes.add(g)
            w.writerow([fname, g, sdir])

    print("--- Review set built ---")
    print(f"Pool available:   {len(pool)} images")
    print(f"Selected:         {n} (seed {args.seed})")
    print(f"Distinct genes:   {len(genes)}")
    print(f"Images copied to: {OUT_DIR}")
    print(f"Manifest:         {MANIFEST}")


if __name__ == "__main__":
    main()
