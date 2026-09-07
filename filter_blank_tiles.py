"""
filter_blank_tiles.py - seed the slice quality labelling worksheet.

The slice quality classifier is trained and evaluated on EVERY tile in the
manifest, so this script never excludes, deletes, or moves a tile. It reads the
cut_images manifest and writes a NEW manifest (manifest_labels.csv) that copies
every original row and appends three columns: tissue_fraction, label, and
label_source.

WHAT THE LABELS MEAN:
  - Tiles whose tissue_fraction falls below AUTO_UNUSABLE_THRESHOLD are
    auto-labelled label="unusable", label_source="auto". These represent ONLY
    the empty-background extreme of the unusable class - the obvious pale
    margin around each core - not the full range of unusable slices.
  - Every other tile is left blank (label="") with label_source="manual", so a
    human can review it and fill in the slice quality label by hand.

NO TILE IS REMOVED FROM THE DATASET. This script is strictly non-destructive:
it reads one manifest and writes another, changing nothing on disk.

TISSUE DETECTION:
  Background is pale AND grey; real tissue is darker, or coloured, or both. So a
  pixel counts as tissue if EITHER:
    - luminance (mean of R,G,B) < LUM_THRESHOLD, OR
    - saturation ((max-min)/max per pixel) > SAT_THRESHOLD
  tissue_fraction = fraction of tissue pixels in the tile.
"""

import os
import csv
import numpy as np
from PIL import Image

# ------------------------------------------------------------------
# CONFIG - edit these only
# ------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
_TILES_DIR    = os.path.join(SCRIPT_DIR, "HPA_small_intestine_tiles")
IN_MANIFEST   = os.path.join(_TILES_DIR, "manifest.csv")
OUT_MANIFEST  = os.path.join(_TILES_DIR, "manifest_labels.csv")
AUTO_UNUSABLE_THRESHOLD = 0.05   # below this tissue fraction, auto-label "unusable"
LUM_THRESHOLD = 200          # pixels darker than this (0..255) count as tissue
SAT_THRESHOLD = 0.10         # pixels more colourful than this (0..1) count as tissue
# ------------------------------------------------------------------


def tissue_fraction(image_path, lum_threshold, sat_threshold):
    """Return the fraction (0..1) of pixels in the tile that look like tissue."""
    with Image.open(image_path) as img:
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)   # H x W x 3

    lum = arr.mean(axis=2)                 # per-pixel brightness, 0..255
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    # saturation = (max-min)/max; define it as 0 where max==0 (pure black) to avoid /0.
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)

    is_tissue = (lum < lum_threshold) | (sat > sat_threshold)
    return float(is_tissue.mean())


def main():
    with open(IN_MANIFEST, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
        fieldnames = list(reader.fieldnames)

    # Append the worksheet columns without disturbing the originals.
    out_fields = fieldnames + ["tissue_fraction", "label", "label_source"]

    total = len(rows)
    auto_unusable = 0

    os.makedirs(os.path.dirname(OUT_MANIFEST), exist_ok=True)
    with open(OUT_MANIFEST, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=out_fields)
        writer.writeheader()

        for i, row in enumerate(rows, start=1):
            tp = row["tile_path"]
            if not os.path.isabs(tp):
                tp = os.path.join(SCRIPT_DIR, tp)
            frac = tissue_fraction(tp, LUM_THRESHOLD, SAT_THRESHOLD)

            if frac < AUTO_UNUSABLE_THRESHOLD:
                label = "unusable"
                label_source = "auto"
                auto_unusable += 1
            else:
                label = ""
                label_source = "manual"

            row["tissue_fraction"] = round(frac, 4)
            row["label"] = label
            row["label_source"] = label_source
            writer.writerow(row)

            if i % 200 == 0:
                print(f"  processed {i}/{total} tiles...")

    manual = total - auto_unusable
    pct = lambda n: (100.0 * n / total) if total else 0.0

    print("\n--- Summary ---")
    print(f"Total tiles:              {total}")
    print(f"Auto-labelled unusable:   {auto_unusable} ({pct(auto_unusable):.1f}%)")
    print(f"Left for manual review:   {manual} ({pct(manual):.1f}%)")
    print(f"Output:                   {OUT_MANIFEST}")
    print("\nThis seeds the slice quality labelling worksheet. Auto-labelled tiles carry")
    print("label_source=auto and cover ONLY the empty-background extreme of the unusable")
    print("class; every other tile is left blank with label_source=manual for human review.")
    print("NO TILE WAS REMOVED - the classifier still sees the entire dataset.")


if __name__ == "__main__":
    main()
