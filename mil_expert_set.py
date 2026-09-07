"""mil_expert_set.py - build the Stage-2 expert tile-annotation set (scope 05).

Selects 20 USABLE, human-labelled slides (4 per Stage-1 test fold, donor-stratified,
preferring slides that also have stored VLM per-tile answers so expert-vs-VLM kappa is
measurable), cuts each into its 8x8 native tiles, and wires them into the review site:

  review_tool/webapp/loc_tiles/{stem}__r{r}c{c}.jpg     the tile images (items)
  review_tool/webapp/loc_tile_manifest.csv              tile_path,tile_id,image_id,row,col
  review_tool/webapp/loc_set.txt                        tile-id whitelist (PARAM_SUBSET)

The expert then answers per tile "qualifying stretch here? -> yes / <- no" through the
EXISTING arrow-key flow (profile loc_stretch) - no new UI. His judgments are evaluation
data only; they never train anything.

Deterministic: seed 42. Non-destructive: writes only the three outputs above.
"""
import csv
import json
import os
import random

from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HUMAN_CSV = os.path.join(SCRIPT_DIR, "review_tool", "results", "final_clean_dataset_20260802.csv")
FOLDS_JSON = os.path.join(SCRIPT_DIR, "models", "quality", "quality_metrics.json")
SILVER_CSV = os.path.join(SCRIPT_DIR, "probe", "prompt_val.csv")
IMG_DIR = os.path.join(SCRIPT_DIR, "review_tool", "webapp", "review_images")
OUT_TILES = os.path.join(SCRIPT_DIR, "review_tool", "webapp", "loc_tiles")
OUT_MANIFEST = os.path.join(SCRIPT_DIR, "review_tool", "webapp", "loc_tile_manifest.csv")
OUT_SET = os.path.join(SCRIPT_DIR, "review_tool", "webapp", "loc_set.txt")
GRID = 8
PER_FOLD = 4
SEED = 42


def tiles_of(im, grid):
    """MUST match tile_label.tiles_of: floor-division tile size, remainder absorbed into
    the last row/col, row-major order. The silver VLM reference uses this convention."""
    w, h = im.size
    tw, th = w // grid, h // grid
    return [(r, c, im.crop((c * tw, r * th,
                            (c + 1) * tw if c < grid - 1 else w,
                            (r + 1) * th if r < grid - 1 else h)))
            for r in range(grid) for c in range(grid)]


def main():
    human = list(csv.DictReader(open(HUMAN_CSV, newline="", encoding="utf-8")))
    usable = [r for r in human if r["label"] == "usable"]
    per_fold_donors = [set(f["test_donors"])
                       for f in json.load(open(FOLDS_JSON, encoding="utf-8"))["per_fold"]]
    silver = {r["image_id"] for r in csv.DictReader(open(SILVER_CSV, newline="", encoding="utf-8"))
              if r["label"] == "usable"}

    rng = random.Random(SEED)
    picked = []
    for k, donors in enumerate(per_fold_donors):
        cand = [r for r in usable if r["patient_id"] in donors]
        # prefer silver-overlap slides, then donor diversity, deterministically
        cand.sort(key=lambda r: (r["image_id"] not in silver, r["image_id"]))
        chosen, seen_donors = [], set()
        for r in cand:                                   # pass 1: one slide per donor
            if len(chosen) >= PER_FOLD:
                break
            if r["patient_id"] not in seen_donors:
                chosen.append(r)
                seen_donors.add(r["patient_id"])
        for r in cand:                                   # pass 2: top up if few donors
            if len(chosen) >= PER_FOLD:
                break
            if r not in chosen:
                chosen.append(r)
        rng.shuffle(chosen)
        for r in chosen:
            picked.append({**r, "fold": k + 1})

    os.makedirs(OUT_TILES, exist_ok=True)
    rows, ids = [], []
    for p in picked:
        stem = os.path.splitext(p["image_id"])[0]
        with Image.open(os.path.join(IMG_DIR, p["image_id"])) as im:
            im = im.convert("RGB")
            for r, c, tile in tiles_of(im, GRID):
                tid = f"{stem}__r{r}c{c}.jpg"
                tile.save(os.path.join(OUT_TILES, tid), "JPEG", quality=92)
                rows.append({"tile_path": f"loc_tiles/{tid}", "tile_id": tid,
                             "image_id": p["image_id"], "row": r, "col": c,
                             "fold": p["fold"]})
                ids.append(tid)

    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["tile_path", "tile_id", "image_id",
                                           "row", "col", "fold"])
        w.writeheader()
        w.writerows(rows)
    with open(OUT_SET, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(ids) + "\n")

    n_silver = sum(1 for p in picked if p["image_id"] in silver)
    print(f"selected {len(picked)} usable slides "
          f"({PER_FOLD}/fold x {len(per_fold_donors)} folds), "
          f"{len({p['patient_id'] for p in picked})} donors, "
          f"{n_silver} overlap the VLM silver set (kappa measurable)")
    print(f"cut {len(rows)} tiles ({GRID}x{GRID}) -> {OUT_TILES}")
    print(f"wrote {OUT_MANIFEST}")
    print(f"wrote {OUT_SET}")
    for p in picked:
        print(f"  fold {p['fold']}  donor {p['patient_id']:>5s}  "
              f"{'[silver]' if p['image_id'] in silver else '        '}  {p['image_id']}")


def blankcut(threshold=0.05):
    """Drop blank tiles from the expert queue (user-approved 2026-08-21, visual check passed).

    Tiles with tissue_fraction < threshold cannot contain a qualifying stretch (~5 adjacent
    epithelial cells), so they are recorded as AUTOMATIC "no" (label_source=auto — the same
    rule and bookkeeping Stage 1's filter_blank_tiles.py used) and removed from loc_set.txt.
    The manifest keeps all 64 tiles per slide; the gold eval merges the expert's answers with
    these auto-no rows so every tile still has a verdict. Non-destructive: rewrites only
    loc_set.txt and writes the auto-no record.
    """
    tf = {r["tile_id"]: float(r["tissue_fraction"])
          for r in csv.DictReader(open(os.path.join(SCRIPT_DIR, "review_tool",
                                       "loc_tile_tissue_fraction.csv"), encoding="utf-8"))}
    man = list(csv.DictReader(open(OUT_MANIFEST, newline="", encoding="utf-8")))
    keep, auto = [], []
    for r in man:
        (auto if tf[r["tile_id"]] < threshold else keep).append(r)

    with open(OUT_SET, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(r["tile_id"] for r in keep) + "\n")
    auto_csv = os.path.join(SCRIPT_DIR, "review_tool", "loc_tile_auto_no.csv")
    with open(auto_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["tile_id", "image_id", "row", "col",
                                           "tissue_fraction", "label", "label_source"])
        w.writeheader()
        for r in auto:
            w.writerow({"tile_id": r["tile_id"], "image_id": r["image_id"],
                        "row": r["row"], "col": r["col"],
                        "tissue_fraction": f"{tf[r['tile_id']]:.4f}",
                        "label": "no", "label_source": "auto"})
    print(f"blank cut @ tissue_fraction<{threshold}: queue {len(man)} -> {len(keep)} tiles "
          f"({len(auto)} auto-no recorded in {os.path.basename(auto_csv)})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "blankcut":
        blankcut()
    else:
        main()
