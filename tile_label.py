"""Native-resolution tiling for the slice-quality labeller (lever 2 of 2).

The whole-slide call downscales a 3000x3000 section to 1512px, roughly halving linear resolution.
A qualifying villus stretch is ~5 adjacent cells -- small on a shrunk whole slide. Tiling cuts the
section into a GRID x GRID mosaic sent at NATIVE resolution (2x2 -> 1500px tiles, 3x3 -> 1000px),
asks each tile the single question "is there a qualifying villus stretch HERE?", and aggregates:

    slide = usable  <=>  (number of tiles answering yes) >= K

K=1 is the user's rule (usable if ANY tile qualifies). Because the per-tile yes-count is recorded,
every K from 1..GRID^2 is scored from the SAME api spend -- a free operating-point sweep.

Scored on probe/dev_eval2.csv (100 fresh usable + 100 unusable), the same set as few_shot_label.py,
so the two levers are directly comparable. The 120-slide TEST set is never touched here.

  python tile_label.py 2        # 2x2 = 4 tiles/slide  (800 calls)
  python tile_label.py 3        # 3x3 = 9 tiles/slide  (1800 calls)

Env: PROBE_MODEL, TILE_MAXEDGE (1512 cap per tile), TILE_WORKERS (16), TILE_EVAL (dev_eval2.csv).
"""
import concurrent.futures as cf
import csv
import json
import os
import sys

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe"))
import spend
from vertex_common import (CLASSES, IMG_DIR, REPO, encode_im, get_client, img_part, parse_yes_no,
                           score, text_part)

GRID = int(sys.argv[1]) if len(sys.argv) > 1 else 2
MAXEDGE = int(os.environ.get("TILE_MAXEDGE", "1512"))
WORKERS = int(os.environ.get("TILE_WORKERS", "16"))
EVAL = os.environ.get("TILE_EVAL", os.path.join(REPO, "probe", "dev_eval2.csv"))
OUT_JSON = os.path.join(REPO, "probe", "tile_results.json")

# Per-tile question. Derived from rubric v1.0 but scoped to a CROP: the whole-slide prompt's
# "search the entire section" instruction is wrong for a tile and is replaced by an explicit
# statement that this is one piece of a larger section, so "no" here is not a verdict on the slide.
TILE_PROMPT = """You are shown ONE CROP taken from a larger Human Protein Atlas small-intestine section, at full resolution. Other crops of the same section are being judged separately, so your answer about this crop alone is not a verdict on the whole section.

Decide ONLY this: does THIS CROP contain at least one "qualifying stretch" of villus epithelium?

A qualifying stretch = a run of about 5 or more adjacent villus epithelial cells in a roughly linear arrangement, where both the apical surface (facing the lumen) and the basal surface (facing the underlying tissue) can be made out, and the staining can be read along it. Five cells is a LOW minimum, not a high bar — a short, modest, tilted or imperfect run still counts. Stain colour is irrelevant, and ABSENCE of staining is fine: an unstained but readable villus stretch still counts.

Answer no if this crop holds no qualifying villus stretch — for example it is empty background, submucosa or muscle only, entirely crypt cross-sections, glands, goblet-dominated regions, crypt-to-villus transition zones, or has no readable epithelium. A stretch that is cut off at the crop edge still counts, as long as about 5 adjacent cells are visible within the crop.

Answer with exactly one word: yes or no."""


def rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def tiles_of(path, grid):
    """Cut into grid x grid equal tiles at native resolution (left-to-right, top-to-bottom)."""
    im = Image.open(path).convert("RGB")
    w, h = im.size
    tw, th = w // grid, h // grid
    return [im.crop((c * tw, r * th, (c + 1) * tw if c < grid - 1 else w,
                     (r + 1) * th if r < grid - 1 else h))
            for r in range(grid) for c in range(grid)]


def main():
    ev = rows(EVAL)
    fns = [r["image_id"] for r in ev]
    truth = [r["label"] for r in ev]
    client, model = get_client()
    ntiles = GRID * GRID
    print(f"tiling {len(ev)} slides at {GRID}x{GRID} = {len(ev) * ntiles} tile calls "
          f"(tile cap {MAXEDGE}px, native crops from ~3000px slides)")

    jobs = [(i, t) for i in range(len(fns)) for t in range(ntiles)]

    def run(job):
        i, t = job
        try:
            tile = tiles_of(os.path.join(IMG_DIR, fns[i]), GRID)[t]
            parts = [text_part(TILE_PROMPT), img_part(encode_im(tile, MAXEDGE))]
            r = client.chat.completions.create(
                model=model, max_tokens=512, temperature=0,
                messages=[{"role": "user", "content": parts}])
            spend.add(model, r.usage)
            return parse_yes_no(r.choices[0].message.content)
        except Exception:
            return "ERR"

    out = list(cf.ThreadPoolExecutor(max_workers=WORKERS).map(run, jobs))
    per_tile = [[None] * ntiles for _ in fns]
    for (i, t), v in zip(jobs, out):
        per_tile[i][t] = v
    yes_counts = [sum(1 for v in row if v == "yes") for row in per_tile]
    errs = sum(1 for v in out if v == "ERR")
    print(f"tile answers: yes={out.count('yes')}  no={out.count('no')}  err={errs}  "
          f"unparsable={out.count('?')}")

    # Tag the output with the manifest: a bare grid-only name silently overwrote the previous
    # split's per-slide predictions and lost them.
    stem = os.path.splitext(os.path.basename(EVAL))[0]
    with open(os.path.join(REPO, "probe", f"tile_preds_{GRID}x{GRID}_{stem}.csv"),
              "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "true", "yes_count", "n_tiles"] + [f"t{t}" for t in range(ntiles)])
        for i, fn in enumerate(fns):
            w.writerow([fn, truth[i], yes_counts[i], ntiles] + per_tile[i])

    # Free operating-point sweep: every K scored from the same spend.
    allr = json.load(open(OUT_JSON, encoding="utf-8")) if os.path.exists(OUT_JSON) else {}
    best = None
    for k in range(1, ntiles + 1):
        preds = ["usable" if c >= k else "unusable" for c in yes_counts]
        m = score(truth, preds, f"tile {GRID}x{GRID} K>={k}")
        if m:
            m.update({"grid": GRID, "K": k, "maxedge": MAXEDGE, "model": model,
                      "manifest": f"{os.path.basename(EVAL)}({len(ev)})", "tile_errors": errs})
            allr[f"{GRID}x{GRID}_K{k}"] = m
            if best is None or m["accuracy"] > best["accuracy"]:
                best = m
    json.dump(allr, open(OUT_JSON, "w", encoding="utf-8"), indent=2)
    print(f"\nwrote {OUT_JSON}")
    if best:
        print(f"BEST for {GRID}x{GRID}: K>={best['K']}  acc {best['accuracy']:.3f}  "
              f"macro-F1 {best['macro_f1']:.3f}  unusable recall {best['unusable_recall']:.3f}  "
              f"usable recall {best['usable_recall']:.3f}")
    print("  zero-shot whole-slide base on the same 200: acc 0.840  macro-F1 0.840  "
          "unusable recall 0.850  usable recall 0.830")
    spend.report(f"tile-{GRID}x{GRID}", model)


if __name__ == "__main__":
    main()
