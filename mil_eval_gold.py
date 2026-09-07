"""Stage-2 localisation evaluation against the expert's gold tile judgments (scope 05, criterion c).

Reads the MIL per-tile scores (mil_tile_scores.csv) and judges localisation on the 20 gold
slides: the expert's 834 per-tile yes/no + 446 auto-no blanks = 1,280 tiles (coverage asserted).
Reviewer filter is hard-coded to the expert — the 6 local test judgments by the author are excluded.

Metrics per scorer (Additive contribution = primary; attention and tissue_fraction = baselines;
random = analytic): pointing game (argmax tile is an expert yes), top-k hit (k=1,3,5), per-slide
ranking AUC. Slide-clustered bootstrap (10k resamples of the 20 slides) for CIs. Also, if any
gold slide has silver 3x3 VLM answers (probe/tile_preds_3x3*.csv), reports the 8x8->3x3
block-pooled agreement as secondary evidence. Pure numpy/csv — runs anywhere. No training here:
the expert's tiles never touch a model.

  python mil_eval_gold.py            # expects models/mil/mil_tile_scores.csv
Env: MIL_OUT_DIR, MIL_SCORE_COL (contribution|attention primary column, default contribution).
Output: models/mil/mil_gold_eval.json + printed table.
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.environ.get("MIL_OUT_DIR", os.path.join(SCRIPT_DIR, "models", "mil"))
GOLD = os.path.join(SCRIPT_DIR, "review_tool", "results", "loc_stretch_export_20260827.csv")
AUTO_NO = os.path.join(SCRIPT_DIR, "review_tool", "loc_tile_auto_no.csv")
MANIFEST = os.path.join(SCRIPT_DIR, "review_tool", "webapp", "loc_tile_manifest.csv")
TISSUE = os.path.join(SCRIPT_DIR, "review_tool", "loc_tile_tissue_fraction.csv")
GRID = 8


def read(path):
    return list(csv.DictReader(open(path, newline="", encoding="utf-8")))


def rc_of(tile_id):
    tag = tile_id.rsplit("__", 1)[1].split(".")[0]           # r{r}c{c}
    r, c = tag[1:].split("c")
    return int(r), int(c)


def slide_of(tile_id):
    return tile_id.rsplit("__", 1)[0] + ".jpg"


def topk_hit(scores, yes, k):
    return int(bool(set(np.argsort(-scores)[:k]) & yes))


def rank_auc(scores, yes_mask):
    pos, neg = scores[yes_mask], scores[~yes_mask]
    if not len(pos) or not len(neg):
        return None
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def main():
    # ---- gold labels: expert only + auto-no, coverage asserted ----
    gold = {}
    for r in read(GOLD):
        if r["reviewer"] == "expert":
            gold[r["image_id"]] = 1 if r["label"] == "yes" else 0
    n_expert = len(gold)
    for r in read(AUTO_NO):
        assert r["tile_id"] not in gold, f"auto-no overlaps expert: {r['tile_id']}"
        gold[r["tile_id"]] = 0
    slides = sorted({slide_of(t) for t in gold})
    assert n_expert == 834 and len(gold) == 1280 and len(slides) == 20, \
        f"gold coverage {n_expert}/{len(gold)}/{len(slides)} != 834/1280/20"

    tissue = {r["tile_id"]: float(r["tissue_fraction"]) for r in read(TISSUE)}

    by_slide = defaultdict(dict)
    for r in read(os.path.join(OUT_DIR, "mil_tile_scores.csv")):
        by_slide[r["image_id"]][(int(r["row"]), int(r["col"]))] = (
            float(r["contribution"]), float(r["attention"]))
    missing = [s for s in slides if len(by_slide.get(s, {})) != GRID * GRID]
    assert not missing, f"MIL scores missing for gold slides: {missing[:3]}"

    scorers = {"contribution": None, "attention": None, "tissue_fraction": None}
    res = {name: defaultdict(list) for name in scorers}
    rand = defaultdict(list)
    per_slide_yes = []
    for s in slides:
        y = np.zeros(GRID * GRID)
        tis = np.zeros(GRID * GRID)
        con = np.zeros(GRID * GRID)
        att = np.zeros(GRID * GRID)
        for (r, c), (cv, av) in by_slide[s].items():
            con[r * GRID + c], att[r * GRID + c] = cv, av
        for t, lab in gold.items():
            if slide_of(t) == s:
                r, c = rc_of(t)
                y[r * GRID + c] = lab
                tis[r * GRID + c] = tissue.get(t, 0.0)
        yes = set(np.where(y == 1)[0])
        assert yes, f"gold slide {s} has zero yes tiles"
        per_slide_yes.append(len(yes))
        n = GRID * GRID
        for name, sc in [("contribution", con), ("attention", att), ("tissue_fraction", tis)]:
            for k in (1, 3, 5):
                res[name][f"top{k}"].append(topk_hit(sc, yes, k))
            res[name]["auc"].append(rank_auc(sc, y == 1))
        ny = len(yes)
        rand["top1"].append(ny / n)
        for k in (3, 5):                                     # P(at least one yes in random k)
            miss = 1.0
            for j in range(k):
                miss *= (n - ny - j) / (n - j)
            rand[f"top{k}"].append(1 - miss)
        rand["auc"].append(0.5)

    rng = np.random.default_rng(42)
    def boot_ci(vals):
        vals = np.array([v for v in vals if v is not None], dtype=float)
        m = rng.choice(len(vals), size=(10000, len(vals))).astype(int)
        means = vals[m].mean(axis=1)
        return [round(float(np.percentile(means, q)), 3) for q in (2.5, 97.5)]

    out = {"n_slides": len(slides), "n_gold_tiles": len(gold),
           "yes_tiles_per_slide": {"min": min(per_slide_yes), "median": int(np.median(per_slide_yes)),
                                   "max": max(per_slide_yes)},
           "random_baseline": {m: round(float(np.mean(v)), 3) for m, v in rand.items()},
           "scorers": {}}
    for name in scorers:
        out["scorers"][name] = {
            m: {"mean": round(float(np.mean([x for x in v if x is not None])), 3),
                "ci95": boot_ci(v)}
            for m, v in res[name].items()}
    c = out["scorers"]["contribution"]
    out["success_criteria"] = {
        "pointing_ge_0.60": c["top1"]["mean"] >= 0.60,
        "pointing_ge_random_plus_0.20": c["top1"]["mean"] >= out["random_baseline"]["top1"] + 0.20,
        "beats_tissue_fraction": c["top1"]["mean"] > out["scorers"]["tissue_fraction"]["top1"]["mean"],
    }

    # ---- silver cross-check (secondary): 8x8 -> 3x3 block max vs VLM per-tile answers ----
    silver = {}
    for p in glob.glob(os.path.join(SCRIPT_DIR, "probe", "tile_preds_3x3*.csv")):
        for r in read(p):
            silver[r["image_id"]] = [r[f"t{i}"] == "yes" for i in range(9)]
    ov = [s for s in slides if s in silver]
    if ov:
        hits = []
        for s in ov:
            con = np.zeros((GRID, GRID))
            for (r, c), (cv, _) in by_slide[s].items():
                con[r, c] = cv
            block = np.zeros(9)
            for r in range(GRID):
                for c in range(GRID):
                    b = (r * 3 // GRID) * 3 + (c * 3 // GRID)
                    block[b] = max(block[b], con[r, c])
            yes = {i for i, v in enumerate(silver[s]) if v}
            hits.append(int(np.argmax(block) in yes) if yes else None)
        hv = [h for h in hits if h is not None]
        out["silver_3x3"] = {"n_overlap": len(ov), "n_scored": len(hv),
                             "pointing": round(float(np.mean(hv)), 3) if hv else None}

    json.dump(out, open(os.path.join(OUT_DIR, "mil_gold_eval.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    print(f"wrote {os.path.join(OUT_DIR, 'mil_gold_eval.json')}")


if __name__ == "__main__":
    main()
