"""
mine_unusable_candidates.py - active-learning miner for MORE unusable slides.

The Stage-1 quality model's binding constraint is the thin unusable class (64
examples -> recall 0.59 +/- 0.28). Unusable slides are rare in HPA (~10-14%), so
labelling a random batch wastes the expert's effort (~9/10 come back usable).
This script flips that: it scores the large UNLABELLED pool (data_bulk, 3000
whole slides) with the trained model and hands the expert only the slides the
model thinks are most likely UNUSABLE, so the hit-rate jumps and each review
buys far more minority-class signal.

    python mine_unusable_candidates.py            # score pool, write shortlist
    python mine_unusable_candidates.py smoke      # first 120 slides only (sanity)

Method:
  - Ensemble the 5 donor-fold checkpoints (models/quality/quality_resnet50_fold*.pt);
    average the softmax so no single fold's quirks dominate. unusable_prob =
    mean P(unusable).
  - tissue_fraction (same rule as train_quality_cnn.py) as a second, independent
    signal - a near-blank slide is a cheap-to-spot unusable candidate.
  - Rank by unusable_prob (desc); surface tissue_fraction and a donor-novelty flag
    so the eventual retrain can stay donor-grouped and leakage-free.

Non-destructive: reads data_bulk/images/ and the model; writes only into mining/
and (the shortlist images) into the reviewer's served folder. Never deletes.

Outputs:
  mining/candidates_ranked_<date>.csv   full ranked pool (audit trail)
  review_tool/webapp/topup_set.txt      top-N image_ids, one per line (server hook)
  review_tool/webapp/review_images/     top-N images copied in (so the server can serve them)
"""

import os
import sys
import csv
import glob
import shutil
import datetime

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import timm

# ------------------------------------------------------------------
# CONFIG - edit these only
# ------------------------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
# Pool to score. Override for a second pool, e.g. the fresh clean-donor download:
#   MINE_POOL_DIR=data_topup/images python mine_unusable_candidates.py
POOL_DIR     = os.environ.get("MINE_POOL_DIR",
                              os.path.join(SCRIPT_DIR, "data_bulk", "images"))     # unlabelled pool
MODEL_GLOB   = os.path.join(SCRIPT_DIR, "models", "quality", "quality_resnet50_fold*.pt")
LABELLED_CSV = os.path.join(SCRIPT_DIR, "review_tool", "results",
                            "final_clean_dataset_20260728.csv")         # already-labelled donors
SERVED_DIR   = os.path.join(SCRIPT_DIR, "review_tool", "webapp", "review_images")  # what the server serves
TOPUP_TXT    = os.path.join(SCRIPT_DIR, "review_tool", "webapp", "topup_set.txt")  # server id-list hook
OUT_DIR      = os.path.join(SCRIPT_DIR, "mining")

SHORTLIST_N  = int(os.environ.get("MINE_SHORTLIST_N", "250"))  # top candidates for the expert (plan: ~150-300)
COPY_IMAGES  = True       # copy the shortlist into the served folder (needed for the reviewer)
BATCH        = 32         # inference batch (no grads -> larger than training is fine on 6 GB)

INPUT_SIZE   = 448        # MUST match the trained model's input_size
LUM_THRESHOLD = 200       # tissue_fraction rule (from train_quality_cnn.py / filter_blank_tiles.py)
SAT_THRESHOLD = 0.10
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASSES = ["unusable", "usable"]   # index 0 = unusable, index 1 = usable (matches training)

# 16 cancer-carrying donors (AUDIT.md D3) - never send these to the expert.
CANCER_DONORS = {"1069", "1090", "1839", "1879", "1904", "2338", "2467", "2806",
                 "3049", "3548", "603", "604", "637", "700", "707", "997"}
# ------------------------------------------------------------------


def donor_of(fname):
    """patient_id = last underscore token of the stem (filename join key)."""
    return os.path.splitext(fname)[0].split("_")[-1]


def labelled_ids_and_donors():
    """Return (set of already-labelled image_ids, set of donors in the trainset).
    Candidates already labelled are skipped; donor overlap is flagged, not skipped."""
    ids, donors = set(), set()
    if os.path.isfile(LABELLED_CSV):
        with open(LABELLED_CSV, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                ids.add(r["image_id"])
                donors.add(r.get("patient_id") or donor_of(r["image_id"]))
    # anything physically in the served folder has been (or is being) reviewed already
    if os.path.isdir(SERVED_DIR):
        for f in os.listdir(SERVED_DIR):
            if f.lower().endswith(".jpg"):
                ids.add(f)
    return ids, donors


def candidate_files(labelled_ids):
    """Pool images that are NEW (not labelled) and NOT from a cancer donor."""
    if not os.path.isdir(POOL_DIR):
        raise SystemExit(f"Pool not found: {POOL_DIR}")
    out = []
    for f in sorted(os.listdir(POOL_DIR)):
        if not f.lower().endswith(".jpg"):
            continue
        if f in labelled_ids:
            continue
        if donor_of(f) in CANCER_DONORS:
            continue
        out.append(f)
    return out


def tissue_fraction(arr):
    """Fraction of pixels that look like tissue (lum<200 OR sat>0.10), on the
    resized slide - a ratio, so scale-invariant enough for the whole-slide proxy."""
    a = arr.astype(np.float32)
    lum = a.mean(axis=2)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    is_tissue = (lum < LUM_THRESHOLD) | (sat > SAT_THRESHOLD)
    return float(is_tissue.mean())


def load_models(dev):
    """Load every fold checkpoint into an eval-mode ResNet-50 ensemble."""
    paths = sorted(glob.glob(MODEL_GLOB))
    if not paths:
        raise SystemExit(f"No model checkpoints match {MODEL_GLOB}")
    models = []
    for p in paths:
        ckpt = torch.load(p, map_location=dev)
        arch = ckpt.get("arch", "resnet50")
        m = timm.create_model(arch, pretrained=False, num_classes=len(ckpt.get("classes", CLASSES)))
        m.load_state_dict(ckpt["state_dict"])
        m.eval().to(dev)
        models.append(m)
        assert ckpt.get("input_size", INPUT_SIZE) == INPUT_SIZE, \
            f"{p} trained at {ckpt.get('input_size')}px != INPUT_SIZE {INPUT_SIZE}"
    print(f"Loaded {len(models)}-model ensemble from {os.path.dirname(paths[0])}")
    return models


def preload(files):
    """Resize each slide once to INPUT_SIZE (uint8 cache) + compute tissue_fraction."""
    imgs = np.zeros((len(files), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    tf = np.zeros(len(files), dtype=np.float32)
    for i, f in enumerate(files):
        with Image.open(os.path.join(POOL_DIR, f)) as im:
            a = np.asarray(im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR))
        imgs[i] = a
        tf[i] = tissue_fraction(a)
        if (i + 1) % 200 == 0:
            print(f"  preloaded {i + 1}/{len(files)} slides")
    return imgs, tf


@torch.no_grad()
def score(models, imgs, dev):
    """Ensemble-mean P(unusable) for every slide."""
    norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    unusable_idx = CLASSES.index("unusable")
    out = np.zeros(len(imgs), dtype=np.float32)
    for s in range(0, len(imgs), BATCH):
        chunk = imgs[s:s + BATCH]
        t = torch.from_numpy(np.ascontiguousarray(chunk).transpose(0, 3, 1, 2)).float() / 255.0
        t = norm(t).to(dev, non_blocking=True)
        probs = None
        with torch.autocast("cuda", enabled=(dev.type == "cuda")):
            for m in models:
                p = F.softmax(m(t).float(), dim=1)
                probs = p if probs is None else probs + p
        probs = (probs / len(models)).cpu().numpy()
        out[s:s + len(chunk)] = probs[:, unusable_idx]
        if (s // BATCH) % 10 == 0:
            print(f"  scored {min(s + BATCH, len(imgs))}/{len(imgs)}")
    return out


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    # smoke = pipeline sanity only: never touch the served folder or the real
    # topup_set.txt (write to _smoke sidecars instead).
    copy_images = COPY_IMAGES and not smoke
    topup_txt = TOPUP_TXT if not smoke else TOPUP_TXT.replace(".txt", "_smoke.txt")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}"
          + (f" ({torch.cuda.get_device_name(0)})" if dev.type == "cuda" else " (CPU - slow)"))

    labelled_ids, train_donors = labelled_ids_and_donors()
    files = candidate_files(labelled_ids)
    if smoke:
        files = files[:120]
    print(f"pool={POOL_DIR}")
    print(f"candidates={len(files)} (excluded {len(labelled_ids)} already-labelled ids "
          f"+ {len(CANCER_DONORS)} cancer donors); trainset donors={len(train_donors)}")
    if not files:
        raise SystemExit("No candidates to score.")

    models = load_models(dev)
    imgs, tf = preload(files)
    unusable_prob = score(models, imgs, dev)

    # rank by model unusable_prob (desc); tissue_fraction + donor novelty as context
    order = np.argsort(-unusable_prob)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d")
    pool_tag = os.path.basename(os.path.dirname(POOL_DIR)) or "pool"   # e.g. data_bulk / data_topup
    ranked_csv = os.path.join(OUT_DIR, f"candidates_ranked_{pool_tag}_{stamp}{'_smoke' if smoke else ''}.csv")
    with open(ranked_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "image_id", "unusable_prob", "tissue_fraction",
                    "patient_id", "donor_in_trainset"])
        for rank, i in enumerate(order, 1):
            f = files[i]
            d = donor_of(f)
            w.writerow([rank, f, round(float(unusable_prob[i]), 4),
                        round(float(tf[i]), 4), d,
                        "yes" if d in train_donors else "no"])

    # top-N shortlist for the reviewer. ACCUMULATE across pools: union with any
    # existing topup_set so a second (fresh-donor) run adds to the first, never
    # clobbers it. Existing ids kept in order; new ones appended.
    shortlist = [files[i] for i in order[:SHORTLIST_N]]
    existing = []
    if os.path.exists(topup_txt):
        with open(topup_txt, encoding="utf-8") as fh:
            existing = [ln.strip() for ln in fh if ln.strip()]
    seen = set(existing)
    added_ids = [f for f in shortlist if f not in seen]
    merged = existing + added_ids
    with open(topup_txt, "w", encoding="utf-8") as fh:
        fh.write("\n".join(merged) + "\n")

    copied = already = 0
    if copy_images:
        os.makedirs(SERVED_DIR, exist_ok=True)
        for f in shortlist:
            dst = os.path.join(SERVED_DIR, f)
            if os.path.exists(dst):
                already += 1
            else:
                shutil.copy2(os.path.join(POOL_DIR, f), dst)
                copied += 1

    top = order[:SHORTLIST_N]
    new_donor = sum(1 for i in top if donor_of(files[i]) not in train_donors)
    n_new_donors = len({donor_of(files[i]) for i in top if donor_of(files[i]) not in train_donors})
    print("\n================ MINING SUMMARY ================")
    print(f"scored {len(files)} candidates; shortlist = top {len(shortlist)}")
    print(f"shortlist unusable_prob range: {unusable_prob[top].min():.3f} - {unusable_prob[top].max():.3f}"
          f"  (median {np.median(unusable_prob[top]):.3f})")
    print(f"shortlist from NEW donors (not in trainset): {new_donor}/{len(shortlist)} "
          f"across {n_new_donors} new donors")
    print(f"topup_set.txt: +{len(added_ids)} new ids this run -> {len(merged)} total to review")
    if copy_images:
        print(f"copied {copied} images into {os.path.relpath(SERVED_DIR, SCRIPT_DIR)} "
              f"({already} already present)")
    else:
        print("image copy SKIPPED (smoke) - served folder untouched")
    print(f"\nwrote:\n  {os.path.relpath(ranked_csv, SCRIPT_DIR)}"
          f"\n  {os.path.relpath(topup_txt, SCRIPT_DIR)}  (server id-list hook)")
    print("\nNext: enable the 'slice_quality_topup' profile and redeploy so the "
          "expert reviews only this shortlist.")


if __name__ == "__main__":
    main()
