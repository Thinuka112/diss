"""Stage-2 MIL feature extraction: fold-matched per-tile embeddings (scope 05).

For each Stage-1 fold k, embeds EVERY slide in the input CSV with that fold's 448px encoder
(models/quality/quality_resnet50_fold{k}.pt). Fold-matched means MIL fold k trains AND tests
purely on encoder-k embeddings — encoder k never saw fold k's donors, so the held-out test stays
clean, and train/test share one feature space (no cross-encoder shift).

Tiles: 8x8 grid, tile_label.tiles_of convention (floor division, remainder into the last
row/col, row-major), each tile resized to 448 and ImageNet-normalised.

  MIL_CSV=review_tool/results/final_clean_dataset_20260802.csv \
  MIL_EMB_DIR=models/mil/emb_human MIL_FOLDS="1 2 3 4 5" python mil_features.py

Env: MIL_CSV, MIL_EMB_DIR, MIL_IMG_DIRS (colon list), MIL_FOLDS (subset for parallelism),
     MIL_BATCH (default 96), MIL_FOLDS_JSON, MIL_CKPT_DIR.
Output per fold: {MIL_EMB_DIR}/fold{k}.npz  ids[N], emb[N,64,2048] float16, label[N],
donor[N], label_source[N], usable_frac[N].
"""
import csv
import json
import os
import sys

import numpy as np
import timm
import torch
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.environ.get("MIL_CSV", os.path.join(
    SCRIPT_DIR, "review_tool", "results", "final_clean_dataset_20260802.csv"))
EMB_DIR = os.environ.get("MIL_EMB_DIR", os.path.join(SCRIPT_DIR, "models", "mil", "emb_human"))
IMG_DIRS = os.environ.get("MIL_IMG_DIRS", os.path.join(
    SCRIPT_DIR, "review_tool", "webapp", "review_images")).split(":")
FOLDS = [int(x) for x in os.environ.get("MIL_FOLDS", "1 2 3 4 5").split()]
BATCH = int(os.environ.get("MIL_BATCH", "96"))
FOLDS_JSON = os.environ.get("MIL_FOLDS_JSON", os.path.join(
    SCRIPT_DIR, "models", "quality", "quality_metrics.json"))
CKPT_DIR = os.environ.get("MIL_CKPT_DIR", os.path.join(SCRIPT_DIR, "models", "quality"))
GRID, TILE_SIZE = 8, 448
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def tiles_of(im, grid=GRID):
    """tile_label.py convention verbatim: equal tiles, remainder into last row/col, row-major."""
    w, h = im.size
    tw, th = w // grid, h // grid
    return [im.crop((c * tw, r * th, (c + 1) * tw if c < grid - 1 else w,
                     (r + 1) * th if r < grid - 1 else h))
            for r in range(grid) for c in range(grid)]


def resolve(img_id):
    for d in IMG_DIRS:
        p = os.path.join(d, img_id)
        if os.path.exists(p):
            return p
    return None


def slide_tensor(path):
    im = Image.open(path).convert("RGB")
    ts = []
    for t in tiles_of(im):
        t = t.resize((TILE_SIZE, TILE_SIZE), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(t, dtype=np.uint8)).permute(2, 0, 1).float() / 255.0
        ts.append((x - MEAN) / STD)
    return torch.stack(ts)                                    # (64,3,448,448)


def main():
    rows = list(csv.DictReader(open(CSV_PATH, newline="", encoding="utf-8")))
    ids = [r["image_id"] for r in rows]
    miss = [i for i in ids if resolve(i) is None]
    assert not miss, f"{len(miss)} images unresolved, e.g. {miss[:3]}"
    per_fold = json.load(open(FOLDS_JSON, encoding="utf-8"))["per_fold"]
    assert [f["fold"] for f in per_fold] == [1, 2, 3, 4, 5], "unexpected fold ids"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(EMB_DIR, exist_ok=True)
    print(f"slides={len(rows)} dirs={IMG_DIRS} folds={FOLDS} device={dev} -> {EMB_DIR}")

    for k in FOLDS:
        out = os.path.join(EMB_DIR, f"fold{k}.npz")
        if os.path.exists(out):
            print(f"fold{k}: exists, skip")
            continue
        ck = os.path.join(CKPT_DIR, f"quality_resnet50_fold{k}.pt")
        blob = torch.load(ck, map_location="cpu", weights_only=True)
        sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
        if isinstance(blob, dict) and "fold" in blob:
            assert int(blob["fold"]) == k, f"checkpoint fold {blob['fold']} != {k}"
        model = timm.create_model("resnet50", num_classes=2)
        model.load_state_dict(sd)
        model.eval().to(dev)
        emb = np.empty((len(rows), GRID * GRID, 2048), dtype=np.float16)
        with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
            for i, img_id in enumerate(ids):
                x = slide_tensor(resolve(img_id))
                fs = []
                for b in range(0, x.shape[0], BATCH):
                    f = model.forward_head(model.forward_features(x[b:b + BATCH].to(dev)),
                                           pre_logits=True)
                    fs.append(f.float().cpu())
                emb[i] = torch.cat(fs).numpy().astype(np.float16)
                if (i + 1) % 200 == 0:
                    print(f"fold{k}: {i + 1}/{len(rows)}", flush=True)
        np.savez_compressed(
            out, ids=np.array(ids), emb=emb,
            label=np.array([r["label"] for r in rows]),
            donor=np.array([r.get("patient_id", "") for r in rows]),
            label_source=np.array([r.get("label_source", "human") for r in rows]),
            usable_frac=np.array([float(r.get("usable_frac", "") or
                                        (1.0 if r["label"] == "usable" else 0.0))
                                  for r in rows], dtype=np.float32))
        print(f"fold{k}: wrote {out} ({os.path.getsize(out) / 1e9:.2f} GB)", flush=True)
    print("features done")


if __name__ == "__main__":
    sys.exit(main())
