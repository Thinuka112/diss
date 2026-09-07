"""
train_cnn_bulk.py - scaled CNN trainer over the full image set (sample + bulk).

    python train_cnn_bulk.py usable        # primary, at scale (~3k slides)
    python train_cnn_bulk.py expression    # needs bulk_metadata.csv

Scale trick: instead of writing ~200k tile files, each SOURCE image is decoded
once, downscaled to SRC_SIZE, and cached in a RAM uint8 array. Tiles are cropped
on the fly (in-RAM, cheap). Per-tile tissue_fraction is computed from the cached
image, so the `usable` label is self-consistent.

Everything else matches train_cnn.py: ResNet-18 (pretrained), CUDA, 60/20/20
GROUPED-BY-SLIDE split (seed 42), model chosen on val, reported on unseen test.
Same honesty note: `usable` labels are the tissue_fraction heuristic.
"""

import os
import sys
import csv
import json
import collections

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIRS = [os.path.join(SCRIPT_DIR, "HPA_small_intestine"),
              os.path.join(SCRIPT_DIR, "data_bulk", "images")]
META = os.path.join(SCRIPT_DIR, "data_bulk", "bulk_metadata.csv")   # for expression
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

MAX_IMAGES = None          # cap for speed/RAM; None = all
SRC_SIZE = 512             # cache each source image at this resolution
COLS = ROWS = 8
INPUT_SIZE = 96
BATCH = 256
EPOCHS = 10
LR = 1e-4
SEED = 42
LUM_T, SAT_T, USABLE_MAX, USABLE_MIN = 200, 0.10, 0.05, 0.50
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LEVELS = ["not detected", "low", "medium", "high"]
SEX = ("Male", "Female", "Unknown")


def gene_of(fname):
    toks = os.path.splitext(fname)[0].split("_")
    for i, t in enumerate(toks):
        if t in SEX and i > 0:
            return toks[i - 1]
    return "Unknown"


def tissue_fraction(tile):
    a = tile.astype(np.float32)
    lum = a.mean(2); mx = a.max(2); mn = a.min(2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    return float(((lum < LUM_T) | (sat > SAT_T)).mean())


def gather_images():
    paths = []
    for d in IMAGE_DIRS:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(".jpg"):
                    paths.append((os.path.join(d, f), f))
    # de-dup by filename (sample vs bulk shouldn't overlap, but be safe)
    seen, uniq = set(), []
    for p, f in paths:
        if f not in seen:
            seen.add(f); uniq.append((p, f))
    if MAX_IMAGES:
        uniq = uniq[:MAX_IMAGES]
    return uniq


def build(task):
    imgs_meta = gather_images()
    n = len(imgs_meta)
    print(f"caching {n} source images at {SRC_SIZE}px...")
    cache = np.zeros((n, SRC_SIZE, SRC_SIZE, 3), dtype=np.uint8)
    genes, sources = [], []
    for i, (p, f) in enumerate(imgs_meta):
        with Image.open(p) as im:
            cache[i] = np.asarray(im.convert("RGB").resize((SRC_SIZE, SRC_SIZE)))
        genes.append(gene_of(f)); sources.append(f)
        if (i + 1) % 500 == 0:
            print(f"  cached {i + 1}/{n}")

    tw = SRC_SIZE // COLS
    expr = {}
    if task == "expression":
        with open(META, newline="", encoding="utf-8") as fh:
            for m in csv.DictReader(fh):
                expr[m["source_image"]] = m["si_expression_level"]
        classes = LEVELS
    else:
        classes = ["unusable", "usable"]

    items = []   # (img_idx, r, c, label)
    lvl_idx = {l: i for i, l in enumerate(LEVELS)}
    for i in range(n):
        if task == "expression":
            lv = expr.get(sources[i], "")
            if lv not in lvl_idx:
                continue
            y = lvl_idx[lv]
            for r in range(ROWS):
                for c in range(COLS):
                    items.append((i, r, c, y))
        else:
            for r in range(ROWS):
                for c in range(COLS):
                    tile = cache[i, r * tw:(r + 1) * tw, c * tw:(c + 1) * tw]
                    f = tissue_fraction(tile)
                    y = 0 if f < USABLE_MAX else (1 if f >= USABLE_MIN else -1)
                    if y != -1:
                        items.append((i, r, c, y))
    return cache, tw, items, genes, sources, classes


class BulkDS(Dataset):
    def __init__(self, cache, tw, items, train):
        self.cache, self.tw, self.items, self.train = cache, tw, items, train
        self.norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, r, c, y = self.items[k]
        tw = self.tw
        a = self.cache[i, r * tw:(r + 1) * tw, c * tw:(c + 1) * tw]
        a = np.asarray(Image.fromarray(a).resize((INPUT_SIZE, INPUT_SIZE)))
        if self.train:
            if np.random.rand() < 0.5:
                a = a[:, ::-1]
            if np.random.rand() < 0.5:
                a = a[::-1, :]
            a = np.rot90(a, np.random.randint(4))
        t = torch.from_numpy(np.ascontiguousarray(a).transpose(2, 0, 1)).float() / 255.0
        return self.norm(t), int(y)


def run_epoch(model, loader, crit, opt, dev, train):
    model.train() if train else model.eval()
    tot, ls, ps, gs = 0, 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            out = model(x); loss = crit(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            ls += loss.item() * len(y); tot += len(y)
            ps.append(out.argmax(1).cpu().numpy()); gs.append(y.cpu().numpy())
    return ls / tot, np.concatenate(ps), np.concatenate(gs)


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "usable"
    np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache, tw, items, genes, sources, classes = build(task)
    ys = np.array([it[3] for it in items])
    grp = np.array([sources[it[0]] for it in items])
    print(f"task={task} device={dev} tiles={len(items)} slides={len(set(grp))} "
          f"classes={classes} counts={dict(collections.Counter(ys.tolist()))}")

    tv, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(grp, ys, grp))
    tr, va = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED).split(
        grp[tv], ys[tv], grp[tv]))
    tr, va = tv[tr], tv[va]
    print(f"train {len(tr)} / val {len(va)} / test {len(te)} tiles | "
          f"test slides {len(set(grp[te]))} (unseen)")

    it = np.array(items, dtype=object)
    dl = lambda idx, s: DataLoader(BulkDS(cache, tw, [items[j] for j in idx], s),
                                   batch_size=BATCH, shuffle=s, num_workers=0, pin_memory=True)
    tl, vl, tel = dl(tr, True), dl(va, False), dl(te, False)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(dev)
    cnt = collections.Counter(ys[tr].tolist())
    w = torch.tensor([len(tr) / (len(classes) * cnt.get(c, 1)) for c in range(len(classes))],
                     dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1, best = -1, None
    for ep in range(1, EPOCHS + 1):
        trl, trp, trg = run_epoch(model, tl, crit, opt, dev, True)
        vll, vp, vg = run_epoch(model, vl, crit, opt, dev, False)
        vf = f1_score(vg, vp, average="macro")
        print(f"epoch {ep:2d} train loss {trl:.3f} f1 {f1_score(trg,trp,average='macro'):.3f} | "
              f"val loss {vll:.3f} acc {accuracy_score(vg,vp):.3f} f1 {vf:.3f}")
        if vf > best_f1:
            best_f1 = vf; best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best)
    _, tp, tg = run_epoch(model, tel, crit, opt, dev, False)
    acc = accuracy_score(tg, tp); f1 = f1_score(tg, tp, average="macro")
    maj = collections.Counter(ys[tr].tolist()).most_common(1)[0][0]
    maj_acc = accuracy_score(tg, np.full(len(tg), maj))
    cm = confusion_matrix(tg, tp, labels=list(range(len(classes))))
    print(f"\n=== TEST (unseen slides) task={task} @scale ===")
    print(f"accuracy {acc:.4f}  macro-F1 {f1:.4f}  (majority {maj_acc:.4f}, lift {acc-maj_acc:+.4f})")
    print("confusion (rows=true, cols=pred):"); print(cm)

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({"state_dict": best, "classes": classes, "input_size": INPUT_SIZE, "arch": "resnet18"},
               os.path.join(MODEL_DIR, f"cnn_bulk_{task}.pt"))
    out = {"task": task, "scale": "bulk", "arch": "resnet18 pretrained",
           "n_images": len(set(grp)), "n_tiles": len(items),
           "split": "60/20/20 grouped by slide, seed 42", "classes": classes,
           "test_accuracy": round(float(acc), 4), "test_macro_f1": round(float(f1), 4),
           "majority_baseline": round(float(maj_acc), 4), "confusion_matrix": cm.tolist(),
           "note": ("usable labels are the tissue_fraction heuristic - learned blank detector"
                    if task == "usable" else "expression: real signal test, reported straight")}
    with open(os.path.join(MODEL_DIR, f"cnn_bulk_{task}_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    fig, ax = plt.subplots(figsize=(4.6, 4)); ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "#222", fontsize=8)
    ax.set_title(f"CNN {task} @scale ({len(set(grp))} slides)\nunseen acc {acc:.3f} "
                 f"(base {maj_acc:.3f})", fontsize=9)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); fig.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, f"cnn_bulk_{task}_confusion.png"), dpi=130)
    print(f"saved models/cnn_bulk_{task}.*")


if __name__ == "__main__":
    main()
