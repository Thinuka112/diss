"""
train_cnn.py - CNN classifier for tile tasks, on GPU, with a leakage-safe split.

    python train_cnn.py usable        # primary: usable vs unusable (weak labels)
    python train_cnn.py expression    # secondary: HPA expression level (4-class)

Design (addresses the audit):
  - Model: ResNet-18 (ImageNet-pretrained, fine-tuned) - a real CNN, GPU/CUDA.
  - Split: 60/20/20 train/val/test, GROUPED BY SLIDE (source_image never spans
    splits). Seed fixed. Model selected on val macro-F1; reported on the unseen
    TEST slides only.
  - Baseline reported alongside (majority class, and for `usable` the
    tissue_fraction rule) so the CNN is judged against what it must beat.

Honesty notes baked in:
  - `usable` labels are the tissue_fraction heuristic (no human labels yet), so a
    high score = a learned blank/tissue detector, NOT a validated quality model.
  - `expression` is a genuine test with real (staining) signal; its number is
    whatever it is - reported straight, win or lose.

Reads a tile manifest (default HPA_small_intestine_tiles/manifest_labels.csv) +
image_metadata.csv. Set MANIFEST/META to a larger set to scale up. Tiles are
preloaded to RAM (fine for <~150k tiles). Writes models/cnn_<task>_*.
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
TILES_DIR = os.path.join(SCRIPT_DIR, "HPA_small_intestine_tiles")
MANIFEST = os.path.join(TILES_DIR, "manifest_labels.csv")
META = os.path.join(SCRIPT_DIR, "image_metadata.csv")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")

INPUT_SIZE = 96
BATCH = 128
EPOCHS = 12
LR = 1e-4
SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
LEVELS = ["not detected", "low", "medium", "high"]


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def build_rows(task):
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if task == "usable":
        classes = ["unusable", "usable"]
        out = []
        for r in rows:
            f = float(r["tissue_fraction"])
            y = 0 if f < 0.05 else (1 if f >= 0.5 else -1)
            if y != -1:
                out.append((r["tile_path"], r["source_image"], y))
        return out, classes
    elif task == "expression":
        meta = {}
        with open(META, newline="", encoding="utf-8") as fh:
            for m in csv.DictReader(fh):
                meta[m["source_image"]] = m["si_expression_level"]
        idx = {l: i for i, l in enumerate(LEVELS)}
        out = []
        for r in rows:
            lvl = meta.get(r["source_image"], "")
            if lvl in idx:
                out.append((r["tile_path"], r["source_image"], idx[lvl]))
        return out, LEVELS
    raise SystemExit("task must be 'usable' or 'expression'")


def preload(rows):
    imgs = np.zeros((len(rows), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    for i, (tp, _, _) in enumerate(rows):
        full = tp if os.path.isabs(tp) else os.path.join(SCRIPT_DIR, tp)
        with Image.open(full) as im:
            imgs[i] = np.asarray(im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE)))
        if (i + 1) % 2000 == 0:
            print(f"  preloaded {i + 1}/{len(rows)} tiles")
    return imgs


class TileDS(Dataset):
    def __init__(self, imgs, ys, train):
        self.imgs, self.ys, self.train = imgs, ys, train
        self.norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        a = self.imgs[i]
        if self.train:
            if np.random.rand() < 0.5:
                a = a[:, ::-1]
            if np.random.rand() < 0.5:
                a = a[::-1, :]
            a = np.rot90(a, np.random.randint(4))
        t = torch.from_numpy(np.ascontiguousarray(a).transpose(2, 0, 1)).float() / 255.0
        return self.norm(t), int(self.ys[i])


def split_60_20_20(groups, y):
    g = np.array(groups)
    tv, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(g, y, g))
    tr, va = next(GroupShuffleSplit(1, test_size=0.25, random_state=SEED).split(
        g[tv], np.array(y)[tv], g[tv]))
    return tv[tr], tv[va], te


def run_epoch(model, loader, crit, opt, dev, train):
    model.train() if train else model.eval()
    tot, loss_sum, preds, gts = 0, 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            out = model(x)
            loss = crit(out, y)
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item() * len(y); tot += len(y)
            preds.append(out.argmax(1).cpu().numpy()); gts.append(y.cpu().numpy())
    p, g = np.concatenate(preds), np.concatenate(gts)
    return loss_sum / tot, accuracy_score(g, p), f1_score(g, p, average="macro"), p, g


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "usable"
    set_seed(SEED)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, classes = build_rows(task)
    ys = np.array([r[2] for r in rows]); groups = [r[1] for r in rows]
    print(f"task={task}  device={dev}  tiles={len(rows)}  classes={classes}")
    print(f"class counts: {dict(collections.Counter(ys.tolist()))}  "
          f"slides={len(set(groups))}")

    imgs = preload(rows)
    tr, va, te = split_60_20_20(groups, ys)
    print(f"train {len(tr)} / val {len(va)} / test {len(te)} tiles | "
          f"test slides={len(set(np.array(groups)[te]))} (unseen)")

    # num_workers=0: tiles are already in RAM, and Windows 'spawn' would copy the
    # whole array into every worker. In-process loading is fast enough here.
    dl = lambda idx, train: DataLoader(
        TileDS(imgs[idx], ys[idx], train), batch_size=BATCH, shuffle=train,
        num_workers=0, pin_memory=True)
    tl, vl, tel = dl(tr, True), dl(va, False), dl(te, False)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model = model.to(dev)

    cnt = collections.Counter(ys[tr].tolist())
    w = torch.tensor([len(tr) / (len(classes) * cnt.get(c, 1)) for c in range(len(classes))],
                     dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1, best_state = -1, None
    for ep in range(1, EPOCHS + 1):
        trl, tra, trf, *_ = run_epoch(model, tl, crit, opt, dev, True)
        vll, vla, vlf, *_ = run_epoch(model, vl, crit, opt, dev, False)
        print(f"epoch {ep:2d}  train loss {trl:.3f} f1 {trf:.3f} | "
              f"val loss {vll:.3f} acc {vla:.3f} f1 {vlf:.3f}")
        if vlf > best_f1:
            best_f1 = vlf
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _, te_acc, te_f1, te_pred, te_gt = run_epoch(model, tel, crit, opt, dev, False)
    maj = collections.Counter(ys[tr].tolist()).most_common(1)[0][0]
    maj_acc = accuracy_score(te_gt, np.full(len(te_gt), maj))
    cm = confusion_matrix(te_gt, te_pred, labels=list(range(len(classes))))

    print(f"\n=== TEST (unseen slides) task={task} ===")
    print(f"accuracy {te_acc:.4f}  macro-F1 {te_f1:.4f}  (majority baseline {maj_acc:.4f}, "
          f"lift {te_acc - maj_acc:+.4f})")
    print("confusion matrix (rows=true, cols=pred):"); print(cm)

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({"state_dict": best_state, "classes": classes, "input_size": INPUT_SIZE,
                "arch": "resnet18"}, os.path.join(MODEL_DIR, f"cnn_{task}.pt"))
    metrics = {"task": task, "arch": "resnet18 (ImageNet-pretrained, fine-tuned)",
               "device": str(dev), "n_tiles": len(rows), "n_slides": len(set(groups)),
               "split": "60/20/20 grouped by slide, seed 42",
               "classes": classes, "test_accuracy": round(float(te_acc), 4),
               "test_macro_f1": round(float(te_f1), 4),
               "majority_baseline": round(float(maj_acc), 4),
               "confusion_matrix": cm.tolist(),
               "note": ("usable labels are the tissue_fraction heuristic (no human "
                        "labels yet) - a high score is a learned blank detector, not a "
                        "validated quality model." if task == "usable" else
                        "expression is a genuine signal test; number reported straight.")}
    with open(os.path.join(MODEL_DIR, f"cnn_{task}_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "#222", fontsize=9)
    ax.set_title(f"CNN {task} — unseen slides\nacc {te_acc:.3f} (base {maj_acc:.3f})", fontsize=10)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    fig.tight_layout(); fig.savefig(os.path.join(MODEL_DIR, f"cnn_{task}_confusion.png"), dpi=130)
    print(f"saved models/cnn_{task}.pt, cnn_{task}_metrics.json, cnn_{task}_confusion.png")


if __name__ == "__main__":
    main()
