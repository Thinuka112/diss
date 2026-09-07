"""
train_quality_cnn.py - the real Stage-1 slice-quality classifier (usable vs
unusable) on WHOLE SLIDES, trained on the human expert labels.

    python train_quality_cnn.py            # full 5-fold donor-grouped CV
    python train_quality_cnn.py smoke      # 1 fold, 2 epochs (data/OOM sanity)

This is the first model trained on real human quality labels. The earlier
train_cnn.py / train_cnn_bulk.py pilots used the tissue_fraction heuristic on
96px tiles and are a learned blank detector, NOT this. Here the unit of
prediction is the whole 3000x3000 slide (matching the expert's whole-slide
judgement), downscaled to a square input.

Design (from research/scoping/03-classifier.md, finalised 2026-07-28):
  - Input: whole slide resized to INPUT_SIZE square (default 448).
  - Model: ResNet-50 via timm (ImageNet-pretrained, fine-tuned), num_classes=2.
  - Split: donor-grouped 5-fold CV (sklearn GroupKFold by patient_id). A donor
    never appears in two folds. Each fold is held out as test; a donor-grouped
    val slice is carved from the other 4 folds for early stopping / selection.
  - Imbalance: class-weighted CrossEntropyLoss (inverse frequency). Light aug
    only (flips, 90-deg rotations, mild brightness/contrast) - IHC stain, so no
    heavy colour distortion.
  - Optim: AdamW, lr 1e-4, AMP (mixed precision) for the 6 GB card, seed 42.
  - Metrics per donor-held-out test fold: macro-F1, AUC-ROC (usable-class prob),
    per-class precision/recall, confusion. Aggregated as mean +/- std over folds.
  - Baselines it must beat: (a) tissue_fraction whole-slide heuristic (threshold
    picked on the train portion), (b) majority class.

Reads review_tool/results/final_clean_dataset_20260728.csv (597 labels) and the
whole slides in review_tool/webapp/review_images/. READ-ONLY on all data.
Writes models/quality/.
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             roc_auc_score, precision_recall_fscore_support)

# ------------------------------------------------------------------
# CONFIG - edit these only
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Enriched 847-label set (597 base + 250 mined top-up) is the default; override with
# QUALITY_CSV to reproduce the 597-label baseline (final_clean_dataset_20260728.csv).
CSV_PATH   = os.environ.get("QUALITY_CSV", os.path.join(SCRIPT_DIR, "review_tool", "results",
                          "final_clean_dataset_20260802.csv"))
# Image lookup: an os.pathsep-separated search list (first hit wins). Default = the single
# baseline dir. The combined dataset needs review_images AND data_bulk/images (88 human-only
# files exist only in the former).
IMG_DIRS   = os.environ.get("QUALITY_IMG_DIRS",
                            os.path.join(SCRIPT_DIR, "review_tool", "webapp", "review_images")
                            ).split(os.pathsep)
OUT_DIR    = os.environ.get("QUALITY_OUT_DIR", os.path.join(SCRIPT_DIR, "models", "quality"))

# Whole-slide resize target + batch. Defaults reproduce the 448px / 6 GB baseline;
# override on a bigger GPU, e.g. QUALITY_INPUT_SIZE=768 QUALITY_BATCH=8 (Hex A100/3080).
INPUT_SIZE = int(os.environ.get("QUALITY_INPUT_SIZE", "448"))   # square resize; drop to 384 if OOM
BATCH      = int(os.environ.get("QUALITY_BATCH", "12"))         # lower this as INPUT_SIZE rises
EPOCHS     = int(os.environ.get("QUALITY_EPOCHS", "20"))  # max epochs/fold; early stop on val F1

# ---- combined-dataset arm flags (scope 03c). All default OFF -> baseline behaviour. ----
# Metrics are ALWAYS computed on human-labelled test rows only (label_source column; a CSV
# without the column is all-human, so the 847 baseline is unchanged).
TRAIN_HUMAN_ONLY = os.environ.get("QUALITY_TRAIN_HUMAN_ONLY", "") == "1"   # Arm B control
SOFT_TARGETS     = os.environ.get("QUALITY_SOFT_TARGETS", "") == "1"       # Arm D: usable_frac targets
USE_SAMPLER      = os.environ.get("QUALITY_SAMPLER", "") == "1"            # Arm D: WeightedRandomSampler + plain CE
GEO_AUG_ONLY     = os.environ.get("QUALITY_AUG", "") == "geo"              # Arm D: drop brightness/contrast jitter
TWO_STAGE        = os.environ.get("QUALITY_TWO_STAGE", "") == "1"          # Arm D: fine-tune on human rows at LR/10
STAGE2_EPOCHS    = int(os.environ.get("QUALITY_STAGE2_EPOCHS", "10"))
STAGE2_PATIENCE  = 3
PATIENCE   = 6            # stop if val macro-F1 does not improve for this many
N_FOLDS    = 5
VAL_FRAC   = 0.2          # donor-grouped val carved from the 4 training folds
LR         = 1e-4
WEIGHT_DECAY = 1e-4
SEED       = 42
ARCH       = "resnet50"

# tissue_fraction baseline rule (reused from filter_blank_tiles.py)
LUM_THRESHOLD = 200       # pixel darker than this (0..255) counts as tissue
SAT_THRESHOLD = 0.10      # pixel more colourful than this (0..1) counts as tissue

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
CLASSES = ["unusable", "usable"]   # index 1 = usable = positive class for AUC
# ------------------------------------------------------------------


def set_seed(s):
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def find_image(img_id):
    """First existing path for img_id across the IMG_DIRS search list, else None."""
    for d in IMG_DIRS:
        p = os.path.join(d, img_id)
        if os.path.isfile(p):
            return p
    return None


def load_rows():
    """Read the label CSV -> (image_ids, y, groups, is_human, fracs). Load by exact
    image_id; filenames may contain commas, so never split on comma.

    label_source and usable_frac are optional columns (combined dataset, scope 03c);
    a CSV without them is treated as all-human hard labels -> baseline unchanged."""
    with open(CSV_PATH, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    idx = {c: i for i, c in enumerate(CLASSES)}
    ids, ys, groups, human, fracs = [], [], [], [], []
    missing = []
    for r in rows:
        img_id = r["image_id"]
        if find_image(img_id) is None:
            missing.append(img_id)
            continue
        ids.append(img_id)
        ys.append(idx[r["label"]])
        groups.append(r["patient_id"])
        human.append(r.get("label_source", "human") == "human")
        fracs.append(float(r.get("usable_frac", "") or float(ys[-1])))
    if missing:
        print(f"WARNING: {len(missing)} image_ids in CSV not found on disk "
              f"(first few: {missing[:5]})")
    return (ids, np.array(ys), np.array(groups),
            np.array(human, dtype=bool), np.array(fracs, dtype=np.float32))


def preload(ids):
    """Resize every whole slide once to INPUT_SIZE and cache in RAM (uint8)."""
    imgs = np.zeros((len(ids), INPUT_SIZE, INPUT_SIZE, 3), dtype=np.uint8)
    for i, img_id in enumerate(ids):
        path = find_image(img_id)
        with Image.open(path) as im:
            imgs[i] = np.asarray(
                im.convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR))
        if (i + 1) % 100 == 0:
            print(f"  preloaded {i + 1}/{len(ids)} slides")
    return imgs


def tissue_fraction(arr):
    """Fraction of pixels that look like tissue (filter_blank_tiles.py rule),
    computed on the cached resized slide. A pixel is tissue if luminance<200 OR
    saturation>0.10. tissue_fraction is a ratio, near scale-invariant, so the
    resized slide is a faithful proxy for the whole-slide heuristic."""
    a = arr.astype(np.float32)
    lum = a.mean(axis=2)
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    is_tissue = (lum < LUM_THRESHOLD) | (sat > SAT_THRESHOLD)
    return float(is_tissue.mean())


class SlideDS(Dataset):
    def __init__(self, imgs, ys, train, fracs=None):
        self.imgs, self.ys, self.train = imgs, ys, train
        # soft target = P(usable); defaults to the hard label when no fracs are given
        self.fracs = fracs if fracs is not None else ys.astype(np.float32)
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
        if self.train and not GEO_AUG_ONLY:
            # light photometric jitter only - IHC stain, keep it mild. QUALITY_AUG=geo drops
            # this entirely: exposure/stain readability can be part of the quality label
            # (Schoemig 2021 logic), so the noise-aware arm runs geometric-only.
            b = 1.0 + (np.random.rand() * 0.2 - 0.1)      # brightness x[0.9,1.1]
            c = 1.0 + (np.random.rand() * 0.2 - 0.1)      # contrast   x[0.9,1.1]
            t = torch.clamp((t * b - 0.5) * c + 0.5, 0.0, 1.0)
        return self.norm(t), int(self.ys[i]), float(self.fracs[i])


def carve_val(tr_idx, y, groups):
    """Donor-grouped val split from the training folds (seed 42)."""
    g = groups[tr_idx]
    inner = GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC, random_state=SEED)
    a, b = next(inner.split(tr_idx, y[tr_idx], g))
    return tr_idx[a], tr_idx[b]


def run_epoch(model, loader, crit, opt, scaler, dev, train, soft=False):
    """One pass. soft=True trains against the usable_frac probabilistic target
    (CE with a (N,2) target distribution); metrics always use the hard labels."""
    model.train() if train else model.eval()
    tot, loss_sum, probs, preds, gts = 0, 0.0, [], [], []
    with torch.set_grad_enabled(train):
        for x, y, fr in loader:
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            with torch.autocast("cuda", enabled=(dev.type == "cuda")):
                out = model(x)
                if soft and train:
                    fr = fr.to(dev, non_blocking=True).float()
                    target = torch.stack([1.0 - fr, fr], dim=1)   # [P(unusable), P(usable)]
                    loss = nn.functional.cross_entropy(out, target)
                else:
                    loss = crit(out, y)
            if train:
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            loss_sum += loss.item() * len(y)
            tot += len(y)
            p = torch.softmax(out.float(), dim=1)
            probs.append(p[:, 1].detach().cpu().numpy())     # usable-class prob
            preds.append(out.argmax(1).cpu().numpy())
            gts.append(y.cpu().numpy())
    return (loss_sum / tot, np.concatenate(probs),
            np.concatenate(preds), np.concatenate(gts))


def fold_metrics(gts, preds, probs):
    """Per-fold metrics dict. AUC is nan if the test fold is single-class."""
    macro_f1 = float(f1_score(gts, preds, average="macro"))
    acc = float(accuracy_score(gts, preds))
    p, r, f, _ = precision_recall_fscore_support(
        gts, preds, labels=[0, 1], zero_division=0)
    if len(set(gts.tolist())) == 2:
        auc = float(roc_auc_score(gts, probs))
    else:
        auc = float("nan")
    cm = confusion_matrix(gts, preds, labels=[0, 1])
    return {
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(acc, 4),
        "auc": None if np.isnan(auc) else round(auc, 4),
        "precision": {"unusable": round(float(p[0]), 4), "usable": round(float(p[1]), 4)},
        "recall":    {"unusable": round(float(r[0]), 4), "usable": round(float(r[1]), 4)},
        "f1":        {"unusable": round(float(f[0]), 4), "usable": round(float(f[1]), 4)},
        "confusion_matrix": cm.tolist(),
        "n_test": int(len(gts)),
        "test_class_counts": {CLASSES[k]: int(v)
                              for k, v in collections.Counter(gts.tolist()).items()},
    }


def make_loader(imgs, y, fracs, idx, dev, train):
    ds = SlideDS(imgs[idx], y[idx], train, fracs[idx])
    if train and USE_SAMPLER:
        cnt = collections.Counter(y[idx].tolist())
        sw = np.array([1.0 / cnt[int(c)] for c in y[idx]], dtype=np.float64)
        sampler = WeightedRandomSampler(torch.as_tensor(sw), num_samples=len(idx),
                                        replacement=True)
        return DataLoader(ds, batch_size=BATCH, sampler=sampler,
                          num_workers=0, pin_memory=(dev.type == "cuda"))
    return DataLoader(ds, batch_size=BATCH, shuffle=train,
                      num_workers=0, pin_memory=(dev.type == "cuda"))


def make_crit(y_idx, dev):
    """Class-weighted CE (baseline). Under the sampler the classes arrive ~balanced,
    so plain CE is used instead (weighting twice would overshoot)."""
    if USE_SAMPLER:
        return nn.CrossEntropyLoss()
    cnt = collections.Counter(y_idx.tolist())
    w = torch.tensor([len(y_idx) / (len(CLASSES) * cnt.get(c, 1)) for c in range(len(CLASSES))],
                     dtype=torch.float32, device=dev)
    return nn.CrossEntropyLoss(weight=w)


def fit(model, tl, vl, crit, opt, scaler, dev, epochs, patience, soft, tag=""):
    """Train with early stopping (restore-best on val macro-F1). Returns
    (best_state, best_val_f1, train_f1_at_best)."""
    best_f1, best_state, best_trf1, wait = -1.0, None, None, 0
    for ep in range(1, epochs + 1):
        trl, _, trpd, trgt = run_epoch(model, tl, crit, opt, scaler, dev, True, soft=soft)
        trf1 = f1_score(trgt, trpd, average="macro")
        _, vpr, vpd, vgt = run_epoch(model, vl, crit, opt, scaler, dev, False)
        vf1 = f1_score(vgt, vpd, average="macro")
        print(f"    {tag}epoch {ep:2d}  train loss {trl:.3f} F1 {trf1:.3f} | "
              f"val macro-F1 {vf1:.3f}  (best {max(best_f1, vf1):.3f})")
        if vf1 > best_f1:
            best_f1, best_trf1, wait = vf1, trf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                print(f"    {tag}early stop at epoch {ep} (no val gain for {patience})")
                break
    return best_state, best_f1, best_trf1


def tune_threshold(vgt, vpr):
    """Operating point on VAL (human rows): usable iff P(usable) >= thr, max macro-F1."""
    best_thr, best_f1 = 0.5, -1.0
    for thr in np.unique(np.round(vpr, 3)):
        f = f1_score(vgt, (vpr >= thr).astype(int), labels=[0, 1], average="macro")
        if f > best_f1:
            best_f1, best_thr = f, float(thr)
    return best_thr


def train_one_fold(imgs, y, fracs, is_human, groups, tr, te, dev, epochs):
    """Stage 1 (+ optional stage-2 human fine-tune), then human-only test eval.
    `te` must already be human-rows-only. Returns (state, val_f1, gts, preds, probs, extras)."""
    set_seed(SEED)
    tr2, va2 = carve_val(tr, y, groups)
    tl = make_loader(imgs, y, fracs, tr2, dev, True)
    vl = make_loader(imgs, y, fracs, va2, dev, False)
    tel = make_loader(imgs, y, fracs, te, dev, False)

    model = timm.create_model(ARCH, pretrained=True, num_classes=len(CLASSES)).to(dev)
    crit = make_crit(y[tr2], dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda"))

    state, best_f1, trf1 = fit(model, tl, vl, crit, opt, scaler, dev,
                               epochs, PATIENCE, soft=SOFT_TARGETS)
    model.load_state_dict(state)
    extras = {"stage1_val_f1": round(float(best_f1), 4),
              "stage1_train_f1_at_best": round(float(trf1), 4),
              "stage1_gap": round(float(trf1 - best_f1), 4)}

    if TWO_STAGE:
        # Fine-tune on the fold's HUMAN train rows only, cold LR, short leash — the
        # "clean data last" stage. Monitored on the human rows of val (overfit danger
        # zone: ~540 images; guards = LR/10, patience 3, restore-best).
        tr2h = tr2[is_human[tr2]]
        va2h = va2[is_human[va2]]
        if len(va2h) == 0 or len(set(y[va2h].tolist())) < 2:
            va2h = va2                       # fallback: mixed val if human val is degenerate
        print(f"    stage2: fine-tune on {len(tr2h)} human rows (val {len(va2h)})")
        tl2 = make_loader(imgs, y, fracs, tr2h, dev, True)
        vl2 = make_loader(imgs, y, fracs, va2h, dev, False)
        crit2 = make_crit(y[tr2h], dev)
        opt2 = torch.optim.AdamW(model.parameters(), lr=LR / 10, weight_decay=WEIGHT_DECAY)
        state, best_f1, trf1 = fit(model, tl2, vl2, crit2, opt2, scaler, dev,
                                   STAGE2_EPOCHS, STAGE2_PATIENCE, soft=False, tag="s2 ")
        model.load_state_dict(state)
        extras.update({"stage2_val_f1": round(float(best_f1), 4),
                       "stage2_train_f1_at_best": round(float(trf1), 4),
                       "stage2_gap": round(float(trf1 - best_f1), 4)})

    # threshold selected on HUMAN val rows only (metrics are judged on human labels)
    va_h = va2[is_human[va2]]
    if len(va_h) > 0 and len(set(y[va_h].tolist())) == 2:
        vl_h = make_loader(imgs, y, fracs, va_h, dev, False)
        _, vpr, _, vgt = run_epoch(model, vl_h, crit, opt, scaler, dev, False)
        extras["tuned_threshold"] = round(tune_threshold(vgt, vpr), 3)
    else:
        extras["tuned_threshold"] = 0.5

    _, tpr, tpd, tgt = run_epoch(model, tel, crit, opt, scaler, dev, False)
    return state, best_f1, tgt, tpd, tpr, extras


def majority_baseline(y, folds):
    """Predict the train-majority class on every test fold. Aggregate macro-F1."""
    f1s = []
    for tr, te in folds:
        maj = collections.Counter(y[tr].tolist()).most_common(1)[0][0]
        pred = np.full(len(te), maj)
        f1s.append(f1_score(y[te], pred, labels=[0, 1], average="macro"))
    return f1s


def tissue_baseline(tf, y, folds):
    """Whole-slide tissue_fraction heuristic. Threshold chosen on the train
    portion (max macro-F1), applied to the test fold. Predict usable if tf>=thr."""
    f1s, thrs = [], []
    cand = np.linspace(0.0, 1.0, 201)
    for tr, te in folds:
        best_thr, best_f1 = 0.5, -1.0
        for thr in cand:
            pred = (tf[tr] >= thr).astype(int)
            f = f1_score(y[tr], pred, labels=[0, 1], average="macro")
            if f > best_f1:
                best_f1, best_thr = f, thr
        pred_te = (tf[te] >= best_thr).astype(int)
        f1s.append(f1_score(y[te], pred_te, labels=[0, 1], average="macro"))
        thrs.append(round(float(best_thr), 3))
    return f1s, thrs


def mean_std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not xs:
        return None, None
    return round(float(np.mean(xs)), 4), round(float(np.std(xs)), 4)


def save_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4.6, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES)), CLASSES, fontsize=9)
    ax.set_yticks(range(len(CLASSES)), CLASSES, fontsize=9)
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thr else "#222", fontsize=11)
    ax.set_title("ResNet-50 slice quality\nsummed over 5 donor-held-out folds", fontsize=10)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "smoke"
    set_seed(SEED)
    torch.backends.cudnn.benchmark = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type != "cuda":
        raise SystemExit("CUDA not available - refusing to CPU-train. Stopping.")

    ids, y, groups, is_human, fracs = load_rows()
    n_h = int(is_human.sum())
    print(f"device={dev} ({torch.cuda.get_device_name(0)})")
    print(f"slides={len(ids)}  donors={len(set(groups.tolist()))}  "
          f"class counts={ {CLASSES[k]: int(v) for k, v in collections.Counter(y.tolist()).items()} }")
    print(f"label sources: human={n_h}  machine={len(ids) - n_h}")
    print(f"input={INPUT_SIZE}px  batch={BATCH}  arch={ARCH}  "
          f"epochs<= {2 if smoke else EPOCHS}  folds={1 if smoke else N_FOLDS}")
    flags = {"train_human_only": TRAIN_HUMAN_ONLY, "soft_targets": SOFT_TARGETS,
             "sampler": USE_SAMPLER, "geo_aug_only": GEO_AUG_ONLY, "two_stage": TWO_STAGE}
    if any(flags.values()):
        print(f"arm flags: { {k: v for k, v in flags.items() if v} }")

    imgs = preload(ids)
    tf = np.array([tissue_fraction(imgs[i]) for i in range(len(ids))])

    gkf = GroupKFold(n_splits=N_FOLDS)
    folds = list(gkf.split(imgs, y, groups))

    # sanity: no donor spans folds
    for k, (tr, te) in enumerate(folds):
        overlap = set(groups[tr].tolist()) & set(groups[te].tolist())
        assert not overlap, f"donor leak in fold {k}: {overlap}"

    os.makedirs(OUT_DIR, exist_ok=True)

    per_fold = []
    all_gts, all_preds, all_probs = [], [], []
    cm_sum = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
    best_overall = {"fold": None, "macro_f1": -1.0, "state": None}

    run_folds = folds[:1] if smoke else folds
    epochs = 2 if smoke else EPOCHS

    donor_err = collections.defaultdict(lambda: [0, 0])   # donor -> [n_test, errors] (human rows)

    for k, (tr, te) in enumerate(run_folds):
        # leakage guard runs on the FULL fold; training/eval subsets are carved after
        te_h = te[is_human[te]]                    # metrics judged on human labels only
        tr_eff = tr[is_human[tr]] if TRAIN_HUMAN_ONLY else tr
        assert len(te_h) > 0, f"fold {k}: no human-labelled test rows"
        te_donors = sorted(set(groups[te].tolist()))
        print(f"\n=== FOLD {k + 1}/{len(run_folds)} ===")
        print(f"  train {len(tr_eff)} ({int(is_human[tr_eff].sum())} human) / "
              f"test {len(te_h)} human-labelled slides (fold holds {len(te)}) | "
              f"test donors={len(te_donors)}  "
              f"test class counts={ {CLASSES[kk]: int(v) for kk, v in collections.Counter(y[te_h].tolist()).items()} }")
        state, val_f1, gts, preds, probs, extras = train_one_fold(
            imgs, y, fracs, is_human, groups, tr_eff, te_h, dev, epochs)

        fm = fold_metrics(gts, preds, probs)
        fm["fold"] = k + 1
        fm["val_macro_f1"] = round(float(val_f1), 4)
        fm["test_donors"] = te_donors
        fm["n_train"] = int(len(tr_eff))
        fm["n_train_human"] = int(is_human[tr_eff].sum())
        fm["n_train_machine"] = int(len(tr_eff) - is_human[tr_eff].sum())
        fm.update(extras)
        # tuned operating point (threshold chosen on human val rows, applied to test)
        thr = extras["tuned_threshold"]
        fm_t = fold_metrics(gts, (probs >= thr).astype(int), probs)
        fm["tuned"] = {"threshold": thr, "macro_f1": fm_t["macro_f1"],
                       "recall": fm_t["recall"], "precision": fm_t["precision"],
                       "confusion_matrix": fm_t["confusion_matrix"]}
        for i_loc, gi in enumerate(te_h):
            donor_err[groups[gi]][0] += 1
            donor_err[groups[gi]][1] += int(preds[i_loc] != gts[i_loc])
        per_fold.append(fm)
        print(f"  TEST fold {k + 1}: macro-F1 {fm['macro_f1']:.4f}  "
              f"AUC {fm['auc']}  "
              f"recall(unusable/usable) {fm['recall']['unusable']:.3f}/{fm['recall']['usable']:.3f}")

        all_gts.append(gts)
        all_preds.append(preds)
        all_probs.append(probs)
        cm_sum += np.array(fm["confusion_matrix"])

        torch.save({"state_dict": state, "classes": CLASSES, "input_size": INPUT_SIZE,
                    "arch": ARCH, "fold": k + 1},
                   os.path.join(OUT_DIR, f"quality_{ARCH}_fold{k + 1}.pt"))
        if fm["macro_f1"] > best_overall["macro_f1"]:
            best_overall = {"fold": k + 1, "macro_f1": fm["macro_f1"], "state": state}

        torch.cuda.empty_cache()

    f1_m, f1_s = mean_std([f["macro_f1"] for f in per_fold])
    auc_m, auc_s = mean_std([f["auc"] for f in per_fold])
    pu_m, pu_s = mean_std([f["precision"]["usable"] for f in per_fold])
    pn_m, pn_s = mean_std([f["precision"]["unusable"] for f in per_fold])
    ru_m, ru_s = mean_std([f["recall"]["usable"] for f in per_fold])
    rn_m, rn_s = mean_std([f["recall"]["unusable"] for f in per_fold])

    # ---- baselines (same folds; train side full, scored on human test rows only) ----
    eval_folds = [((tr[is_human[tr]] if TRAIN_HUMAN_ONLY else tr), te[is_human[te]])
                  for tr, te in run_folds]
    maj_f1 = majority_baseline(y, eval_folds)
    tf_f1, tf_thr = tissue_baseline(tf, y, eval_folds)
    maj_m, maj_s = mean_std(maj_f1)
    tf_m, tf_s = mean_std(tf_f1)

    save_confusion(cm_sum, os.path.join(OUT_DIR, f"quality_{ARCH}_confusion.png"))
    if best_overall["state"] is not None:
        torch.save({"state_dict": best_overall["state"], "classes": CLASSES,
                    "input_size": INPUT_SIZE, "arch": ARCH,
                    "fold": best_overall["fold"]},
                   os.path.join(OUT_DIR, f"quality_{ARCH}_best.pt"))

    summary = {
        "task": "slice_quality_whole_slide_usable_vs_unusable",
        "smoke": smoke,
        "config": {
            "arch": f"{ARCH} (timm, ImageNet-pretrained, fine-tuned)",
            "input_size": INPUT_SIZE, "batch": BATCH, "max_epochs": epochs,
            "patience": PATIENCE, "lr": LR, "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW", "amp": True, "seed": SEED,
            "n_folds": len(run_folds), "val_frac": VAL_FRAC,
            "loss": ("plain CE + WeightedRandomSampler" if USE_SAMPLER
                     else "class-weighted CrossEntropyLoss (inverse frequency)")
                    + (" | soft usable_frac targets (stage 1)" if SOFT_TARGETS else ""),
            "augmentation": "hflip, vflip, rot90" + ("" if GEO_AUG_ONLY
                            else ", brightness/contrast x[0.9,1.1]"),
            "split": "donor-grouped GroupKFold by patient_id (no donor in two folds)",
            "arm_flags": flags,
            "stage2": ({"epochs": STAGE2_EPOCHS, "patience": STAGE2_PATIENCE,
                        "lr": LR / 10, "data": "fold's human train rows"}
                       if TWO_STAGE else None),
            "img_dirs": [os.path.relpath(d, SCRIPT_DIR) for d in IMG_DIRS],
            "eval": "test metrics on human-labelled rows only; threshold tuned on human val rows",
            "device": torch.cuda.get_device_name(0),
        },
        "data": {
            "csv": os.path.relpath(CSV_PATH, SCRIPT_DIR),
            "n_slides": len(ids), "n_donors": len(set(groups.tolist())),
            "class_counts": {CLASSES[k]: int(v)
                             for k, v in collections.Counter(y.tolist()).items()},
            "n_rows_by_source": {"human": n_h, "machine": len(ids) - n_h},
            "n_test_human_total": int(sum(f["n_test"] for f in per_fold)),
        },
        "per_donor_test_errors": {d: {"n": v[0], "errors": v[1]}
                                  for d, v in sorted(donor_err.items(),
                                                     key=lambda kv: -kv[1][1])},
        "tuned_aggregate": {
            "macro_f1_mean": mean_std([f["tuned"]["macro_f1"] for f in per_fold])[0],
            "macro_f1_std": mean_std([f["tuned"]["macro_f1"] for f in per_fold])[1],
            "recall_unusable_mean": mean_std([f["tuned"]["recall"]["unusable"]
                                              for f in per_fold])[0],
            "recall_unusable_std": mean_std([f["tuned"]["recall"]["unusable"]
                                             for f in per_fold])[1],
            "per_fold_threshold": [f["tuned"]["threshold"] for f in per_fold],
        },
        "cnn_aggregate": {
            "macro_f1_mean": f1_m, "macro_f1_std": f1_s,
            "auc_mean": auc_m, "auc_std": auc_s,
            "precision_usable_mean": pu_m, "precision_usable_std": pu_s,
            "precision_unusable_mean": pn_m, "precision_unusable_std": pn_s,
            "recall_usable_mean": ru_m, "recall_usable_std": ru_s,
            "recall_unusable_mean": rn_m, "recall_unusable_std": rn_s,
            "confusion_matrix_summed": cm_sum.tolist(),
            "confusion_rows_true_cols_pred_order": CLASSES,
        },
        "baselines": {
            "majority_class": {"per_fold_macro_f1": [round(x, 4) for x in maj_f1],
                               "macro_f1_mean": maj_m, "macro_f1_std": maj_s},
            "tissue_fraction_heuristic": {
                "per_fold_macro_f1": [round(x, 4) for x in tf_f1],
                "per_fold_threshold": tf_thr,
                "macro_f1_mean": tf_m, "macro_f1_std": tf_s,
                "rule": "usable if mean tissue_fraction >= thr; thr max-F1 on train",
            },
        },
        "beats_majority_by": (round(f1_m - maj_m, 4) if f1_m is not None else None),
        "beats_tissue_fraction_by": (round(f1_m - tf_m, 4) if f1_m is not None else None),
        "per_fold": per_fold,
        "best_fold": best_overall["fold"],
    }

    with open(os.path.join(OUT_DIR, "quality_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print("\n================ AGGREGATE (donor-held-out CV) ================")
    print(f"CNN    macro-F1 {f1_m} +/- {f1_s}   AUC {auc_m} +/- {auc_s}")
    print(f"       precision usable {pu_m}+/-{pu_s}  unusable {pn_m}+/-{pn_s}")
    print(f"       recall    usable {ru_m}+/-{ru_s}  unusable {rn_m}+/-{rn_s}")
    print(f"BASE   majority        macro-F1 {maj_m} +/- {maj_s}")
    print(f"BASE   tissue_fraction macro-F1 {tf_m} +/- {tf_s}  (thr {tf_thr})")
    print(f"summed confusion (rows=true unusable/usable, cols=pred):")
    print(cm_sum)
    print(f"\nwrote {OUT_DIR}\\quality_metrics.json, quality_{ARCH}_confusion.png, "
          f"quality_{ARCH}_fold*.pt, quality_{ARCH}_best.pt")


if __name__ == "__main__":
    main()
