"""ensemble_arms_eval.py - donor-held-out ensemble of the B/C/D arm checkpoints.

The three arms trained on IDENTICAL GroupKFold folds (same CSV, same seed), so for any
slide, each arm has exactly one fold model that never saw that slide's donor. Averaging
those per-slide probabilities is therefore still a donor-held-out prediction.

Honesty protocol:
  - Arm combo and decision thresholds are selected on the fold's VAL human rows only
    (carve_val on the full union train indices, deterministic seed 42).
  - The held-out TEST human rows are scored ONCE with the val-chosen combo + thresholds.
  - Reported: accuracy / macro-F1 / unusable recall at (a) the val-accuracy threshold and
    (b) the val-macroF1 threshold, plus the plain 0.5 cut. Per-arm singles reported too.

Run on garlick (checkpoints + images live there):
  cd ~/quality && .venv/bin/python ensemble_arms_eval.py
Writes models/ensemble_arms_eval.json
"""
import json
import os

import numpy as np
import torch
import timm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

os.environ.setdefault("QUALITY_CSV",
    "review_tool/results/combined_human_machine_20260818.csv")
os.environ.setdefault("QUALITY_IMG_DIRS",
    "review_tool/webapp/review_images" + os.pathsep + "data_bulk/images")
os.environ.setdefault("QUALITY_INPUT_SIZE", "640")
import train_quality_cnn as tq

ARMS = {"B": "models/quality_union_humanonly",
        "C": "models/quality_combined_hard",
        "D": "models/quality_combined_soft2stage"}
OUT = "models/ensemble_arms_eval.json"
BATCH = 12


def probs_for(model, imgs, idx, dev):
    dl = torch.utils.data.DataLoader(tq.SlideDS(imgs[idx], np.zeros(len(idx)), False),
                                     batch_size=BATCH, shuffle=False)
    out = []
    model.eval()
    with torch.no_grad():
        for x, _, _ in dl:
            with torch.autocast("cuda"):
                p = torch.softmax(model(x.to(dev)).float(), dim=1)[:, 1]
            out.append(p.cpu().numpy())
    return np.concatenate(out)


def pick_thr(y, p, metric):
    best_thr, best = 0.5, -1.0
    for thr in np.unique(np.round(p, 3)):
        pred = (p >= thr).astype(int)
        v = (pred == y).mean() if metric == "acc" else f1_score(y, pred, average="macro")
        if v > best:
            best, best_thr = v, float(thr)
    return best_thr


def main():
    dev = torch.device("cuda")
    tq.set_seed(tq.SEED)
    ids, y, groups, is_human, fracs = tq.load_rows()
    imgs = tq.preload(ids)
    folds = list(GroupKFold(n_splits=tq.N_FOLDS).split(imgs, y, groups))

    # per-slide held-out probs per arm (human rows only, each slide from its test fold)
    P = {a: np.full(len(ids), np.nan) for a in ARMS}          # test probs
    V = {a: [] for a in ARMS}                                  # (fold val) probs
    Vy, Vfold = [], []
    Tidx, Tfold = [], []

    for k, (tr, te) in enumerate(folds):
        te_h = te[is_human[te]]
        tr2, va2 = tq.carve_val(tr, y, groups)
        va_h = va2[is_human[va2]]
        Tidx.extend(te_h.tolist()); Tfold.extend([k] * len(te_h))
        Vy.extend(y[va_h].tolist()); Vfold.extend([k] * len(va_h))
        for a, d in ARMS.items():
            ck = torch.load(os.path.join(d, f"quality_resnet50_fold{k+1}.pt"),
                            map_location="cpu", weights_only=True)
            model = timm.create_model(tq.ARCH, pretrained=False, num_classes=2).to(dev)
            model.load_state_dict(ck["state_dict"])
            P[a][te_h] = probs_for(model, imgs, te_h, dev)
            V[a].append(probs_for(model, imgs, va_h, dev))
            del model
            torch.cuda.empty_cache()
        print(f"fold {k+1}: test {len(te_h)} val {len(va_h)} done")

    Tidx = np.array(Tidx); Vy = np.array(Vy)
    Vp = {a: np.concatenate(V[a]) for a in ARMS}
    yt = y[Tidx]

    combos = [("B",), ("C",), ("D",), ("B", "C"), ("B", "D"), ("C", "D"), ("B", "C", "D")]
    results = {}
    best_combo, best_val_acc = None, -1
    for combo in combos:
        vp = np.mean([Vp[a] for a in combo], axis=0)
        tp = np.mean([P[a][Tidx] for a in combo], axis=0)
        thr_a = pick_thr(Vy, vp, "acc")
        thr_f = pick_thr(Vy, vp, "f1")
        val_acc = ((vp >= thr_a).astype(int) == Vy).mean()
        row = {}
        for tag, thr in [("thr0.5", 0.5), ("val_acc_thr", thr_a), ("val_f1_thr", thr_f)]:
            pred = (tp >= thr).astype(int)
            cm = np.zeros((2, 2), int)
            for t, pr in zip(yt, pred):
                cm[t, pr] += 1
            row[tag] = {"threshold": round(float(thr), 3),
                        "accuracy": round(float((pred == yt).mean()), 4),
                        "macro_f1": round(float(f1_score(yt, pred, average="macro")), 4),
                        "unusable_recall": round(float(cm[0, 0] / cm[0].sum()), 4),
                        "confusion": cm.tolist()}
        results["+".join(combo)] = {"val_acc_at_val_thr": round(float(val_acc), 4), **row}
        if len(combo) > 1 and val_acc > best_val_acc:
            best_val_acc, best_combo = val_acc, "+".join(combo)
        print("+".join(combo), "-> test:", row["val_acc_thr"])

    out = {"protocol": "combo+thresholds selected on fold-val human rows; test scored once",
           "n_test": int(len(Tidx)), "val_selected_combo": best_combo,
           "results": results}
    json.dump(out, open(OUT, "w"), indent=2)
    print(f"\nval-selected ensemble: {best_combo}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
