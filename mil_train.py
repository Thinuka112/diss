"""Stage-2 MIL training: slide verdict + per-tile localisation from slide labels only (scope 05).

Folds are the Stage-1 folds (donor identity read from quality_metrics.json per_fold.test_donors
— asserted, never recomputed). MIL fold k: test = HUMAN slides of fold k's donors, embedded by
encoder k; train = all other slides (human-only primary arm; human+machine bags in the secondary
arm when MIL_EMB_DIR points at combined embeddings), same encoder. Model selection on a
donor-grouped val split of the train donors (seed 42). Test metrics are argmax so the parity
comparison vs the Stage-1 448px per-fold macro-F1 is like-for-like.

  MIL_EMB_DIR=models/mil/emb_human MIL_OUT_DIR=models/mil python mil_train.py

Env: MIL_EMB_DIR, MIL_OUT_DIR, MIL_EPOCHS(200), MIL_PATIENCE(20), MIL_LR(3e-4), MIL_SEED(42),
     MIL_SOFT=1 to train on usable_frac soft targets (secondary arm), MIL_FOLDS_JSON.
Outputs: mil_metrics.json, mil_slide_preds.csv, mil_tile_scores.csv (contribution + attention
per tile for every donor-held-out test slide).
"""
import csv
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from mil_model import GatedAdditiveMIL

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_DIR = os.environ.get("MIL_EMB_DIR", os.path.join(SCRIPT_DIR, "models", "mil", "emb_human"))
OUT_DIR = os.environ.get("MIL_OUT_DIR", os.path.join(SCRIPT_DIR, "models", "mil"))
FOLDS_JSON = os.environ.get("MIL_FOLDS_JSON", os.path.join(
    SCRIPT_DIR, "models", "quality", "quality_metrics.json"))
EPOCHS = int(os.environ.get("MIL_EPOCHS", "200"))
PATIENCE = int(os.environ.get("MIL_PATIENCE", "20"))
LR = float(os.environ.get("MIL_LR", "3e-4"))
SEED = int(os.environ.get("MIL_SEED", "42"))
SOFT = os.environ.get("MIL_SOFT", "") == "1"
GRID = 8
CLASSES = ["unusable", "usable"]          # matches the whole project; usable = positive logit


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def macro_f1(y, p):
    return f1_score(y, p, average="macro", zero_division=0)


def epoch_pass(model, emb, y, w, idx, opt=None, bs=64, dev="cpu"):
    losses, preds = [], np.zeros(len(idx))
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        x = torch.from_numpy(emb[b].astype(np.float32)).to(dev)
        t = torch.from_numpy(y[b].astype(np.float32)).to(dev)
        wt = torch.from_numpy(w[b].astype(np.float32)).to(dev)
        logit, _, _ = model(x)
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(
            logit, t, reduction="none") * wt).mean()
        if opt is not None:
            opt.zero_grad(); loss.backward(); opt.step()
        losses.append(float(loss))
        preds[s:s + bs] = (logit.detach().cpu().numpy() > 0).astype(int)
    return float(np.mean(losses)), preds


def main():
    set_seed(SEED)
    per_fold_meta = json.load(open(FOLDS_JSON, encoding="utf-8"))["per_fold"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT_DIR, exist_ok=True)

    agg, slide_rows, tile_rows = [], [], []
    for meta in per_fold_meta:
        k = meta["fold"]
        d = np.load(os.path.join(EMB_DIR, f"fold{k}.npz"), allow_pickle=False)
        ids, emb = d["ids"], d["emb"]
        bad = int((~np.isfinite(emb.astype(np.float32))).any(axis=2).sum())
        if bad:
            # float16 overflow on a handful of extreme tiles during extraction — zero them
            print(f"fold{k}: sanitising {bad} non-finite tiles of {emb.shape[0] * emb.shape[1]}")
            emb = np.nan_to_num(emb.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        lab = (d["label"] == "usable").astype(int)
        donor, src = d["donor"], d["label_source"]
        frac = d["usable_frac"]
        raw = (meta["test_donors"].split(",") if isinstance(meta["test_donors"], str)
               else meta["test_donors"])
        test_donors = {str(x) for x in raw}
        donor = donor.astype(str)
        te = np.where(np.isin(donor, list(test_donors)) & (src == "human"))[0]
        tr = np.where(~np.isin(donor, list(test_donors)))[0]
        assert len(te) and not (set(donor[te]) & set(donor[tr])), f"fold{k} donor leak"

        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        tr2_i, va_i = next(gss.split(tr, groups=donor[tr]))
        tr2, va = tr[tr2_i], tr[va_i]
        target = frac if SOFT else lab.astype(np.float32)
        n_un = max(1, int((lab[tr2] == 0).sum()))
        w = np.where(lab == 0, (lab[tr2] == 1).sum() / n_un, 1.0)

        set_seed(SEED + k)
        model = GatedAdditiveMIL().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        best, best_state, wait = -1.0, None, 0
        for ep in range(EPOCHS):
            model.train()
            order = np.random.permutation(tr2)
            epoch_pass(model, emb, target, w, order, opt, dev=dev)
            model.eval()
            with torch.no_grad():
                _, vp = epoch_pass(model, emb, lab.astype(np.float32), w, va, None, dev=dev)
            vf1 = macro_f1(lab[va], vp)
            if vf1 > best:
                best, wait = vf1, 0
                best_state = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        model.load_state_dict(best_state); model.eval()

        with torch.no_grad():
            x = torch.from_numpy(emb[te].astype(np.float32)).to(dev)
            logit, contrib, att = model(x)
            prob = torch.sigmoid(logit).cpu().numpy()
            contrib, att = contrib.cpu().numpy(), att.cpu().numpy()
        pred = (prob > 0.5).astype(int)
        mf1 = macro_f1(lab[te], pred)
        auc = roc_auc_score(lab[te], prob) if len(set(lab[te])) > 1 else float("nan")
        cm = confusion_matrix(lab[te], pred, labels=[0, 1]).tolist()
        s1 = meta["macro_f1"]
        agg.append({"fold": k, "n_test": int(len(te)), "macro_f1": round(mf1, 4),
                    "auc": round(float(auc), 4), "confusion_unusable_usable": cm,
                    "val_macro_f1": round(best, 4), "stage1_448_macro_f1": s1,
                    "delta_vs_stage1": round(mf1 - s1, 4),
                    "test_donors": sorted(test_donors)})
        print(f"fold{k}: MIL macroF1 {mf1:.3f} (val {best:.3f})  stage1 {s1:.3f}  "
              f"delta {mf1 - s1:+.3f}  n={len(te)}", flush=True)
        for j, i in enumerate(te):
            slide_rows.append([ids[i], k, CLASSES[lab[i]], CLASSES[pred[j]], f"{prob[j]:.4f}"])
            for t in range(GRID * GRID):
                tile_rows.append([ids[i], k, t // GRID, t % GRID,
                                  f"{contrib[j, t]:.6f}", f"{att[j, t]:.6f}"])

    deltas = [a["delta_vs_stage1"] for a in agg]
    summary = {
        "arm": "soft_combined" if SOFT else "human_only_primary",
        "emb_dir": EMB_DIR, "seed": SEED, "classes": CLASSES,
        "aggregate": {"macro_f1_mean": round(float(np.mean([a["macro_f1"] for a in agg])), 4),
                      "macro_f1_std": round(float(np.std([a["macro_f1"] for a in agg])), 4),
                      "mean_delta_vs_stage1": round(float(np.mean(deltas)), 4),
                      "worst_fold_delta": round(float(min(deltas)), 4),
                      "parity_pass": bool(np.mean(deltas) >= -0.02 and min(deltas) > -0.10)},
        "per_fold": agg}
    json.dump(summary, open(os.path.join(OUT_DIR, "mil_metrics.json"), "w"), indent=1)
    with open(os.path.join(OUT_DIR, "mil_slide_preds.csv"), "w", newline="") as fh:
        w_ = csv.writer(fh); w_.writerow(["image_id", "fold", "true", "pred", "prob_usable"])
        w_.writerows(slide_rows)
    with open(os.path.join(OUT_DIR, "mil_tile_scores.csv"), "w", newline="") as fh:
        w_ = csv.writer(fh); w_.writerow(["image_id", "fold", "row", "col",
                                          "contribution", "attention"])
        w_.writerows(tile_rows)
    print(json.dumps(summary["aggregate"], indent=1))
    print(f"wrote {OUT_DIR}/mil_metrics.json, mil_slide_preds.csv, mil_tile_scores.csv")


if __name__ == "__main__":
    main()
