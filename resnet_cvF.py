"""ResNet-50 on the Arm F combined dataset (5,681), donor-grouped CV on Arm B's fold
partition — the engine behind notebooks/06_resnet_final_cv.ipynb.

Reuses train_quality_cnn.py (imported, never copied): the caller must set the QUALITY_*
env vars BEFORE importing this module (the notebook's config cell does), in particular
QUALITY_CSV -> combined_human_s2usable_20260828.csv and the noise-aware recipe flags
(QUALITY_SOFT_TARGETS/SAMPLER/AUG=geo/TWO_STAGE — the scope-03c Arm D recipe).

Per fold k (Arm B partition, read from models/quality_union_humanonly/quality_metrics.json):
  TRAIN = every row (human AND machine) whose donor is not a fold-k test donor
  TEST  = the human rows of the fold-k test donors (Σ over folds = 847, asserted)
  val   = donor-grouped 20% carved from TRAIN by tq.carve_val (seed 42); threshold
          tuned on the human val rows inside tq.train_one_fold (project rule)
Fold-resume: a finished fold leaves fold{k}_metrics.json + fold{k}_preds.csv in RESF_OUT
and is skipped on re-run, so an interrupted notebook Run All continues where it stopped.
"""
import csv
import json
import os

import numpy as np

import train_quality_cnn as tq

OUT = os.environ.get("RESF_OUT", os.path.join(tq.SCRIPT_DIR, "models", "quality_resnet_cvF"))
ARMB_JSON = os.environ.get("RESF_ARMB_JSON", os.path.join(
    tq.SCRIPT_DIR, "models", "quality_union_humanonly", "quality_metrics.json"))

_cache = {}


def _csv_rows():
    """The dataset as defined by the CSV — no image-on-disk requirement, so the
    census and fold tables work on any machine. Training (run_fold) separately
    asserts that every image resolves."""
    if "csv_rows" not in _cache:
        with open(tq.CSV_PATH, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        for r in rows:
            if not r.get("label_source"):
                r["label_source"] = "human"
        _cache["csv_rows"] = rows
    return _cache["csv_rows"]


def _data():
    if "rows" not in _cache:
        ids, y, groups, is_human, fracs = tq.load_rows()
        n_csv = len(_csv_rows())
        assert len(ids) == n_csv, (
            f"only {len(ids)}/{n_csv} images resolve on this machine - training must "
            f"run where the full image set exists (garlick: review_images + data_bulk)")
        _cache["rows"] = (ids, y, groups, is_human, fracs)
    return _cache["rows"]


def _imgs():
    if "imgs" not in _cache:
        ids = _data()[0]
        print(f"preloading {len(ids)} slides at {tq.INPUT_SIZE}px "
              f"(~{len(ids) * tq.INPUT_SIZE * tq.INPUT_SIZE * 3 / 1e9:.1f} GB RAM)...")
        _cache["imgs"] = tq.preload(ids)
    return _cache["imgs"]


def census():
    """Dataset counts for the notebook's census cell. Cheap (CSV only, no images)."""
    rows = _csv_rows()
    c = {"rows": len(rows),
         "human": sum(1 for r in rows if r["label_source"] == "human"),
         "machine": sum(1 for r in rows if r["label_source"] != "human"),
         "donors": len({r["patient_id"] for r in rows}),
         "unusable": sum(1 for r in rows if r["label"] == "unusable"),
         "usable": sum(1 for r in rows if r["label"] == "usable"),
         "csv": os.path.basename(tq.CSV_PATH)}
    assert c["human"] == 847, c
    return c


def fold_partition():
    """Arm B's five test-donor groups, with this dataset's per-fold row counts
    (CSV-based, so it works on any machine)."""
    per_fold = json.load(open(ARMB_JSON, encoding="utf-8"))["per_fold"]
    assert len(per_fold) == 5
    rows = _csv_rows()
    out = []
    for k, f in enumerate(per_fold, 1):
        d = set(f["test_donors"])
        n_te = sum(1 for r in rows if r["patient_id"] in d and r["label_source"] == "human")
        n_tr = sum(1 for r in rows if r["patient_id"] not in d)
        out.append({"fold": k, "test_donors": sorted(d), "n_test_human": n_te,
                    "n_train_rows": n_tr, "armB_macro_f1": f["macro_f1"],
                    "armB_tuned_macro_f1": f["tuned"]["macro_f1"]})
    assert sum(f["n_test_human"] for f in out) == 847, "fold test rows must sum to 847"
    return out


def run_fold(k, epochs=None, demo=False):
    """Train + evaluate one fold. demo=True: train with the given epoch budget and
    return metrics WITHOUT writing definitive outputs. Otherwise resume-aware."""
    import torch
    os.makedirs(OUT, exist_ok=True)
    done = os.path.join(OUT, f"fold{k}_metrics.json")
    if not demo and os.path.exists(done):
        print(f"fold {k}: already complete - skipping (delete {done} to re-run)")
        return json.load(open(done, encoding="utf-8"))

    ids, y, groups, is_human, fracs = _data()
    part = fold_partition()[k - 1]
    d = set(part["test_donors"])
    te = np.array([i for i in range(len(ids)) if groups[i] in d and is_human[i]])
    tr = np.array([i for i in range(len(ids)) if groups[i] not in d])
    assert not (set(groups[tr].tolist()) & d), f"fold {k}: donor leak into training"
    assert len(te) == part["n_test_human"]
    print(f"fold {k}: train {len(tr)} rows ({int(is_human[tr].sum())} human, "
          f"{len(tr) - int(is_human[tr].sum())} machine) | test {len(te)} human rows "
          f"from {len(d)} unseen donors")

    imgs = _imgs()
    dev = torch.device("cuda")
    ep = epochs or tq.EPOCHS
    tq.set_seed(tq.SEED)
    state, val_f1, gts, preds, probs, extras = tq.train_one_fold(
        imgs, y, fracs, is_human, groups, tr, te, dev, ep)

    fm = tq.fold_metrics(gts, preds, probs)
    thr = extras["tuned_threshold"]
    fm_t = tq.fold_metrics(gts, (probs >= thr).astype(int), probs)
    fm.update({"fold": k, "tuned": {**fm_t, "threshold": thr}, **extras,
               "n_train": int(len(tr)), "n_train_human": int(is_human[tr].sum()),
               "val_macro_f1": round(float(val_f1), 4)})
    print(f"fold {k} TEST: macro-F1 {fm['macro_f1']:.4f} (tuned {fm_t['macro_f1']:.4f})  "
          f"unusable recall {fm['recall']['unusable']:.3f}  AUC {fm['auc']}")
    if demo:
        return fm

    with open(os.path.join(OUT, f"fold{k}_preds.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "true", "p_usable"])
        for i_loc, gi in enumerate(te):
            w.writerow([ids[gi], tq.CLASSES[int(gts[i_loc])], f"{probs[i_loc]:.4f}"])
    torch.save({"state_dict": state, "classes": tq.CLASSES, "input_size": tq.INPUT_SIZE,
                "arch": tq.ARCH, "fold": k},
               os.path.join(OUT, f"fold{k}_{tq.ARCH}.pt"))
    json.dump(fm, open(done, "w", encoding="utf-8"), indent=2)
    return fm


def aggregate():
    """Combine the five fold files; paired deltas vs Arm B at MATCHED operating points."""
    part = fold_partition()
    per_fold, cm, cm_t = [], np.zeros((2, 2), int), np.zeros((2, 2), int)
    for k in range(1, 6):
        f = json.load(open(os.path.join(OUT, f"fold{k}_metrics.json"), encoding="utf-8"))
        f["delta_vs_armB_argmax"] = round(f["macro_f1"] - part[k - 1]["armB_macro_f1"], 4)
        f["delta_vs_armB_tuned"] = round(
            f["tuned"]["macro_f1"] - part[k - 1]["armB_tuned_macro_f1"], 4)
        per_fold.append(f)
        cm += np.array(f["confusion_matrix"])
        cm_t += np.array(f["tuned"]["confusion_matrix"])
    n = sum(f["n_test"] for f in per_fold)
    assert n == 847, n

    def ms(get):
        xs = [get(f) for f in per_fold]
        return round(float(np.mean(xs)), 4), round(float(np.std(xs)), 4)

    summary = {
        "task": "resnet50_armF_combined_donor_grouped_cv_foldmatched_armB",
        "config": {"arch": tq.ARCH, "input_size": tq.INPUT_SIZE, "batch": tq.BATCH,
                   "max_epochs": tq.EPOCHS, "seed": tq.SEED,
                   "csv": os.path.basename(tq.CSV_PATH),
                   "recipe": {"soft_targets": tq.SOFT_TARGETS, "sampler": tq.USE_SAMPLER,
                              "geo_aug_only": tq.GEO_AUG_ONLY, "two_stage": tq.TWO_STAGE},
                   "folds": "Arm B test-donor partition; test = human rows only"},
        "aggregate": {"macro_f1": ms(lambda f: f["macro_f1"]),
                      "tuned_macro_f1": ms(lambda f: f["tuned"]["macro_f1"]),
                      "accuracy": ms(lambda f: f["accuracy"]),
                      "tuned_accuracy": ms(lambda f: f["tuned"]["accuracy"]),
                      "auc": ms(lambda f: f["auc"]),
                      "recall_unusable": ms(lambda f: f["recall"]["unusable"]),
                      "confusion_summed": cm.tolist(),
                      "confusion_summed_tuned": cm_t.tolist(),
                      "mean_delta_vs_armB_argmax": round(float(np.mean(
                          [f["delta_vs_armB_argmax"] for f in per_fold])), 4),
                      "mean_delta_vs_armB_tuned": round(float(np.mean(
                          [f["delta_vs_armB_tuned"] for f in per_fold])), 4)},
        "armB_reference": {"argmax_macro_f1": [p_["armB_macro_f1"] for p_ in part],
                           "tuned_macro_f1": [p_["armB_tuned_macro_f1"] for p_ in part]},
        "per_fold": per_fold,
    }
    json.dump(summary, open(os.path.join(OUT, "cv_metrics.json"), "w", encoding="utf-8"),
              indent=2)
    a = summary["aggregate"]
    print(f"AGGREGATE macro-F1 {a['macro_f1'][0]} +/- {a['macro_f1'][1]}  "
          f"(tuned {a['tuned_macro_f1'][0]} +/- {a['tuned_macro_f1'][1]})  "
          f"AUC {a['auc'][0]}")
    print(f"paired vs Arm B: argmax {a['mean_delta_vs_armB_argmax']:+.4f}  "
          f"tuned {a['mean_delta_vs_armB_tuned']:+.4f}")
    return summary


if __name__ == "__main__":
    import sys as _sys
    _cmd = _sys.argv[1]
    if _cmd == "fold":
        run_fold(int(_sys.argv[2]))
    elif _cmd == "agg":
        aggregate()
    elif _cmd == "census":
        print(census())
