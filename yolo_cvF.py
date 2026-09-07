"""YOLOv8s-cls on the Arm F combined dataset (5,681), donor-grouped CV on Arm B's fold
partition — the engine behind notebooks/05_yolo_final_cv.ipynb. The honest version of
the leaky-split "91%" experiment.

Per fold k (Arm B partition): TRAIN = every row (human AND machine) whose donor is not a
fold-k test donor, unusable links duplicated x8 (oversampling); donor-grouped val carved
from the train pool (GroupShuffleSplit 0.2, seed 42) for ultralytics model selection;
threshold tuned on the HUMAN val rows only (project rule); TEST = the human rows of the
fold-k test donors (sum over folds = 847, asserted). Fold-resume: finished folds leave
fold{k}_done.json + fold{k}_preds.csv in YCVF_OUT and are skipped on re-run.

Images: sources are the full-resolution slides; prepare_images() builds a one-time
1024px cache (same convention as the YOLO/847 CV — crops for the 640px model come from
1024px sources; recorded in the output config). Env overrides let the same module run
locally (defaults) or on garlick.
"""
import csv
import json
import os

import numpy as np

_WT = os.path.dirname(os.path.abspath(__file__))
_MAIN = r"C:\Users\<user>\Documents\DissProject"
_LOCAL = os.path.isdir(os.path.join(_MAIN, "data_bulk", "images"))

CSV = os.environ.get("YCVF_CSV", os.path.join(
    _WT, "review_tool", "results", "combined_human_s2usable_20260828.csv"))
ARMB_JSON = os.environ.get("YCVF_ARMB_JSON", os.path.join(
    _WT, "models", "quality_union_humanonly", "quality_metrics.json"))
IMG_SRC = os.environ.get("YCVF_IMG_SRC", os.pathsep.join(
    [os.path.join(_MAIN, "review_tool", "webapp", "review_images"),
     os.path.join(_MAIN, "data_bulk", "images")] if _LOCAL else
    [os.path.expanduser("~/quality/review_tool/webapp/review_images"),
     os.path.expanduser("~/quality/data_bulk/images")])).split(os.pathsep)
CACHE = os.environ.get("YCVF_CACHE", os.path.join(_MAIN, "data_cache_1024") if _LOCAL
                       else "/dev/shm/tnm31_imgs1024_all")
TREES = os.environ.get("YCVF_TREES", os.path.join(CACHE, "yolo_cvF_trees"))
OUT = os.environ.get("YCVF_OUT", os.path.join(_WT, "models", "quality_yolo_cvF"))
DEVICE = int(os.environ.get("YCVF_DEVICE", "0"))
BATCH = int(os.environ.get("YCVF_BATCH", "16"))
WORKERS = int(os.environ.get("YCVF_WORKERS", "2"))
MAXEDGE, IMGSZ, EPOCHS, PATIENCE, DUP, SEED = 1024, 640, 40, 12, 8, 42
CLASSES = ["unusable", "usable"]

_cache = {}


def rows():
    if "rows" not in _cache:
        with open(CSV, newline="", encoding="utf-8") as fh:
            rs = list(csv.DictReader(fh))
        for r in rs:
            if not r.get("label_source"):
                r["label_source"] = "human"
        assert sum(1 for r in rs if r["label_source"] == "human") == 847
        _cache["rows"] = rs
    return _cache["rows"]


def census():
    rs = rows()
    return {"rows": len(rs),
            "human": sum(1 for r in rs if r["label_source"] == "human"),
            "machine": sum(1 for r in rs if r["label_source"] != "human"),
            "donors": len({r["patient_id"] for r in rs}),
            "unusable": sum(1 for r in rs if r["label"] == "unusable"),
            "usable": sum(1 for r in rs if r["label"] == "usable"),
            "csv": os.path.basename(CSV)}


def fold_partition():
    per_fold = json.load(open(ARMB_JSON, encoding="utf-8"))["per_fold"]
    assert len(per_fold) == 5
    rs = rows()
    out = []
    for k, f in enumerate(per_fold, 1):
        d = set(f["test_donors"])
        out.append({"fold": k, "test_donors": sorted(d),
                    "n_test_human": sum(1 for r in rs if r["patient_id"] in d
                                        and r["label_source"] == "human"),
                    "n_train_rows": sum(1 for r in rs if r["patient_id"] not in d),
                    "armB_macro_f1": f["macro_f1"],
                    "armB_tuned_macro_f1": f["tuned"]["macro_f1"]})
    assert sum(f["n_test_human"] for f in out) == 847
    return out


def _find_src(fn):
    for d in IMG_SRC:
        q = os.path.join(d, fn)
        if os.path.isfile(q):
            return q
    raise FileNotFoundError(fn)


def prepare_images():
    """One-time 1024px cache of all dataset slides (skips files already cached)."""
    from PIL import Image
    os.makedirs(CACHE, exist_ok=True)
    rs = rows()
    n_new = 0
    for i, r in enumerate(rs):
        p = os.path.join(CACHE, r["image_id"])
        if os.path.exists(p):
            continue
        with Image.open(_find_src(r["image_id"])) as im:
            im = im.convert("RGB")
            s = MAXEDGE / max(im.size)
            if s < 1:
                im = im.resize((round(im.width * s), round(im.height * s)))
            im.save(p, quality=90)
        n_new += 1
        if n_new % 250 == 0:
            print(f"  cached {n_new} new images ({i + 1}/{len(rs)} checked)")
    n_have = sum(1 for r in rs if os.path.exists(os.path.join(CACHE, r["image_id"])))
    print(f"image cache ready: {n_have}/{len(rs)} at <= {MAXEDGE}px ({n_new} newly resized)")
    assert n_have == len(rs)


def _link(dst_dir, name, src):
    os.makedirs(dst_dir, exist_ok=True)
    q = os.path.join(dst_dir, name)
    if os.path.exists(q) or os.path.islink(q):
        return
    try:
        os.symlink(src, q)
    except OSError:
        try:
            os.link(src, q)
        except OSError:
            import shutil
            shutil.copy(src, q)


def build_trees():
    """Per-fold ImageFolder trees (train/val/test) from the cache. Idempotent."""
    from sklearn.model_selection import GroupShuffleSplit
    rs = rows()
    for part in fold_partition():
        k, d = part["fold"], set(part["test_donors"])
        te = [r for r in rs if r["patient_id"] in d and r["label_source"] == "human"]
        pool = [r for r in rs if r["patient_id"] not in d]
        tr_i, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                          .split(pool, None, [r["patient_id"] for r in pool]))
        tr, va = [pool[i] for i in tr_i], [pool[i] for i in va_i]
        assert not ({r["patient_id"] for r in tr} & {r["patient_id"] for r in va})
        assert not ({r["patient_id"] for r in pool} & d), f"fold {k} donor leak"
        for split, part_rows in (("train", tr), ("val", va), ("test", te)):
            for r in part_rows:
                src = os.path.join(CACHE, r["image_id"])
                _link(os.path.join(TREES, f"fold{k}", split, r["label"]), r["image_id"], src)
                if split == "train" and r["label"] == "unusable":
                    for j in range(1, DUP):
                        _link(os.path.join(TREES, f"fold{k}", split, r["label"]),
                              f"dup{j}_{r['image_id']}", src)
        n_un = sum(1 for r in tr if r["label"] == "unusable")
        print(f"fold{k}: train {len(tr)} (+{(DUP - 1) * n_un} dups) "
              f"val {len(va)} ({sum(1 for r in va if r['label_source'] == 'human')} human) "
              f"test {len(te)}")
    print("fold trees ready")


def _predict(model, rs_, tree):
    u = [k for k, v in model.names.items() if v == "usable"][0]
    paths = [os.path.join(tree, r["label"], r["image_id"]) for r in rs_]
    out = []
    for i in range(0, len(paths), 64):
        for res in model.predict(paths[i:i + 64], imgsz=IMGSZ, device=DEVICE, verbose=False):
            out.append(float(res.probs.data[u]))
    return np.array(out)


def run_fold(k, epochs=None, demo=False):
    from sklearn.metrics import f1_score
    from sklearn.model_selection import GroupShuffleSplit
    from ultralytics import YOLO
    os.makedirs(OUT, exist_ok=True)
    done = os.path.join(OUT, f"fold{k}_done.json")
    if not demo and os.path.exists(done):
        print(f"fold {k}: already complete - skipping (delete {done} to re-run)")
        return json.load(open(done, encoding="utf-8"))

    rs = rows()
    part = fold_partition()[k - 1]
    d = set(part["test_donors"])
    te = [r for r in rs if r["patient_id"] in d and r["label_source"] == "human"]
    pool = [r for r in rs if r["patient_id"] not in d]
    _, va_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                   .split(pool, None, [r["patient_id"] for r in pool]))
    va_h = [pool[i] for i in va_i if pool[i]["label_source"] == "human"]
    assert len({v["label"] for v in va_h}) == 2, f"fold {k}: degenerate human val"

    m = YOLO("yolov8s-cls.pt")
    m.train(data=os.path.join(TREES, f"fold{k}"), epochs=epochs or EPOCHS, imgsz=IMGSZ,
            batch=BATCH, seed=SEED, device=DEVICE, patience=PATIENCE, project=OUT,
            name=f"fold{k}" + ("_demo" if demo else ""), exist_ok=True, workers=WORKERS)
    best = YOLO(os.path.join(OUT, f"fold{k}" + ("_demo" if demo else ""),
                             "weights", "best.pt"))

    vp = _predict(best, va_h, os.path.join(TREES, f"fold{k}", "val"))
    vg = np.array([CLASSES.index(r["label"]) for r in va_h])
    thr, bf = 0.5, -1.0
    for t in np.unique(np.round(vp, 3)):
        f = f1_score(vg, (vp >= t).astype(int), labels=[0, 1], average="macro")
        if f > bf:
            bf, thr = f, float(t)
    print(f"fold {k}: threshold tuned on {len(va_h)} human val rows -> {thr:.3f}")

    tp = _predict(best, te, os.path.join(TREES, f"fold{k}", "test"))
    if demo:
        tg = np.array([CLASSES.index(r["label"]) for r in te])
        print(f"fold {k} DEMO test macro-F1 "
              f"{f1_score(tg, (tp >= 0.5).astype(int), average='macro'):.4f} "
              f"(n={len(te)}; demo numbers are never quoted)")
        return None
    with open(os.path.join(OUT, f"fold{k}_preds.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "true", "p_usable"])
        for r, q in zip(te, tp):
            w.writerow([r["image_id"], r["label"], f"{q:.4f}"])
    rec = {"fold": k, "n_test": len(te), "tuned_threshold": round(thr, 3)}
    json.dump(rec, open(done, "w", encoding="utf-8"))
    print(f"FOLD {k} DONE n_test={len(te)}")
    return rec


def _metrics(gts, preds, probs):
    from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                                 precision_recall_fscore_support, roc_auc_score)
    p, r, f, _ = precision_recall_fscore_support(gts, preds, labels=[0, 1], zero_division=0)
    return {"macro_f1": round(float(f1_score(gts, preds, average="macro")), 4),
            "accuracy": round(float(accuracy_score(gts, preds)), 4),
            "auc": round(float(roc_auc_score(gts, probs)), 4) if len(set(gts)) == 2 else None,
            "recall": {"unusable": round(float(r[0]), 4), "usable": round(float(r[1]), 4)},
            "precision": {"unusable": round(float(p[0]), 4), "usable": round(float(p[1]), 4)},
            "confusion_matrix": confusion_matrix(gts, preds, labels=[0, 1]).tolist(),
            "n_test": int(len(gts))}


def aggregate():
    part = fold_partition()
    per_fold, cm, cm_t, n = [], np.zeros((2, 2), int), np.zeros((2, 2), int), 0
    for k in range(1, 6):
        rec = json.load(open(os.path.join(OUT, f"fold{k}_done.json"), encoding="utf-8"))
        rs_ = list(csv.DictReader(open(os.path.join(OUT, f"fold{k}_preds.csv"),
                                       encoding="utf-8")))
        g = np.array([CLASSES.index(r["true"]) for r in rs_])
        pr = np.array([float(r["p_usable"]) for r in rs_])
        m = _metrics(g, (pr >= 0.5).astype(int), pr)
        mt = _metrics(g, (pr >= rec["tuned_threshold"]).astype(int), pr)
        m.update({"fold": k, "tuned": {**mt, "threshold": rec["tuned_threshold"]},
                  "delta_vs_armB_argmax": round(m["macro_f1"] - part[k - 1]["armB_macro_f1"], 4),
                  "delta_vs_armB_tuned": round(mt["macro_f1"] - part[k - 1]["armB_tuned_macro_f1"], 4)})
        per_fold.append(m)
        cm += np.array(m["confusion_matrix"])
        cm_t += np.array(mt["confusion_matrix"])
        n += m["n_test"]
        print(f"fold {k}: macro-F1 {m['macro_f1']:.4f} (tuned {mt['macro_f1']:.4f})  "
              f"unus recall {m['recall']['unusable']:.3f}  AUC {m['auc']}  "
              f"delta vs ArmB {m['delta_vs_armB_argmax']:+.4f}")
    assert n == 847, n

    def ms(get):
        xs = [get(f) for f in per_fold]
        return round(float(np.mean(xs)), 4), round(float(np.std(xs)), 4)

    summary = {
        "task": "yolov8s_cls_armF_combined_donor_grouped_cv_foldmatched_armB",
        "config": {"model": "yolov8s-cls (ultralytics as shipped)", "imgsz": IMGSZ,
                   "batch": BATCH, "epochs_max": EPOCHS, "patience": PATIENCE,
                   "seed": SEED, "oversample": f"train unusable x{DUP}",
                   "image_cache": f"{MAXEDGE}px sources", "csv": os.path.basename(CSV),
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
        "armB_reference": {"argmax_macro_f1": [q["armB_macro_f1"] for q in part],
                           "tuned_macro_f1": [q["armB_tuned_macro_f1"] for q in part]},
        "per_fold": per_fold,
    }
    json.dump(summary, open(os.path.join(OUT, "cv_metrics.json"), "w", encoding="utf-8"),
              indent=2)
    a = summary["aggregate"]
    print(f"\nAGGREGATE macro-F1 {a['macro_f1'][0]} +/- {a['macro_f1'][1]}  "
          f"(tuned {a['tuned_macro_f1'][0]} +/- {a['tuned_macro_f1'][1]})  AUC {a['auc'][0]}")
    print(f"paired vs Arm B: argmax {a['mean_delta_vs_armB_argmax']:+.4f}  "
          f"tuned {a['mean_delta_vs_armB_tuned']:+.4f}")
    return summary


if __name__ == "__main__":
    import sys as _sys
    _cmd = _sys.argv[1]
    if _cmd == "prep":
        prepare_images()
        build_trees()
    elif _cmd == "fold":
        run_fold(int(_sys.argv[2]))
    elif _cmd == "agg":
        aggregate()
