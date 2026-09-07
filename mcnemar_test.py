"""mcnemar_test.py - McNemar's test between the two Ch5 finalists on the 847
paired predictions. Gold-only ResNet vs combined-data YOLO at both operating points.

The shipped fold*_preds.csv in models/quality_union_humanonly were written under a
hard gate. Each fold's regenerated macro-F1 had to match the stored
quality_metrics.json to 4 dp before the file was written. When those CSVs are
present this script loads them directly and needs only scipy. When they are absent
it regenerates them from the fold checkpoints under the same gate. Regeneration
needs torch, timm and the raw images.

Run from the repo root. Deterministic. Inference only.
"""
import csv
import json
import os
import sys

from scipy import stats

ARMB_DIR = os.path.join("models", "quality_union_humanonly")
YOLO_JSON = os.path.join("models", "quality_yolo_cvF", "cv_metrics.json")

armB = json.load(open(os.path.join(ARMB_DIR, "quality_metrics.json")))
stored_f1 = [f["macro_f1"] for f in armB["per_fold"]]
thr_res = [f["tuned_threshold"] for f in armB["per_fold"]]

preds_files = [os.path.join(ARMB_DIR, f"fold{k + 1}_preds.csv") for k in range(5)]

if all(os.path.exists(p) for p in preds_files):
    all_rows = []
    for k, path in enumerate(preds_files):
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                all_rows.append((r["image_id"], r["true"], float(r["p_usable"]), k + 1))
    assert len(all_rows) == 847, f"expected 847 rows, got {len(all_rows)}"
    print(f"loaded {len(all_rows)} stored per-image predictions (gate-verified at write time)")
else:
    # Regenerate from the fold checkpoints. Heavy path. Needs torch, timm and the images.
    os.environ.setdefault("QUALITY_INPUT_SIZE", "640")
    os.environ.setdefault("QUALITY_IMG_DIRS", os.pathsep.join([
        os.path.join("review_tool", "webapp", "review_images"),
        os.path.join("data_bulk", "images")]))

    import numpy as np
    import torch
    import timm
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score

    import train_quality_cnn as tqc

    test_donors = [set(map(str, f["test_donors"])) for f in armB["per_fold"]]
    ids, y, groups, human, fracs = tqc.load_rows()
    ids = np.array(ids)
    print(f"gold rows: {len(ids)}")
    imgs = tqc.preload(list(ids))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_rows = []
    for k in range(5):
        te = np.where(np.isin(groups.astype(str), list(test_donors[k])))[0]
        ck = torch.load(os.path.join(ARMB_DIR, f"quality_resnet50_fold{k + 1}.pt"),
                        map_location=dev, weights_only=False)
        assert ck.get("input_size", 640) == tqc.INPUT_SIZE, "input size mismatch"
        model = timm.create_model(tqc.ARCH, num_classes=2)
        model.load_state_dict(ck["state_dict"])
        model.to(dev).eval()
        ds = tqc.SlideDS(imgs[te], y[te], train=False)
        dl = DataLoader(ds, batch_size=8, shuffle=False)
        probs = []
        with torch.no_grad():
            for x, _, _ in dl:
                with torch.autocast("cuda", enabled=(dev.type == "cuda")):
                    p = torch.softmax(model(x.to(dev)).float(), dim=1)
                probs.append(p[:, 1].cpu().numpy())
        p_us = np.concatenate(probs)
        mf1 = f1_score(y[te], (p_us >= 0.5).astype(int), average="macro")
        ok = abs(mf1 - stored_f1[k]) < 5e-4
        print(f"fold {k + 1}: n={len(te)}  regenerated macro-F1 {mf1:.4f}  "
              f"stored {stored_f1[k]:.4f}  {'OK' if ok else 'MISMATCH'}")
        if not ok:
            sys.exit("VERIFICATION FAILED - refusing to write predictions or run the test.")
        for i, pi in zip(te, p_us):
            all_rows.append((ids[i], tqc.CLASSES[y[i]], float(pi), k + 1))

    assert len(all_rows) == 847, f"expected 847 rows, got {len(all_rows)}"
    for k in range(5):
        with open(os.path.join(ARMB_DIR, f"fold{k + 1}_preds.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image_id", "true", "p_usable"])
            for img, t, p, fk in all_rows:
                if fk == k + 1:
                    w.writerow([img, t, f"{p:.4f}"])
    print("per-fold predictions written to", ARMB_DIR)

yolo = json.load(open(YOLO_JSON))
thr_yolo = [f["tuned"]["threshold"] for f in yolo["per_fold"]]
ypred = {}
for k in range(5):
    with open(os.path.join("models", "quality_yolo_cvF", f"fold{k + 1}_preds.csv"), newline="") as f:
        for r in csv.DictReader(f):
            ypred[r["image_id"]] = (float(r["p_usable"]), k)


def mcnemar(point):
    b = c = agree = 0
    for img, t, p_res, fk in all_rows:
        p_y, ky = ypred[img]
        if point == "argmax":
            pr, py = p_res >= 0.5, p_y >= 0.5
        else:
            pr, py = p_res >= thr_res[fk - 1], p_y >= thr_yolo[ky]
        truth = (t == "usable")
        res_ok, y_ok = (pr == truth), (py == truth)
        if res_ok and not y_ok:
            b += 1
        elif y_ok and not res_ok:
            c += 1
        else:
            agree += 1
    p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if (b + c) else float("nan")
    print(f"{point:6s}: ResNet-only-correct b={b}  YOLO-only-correct c={c}  agree={agree}"
          f"  exact McNemar p = {p:.3f}")


print(f"\nMcNemar on {len(all_rows)} paired predictions (discordant pairs b vs c):")
mcnemar("argmax")
mcnemar("tuned")
