"""
fable_probe.py - can Claude Fable 5 (multimodal) label slice quality per the expert's rubric?

Honest probe: a FIXED prompt (the expert's rubric verbatim) is run over a held-out balanced
slice of the human labels (probe/probe_testset.csv, 60 unusable / 60 usable). The prompt
never sees the labels. Scored vs the expert's ground truth; the number that matters is
UNUSABLE RECALL (compare to ResNet-50 @ 640px ~0.74).

    ANTHROPIC_API_KEY=... python probe/fable_probe.py         # full 120
    ANTHROPIC_API_KEY=... python probe/fable_probe.py 10      # quick 10-image smoke

Needs an Anthropic API key (env ANTHROPIC_API_KEY, or an `ant auth login` profile).
Cost estimate: ~120 images x ~1.8k input tokens @ Fable 5 ($10/1M) ~= a couple of dollars.
"""
import os
import sys
import csv
import io
import base64

import numpy as np
from PIL import Image
import anthropic
from sklearn.metrics import (f1_score, confusion_matrix,
                             precision_recall_fscore_support, accuracy_score)

MODEL   = "claude-fable-5"
CSV     = "probe/probe_testset.csv"
IMG_DIR = "review_tool/webapp/review_images"
OUT     = "probe/fable_probe_results.csv"
MAXEDGE = 1512          # resize long edge before sending (detail vs token cost)
CLASSES = ["unusable", "usable"]

# --- the fixed prompt: the expert's rubric, verbatim (Slice_Usability_Rubric.docx) ---
PROMPT = (
    "You are a histopathology quality reviewer assessing a Human Protein Atlas small-intestine "
    "tissue section. A slice is USABLE if it contains AT LEAST ONE 'qualifying stretch' of "
    "epithelium anywhere in the image, at any orientation. A qualifying stretch requires ALL of: "
    "(1) at least about 5 adjacent epithelial cells in a clear LINEAR arrangement; "
    "(2) BOTH surfaces visible - the apical surface facing the lumen and the basal surface facing "
    "the underlying tissue; (3) staining that can be read consistently along the stretch. "
    "Stain colour does not matter (brown signal or blue nuclear counterstain are both fine). "
    "Poor or complex regions elsewhere in the image do NOT matter - one qualifying stretch is enough. "
    "A slice is UNUSABLE if NO qualifying stretch exists - for example crypt cross-sections "
    "(circular, tightly packed profiles from a horizontal cut through a villus base), gland cells, "
    "goblet-cell-dominated regions, or crypt-to-villus transition zones where mixed differentiation "
    "makes reading unreliable. If the image is genuinely ambiguous, answer unusable. "
    "Answer with exactly one word: usable or unusable."
)


def encode(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = MAXEDGE / max(w, h)
    if s < 1:
        im = im.resize((int(w * s), int(h * s)), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def classify(client, path):
    resp = client.messages.create(
        model=MODEL, max_tokens=16,          # Fable 5: no thinking/sampling params (they 400)
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                         "data": encode(path)}},
            {"type": "text", "text": PROMPT},
        ]}],
    )
    txt = " ".join(b.text for b in resp.content if b.type == "text").strip().lower()
    if "unusable" in txt:   return "unusable", txt      # check 'unusable' first (contains 'usable')
    if "usable" in txt:     return "usable", txt
    return "?", txt


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    if limit:
        rows = rows[:limit]
    client = anthropic.Anthropic()
    print(f"probing {len(rows)} slides with {MODEL} ...")

    y_true, y_pred, records = [], [], []
    for i, r in enumerate(rows, 1):
        path = os.path.join(IMG_DIR, r["image_id"])
        try:
            pred, txt = classify(client, path)
        except Exception as e:
            pred, txt = "ERR", str(e)[:120]
        y_true.append(r["true_label"]); y_pred.append(pred)
        records.append([r["image_id"], r["true_label"], pred, txt])
        if i % 10 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    os.makedirs("probe", exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["image_id", "true_label", "fable_pred", "raw_answer"])
        w.writerows(records)

    # score on the slides that got a clean label
    idx = {c: i for i, c in enumerate(CLASSES)}
    keep = [i for i, p in enumerate(y_pred) if p in CLASSES]
    dropped = len(y_pred) - len(keep)
    yt = np.array([idx[y_true[i]] for i in keep])
    yp = np.array([idx[y_pred[i]] for i in keep])
    macro = f1_score(yt, yp, average="macro")
    acc = accuracy_score(yt, yp)
    pr, rc, f1c, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(yt, yp, labels=[0, 1])

    print("\n================ FABLE 5 PROBE (vs the expert's labels) ================")
    print(f"scored {len(keep)}/{len(y_pred)}  ({dropped} unparsable/errored)")
    print(f"accuracy {acc:.3f}   macro-F1 {macro:.3f}")
    for k, c in enumerate(CLASSES):
        print(f"  {c:9s}  precision {pr[k]:.3f}  recall {rc[k]:.3f}")
    print(f"** UNUSABLE recall {rc[0]:.3f}  (ResNet-50 @640px = 0.738) **")
    print(f"confusion [rows=true, cols=pred] order {CLASSES}:\n{cm}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
