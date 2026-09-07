"""Shared Vertex/Gemini plumbing for the labeller experiments.

Factored out of tune_prompt.py / make_test_audit.py so few_shot_label.py and tile_label.py score
on IDENTICAL machinery — same client, same image encoding, same one-word parse, same sklearn
metrics and label order (CLASSES = ["unusable", "usable"], matching train_quality_cnn.py).
Any comparison between experiments is therefore apples-to-apples.
"""
import base64
import io
import json
import os

import numpy as np
from PIL import Image
from openai import OpenAI
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG_DIR = os.path.join(REPO, "review_tool", "webapp", "review_images")
CLASSES = ["unusable", "usable"]
REGION = os.environ.get("VERTEX_REGION", "us-central1")


def get_client(model=None):
    """ADC-authenticated OpenAI-compatible client on the Vertex openapi endpoint."""
    from google.auth import default
    import google.auth.transport.requests as greq
    creds, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    creds.refresh(greq.Request())
    adc = os.path.join(os.environ.get("APPDATA", ""), "gcloud",
                       "application_default_credentials.json")
    project = os.environ.get("VERTEX_PROJECT") or json.load(open(adc)).get("quota_project_id")
    base = (f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{project}"
            f"/locations/{REGION}/endpoints/openapi")
    model = model or os.environ.get("PROBE_MODEL", "google/gemini-2.5-flash")
    print(f"[vertex] project={project} region={REGION} model={model}")
    return OpenAI(base_url=base, api_key=creds.token, max_retries=5, timeout=120), model


def encode_im(im, maxedge, quality=90):
    """PIL image -> base64 JPEG, downscaled only if it exceeds maxedge."""
    w, h = im.size
    s = maxedge / max(w, h)
    if s < 1:
        im = im.resize((int(w * s), int(h * s)), Image.BILINEAR)
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=quality)
    return base64.standard_b64encode(buf.getvalue()).decode()


def encode(path, maxedge, quality=90):
    return encode_im(Image.open(path), maxedge, quality)


def img_part(b64):
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}


def text_part(t):
    return {"type": "text", "text": t}


def parse_label(t):
    """Answer -> class. 'unusable' is checked first (it contains 'usable').

    Prompts that reason before deciding must end with a 'FINAL:' marker; only the text after the
    LAST marker is parsed, so a verdict is never contaminated by the word 'unusable' appearing in
    the preceding rationale.
    """
    t = (t or "").strip().lower()
    if "final:" in t:
        t = t.rsplit("final:", 1)[1]
    return "unusable" if "unusable" in t else ("usable" if "usable" in t else "?")


def parse_yes_no(t):
    t = (t or "").strip().lower()
    if "yes" in t:
        return "yes"
    return "no" if "no" in t else "?"


def score(truth, preds, tag, n_total=None):
    """Print + return metrics over the parseable subset. Mirrors tune_prompt.py exactly."""
    idx = {c: i for i, c in enumerate(CLASSES)}
    keep = [i for i, p in enumerate(preds) if p in CLASSES]
    if not keep:
        print(f"[{tag}] ALL {len(preds)} unparsable/errored — nothing to score.")
        return None
    yt = np.array([idx[truth[i]] for i in keep])
    yp = np.array([idx[preds[i]] for i in keep])
    acc = accuracy_score(yt, yp)
    macro = f1_score(yt, yp, average="macro")
    pr, rc, _, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    n = n_total or len(preds)
    print(f"\n[{tag}]  n={n} scored={len(keep)} err={len(preds) - len(keep)}")
    print(f"  ACCURACY {acc:.3f}   macro-F1 {macro:.3f}")
    print(f"  unusable  recall {rc[0]:.3f}  precision {pr[0]:.3f}")
    print(f"  usable    recall {rc[1]:.3f}  precision {pr[1]:.3f}")
    print(f"  confusion [true x pred] {CLASSES}: {cm.tolist()}")
    return {"tag": tag, "n": n, "scored": len(keep), "errors": len(preds) - len(keep),
            "accuracy": round(float(acc), 4), "macro_f1": round(float(macro), 4),
            "unusable_recall": round(float(rc[0]), 4), "unusable_precision": round(float(pr[0]), 4),
            "usable_recall": round(float(rc[1]), 4), "usable_precision": round(float(pr[1]), 4),
            "confusion": cm.tolist()}
