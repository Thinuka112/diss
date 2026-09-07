"""
tune_prompt.py - score a Gemini prompt against a gold manifest (concurrent), list disagreements.
Used in the prompt-optimization loop: run -> inspect misses -> revise prompt -> re-run.

  set -a; source probe/keys.local.env; set +a
  PROBE_MAXEDGE=3072 python tune_prompt.py <manifest.csv> <prompt.txt> [tag]

Env: PROBE_MODEL (gemini-flash-latest), PROBE_MAXEDGE (3072), TUNE_WORKERS (24),
     TUNE_REASONING ('' = model default; or 'low'/'medium'/'high' to force thinking).
"""
import os, sys, csv, io, base64
import concurrent.futures as cf
import numpy as np
from PIL import Image
from openai import OpenAI
from sklearn.metrics import (f1_score, accuracy_score,
                             precision_recall_fscore_support, confusion_matrix)

REPO = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(REPO, "review_tool", "webapp", "review_images")
CLASSES = ["unusable", "usable"]
BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"
MODEL = os.environ.get("PROBE_MODEL", "gemini-flash-latest")
MAXEDGE = int(os.environ.get("PROBE_MAXEDGE", "3072"))
WORKERS = int(os.environ.get("TUNE_WORKERS", "24"))
REASON = os.environ.get("TUNE_REASONING", "")
MAXTOK = int(os.environ.get("TUNE_MAXTOK", "1024"))  # bump when thinking is on (it eats the budget)

manifest = sys.argv[1]
prompt_path = sys.argv[2]
tag = sys.argv[3] if len(sys.argv) > 3 else os.path.splitext(os.path.basename(prompt_path))[0]

rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
PROMPT = open(prompt_path, encoding="utf-8").read().strip()
if os.environ.get("PROBE_PROVIDER") == "vertex":
    import json as _json
    from google.auth import default as _gdefault
    import google.auth.transport.requests as _greq
    _creds, _ = _gdefault(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    _creds.refresh(_greq.Request())
    _adc = os.path.join(os.environ.get("APPDATA", ""), "gcloud", "application_default_credentials.json")
    PROJECT = os.environ.get("VERTEX_PROJECT") or _json.load(open(_adc)).get("quota_project_id")
    REGION = os.environ.get("VERTEX_REGION", "us-central1")
    BASE = f"https://{REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/{REGION}/endpoints/openapi"
    MODEL = os.environ.get("PROBE_MODEL", "google/gemini-2.5-flash")
    client = OpenAI(base_url=BASE, api_key=_creds.token, max_retries=5, timeout=90)
    print(f"[vertex] project={PROJECT} region={REGION} model={MODEL}")
else:
    client = OpenAI(base_url=BASE, api_key=os.environ["GEMINI_API_KEY"], max_retries=5, timeout=60)


def encode(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = MAXEDGE / max(w, h)
    if s < 1:
        im = im.resize((int(w * s), int(h * s)), Image.BILINEAR)
    b = io.BytesIO(); im.save(b, "JPEG", quality=90)
    return base64.standard_b64encode(b.getvalue()).decode()


def parse(t):
    t = (t or "").strip().lower()
    return "unusable" if "unusable" in t else ("usable" if "usable" in t else "?")


def classify(fname):
    kw = dict(model=MODEL, max_tokens=MAXTOK, temperature=0,
              messages=[{"role": "user", "content": [
                  {"type": "text", "text": PROMPT},
                  {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encode(os.path.join(IMG_DIR, fname))}}]}])
    if REASON:
        kw["reasoning_effort"] = REASON
    try:
        r = client.chat.completions.create(**kw)
        return parse(r.choices[0].message.content)
    except Exception as e:
        return "ERR"


fnames = [r["image_id"] for r in rows]
truth = [r["label"] for r in rows]
preds = list(cf.ThreadPoolExecutor(max_workers=WORKERS).map(classify, fnames))

idx = {c: i for i, c in enumerate(CLASSES)}
keep = [i for i, p in enumerate(preds) if p in CLASSES]
if not keep:
    from collections import Counter
    print(f"[{tag}] ALL {len(preds)} unparsable/errored ({dict(Counter(preds))}). "
          f"Likely token/rate limit with thinking on — lower TUNE_MAXTOK / TUNE_WORKERS / REASONING.")
    sys.exit(1)
yt = np.array([idx[truth[i]] for i in keep])
yp = np.array([idx[preds[i]] for i in keep])
acc = accuracy_score(yt, yp)
macro = f1_score(yt, yp, average="macro")
pr, rc, f1, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
cm = confusion_matrix(yt, yp, labels=[0, 1])

dis = os.path.join(REPO, "probe", f"tune_disagree_{tag}.csv")
with open(dis, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh); w.writerow(["image_id", "true", "pred"])
    for i in range(len(fnames)):
        if preds[i] in CLASSES and preds[i] != truth[i]:
            w.writerow([fnames[i], truth[i], preds[i]])

print(f"[{tag}]  n={len(rows)} scored={len(keep)} err={len(preds)-len(keep)}  maxedge={MAXEDGE} reason={REASON or 'default'}")
print(f"  ACCURACY {acc:.3f}   macro-F1 {macro:.3f}")
print(f"  unusable  recall {rc[0]:.3f}  precision {pr[0]:.3f}")
print(f"  usable    recall {rc[1]:.3f}  precision {pr[1]:.3f}")
print(f"  confusion [true x pred] {CLASSES}: {cm.tolist()}")
print(f"  {sum(1 for i in range(len(fnames)) if preds[i] in CLASSES and preds[i]!=truth[i])} disagreements -> {dis}")
