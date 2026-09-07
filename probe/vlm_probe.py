"""
vlm_probe.py - can a cheap/free multimodal model label slice quality per the expert's rubric v1.0?

Provider-parameterized, HONEST blind probe: a FIXED prompt (rubric v1.0 verbatim) is run over a
held-out balanced slice of the human labels (probe/probe_testset.csv, 60 unusable / 60 usable).
The prompt never sees the labels. Scored vs the expert's ground truth; the number that matters is
UNUSABLE RECALL. Anchors already on record: ResNet-50 @640px = 0.738 ; Quilt-LLaVA-7B = 0.000.

    PROBE_PROVIDER=gemini  GEMINI_API_KEY=...     python probe/vlm_probe.py [limit]
    PROBE_PROVIDER=nvidia  NVIDIA_API_KEY=...     python probe/vlm_probe.py [limit]
    PROBE_PROVIDER=openrouter OPENROUTER_API_KEY=... PROBE_MODEL=... python probe/vlm_probe.py
    PROBE_PROVIDER=claude                          python probe/vlm_probe.py [limit]

The `claude` provider shells out to `claude -p` (Claude Code headless) on the Pro plan's Agent
SDK credit - no ANTHROPIC_API_KEY, no API billing. `gemini`/`nvidia`/`openrouter` use the OpenAI
SDK against each provider's OpenAI-compatible endpoint. `limit` (optional) probes only the first
N slides for a quick smoke test.
"""
import os
import sys
import csv
import io
import json
import time
import base64
import shutil
import subprocess

import numpy as np
from PIL import Image
from sklearn.metrics import (f1_score, confusion_matrix,
                             precision_recall_fscore_support, accuracy_score)

# ------------------------------------------------------------------
# CONFIG - edit these only
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR   = os.path.dirname(SCRIPT_DIR)
CSV        = os.path.join(SCRIPT_DIR, "probe_testset.csv")
IMG_DIR    = os.path.abspath(os.path.join(REPO_DIR, "review_tool", "webapp", "review_images"))
MAXEDGE    = int(os.environ.get("PROBE_MAXEDGE", "1512"))   # resize long edge (detail vs token cost)
CLASSES    = ["unusable", "usable"]
PROVIDER   = os.environ.get("PROBE_PROVIDER", "gemini").lower()

# per-provider OpenAI-compatible endpoint + default model + api-key env var + min seconds/call
OAI = {
    "gemini":     dict(base="https://generativelanguage.googleapis.com/v1beta/openai/",
                       model="gemini-flash-latest", key="GEMINI_API_KEY",  gap=4.5),  # current stable Flash; ~15 RPM free
    "nvidia":     dict(base="https://integrate.api.nvidia.com/v1",
                       model="meta/llama-3.2-90b-vision-instruct", key="NVIDIA_API_KEY", gap=1.6),  # 40 RPM
    "openrouter": dict(base="https://openrouter.ai/api/v1",
                       model="meta-llama/llama-3.2-90b-vision-instruct", key="OPENROUTER_API_KEY", gap=3.2),
}
MODEL = os.environ.get("PROBE_MODEL", OAI.get(PROVIDER, {}).get("model", ""))
GAP   = float(os.environ.get("PROBE_SLEEP", OAI.get(PROVIDER, {}).get("gap", 0)))
# 'none' disables Gemini 'thinking' (cheaper/faster for a one-word classification); "" = leave default
REASONING = os.environ.get("PROBE_REASONING", "none" if PROVIDER == "gemini" else "")

# --- the fixed prompt: rubric v1.0 (Slice_Usability_Rubric_v1_0.docx, frozen 2026-08-12) ---
PROMPT = (
    "You are a histopathology quality reviewer assessing a Human Protein Atlas small-intestine "
    "tissue section at native resolution. A slice is USABLE if it contains AT LEAST ONE "
    "'qualifying stretch' of villus epithelium anywhere in the image, at any orientation; poor "
    "or complex regions elsewhere do not matter. A qualifying stretch requires ALL of: "
    "(1) at least about 5 adjacent villus epithelial cells in a clear LINEAR arrangement; "
    "(2) BOTH surfaces visible - the apical surface facing the lumen and the basal surface facing "
    "the underlying tissue; (3) staining that can be read consistently along the stretch. "
    "Stain colour does NOT matter, and ABSENCE of signal does NOT matter: an image with a "
    "qualifying stretch but no staining in the epithelium is still USABLE. A slice is UNUSABLE if "
    "NO qualifying stretch exists - for example crypt cross-sections (circular, tightly packed "
    "profiles from a horizontal cut through a villus base), gland cells, goblet-cell-dominated "
    "regions, or stretches that fall within the crypt-to-villus transition zone (these never "
    "qualify, because mixed differentiation makes reading unreliable). If the image is genuinely "
    "ambiguous, answer unusable. Answer with exactly one word: usable or unusable."
)

# go/no-go bar (research/scoping/03b-vlm-labeller.md)
BAR = dict(macro_f1=0.80, unusable_recall=0.70, unusable_precision=0.60)
ANCHORS = dict(quilt_llava_unusable_recall=0.000, resnet50_640px_unusable_recall=0.738)


def _parse(txt):
    t = (txt or "").strip().lower()
    if "unusable" in t:  return "unusable"     # check 'unusable' first (contains 'usable')
    if "usable" in t:    return "usable"
    return "?"


def encode(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = MAXEDGE / max(w, h)
    if s < 1:
        im = im.resize((int(w * s), int(h * s)), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def classify_openai(client, path):
    kwargs = dict(
        model=MODEL, max_tokens=512, temperature=0,   # 512 = room if 'thinking' stays on
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {
                "url": "data:image/jpeg;base64," + encode(path)}},
        ]}],
    )
    if PROVIDER == "gemini" and REASONING:
        kwargs["reasoning_effort"] = REASONING
    resp = client.chat.completions.create(**kwargs)
    ch = resp.choices[0]
    txt = (ch.message.content or "").strip()
    if not txt:
        txt = f"[empty; finish_reason={ch.finish_reason}]"
    return _parse(txt), txt


_CLAUDE_BIN = shutil.which("claude") or "claude"


def classify_claude(path):
    # `claude -p` reads the image via its Read tool (forward-slash abs path), one-word answer.
    prompt = (PROMPT + "\n\nRead the image at " + path.replace("\\", "/") +
              " and answer with exactly one word: usable or unusable.")
    cmd = [_CLAUDE_BIN, "-p", "--allowedTools", "Read"]
    if os.environ.get("PROBE_MODEL"):
        cmd += ["--model", os.environ["PROBE_MODEL"]]
    out = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=180)
    txt = (out.stdout or "").strip()
    return _parse(txt), txt


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
    if limit:
        rows = rows[:limit]

    if PROVIDER == "claude":
        model_tag = os.environ.get("PROBE_MODEL", "claude-code-default")
        client = None
    elif PROVIDER in OAI:
        from openai import OpenAI
        key = os.environ.get(OAI[PROVIDER]["key"])
        if not key:
            sys.exit(f"set {OAI[PROVIDER]['key']} for provider '{PROVIDER}'")
        client = OpenAI(base_url=OAI[PROVIDER]["base"], api_key=key, max_retries=6)
        model_tag = MODEL
    else:
        sys.exit(f"unknown PROBE_PROVIDER '{PROVIDER}' (use gemini|nvidia|openrouter|claude)")

    print(f"probing {len(rows)} slides | provider={PROVIDER} model={model_tag} gap={GAP}s")
    y_true, y_pred, records = [], [], []
    for i, r in enumerate(rows, 1):
        path = os.path.join(IMG_DIR, r["image_id"])
        try:
            pred, txt = classify_claude(path) if PROVIDER == "claude" else classify_openai(client, path)
        except Exception as e:
            pred, txt = "ERR", str(e)[:160]
        y_true.append(r["true_label"]); y_pred.append(pred)
        records.append([r["image_id"], r["true_label"], pred, txt.replace("\n", " ")[:300]])
        if i % 5 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}  last={pred}")
        if GAP and i < len(rows):
            time.sleep(GAP)

    safe = "".join(c if c.isalnum() else "_" for c in f"{PROVIDER}_{model_tag}")
    out_csv = os.path.join(SCRIPT_DIR, f"vlm_probe_results_{safe}.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["image_id", "true_label", "pred", "raw_answer"])
        w.writerows(records)

    # score on the slides that got a clean label
    idx = {c: i for i, c in enumerate(CLASSES)}
    keep = [i for i, p in enumerate(y_pred) if p in CLASSES]
    dropped = len(y_pred) - len(keep)
    if not keep:
        print(f"\nNO CLEAN PREDICTIONS - all {dropped} errored/unparsable. First raw answer:")
        print("  " + (records[0][3] if records else "(none)"))
        sys.exit(1)
    yt = np.array([idx[y_true[i]] for i in keep])
    yp = np.array([idx[y_pred[i]] for i in keep])
    macro = float(f1_score(yt, yp, average="macro")) if keep else float("nan")
    acc = float(accuracy_score(yt, yp)) if keep else float("nan")
    pr, rc, f1c, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1], zero_division=0)
    cm = confusion_matrix(yt, yp, labels=[0, 1])
    u_recall, u_prec = float(rc[0]), float(pr[0])
    passed = (macro >= BAR["macro_f1"] and u_recall >= BAR["unusable_recall"]
              and u_prec >= BAR["unusable_precision"])

    summary = dict(
        provider=PROVIDER, model=model_tag, n=len(y_pred), n_scored=len(keep), n_dropped=dropped,
        accuracy=round(acc, 4), macro_f1=round(macro, 4),
        precision={c: round(float(pr[k]), 4) for k, c in enumerate(CLASSES)},
        recall={c: round(float(rc[k]), 4) for k, c in enumerate(CLASSES)},
        unusable_recall=round(u_recall, 4), unusable_precision=round(u_prec, 4),
        confusion_matrix=cm.tolist(), confusion_rows_true_cols_pred_order=CLASSES,
        go_no_go_bar=BAR, passes_bar=bool(passed), anchors=ANCHORS,
    )
    out_json = os.path.join(SCRIPT_DIR, f"vlm_probe_summary_{safe}.json")
    json.dump(summary, open(out_json, "w", encoding="utf-8"), indent=2)

    print(f"\n============ VLM PROBE : {PROVIDER} / {model_tag} (vs the expert's labels) ============")
    print(f"scored {len(keep)}/{len(y_pred)}  ({dropped} unparsable/errored)")
    print(f"accuracy {acc:.3f}   macro-F1 {macro:.3f}")
    for k, c in enumerate(CLASSES):
        print(f"  {c:9s}  precision {pr[k]:.3f}  recall {rc[k]:.3f}")
    print(f"** UNUSABLE recall {u_recall:.3f}  precision {u_prec:.3f} "
          f"(ResNet-50 @640px = 0.738 ; Quilt-LLaVA = 0.000) **")
    print(f"confusion [rows=true, cols=pred] order {CLASSES}:\n{cm}")
    print(f"GO/NO-GO bar {BAR} -> {'PASS' if passed else 'FAIL'}")
    print(f"\nwrote {out_csv}\nwrote {out_json}")


if __name__ == "__main__":
    main()
