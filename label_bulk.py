"""
label_bulk.py - label the clean HPA small-intestine pool with a VLM (default Gemini) to build a
ResNet-50 training set. Reuses probe/vlm_probe.py's PROMPT / encode / classify_openai (rubric v1.0),
excludes the 16 cancer-carrying donors (AUDIT.md D3), and is concurrent + resumable + incremental.

    set -a; source probe/keys.local.env; set +a
    PROBE_PROVIDER=gemini python label_bulk.py [limit]

Output: models/vlm_labels/{provider}_{model}_labels.csv  columns: source_image,label,donor,raw
"""
import os, sys, csv, time, threading
import concurrent.futures as cf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "probe"))
import vlm_probe as vp                     # PROMPT, encode, classify_openai, MODEL, PROVIDER, OAI, _parse
from openai import OpenAI

# ---- CONFIG ----
POOL_DIR = os.environ.get("LABEL_POOL_DIR", os.path.join(SCRIPT_DIR, "data_bulk", "images"))
OUT_DIR  = os.path.join(SCRIPT_DIR, "models", "vlm_labels")
WORKERS  = int(os.environ.get("LABEL_WORKERS", "16"))
# 16 cancer-carrying donors (AUDIT.md D3) - never enter the training set (matches assemble_enriched_dataset.py)
CANCER_DONORS = {"1069", "1090", "1839", "1879", "1904", "2338", "2467", "2806",
                 "3049", "3548", "603", "604", "637", "700", "707", "997"}


def donor_of(fname):
    return os.path.basename(fname).rsplit(".", 1)[0].split("_")[-1]


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    if vp.PROVIDER not in vp.OAI:
        sys.exit(f"label_bulk supports OpenAI-compatible providers only (got '{vp.PROVIDER}')")
    key = os.environ.get(vp.OAI[vp.PROVIDER]["key"])
    if not key:
        sys.exit(f"set {vp.OAI[vp.PROVIDER]['key']}")
    client = OpenAI(base_url=vp.OAI[vp.PROVIDER]["base"], api_key=key, max_retries=4, timeout=40)

    safe = "".join(c if c.isalnum() else "_" for c in f"{vp.PROVIDER}_{vp.MODEL}")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, f"{safe}_labels.csv")

    all_imgs = sorted(f for f in os.listdir(POOL_DIR) if f.lower().endswith(".jpg"))
    clean = [f for f in all_imgs if donor_of(f) not in CANCER_DONORS]
    excluded = len(all_imgs) - len(clean)

    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="", encoding="utf-8") as fh:
            done = {r["source_image"] for r in csv.DictReader(fh)}
    todo = [f for f in clean if f not in done]
    if limit:
        todo = todo[:limit]

    print(f"pool={len(all_imgs)}  clean={len(clean)} (excluded {excluded} cancer-donor)  "
          f"already_labelled={len(done)}  to_label={len(todo)}  provider={vp.PROVIDER} model={vp.MODEL} workers={WORKERS}")
    if not todo:
        print("nothing to do."); return

    lock = threading.Lock()
    new_file = not os.path.exists(out_csv)
    fh = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new_file:
        w.writerow(["source_image", "label", "donor", "raw"])
    counts = {"usable": 0, "unusable": 0, "?": 0, "ERR": 0}
    t0 = time.time()

    def work(fname):
        path = os.path.join(POOL_DIR, fname)
        try:
            pred, txt = vp.classify_openai(client, path)
        except Exception as e:
            pred, txt = "ERR", str(e)[:120]
        return fname, pred, txt

    n = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fname, pred, txt in ex.map(work, todo):
            n += 1
            counts[pred if pred in counts else "?"] = counts.get(pred, 0) + 1
            with lock:
                w.writerow([fname, pred, donor_of(fname), txt.replace("\n", " ")[:120]])
                if n % 25 == 0:
                    fh.flush()
                    rate = n / (time.time() - t0)
                    eta = (len(todo) - n) / rate / 60 if rate else 0
                    print(f"  {n}/{len(todo)}  {rate:.1f}/s  eta {eta:.1f}min  "
                          f"usable={counts['usable']} unusable={counts['unusable']} err={counts['ERR']}")
    fh.flush(); fh.close()
    dt = time.time() - t0
    print(f"\nDONE {n} in {dt/60:.1f}min ({n/dt:.1f}/s). "
          f"usable={counts['usable']} unusable={counts['unusable']} unparsable={counts['?']} err={counts['ERR']}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
