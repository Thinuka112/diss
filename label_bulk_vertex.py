"""label_bulk_vertex.py - relabel the clean HPA small-intestine pool with the VALIDATED v2 config.

Supersedes label_bulk.py, which targets the depleted AI-Studio endpoint and the v1 prompt. This
goes through probe/vertex_common.py, i.e. the byte-identical call path that scored 0.767 accuracy /
0.800 unusable recall on the 120 held-out test (LOG 2026-08-16): whole-slide, prompt v2_lenient,
MAXEDGE 1512, temperature 0, max_tokens 512, google/gemini-2.5-flash on Vertex.

Do not "improve" the config here. If it drifts from the line above, the 0.767 measurement no longer
describes these labels.

Concurrent, resumable (re-running skips rows already written), and non-destructive: writes a NEW
file and never touches the v1 labels. Excludes the 16 cancer-carrying donors (AUDIT.md D3).

    python label_bulk_vertex.py [limit]        # limit = label only the first N outstanding images

Output: models/vlm_labels/vertex_gemini_2_5_flash_v2_labels.csv  (source_image,label,donor,raw)
"""
import concurrent.futures as cf
import csv
import os
import sys
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "probe"))
import spend
from vertex_common import encode, get_client, img_part, parse_label, text_part

POOL_DIR = os.environ.get("LABEL_POOL_DIR", os.path.join(SCRIPT_DIR, "data_bulk", "images"))
OUT_DIR = os.path.join(SCRIPT_DIR, "models", "vlm_labels")
OUT_CSV = os.path.join(OUT_DIR, "vertex_gemini_2_5_flash_v2_labels.csv")
PROMPT_PATH = os.path.join(SCRIPT_DIR, "probe", "prompts", "v2_lenient.txt")
WORKERS = int(os.environ.get("LABEL_WORKERS", "24"))
MAXEDGE = int(os.environ.get("PROBE_MAXEDGE", "1512"))

# 16 cancer-carrying donors (AUDIT.md D3) - never enter the training set.
CANCER_DONORS = {"1069", "1090", "1839", "1879", "1904", "2338", "2467", "2806",
                 "3049", "3548", "603", "604", "637", "700", "707", "997"}

PROMPT = open(PROMPT_PATH, encoding="utf-8").read().strip()


def donor_of(fname):
    return os.path.basename(fname).rsplit(".", 1)[0].split("_")[-1]


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    client, model = get_client()

    all_imgs = sorted(f for f in os.listdir(POOL_DIR) if f.lower().endswith(".jpg"))
    clean = [f for f in all_imgs if donor_of(f) not in CANCER_DONORS]

    done = set()
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, newline="", encoding="utf-8") as fh:
            done = {r["source_image"] for r in csv.DictReader(fh)}
    todo = [f for f in clean if f not in done]
    if limit:
        todo = todo[:limit]

    print(f"pool={len(all_imgs)}  clean={len(clean)} (excluded {len(all_imgs) - len(clean)} "
          f"cancer-donor)  already_done={len(done)}  to_label={len(todo)}")
    print(f"config: {model} | v2_lenient | {MAXEDGE}px | temperature 0 | workers {WORKERS}")
    if not todo:
        print("nothing to do.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    new_file = not os.path.exists(OUT_CSV)
    fh = open(OUT_CSV, "a", newline="", encoding="utf-8")
    w = csv.writer(fh)
    if new_file:
        w.writerow(["source_image", "label", "donor", "raw"])
    lock = threading.Lock()
    counts = {"usable": 0, "unusable": 0, "?": 0, "ERR": 0}
    t0 = time.time()

    def work(fname):
        try:
            parts = [text_part(PROMPT), img_part(encode(os.path.join(POOL_DIR, fname), MAXEDGE))]
            r = client.chat.completions.create(
                model=model, max_tokens=512, temperature=0,
                messages=[{"role": "user", "content": parts}])
            spend.add(model, r.usage)
            txt = (r.choices[0].message.content or "").strip()
            return fname, parse_label(txt), txt
        except Exception as e:
            return fname, "ERR", f"{type(e).__name__}: {e}"[:120]

    n = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for fname, pred, txt in ex.map(work, todo):
            n += 1
            counts[pred if pred in counts else "?"] += 1
            with lock:
                w.writerow([fname, pred, donor_of(fname), txt.replace("\n", " ")[:120]])
                if n % 100 == 0:
                    fh.flush()
                    rate = n / (time.time() - t0)
                    print(f"  {n}/{len(todo)}  {rate:.2f}/s  eta {(len(todo) - n) / rate / 60:5.1f}min"
                          f"  usable={counts['usable']} unusable={counts['unusable']} "
                          f"err={counts['ERR']}", flush=True)
    fh.flush()
    fh.close()

    dt = time.time() - t0
    tot = counts["usable"] + counts["unusable"]
    print(f"\nDONE {n} in {dt / 60:.1f}min ({n / dt:.2f}/s)")
    print(f"  usable={counts['usable']}  unusable={counts['unusable']}"
          f"  ({counts['unusable'] / tot * 100:.1f}% unusable)" if tot else "")
    print(f"  unparsable={counts['?']}  errors={counts['ERR']}")
    print(f"wrote {OUT_CSV}")
    spend.report("label_bulk_v2", model)


if __name__ == "__main__":
    main()
