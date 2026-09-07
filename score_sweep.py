"""Graded-confidence labelling: ask for a 0-10 score, then sweep the usable/unusable threshold.

A binary prompt throws away everything except the side of the line. A graded score keeps the
ordering, so ONE run yields every operating point for free -- the same trick that made the tiling
K-sweep cheap.

  python score_sweep.py <prompt.txt> <manifest.csv> <tag>
"""
import concurrent.futures as cf, csv, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe"))
import spend
from vertex_common import IMG_DIR, REPO, encode, get_client, img_part, score, text_part

MAXEDGE = int(os.environ.get("PROBE_MAXEDGE", "1512"))
WORKERS = int(os.environ.get("LOOP_WORKERS", "24"))
MAXTOK = int(os.environ.get("LOOP_MAXTOK", "1024"))
prompt_path, manifest, tag = sys.argv[1], sys.argv[2], sys.argv[3]
PROMPT = open(prompt_path, encoding="utf-8").read().strip()
rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
fns = [r["image_id"] for r in rows]; truth = [r["label"] for r in rows]
client, model = get_client()

def run(fn):
    try:
        r = client.chat.completions.create(model=model, max_tokens=MAXTOK, temperature=0,
            messages=[{"role": "user", "content": [text_part(PROMPT),
                       img_part(encode(os.path.join(IMG_DIR, fn), MAXEDGE))]}])
        spend.add(model, r.usage)
        t = (r.choices[0].message.content or "")
        m = re.findall(r"SCORE:\s*(\d+)", t) or re.findall(r"\b(\d+)\b", t)
        return int(m[-1]) if m else None
    except Exception:
        return None

sc = list(cf.ThreadPoolExecutor(max_workers=WORKERS).map(run, fns))
bad = sum(1 for s in sc if s is None)
print(f"scores parsed {len(sc)-bad}/{len(sc)}  (unparsable {bad})")
import collections; print("score distribution:", dict(sorted(collections.Counter(sc).items(), key=lambda kv:(kv[0] is None, kv[0]))))
with open(os.path.join(REPO,"probe",f"sweep_scores_{tag}.csv"),"w",newline="",encoding="utf-8") as fh:
    w=csv.writer(fh); w.writerow(["image_id","true","score"])
    for i,fn in enumerate(fns): w.writerow([fn,truth[i],sc[i]])
best=None
for thr in range(1,11):
    preds=["usable" if (s is not None and s>=thr) else "unusable" for s in sc]
    m=score(truth,preds,f"{tag} usable if score>={thr}")
    if m and (best is None or m["accuracy"]>best["accuracy"]): best=m
if best: print(f"\nBEST {tag}: {best['tag']}  acc {best['accuracy']:.3f}  macro-F1 {best['macro_f1']:.3f}  unus.rec {best['unusable_recall']:.3f}  use.rec {best['usable_recall']:.3f}")
spend.report(f"sweep-{tag}", model)
