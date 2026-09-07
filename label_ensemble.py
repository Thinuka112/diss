"""label_ensemble.py - relabel the clean HPA pool with the 7-member cascade ensemble.

Held-out accuracy 0.840 (vs 0.790 for the single v2 prompt), measured on probe/prompt_val.csv with
the member set fixed a priori and the vote threshold fitted on probe/prompt_tune.csv only.
See LOG 2026-08-16 (diverse ensemble + cascade).

CASCADE (this is what makes it affordable):
  Stage 1 - run the three cheap flash members on every slide.
            If all three AGREE, accept that label and stop paying.
  Stage 2 - otherwise buy the remaining four members (incl. 2.5-pro and 3x3 tiling) and take a
            7-member vote: unusable if >= VOTE_UNUSABLE members say unusable.
  ~34% of slides escalate, which reproduces the full-panel accuracy for ~44% of the full-panel cost.

MEMBERS (frozen - changing any of these invalidates the 0.840 measurement):
  1 flash_v2   v2_lenient   binary  512 tok
  2 flash_v5   v5_recog     binary  512 tok          (lenient bias)
  3 flash_v6   v6_score     graded  1024 tok -> usable if score >= 3
  4 flash_v4   v4_balanced  binary  3072 tok         (strict bias, reason-then-verdict)
  5 flash_tile tile 3x3     spatial 512 tok/tile -> usable if >= 2 of 9 tiles show a stretch
  6 pro_v2     v2_lenient   binary  512 tok   on gemini-2.5-pro
  7 pro_v6     v6_score     graded  2048 tok  on gemini-2.5-pro -> usable if score >= 3

Resumable: every finished slide is appended with all seven member votes, so a restart skips it.
Non-destructive: writes its own file and never touches earlier label sets.

  python label_ensemble.py                          # label the full clean pool
  python label_ensemble.py --manifest probe/prompt_val.csv --tag val   # validate the cascade
  python label_ensemble.py --limit 200              # smoke test on the pool

Env: ENS_WORKERS (12 slides in flight), ENS_ALERT_USD (spend alert, default 160).
"""
import argparse
import concurrent.futures as cf
import csv
import os
import re
import sys
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "probe"))
import spend
from PIL import Image
from vertex_common import (IMG_DIR, REPO, encode, encode_im, get_client, img_part, parse_label,
                           parse_yes_no, score, text_part)

FLASH = "google/gemini-2.5-flash"
PRO = "google/gemini-2.5-pro"
MAXEDGE = 1512
GRID = 3
TILE_K = 2            # usable if >= 2 of 9 tiles show a qualifying stretch
SCORE_THR = 3         # graded members: usable if score >= 3
VOTE_UNUSABLE = 5     # of 7 members, fitted on prompt_tune only
POOL_DIR = os.environ.get("LABEL_POOL_DIR", os.path.join(SCRIPT_DIR, "data_bulk", "images"))
OUT_CSV = os.path.join(SCRIPT_DIR, "models", "vlm_labels", "ensemble_cascade_labels.csv")
CANCER_DONORS = {"1069", "1090", "1839", "1879", "1904", "2338", "2467", "2806",
                 "3049", "3548", "603", "604", "637", "700", "707", "997"}
FIELDS = ["source_image", "donor", "label", "stage", "flash_v2", "flash_v5", "flash_v6_score",
          "flash_v4", "tile_yes", "pro_v2", "pro_v6_score", "unusable_votes"]

spend.ALERT_USD = float(os.environ.get("ENS_ALERT_USD", "160"))


def P(name):
    return open(os.path.join(REPO, "probe", "prompts", name), encoding="utf-8").read().strip()


PROMPTS = {"v2": P("v2_lenient.txt"), "v5": P("v5_recog.txt"), "v4": P("v4_balanced.txt"),
           "v6": P("v6_score.txt")}
TILE_PROMPT = None


def load_tile_prompt():
    """Lift TILE_PROMPT out of tile_label.py by AST.

    Importing it would execute its module-level `sys.argv[1]` grid parsing. Reading the literal
    guarantees this ensemble uses the byte-identical prompt the 0.840 measurement was made with,
    rather than a hand-copied near-duplicate that could silently drift.
    """
    global TILE_PROMPT
    import ast
    src = open(os.path.join(SCRIPT_DIR, "tile_label.py"), encoding="utf-8").read()
    for node in ast.parse(src).body:
        if (isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "TILE_PROMPT"
                and isinstance(node.value, ast.Constant)):
            TILE_PROMPT = node.value.value
            return
    raise RuntimeError("TILE_PROMPT not found in tile_label.py")


def donor_of(f):
    return os.path.basename(f).rsplit(".", 1)[0].split("_")[-1]


class Runner:
    """Holds the Vertex clients and keeps their ADC token fresh.

    The ADC token expires after ~1 hour and the OpenAI client pins it at construction. A multi-hour
    run therefore starts 401-ing partway through unless the clients are rebuilt, so refresh
    proactively well inside the window and also force a rebuild on any auth error.
    """
    REFRESH_S = 2400        # 40 min, comfortably inside the ~60 min token lifetime

    def __init__(self):
        self._lock = threading.Lock()
        self._build()

    def _build(self):
        self.flash, _ = get_client(FLASH)
        self.pro, _ = get_client(PRO)
        self._built = time.time()

    def _refresh(self, force=False):
        with self._lock:
            if force or time.time() - self._built > self.REFRESH_S:
                self._build()

    def _client(self, model):
        self._refresh()
        return self.pro if model == PRO else self.flash

    def call(self, model, prompt, b64, maxtok):
        for attempt in (0, 1):
            try:
                r = self._client(model).chat.completions.create(
                    model=model, max_tokens=maxtok, temperature=0,
                    messages=[{"role": "user", "content": [text_part(prompt), img_part(b64)]}])
                spend.add(model, r.usage)
                return r.choices[0].message.content or ""
            except Exception as e:
                msg = str(e).lower()
                if attempt == 0 and ("401" in msg or "unauthenticated" in msg
                                     or "expired" in msg or "invalid authentication" in msg):
                    self._refresh(force=True)
                    continue
                raise

    def binary(self, model, prompt, b64, maxtok):
        return parse_label(self.call(model, prompt, b64, maxtok))

    def graded(self, model, b64, maxtok):
        t = self.call(model, PROMPTS["v6"], b64, maxtok)
        m = re.findall(r"SCORE:\s*(\d+)", t) or re.findall(r"\b(\d+)\b", t)
        return int(m[-1]) if m else None

    def tiles(self, path):
        im = Image.open(path).convert("RGB")
        w, h = im.size
        tw, th = w // GRID, h // GRID
        yes = 0
        for r in range(GRID):
            for c in range(GRID):
                crop = im.crop((c * tw, r * th, (c + 1) * tw if c < GRID - 1 else w,
                                (r + 1) * th if r < GRID - 1 else h))
                a = parse_yes_no(self.call(FLASH, TILE_PROMPT, encode_im(crop, MAXEDGE), 512))
                yes += (a == "yes")
        return yes


def label_one(rn, path, fname):
    """Returns a FIELDS-shaped dict. Raises on failure so the caller can record it."""
    b64 = encode(path, MAXEDGE)
    v2 = rn.binary(FLASH, PROMPTS["v2"], b64, 512)
    v5 = rn.binary(FLASH, PROMPTS["v5"], b64, 512)
    s6 = rn.graded(FLASH, b64, 1024)
    v6 = "usable" if (s6 is not None and s6 >= SCORE_THR) else "unusable"

    row = dict(source_image=fname, donor=donor_of(fname), flash_v2=v2, flash_v5=v5,
               flash_v6_score=("" if s6 is None else s6), flash_v4="", tile_yes="",
               pro_v2="", pro_v6_score="")

    cheap = [v2, v5, v6]
    if len(set(cheap)) == 1 and "?" not in cheap:          # unanimous -> stop paying
        row.update(label=cheap[0], stage=1, unusable_votes="")
        return row

    v4 = rn.binary(FLASH, PROMPTS["v4"], b64, 3072)
    ty = rn.tiles(path)
    pv2 = rn.binary(PRO, PROMPTS["v2"], b64, 512)
    ps6 = rn.graded(PRO, b64, 2048)
    pv6 = "usable" if (ps6 is not None and ps6 >= SCORE_THR) else "unusable"
    tile = "usable" if ty >= TILE_K else "unusable"

    votes = [v2, v5, v6, v4, tile, pv2, pv6]
    nu = sum(1 for v in votes if v == "unusable")
    row.update(flash_v4=v4, tile_yes=ty, pro_v2=pv2, pro_v6_score=("" if ps6 is None else ps6),
               label=("unusable" if nu >= VOTE_UNUSABLE else "usable"), stage=2, unusable_votes=nu)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", help="score a labelled manifest instead of the pool (validation)")
    ap.add_argument("--tag", default="pool")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()
    load_tile_prompt()

    out_csv = OUT_CSV if not args.manifest else os.path.join(
        REPO, "probe", f"ensemble_cascade_{args.tag}.csv")

    if args.manifest:
        data = list(csv.DictReader(open(args.manifest, encoding="utf-8")))
        todo = [r["image_id"] for r in data]
        truth = {r["image_id"]: r["label"] for r in data}
        img_dir = IMG_DIR
    else:
        allimg = sorted(f for f in os.listdir(POOL_DIR) if f.lower().endswith(".jpg"))
        todo = [f for f in allimg if donor_of(f) not in CANCER_DONORS]
        truth, img_dir = None, POOL_DIR
        print(f"pool={len(allimg)}  clean={len(todo)} (excluded {len(allimg)-len(todo)} cancer-donor)")

    # Resume on SUCCESSES only. Treating an ERR row as done would permanently strand every slide
    # that failed transiently (e.g. an expired token), so failed rows are dropped from the file and
    # re-queued. Rewrite via a temp file so an interrupted rewrite cannot truncate real results.
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv, newline="", encoding="utf-8") as fh:
            existing = list(csv.DictReader(fh))
        good = [r for r in existing if r.get("label") in ("usable", "unusable")]
        if len(good) != len(existing):
            tmp = out_csv + ".tmp"
            with open(tmp, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows({k: r.get(k, "") for k in FIELDS} for r in good)
            os.replace(tmp, out_csv)
            print(f"dropped {len(existing) - len(good)} failed row(s) for retry")
        done = {r["source_image"] for r in good}
    pending = [f for f in todo if f not in done]
    if args.limit:
        pending = pending[:args.limit]
    print(f"already_done={len(done)}  to_label={len(pending)}  workers={os.environ.get('ENS_WORKERS','12')}")
    print(f"members: 3 cheap flash -> escalate to 7 (incl 2.5-pro + {GRID}x{GRID} tiling), "
          f"vote unusable if >={VOTE_UNUSABLE}/7   alert at ${spend.ALERT_USD:.0f}")

    if pending:
        rn = Runner()
        new = not os.path.exists(out_csv)
        fh = open(out_csv, "a", newline="", encoding="utf-8")
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        lock = threading.Lock()
        t0, n, esc, bad = time.time(), 0, 0, 0

        def work(fname):
            try:
                return label_one(rn, os.path.join(img_dir, fname), fname)
            except Exception as e:
                return dict(source_image=fname, donor=donor_of(fname), label="ERR", stage=0,
                            flash_v2=f"{type(e).__name__}"[:40], flash_v5="", flash_v6_score="",
                            flash_v4="", tile_yes="", pro_v2="", pro_v6_score="", unusable_votes="")

        with cf.ThreadPoolExecutor(max_workers=int(os.environ.get("ENS_WORKERS", "12"))) as ex:
            for row in ex.map(work, pending):
                n += 1
                esc += (row["stage"] == 2)
                bad += (row["label"] == "ERR")
                with lock:
                    w.writerow(row)
                    if n % 50 == 0:
                        fh.flush()
                        rate = n / (time.time() - t0)
                        print(f"  {n}/{len(pending)}  {rate:.2f} slides/s  "
                              f"eta {(len(pending)-n)/rate/60:6.1f}min  escalated {esc/n*100:.0f}%  "
                              f"err {bad}", flush=True)
        fh.flush()
        fh.close()
        print(f"\nDONE {n} slides in {(time.time()-t0)/60:.1f}min  escalated {esc} ({esc/n*100:.0f}%)  errors {bad}")
        spend.report(f"ensemble-{args.tag}", "cascade")

    with open(out_csv, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    ok = [r for r in rows if r["label"] in ("usable", "unusable")]
    import collections
    print(f"\ntotal rows {len(rows)}  valid {len(ok)}  "
          f"{dict(collections.Counter(r['label'] for r in ok))}  "
          f"stage2 {sum(1 for r in ok if r['stage']=='2')}")
    if truth:
        sub = [r for r in ok if r["source_image"] in truth]
        score([truth[r["source_image"]] for r in sub], [r["label"] for r in sub],
              f"cascade {args.tag}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
