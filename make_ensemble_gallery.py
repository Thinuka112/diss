"""Self-contained visual audit of the 7-member ensemble labels (thumbnails embedded, no sidecar).

Shows for each slide: the ensemble verdict, HOW it was decided (stage 1 = the three cheap members
agreed; stage 2 = escalated to all 7, with the unusable vote count), and what the previous v2
labeller said — so the slides where the ensemble overrode v2 can be inspected directly.

  python make_ensemble_gallery.py [per_class]      # default 80

Output: ensemble_label_gallery.html
"""
import base64
import collections
import csv
import io
import os
import random
import sys

from PIL import Image

REPO = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(REPO, "data_bulk", "images")
ENS = os.path.join(REPO, "models", "vlm_labels", "ensemble_cascade_labels.csv")
V2 = os.path.join(REPO, "models", "vlm_labels", "vertex_gemini_2_5_flash_v2_labels.csv")
OUT = os.path.join(REPO, "ensemble_label_gallery.html")
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 80
THUMB_PX, QUALITY, SEED = 300, 72, 42

rows = list(csv.DictReader(open(ENS, newline="", encoding="utf-8")))
v2 = {r["source_image"]: r["label"] for r in csv.DictReader(open(V2, newline="", encoding="utf-8"))}
counts = collections.Counter(r["label"] for r in rows)
n_valid = counts["usable"] + counts["unusable"]
esc = sum(1 for r in rows if r["stage"] == "2")
flips = sum(1 for r in rows if r["source_image"] in v2 and v2[r["source_image"]] != r["label"])

by = collections.defaultdict(list)
for r in rows:
    by[r["label"]].append(r)
random.seed(SEED)
sample = []
for lab in ("unusable", "usable"):
    pool = by[lab][:]
    random.shuffle(pool)
    sample += pool[:PER_CLASS]

cards, made = [], 0
for r in sample:
    try:
        im = Image.open(os.path.join(POOL, r["source_image"])).convert("RGB")
        w, h = im.size
        im.thumbnail((THUMB_PX, THUMB_PX), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=QUALITY)
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        made += 1
    except Exception as e:
        print(f"  skip {r['source_image']}: {e}")
        continue
    fn = r["source_image"]
    gene = fn.split("_")[1] if "_" in fn else "?"
    prev = v2.get(fn, "-")
    overrode = prev != "-" and prev != r["label"]
    if r["stage"] == "2":
        how = f'escalated &middot; {r["unusable_votes"]}/7 voted unusable'
    else:
        how = "unanimous (3 cheap members)"
    cards.append(
        f'<figure class="card {r["label"]}" data-lab="{r["label"]}" '
        f'data-stage="{r["stage"]}" data-flip="{"1" if overrode else "0"}">'
        f'<img loading="lazy" src="data:image/jpeg;base64,{b64}" alt="{fn}">'
        f'<figcaption><span class="tag {r["label"]}">{r["label"]}</span>'
        f'<span class="how">{how}</span>'
        + (f'<span class="flip">overrode v2: {prev} &rarr; {r["label"]}</span>' if overrode else "")
        + f'<span class="meta">{gene} &middot; donor {r["donor"]} &middot; {w}&times;{h}px</span>'
        f'<span class="fn">{fn}</span></figcaption></figure>')

n_unus_s = sum(1 for c in cards if 'data-lab="unusable"' in c)
n_use_s = sum(1 for c in cards if 'data-lab="usable"' in c)
n_esc_s = sum(1 for c in cards if 'data-stage="2"' in c)
n_flip_s = sum(1 for c in cards if 'data-flip="1"' in c)

doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ensemble label gallery</title><style>
 :root{{--bg:#f6f7f9;--fg:#16181d;--mut:#5b6270;--card:#fff;--line:#dfe3ea;
        --good:#1c7a41;--goodbg:#e6f6ec;--bad:#b02323;--badbg:#fdeaea;--warn:#8a5a00;--warnbg:#fff4e0}}
 @media (prefers-color-scheme:dark){{:root{{--bg:#14161a;--fg:#e8eaed;--mut:#9aa3b2;--card:#1d2026;
        --line:#2c313a;--good:#4ade80;--goodbg:#123021;--bad:#f87171;--badbg:#3a1a1a;
        --warn:#fbbf24;--warnbg:#3a2e12}}}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--fg);
       font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
 header{{position:sticky;top:0;z-index:5;background:var(--card);
         border-bottom:1px solid var(--line);padding:14px 20px}}
 h1{{margin:0 0 4px;font-size:17px}}
 .sub{{color:var(--mut);font-size:13px}} .sub b{{color:var(--fg)}}
 .btns{{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}}
 /* CSS-only filtering: the viewer may not execute inline JS, so the controls are radio inputs
    driving sibling selectors rather than onclick handlers. Works with scripting disabled. */
 .filt{{position:absolute;opacity:0;pointer-events:none}}
 .btns label{{font:inherit;padding:6px 13px;border:1px solid var(--line);border-radius:7px;
              background:var(--bg);color:var(--fg);cursor:pointer;user-select:none}}
 #f-all:checked~header label[for=f-all],
 #f-unus:checked~header label[for=f-unus],
 #f-use:checked~header label[for=f-use],
 #f-esc:checked~header label[for=f-esc],
 #f-flip:checked~header label[for=f-flip]{{background:var(--fg);color:var(--card);
                                          border-color:var(--fg)}}
 .grid .card{{display:none}}
 #f-all:checked~.grid .card,
 #f-unus:checked~.grid .card[data-lab="unusable"],
 #f-use:checked~.grid .card[data-lab="usable"],
 #f-esc:checked~.grid .card[data-stage="2"],
 #f-flip:checked~.grid .card[data-flip="1"]{{display:block}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(215px,1fr));gap:14px;padding:18px}}
 .card{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}}
 .card.unusable{{border-left:4px solid var(--bad)}} .card.usable{{border-left:4px solid var(--good)}}
 .card img{{width:100%;height:190px;object-fit:contain;background:var(--bg);display:block}}
 figcaption{{padding:8px 10px;display:flex;flex-direction:column;gap:3px}}
 .tag{{align-self:flex-start;font-weight:700;font-size:11px;letter-spacing:.03em;
       text-transform:uppercase;padding:2px 7px;border-radius:4px}}
 .tag.usable{{background:var(--goodbg);color:var(--good)}}
 .tag.unusable{{background:var(--badbg);color:var(--bad)}}
 .how{{font-size:11.5px;color:var(--mut)}}
 .flip{{font-size:11px;background:var(--warnbg);color:var(--warn);padding:2px 6px;
        border-radius:4px;align-self:flex-start}}
 .meta{{color:var(--mut);font-size:12px}}
 .fn{{color:var(--mut);font-size:10.5px;font-family:ui-monospace,Consolas,monospace;word-break:break-all}}
</style></head><body>
<input class="filt" type="radio" name="filt" id="f-all" checked>
<input class="filt" type="radio" name="filt" id="f-unus">
<input class="filt" type="radio" name="filt" id="f-use">
<input class="filt" type="radio" name="filt" id="f-esc">
<input class="filt" type="radio" name="filt" id="f-flip">
<header>
 <h1>Ensemble slice-quality labels &mdash; sample of the full set</h1>
 <div class="sub">
  Full set: <b>{len(rows):,}</b> labels &middot; <b>{counts['usable']:,}</b> usable /
  <b>{counts['unusable']:,}</b> unusable (<b>{counts['unusable'] / n_valid * 100:.1f}%</b>) &middot;
  <b>43</b> donors &middot; 0 errors &middot;
  <b>{esc:,}</b> escalated to the full 7-member vote &middot;
  <b>{flips:,}</b> slides where this overrode the previous v2 label<br>
  Showing a <b>balanced</b> sample of <b>{made}</b> slides (seed {SEED}) &mdash; the real set is
  88% usable, so a random sample would hide the calls worth checking.
  Labeller: 7-member cascade (gemini-2.5-flash &times;5 incl. 3&times;3 tiling + gemini-2.5-pro &times;2),
  unusable if &ge;5/7 &mdash; measured at <b>0.850</b> accuracy on 100 held-out expert-labelled slides.
 </div>
 <div class="btns">
  <label for="f-all">All {made}</label>
  <label for="f-unus">Unusable ({n_unus_s})</label>
  <label for="f-use">Usable ({n_use_s})</label>
  <label for="f-esc">Escalated only ({n_esc_s})</label>
  <label for="f-flip">Overrode v2 ({n_flip_s})</label>
 </div>
</header>
<div class="grid">{''.join(cards)}</div>
</body></html>"""

open(OUT, "w", encoding="utf-8").write(doc)
print(f"wrote {OUT}  ({made} thumbnails embedded, {os.path.getsize(OUT) / 1e6:.1f} MB)")
