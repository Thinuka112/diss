"""Build a SELF-CONTAINED visual audit of the v2 machine labels (thumbnails embedded as data URIs).

Single HTML file, no sidecar folder, so it can be opened or sent anywhere. Balanced sample so the
minority class is actually inspectable -- a random sample would be ~78% usable and tell you little
about the calls that matter.

  python make_v2_gallery.py [per_class]        # default 80 per class

Output: v2_label_gallery.html
"""
import base64
import csv
import collections
import io
import os
import random
import sys

from PIL import Image

REPO = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(REPO, "data_bulk", "images")
LABELS = os.path.join(REPO, "models", "vlm_labels", "vertex_gemini_2_5_flash_v2_labels.csv")
OUT = os.path.join(REPO, "v2_label_gallery.html")
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 80
THUMB_PX, QUALITY, SEED = 300, 72, 42

rows = list(csv.DictReader(open(LABELS, encoding="utf-8")))
counts = collections.Counter(r["label"] for r in rows)
donors = len({r["donor"] for r in rows})
by_label = collections.defaultdict(list)
for r in rows:
    by_label[r["label"]].append(r)

random.seed(SEED)
sample = []
for lab in ("unusable", "usable"):            # unusable first -- the calls worth scrutinising
    pool = by_label[lab][:]
    random.shuffle(pool)
    sample += pool[:PER_CLASS]

cards, made = [], 0
for r in sample:
    path = os.path.join(POOL, r["source_image"])
    try:
        im = Image.open(path).convert("RGB")
        w, h = im.size
        im.thumbnail((THUMB_PX, THUMB_PX), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=QUALITY)
        b64 = base64.standard_b64encode(buf.getvalue()).decode()
        made += 1
    except Exception as e:
        print(f"  skip {r['source_image']}: {e}")
        continue
    gene = r["source_image"].split("_")[1] if "_" in r["source_image"] else "?"
    cards.append(
        f'<figure class="card {r["label"]}" data-lab="{r["label"]}">'
        f'<img loading="lazy" src="data:image/jpeg;base64,{b64}" alt="{r["source_image"]}">'
        f'<figcaption><span class="tag {r["label"]}">{r["label"]}</span>'
        f'<span class="meta">{gene} &middot; donor {r["donor"]} &middot; {w}&times;{h}px</span>'
        f'<span class="fn">{r["source_image"]}</span></figcaption></figure>')

n_valid = counts["usable"] + counts["unusable"]
doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>v2 label gallery</title><style>
 :root{{--bg:#f6f7f9;--fg:#16181d;--mut:#5b6270;--card:#fff;--line:#dfe3ea;
        --good:#1c7a41;--goodbg:#e6f6ec;--bad:#b02323;--badbg:#fdeaea}}
 @media (prefers-color-scheme:dark){{:root{{--bg:#14161a;--fg:#e8eaed;--mut:#9aa3b2;--card:#1d2026;
        --line:#2c313a;--good:#4ade80;--goodbg:#123021;--bad:#f87171;--badbg:#3a1a1a}}}}
 *{{box-sizing:border-box}}
 body{{margin:0;background:var(--bg);color:var(--fg);
       font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
 header{{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);
         padding:14px 20px}}
 h1{{margin:0 0 4px;font-size:17px}}
 .sub{{color:var(--mut);font-size:13px}}
 .sub b{{color:var(--fg)}}
 .btns{{margin-top:10px;display:flex;gap:8px;flex-wrap:wrap}}
 button{{font:inherit;padding:6px 13px;border:1px solid var(--line);border-radius:7px;
         background:var(--bg);color:var(--fg);cursor:pointer}}
 button.on{{background:var(--fg);color:var(--card);border-color:var(--fg)}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px;padding:18px}}
 .card{{margin:0;background:var(--card);border:1px solid var(--line);border-radius:10px;
        overflow:hidden}}
 .card.unusable{{border-left:4px solid var(--bad)}} .card.usable{{border-left:4px solid var(--good)}}
 .card img{{width:100%;height:190px;object-fit:contain;background:var(--bg);display:block}}
 figcaption{{padding:8px 10px;display:flex;flex-direction:column;gap:3px}}
 .tag{{align-self:flex-start;font-weight:700;font-size:11px;letter-spacing:.03em;
       text-transform:uppercase;padding:2px 7px;border-radius:4px}}
 .tag.usable{{background:var(--goodbg);color:var(--good)}}
 .tag.unusable{{background:var(--badbg);color:var(--bad)}}
 .meta{{color:var(--mut);font-size:12px}}
 .fn{{color:var(--mut);font-size:10.5px;font-family:ui-monospace,Consolas,monospace;
      word-break:break-all}}
</style></head><body>
<header>
 <h1>Machine-labelled slice quality &mdash; sample of the full set</h1>
 <div class="sub">
  Full set: <b>{len(rows):,}</b> labels &middot; <b>{counts['usable']:,}</b> usable /
  <b>{counts['unusable']:,}</b> unusable (<b>{counts['unusable'] / n_valid * 100:.1f}%</b>) &middot;
  <b>{donors}</b> donors &middot; 0 errors &middot; 0 cancer-donor slides<br>
  Showing a <b>balanced</b> sample of <b>{made}</b> slides (seed {SEED}) &mdash; the real set is
  ~78% usable, so a random sample would hide the calls worth checking.
  Labeller: gemini-2.5-flash, prompt v2, 1512px, temperature 0 &mdash;
  the config measured at <b>0.767</b> accuracy / <b>0.800</b> unusable recall on 120 held-out
  expert-labelled slides.
 </div>
 <div class="btns">
  <button class="on" onclick="f(this,'all')">All {made}</button>
  <button onclick="f(this,'unusable')">Unusable ({sum(1 for c in cards if 'data-lab="unusable"' in c)})</button>
  <button onclick="f(this,'usable')">Usable ({sum(1 for c in cards if 'data-lab="usable"' in c)})</button>
 </div>
</header>
<div class="grid">{''.join(cards)}</div>
<script>function f(b,s){{document.querySelectorAll('button').forEach(x=>x.classList.remove('on'));
b.classList.add('on');document.querySelectorAll('.card').forEach(el=>{{
el.style.display=(s==='all'||el.dataset.lab===s)?'':'none';}});}}</script>
</body></html>"""

open(OUT, "w", encoding="utf-8").write(doc)
print(f"wrote {OUT}  ({made} thumbnails embedded, {os.path.getsize(OUT) / 1e6:.1f} MB)")
