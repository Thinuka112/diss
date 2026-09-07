"""Build a visual audit gallery of Gemini's slice-quality labels (thumbnails + label)."""
import os, sys, csv, random, html
from collections import defaultdict
from PIL import Image

REPO = os.path.dirname(os.path.abspath(__file__))
POOL = os.path.join(REPO, "data_bulk", "images")
LABELS = os.path.join(REPO, "models", "vlm_labels", "gemini_gemini_flash_latest_labels.csv")
OUT_HTML = os.path.join(REPO, "audit_gallery.html")
THUMB_DIR = os.path.join(REPO, "audit_gallery_files")
THUMB_PX = 260
PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
SEED = 42

rows = defaultdict(list)
total = defaultdict(int)
for r in csv.DictReader(open(LABELS, encoding="utf-8")):
    total[r["label"]] += 1
    if r["label"] in ("usable", "unusable"):
        rows[r["label"]].append(r["source_image"])

random.seed(SEED)
sample = []
for lab in ("unusable", "usable"):            # unusable first (the interesting calls)
    pics = rows[lab][:]
    random.shuffle(pics)
    sample += [(f, lab) for f in pics[:PER_CLASS]]

os.makedirs(THUMB_DIR, exist_ok=True)
cards = []
made = 0
for i, (fname, lab) in enumerate(sample):
    src = os.path.join(POOL, fname)
    thumb_rel = f"audit_gallery_files/{i}.jpg"
    try:
        im = Image.open(src).convert("RGB")
        im.thumbnail((THUMB_PX, THUMB_PX), Image.BILINEAR)
        im.save(os.path.join(THUMB_DIR, f"{i}.jpg"), "JPEG", quality=82)
        made += 1
    except Exception as e:
        continue
    short = html.escape(fname if len(fname) <= 46 else fname[:43] + "...")
    cards.append(
        f'<div class="card {lab}" data-label="{lab}">'
        f'<img loading="lazy" src="{thumb_rel}">'
        f'<div class="cap">{short}</div>'
        f'<div class="badge {lab}">{lab.upper()}</div></div>')

n_un = sum(1 for _, l in sample if l == "unusable")
n_us = sum(1 for _, l in sample if l == "usable")
doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Gemini slice-quality labels — visual audit</title>
<style>
 body{{font-family:system-ui,Arial,sans-serif;margin:0;background:#f4f4f6;color:#222}}
 header{{position:sticky;top:0;background:#fff;border-bottom:1px solid #ddd;padding:12px 18px;z-index:5}}
 h1{{font-size:18px;margin:0 0 4px}} .sub{{font-size:13px;color:#555}}
 .btns{{margin-top:8px}} button{{font-size:13px;padding:6px 12px;margin-right:6px;border:1px solid #bbb;border-radius:6px;background:#fafafa;cursor:pointer}}
 button.on{{background:#222;color:#fff;border-color:#222}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px;padding:16px}}
 .card{{background:#fff;border:3px solid #ccc;border-radius:8px;overflow:hidden;text-align:center}}
 .card.usable{{border-color:#2e9e57}} .card.unusable{{border-color:#d23b3b}}
 .card img{{width:100%;height:180px;object-fit:contain;background:#fafafa;display:block}}
 .cap{{font-size:10px;color:#666;padding:4px 6px;word-break:break-all}}
 .badge{{font-size:12px;font-weight:700;color:#fff;padding:3px}} .badge.usable{{background:#2e9e57}} .badge.unusable{{background:#d23b3b}}
</style></head><body>
<header>
 <h1>Gemini slice-quality labels — visual audit</h1>
 <div class="sub">Showing {n_us} usable + {n_un} unusable ({made} of a random sample) from the {total['usable']+total['unusable']:,}-image labelled pool
 ({total['usable']:,} usable / {total['unusable']:,} unusable overall). <b>These are Gemini's calls</b> on unlabelled training images —
 no human ground truth here; judge by eye. Green border = called USABLE, red = UNUSABLE.</div>
 <div class="btns">
  <button class="on" onclick="filt(this,'all')">All</button>
  <button onclick="filt(this,'unusable')">Unusable only</button>
  <button onclick="filt(this,'usable')">Usable only</button>
 </div>
</header>
<div class="grid">
{''.join(cards)}
</div>
<script>
function filt(b,c){{document.querySelectorAll('button').forEach(x=>x.classList.remove('on'));b.classList.add('on');
 document.querySelectorAll('.card').forEach(el=>{{el.style.display=(c==='all'||el.dataset.label===c)?'':'none';}});}}
</script></body></html>"""

open(OUT_HTML, "w", encoding="utf-8").write(doc)
print(f"thumbnails made: {made}/{len(sample)}")
print(f"wrote {OUT_HTML}")
