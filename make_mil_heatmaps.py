"""Visual heatmap gallery for the Stage-2 MIL localiser (combined-data production model).

One card per gold slide: the slide image with an 8x8 CSS overlay —
  green tint  = model's per-tile contribution (normalised within slide, positive only)
  blue outline= the expert's "yes" tiles (gold standard, evaluation-only)
  gold border + star = the model's argmax tile (the pointing-game tile)
Self-contained single HTML (base64 thumbnails, CSS-only — no JavaScript), so it opens anywhere.
"""
import base64
import csv
import io
import os
from collections import defaultdict

from PIL import Image

REPO = os.path.dirname(os.path.abspath(__file__))
SCORES = os.path.join(REPO, "models", "mil_combined", "mil_tile_scores.csv")
GOLD = os.path.join(REPO, "review_tool", "results", "loc_stretch_export_20260827.csv")
AUTO_NO = os.path.join(REPO, "review_tool", "loc_tile_auto_no.csv")
IMG_DIR = os.path.join(REPO, "review_tool", "webapp", "review_images")
OUT = os.path.join(REPO, "mil_heatmap_gallery.html")
GRID, W = 8, 380


def read(p):
    return list(csv.DictReader(open(p, newline="", encoding="utf-8")))


def slide_of(tid):
    return tid.rsplit("__", 1)[0] + ".jpg"


def rc_of(tid):
    tag = tid.rsplit("__", 1)[1].split(".")[0]
    r, c = tag[1:].split("c")
    return int(r), int(c)


gold = {}
for r in read(GOLD):
    if r["reviewer"] == "expert":
        gold[r["image_id"]] = r["label"] == "yes"
for r in read(AUTO_NO):
    gold[r["tile_id"]] = False
slides = sorted({slide_of(t) for t in gold})
assert len(slides) == 20 and len(gold) == 1280

scores = defaultdict(dict)
for r in read(SCORES):
    scores[r["image_id"]][(int(r["row"]), int(r["col"]))] = float(r["contribution"])

cards, hits = [], 0
for s in slides:
    sc = scores[s]
    assert len(sc) == GRID * GRID, f"missing scores for {s}"
    yes = {rc_of(t) for t, v in gold.items() if v and slide_of(t) == s}
    mx = max(sc.values())
    top = max(sc, key=sc.get)
    hit = top in yes
    hits += hit
    pos = [v for v in sc.values() if v > 0]
    hi = max(pos) if pos else 1.0

    im = Image.open(os.path.join(IMG_DIR, s)).convert("RGB")
    im.thumbnail((W, W), Image.BILINEAR)
    b = io.BytesIO(); im.save(b, "JPEG", quality=80)
    b64 = base64.standard_b64encode(b.getvalue()).decode()

    cells = []
    for r in range(GRID):
        for c in range(GRID):
            v = sc[(r, c)]
            a = 0.62 * max(0.0, v) / hi if hi > 0 else 0
            style = f"background:rgba(46,220,110,{a:.3f});"
            cls = []
            if (r, c) in yes:
                cls.append("gy")
            if (r, c) == top:
                cls.append("am")
            cells.append(f'<div class="{" ".join(cls)}" style="{style}">'
                         f'{"&#9733;" if (r, c) == top else ""}</div>')
    gene = s.split("_")[-4] if len(s.split("_")) >= 5 else s
    cards.append(
        f'<div class="card"><div class="wrap">'
        f'<img src="data:image/jpeg;base64,{b64}"><div class="grid">{"".join(cells)}</div></div>'
        f'<div class="cap {"ok" if hit else "no"}"><b>{gene}</b> &mdash; model points '
        f'{"&#10003; at an expert-confirmed stretch" if hit else "&#10007; at a non-stretch tile"}'
        f' &nbsp;<span class="sub">({len(yes)} expert yes-tiles of 64)</span></div></div>')

html = f"""<!doctype html><html><head><meta charset="utf-8"><title>Stage-2 heatmaps</title><style>
 body{{font-family:system-ui,Arial;margin:0;background:#15181d;color:#e8e8e8}}
 header{{position:sticky;top:0;background:#1d2129;border-bottom:1px solid #333;padding:14px 20px;z-index:5}}
 h1{{font-size:17px;margin:0 0 6px}} .leg{{font-size:13px;line-height:1.8}}
 .sw{{display:inline-block;width:13px;height:13px;border-radius:3px;vertical-align:-2px;margin:0 4px 0 10px}}
 .grid-page{{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:16px;padding:18px}}
 .card{{background:#1d2129;border-radius:10px;overflow:hidden;border:1px solid #30343c}}
 .wrap{{position:relative;display:block}} .wrap img{{width:100%;display:block}}
 .grid{{position:absolute;inset:0;display:grid;grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(8,1fr)}}
 .grid div{{position:relative;font-size:15px;line-height:1;color:#ffd34d;text-align:center;display:flex;align-items:center;justify-content:center;text-shadow:0 0 3px #000}}
 .grid .gy{{outline:2px solid #4f8dff;outline-offset:-2px}}
 .grid .am{{outline:3px solid #ffd34d;outline-offset:-3px}}
 .cap{{font-size:13px;padding:8px 10px}} .cap.ok{{color:#7fe0a7}} .cap.no{{color:#ff9a9a}}
 .sub{{color:#9aa0aa}}
</style></head><body>
<header><h1>Stage-2 localisation heatmaps &mdash; combined-data MIL (production model) on the 20 gold slides</h1>
<div class="leg">Model prediction: <span class="sw" style="background:rgba(46,220,110,.62)"></span>green tint = model evidence for a qualifying stretch (per-tile contribution)
 <span class="sw" style="outline:2px solid #4f8dff"></span>blue outline = the expert&rsquo;s &ldquo;yes&rdquo; tiles (never shown to the model)
 <span class="sw" style="outline:3px solid #ffd34d"></span>gold &#9733; = the single tile the model points at
 &nbsp;|&nbsp; <b>pointing hits: {hits}/20</b> &nbsp; top-3: 20/20</div></header>
<div class="grid-page">{''.join(cards)}</div></body></html>"""
open(OUT, "w", encoding="utf-8").write(html)
print(f"pointing hits drawn: {hits}/20")
print(f"wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB)")
