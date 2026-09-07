"""
server.py - web backend for the slide-quality validation tool.

Serves a single-page OpenSeadragon (deep-zoom) UI over the curated review set,
records judgments through the EXISTING review_store.SessionLog (so the output is
identical to the desktop tool and build_gold_standard.py works unchanged), and hands
out images with a queue that serves each reviewer the next slide they haven't judged.

Run locally:
  cd review_tool/webapp
  python -m uvicorn server:app --reload --port 8000
  # open http://localhost:8000  (default passcode: review123)

Config via env: REVIEW_PASSCODE, ADMIN_TOKEN, REVIEW_SECRET, TARGET_REVIEWERS,
REVIEW_IMAGE_DIR, REVIEW_SESSIONS_DIR.
"""

import os
import sys
import glob
import json
import io
import threading
import collections

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import (HTMLResponse, JSONResponse, FileResponse,
                               StreamingResponse, Response)
from fastapi.staticfiles import StaticFiles
from itsdangerous import URLSafeSerializer, BadSignature

HERE = os.path.dirname(os.path.abspath(__file__))
REVIEW_TOOL = os.path.dirname(HERE)
sys.path.insert(0, REVIEW_TOOL)
import review_store as store          # noqa: E402

# ---- config ----
PASSCODE = os.environ.get("REVIEW_PASSCODE", "review123")
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "admin123")
SECRET = os.environ.get("REVIEW_SECRET", "dev-secret-change-me")
TARGET_REVIEWERS = int(os.environ.get("TARGET_REVIEWERS", "1"))
# Optional 2-user lock: comma-separated names allowed to sign in (case-insensitive).
# Empty = any name (local dev). In prod set e.g. ALLOWED_REVIEWERS="Reviewer1,Reviewer2".
ALLOWED = [n.strip().lower() for n in os.environ.get("ALLOWED_REVIEWERS", "").split(",") if n.strip()]
# Which judging features to offer (empty = all). Prod: "slice_quality" only.
ENABLED = [p.strip() for p in os.environ.get("ENABLED_PROFILES", "").split(",") if p.strip()]
IMAGE_DIR = os.environ.get("REVIEW_IMAGE_DIR", os.path.join(HERE, "review_images"))
SESSIONS_DIR = os.environ.get("REVIEW_SESSIONS_DIR",
                              os.path.join(REVIEW_TOOL, "review_data", "sessions"))
PROFILES_DIR = os.path.join(REVIEW_TOOL, "profiles")
STATIC_DIR = os.path.join(HERE, "static")

serializer = URLSafeSerializer(SECRET, salt="review-cookie")

# ---- imageset + profiles (loaded once) ----
ITEMS = store.load_items_from_folder(IMAGE_DIR) if os.path.isdir(IMAGE_DIR) else []
ITEM_BY_ID = {it["image_id"]: it for it in ITEMS}
IMAGE_IDS = [it["image_id"] for it in ITEMS]

# Optional per-param image subset: a fixed list of slides a profile may serve.
# Used for the intra-rater re-code (slice_quality_recode serves only recode_set.txt).
def _load_id_list(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]

_recode = _load_id_list(os.path.join(HERE, "recode_set.txt"))
PARAM_SUBSET = {}
if _recode:
    PARAM_SUBSET["slice_quality_recode"] = [i for i in _recode if i in ITEM_BY_ID]

# Active-learning top-up: the model-mined likely-unusable shortlist
# (topup_set.txt, from mine_unusable_candidates.py). Same subset mechanism as the
# re-code pass - the topup profile serves only these ids.
_topup = _load_id_list(os.path.join(HERE, "topup_set.txt"))
if _topup:
    PARAM_SUBSET["slice_quality_topup"] = [i for i in _topup if i in ITEM_BY_ID]

# Stage-2 gold localisation set (scope 05): 8x8 TILES of 20 usable slides, served as
# ordinary items through the same flow (built by mil_expert_set.py). Tiles are loaded
# from their own manifest so the whole-slide corpus stays untouched; the loc_stretch
# profile serves ONLY these ids, and no other profile can reach them because every other
# enabled profile is subset-gated or predates the tiles.
_loc_manifest = os.path.join(HERE, "loc_tile_manifest.csv")
if os.path.exists(_loc_manifest):
    _tiles = store.load_items_from_manifest(_loc_manifest, "tile_path", "tile_id")
    for _t in _tiles:                       # manifest paths are relative to webapp/
        if not os.path.isabs(_t["image_path"]):
            _t["image_path"] = os.path.join(HERE, _t["image_path"])
    ITEMS.extend(_tiles)
    ITEM_BY_ID.update({t["image_id"]: t for t in _tiles})
    _loc = _load_id_list(os.path.join(HERE, "loc_set.txt")) or []
    PARAM_SUBSET["loc_stretch"] = [i for i in _loc if i in ITEM_BY_ID]


def load_profiles():
    profs = {}
    for p in sorted(glob.glob(os.path.join(PROFILES_DIR, "*.json"))):
        with open(p, encoding="utf-8") as fh:
            d = json.load(fh)
        if "param" in d and "bindings" in d:
            profs[d["param"]] = d
    return profs


PROFILES = load_profiles()

# ---- per-reviewer append-only logs (reused across requests) ----
_lock = threading.Lock()
_logs = {}


def get_log(reviewer, param):
    with _lock:
        log = _logs.get((reviewer, param))
        if log is None:
            os.makedirs(SESSIONS_DIR, exist_ok=True)
            log = store.SessionLog(store.session_path(SESSIONS_DIR, param, reviewer),
                                   reviewer, param)
            _logs[(reviewer, param)] = log
        return log


def coverage(param):
    """image_id -> number of distinct reviewers with a current judgment."""
    cov = collections.Counter()
    for rv in store.list_reviewers(SESSIONS_DIR):
        for iid in store.resolve_judgments(SESSIONS_DIR, param, reviewer=rv):
            cov[iid] += 1
    return cov


def sess(request):
    raw = request.cookies.get("review")
    if not raw:
        return None
    try:
        return serializer.loads(raw)
    except BadSignature:
        return None


def public_profile(param):
    p = PROFILES[param]
    return {"param": p["param"], "prompt": p.get("prompt", f"Judge: {param}"),
            "bindings": p["bindings"], "blind": p.get("blind", True),
            "skip_key": p.get("skip_key", "space"), "undo_key": p.get("undo_key", "BackSpace")}


app = FastAPI(title="Slice-quality review")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    idx = os.path.join(STATIC_DIR, "index.html")
    if not os.path.exists(idx):
        return HTMLResponse("<h1>Frontend not built yet</h1>", status_code=500)
    return FileResponse(idx)


@app.get("/api/profiles")
def api_profiles():
    profs = [public_profile(p) for p in sorted(PROFILES) if not ENABLED or p in ENABLED]
    return {"profiles": profs, "images": len(IMAGE_IDS), "target_reviewers": TARGET_REVIEWERS}


@app.post("/api/signin")
async def api_signin(request: Request):
    body = await request.json()
    if body.get("passcode") != PASSCODE:
        raise HTTPException(status_code=403, detail="Wrong passcode.")
    name = (body.get("name") or "").strip()
    param = body.get("profile")
    if not name:
        raise HTTPException(status_code=400, detail="Enter your name.")
    if ALLOWED and name.lower() not in ALLOWED:
        raise HTTPException(status_code=403, detail="This reviewer name isn't on the allowed list.")
    if param not in PROFILES or (ENABLED and param not in ENABLED):
        raise HTTPException(status_code=400, detail="Unknown feature.")
    resp = JSONResponse({"ok": True, "profile": public_profile(param), "name": name})
    resp.set_cookie("review", serializer.dumps({"name": name, "param": param}),
                    httponly=True, samesite="lax", max_age=60 * 60 * 24 * 30)
    return resp


@app.post("/api/signout")
def api_signout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie("review")
    return resp


@app.get("/api/whoami")
def api_whoami(request: Request):
    s = sess(request)
    # A stale cookie for a now-disabled profile counts as signed out, so the
    # reviewer re-enters into the currently enabled pass (e.g. the re-code).
    if not s or s["param"] not in PROFILES or (ENABLED and s["param"] not in ENABLED):
        return {"signed_in": False}
    return {"signed_in": True, "name": s["name"], "profile": public_profile(s["param"])}


@app.get("/api/next")
def api_next(request: Request):
    s = sess(request)
    if not s:
        raise HTTPException(status_code=401, detail="Not signed in.")
    name, param = s["name"], s["param"]
    pool = PARAM_SUBSET.get(param, IMAGE_IDS)   # restricted set for the re-code pass
    mine = set(store.resolve_judgments(SESSIONS_DIR, param, reviewer=name))
    cov = coverage(param)
    # images this reviewer hasn't judged, fewest-covered first (stable -> keeps order)
    candidates = [iid for iid in pool if iid not in mine]
    covered = sum(1 for iid in pool if cov.get(iid, 0) >= TARGET_REVIEWERS)
    progress = {"judged_by_me": len([i for i in mine if i in set(pool)]), "total": len(pool),
                "covered_at_target": covered, "target": TARGET_REVIEWERS}
    if not candidates:
        return {"done": True, "progress": progress}
    candidates.sort(key=lambda iid: cov.get(iid, 0))
    nxt = candidates[0]
    return {"done": False, "image_id": nxt, "image_url": f"/api/image/{nxt}",
            "coverage": cov.get(nxt, 0), "progress": progress}


@app.get("/api/image/{image_id}")
def api_image(image_id: str):
    it = ITEM_BY_ID.get(image_id)   # only serve known ids (no path traversal)
    if not it or not os.path.exists(it["image_path"]):
        raise HTTPException(status_code=404, detail="Unknown image.")
    return FileResponse(it["image_path"], media_type="image/jpeg")


@app.post("/api/judge")
async def api_judge(request: Request):
    s = sess(request)
    if not s:
        raise HTTPException(status_code=401, detail="Not signed in.")
    body = await request.json()
    image_id = body.get("image_id")
    judgment = body.get("judgment")
    key = body.get("key", "")
    it = ITEM_BY_ID.get(image_id)
    if not it:
        raise HTTPException(status_code=404, detail="Unknown image.")
    valid = {b["label"] for b in PROFILES[s["param"]]["bindings"]}
    if judgment not in valid:
        raise HTTPException(status_code=400, detail="Invalid judgment.")
    get_log(s["name"], s["param"]).label(image_id, it["image_path"], judgment, key)
    return {"ok": True}


@app.post("/api/undo")
async def api_undo(request: Request):
    s = sess(request)
    if not s:
        raise HTTPException(status_code=401, detail="Not signed in.")
    body = await request.json()
    image_id = body.get("image_id")
    it = ITEM_BY_ID.get(image_id)
    if not it:
        raise HTTPException(status_code=404, detail="Unknown image.")
    get_log(s["name"], s["param"]).unset(image_id, it["image_path"], "undo")
    return {"ok": True}


@app.get("/api/export")
def api_export(token: str = "", param: str = "slice_quality", fmt: str = "csv"):
    """The labelled dataset: one row per slide with the reviewer's judgment."""
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Bad admin token.")
    resolved = store.resolve_judgments(SESSIONS_DIR, param)
    rows = sorted(resolved.values(), key=lambda r: r["image_id"])
    fields = ["image_id", "label", "reviewer", "event_ts"]
    vals = lambda r: [r["image_id"], r["judgment"], r.get("reviewer", ""), r.get("event_ts", "")]
    if fmt == "xlsx":
        buf = io.BytesIO()
        from openpyxl import Workbook
        wb = Workbook(); ws = wb.active; ws.title = param; ws.append(fields)
        for r in rows:
            ws.append(vals(r))
        wb.save(buf); buf.seek(0)
        return StreamingResponse(
            buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{param}_labels.xlsx"'})
    # CSV via the csv module so image_ids containing commas are properly quoted.
    import csv as _csv
    sbuf = io.StringIO()
    w = _csv.writer(sbuf)
    w.writerow(fields)
    for r in rows:
        w.writerow(vals(r))
    return Response(sbuf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{param}_labels.csv"'})


@app.get("/api/stats")
def api_stats(token: str = "", param: str = "slice_quality"):
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Bad admin token.")
    resolved = store.resolve_judgments(SESSIONS_DIR, param)
    counts = collections.Counter(r["judgment"] for r in resolved.values())
    return {"param": param, "labelled": len(resolved), "total_images": len(IMAGE_IDS),
            "label_counts": dict(counts),
            "reviewers": sorted({r.get("reviewer", "") for r in resolved.values()})}


@app.get("/api/agreement")
def api_agreement(token: str = "", param_a: str = "slice_quality",
                  param_b: str = "slice_quality_recode"):
    """Intra-rater agreement between two labelling passes (Cohen's kappa)."""
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="Bad admin token.")
    a = store.resolve_judgments(SESSIONS_DIR, param_a)
    b = store.resolve_judgments(SESSIONS_DIR, param_b)
    shared = sorted(set(a) & set(b))
    pairs = [(a[i]["judgment"], b[i]["judgment"]) for i in shared]
    labels = sorted({j for p in pairs for j in p})
    agree = sum(1 for x, y in pairs if x == y)
    disagreements = [{"image_id": i, param_a: a[i]["judgment"], param_b: b[i]["judgment"]}
                     for i in shared if a[i]["judgment"] != b[i]["judgment"]]
    return {"param_a": param_a, "param_b": param_b,
            "n_shared": len(shared), "n_agree": agree,
            "percent_agree": round(agree / len(pairs), 4) if pairs else None,
            "cohen_kappa": round(store.cohen_kappa(pairs, labels), 4) if pairs else None,
            "labels": labels,
            "confusion_matrix": store.confusion_matrix(pairs, labels) if pairs else [],
            "disagreements": disagreements}
