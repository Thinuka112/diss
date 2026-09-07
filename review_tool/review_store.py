"""
review_store.py - shared storage, resolution and metrics for the swipe reviewer.

Everything the domain specialist judges is written to an APPEND-ONLY event log.
One session = one CSV file whose name carries a microsecond timestamp, so two
reviewers (or two runs) can never overwrite each other's data (collision
protection). Nothing is ever edited in place: a judgment, a re-judgment and an
undo are all just new rows appended in order. The "current truth" for an image
is whatever its most recent event says - this is resolved on read, never on
write, so a crash mid-session can lose at most the single keypress in flight.

This module is imported by:
  - swipe_review.py          (writes the log, reads it to resume)
  - build_gold_standard.py   (folds every log into one gold-standard CSV)
  - compare_to_classifier.py (scores the classifier against the gold standard)

It has no GUI dependencies so it can be imported headless and unit-tested.
"""

import os
import csv
import glob
import datetime

# The event-log schema. One row per keypress.
#   action = "label"  -> judgment holds the class the reviewer chose
#   action = "unset"  -> the reviewer undid their judgment for this image
SESSION_FIELDS = [
    "event_ts",     # ISO-8601 with microseconds, sortable, when the key was pressed
    "reviewer",     # who judged it (free text, from --reviewer)
    "param",        # which feature was being judged, e.g. "slice_quality"
    "image_id",     # stable join key (basename, or the manifest id/path column)
    "image_path",   # absolute path actually shown on screen
    "action",       # "label" | "unset"
    "judgment",     # the chosen class for a "label"; "" for an "unset"
    "key",          # the physical key pressed, for auditing (e.g. "Right")
]

VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")


# ------------------------------------------------------------------
# Time / filename helpers (collision protection)
# ------------------------------------------------------------------
def timestamp_now():
    """Sortable ISO-8601 timestamp with microseconds, e.g. 2026-07-18T14:03:11.482173."""
    return datetime.datetime.now().isoformat(timespec="microseconds")


def _fs_stamp():
    """Filesystem-safe timestamp with microseconds for unique session filenames."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _slug(text):
    """Make a token safe for a filename (reviewer names, params)."""
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in (text or "").strip()]
    return "".join(keep).strip("-") or "anon"


def session_path(sessions_dir, param, reviewer):
    """Build a brand-new, collision-proof session file path.

    The microsecond timestamp guarantees uniqueness even if two sessions start in
    the same second; reviewer + param are embedded so the folder is browsable.
    """
    name = f"session_{_slug(param)}_{_slug(reviewer)}_{_fs_stamp()}.csv"
    return os.path.join(sessions_dir, name)


# ------------------------------------------------------------------
# Append-only session log
# ------------------------------------------------------------------
class SessionLog:
    """Owns one session CSV and appends+flushes every judgment as it happens."""

    def __init__(self, path, reviewer, param):
        self.path = path
        self.reviewer = reviewer
        self.param = param
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        new = not os.path.exists(path) or os.path.getsize(path) == 0
        # Line-buffered so each row hits the OS immediately; we still fsync below.
        self._fh = open(path, "a", newline="", encoding="utf-8", buffering=1)
        self._writer = csv.DictWriter(self._fh, fieldnames=SESSION_FIELDS)
        if new:
            self._writer.writeheader()
            self._fh.flush()

    def _write(self, image_id, image_path, action, judgment, key):
        self._writer.writerow({
            "event_ts": timestamp_now(),
            "reviewer": self.reviewer,
            "param": self.param,
            "image_id": image_id,
            "image_path": image_path,
            "action": action,
            "judgment": judgment,
            "key": key,
        })
        self._fh.flush()
        try:
            os.fsync(self._fh.fileno())   # survive a hard crash / power loss
        except OSError:
            pass

    def label(self, image_id, image_path, judgment, key):
        self._write(image_id, image_path, "label", judgment, key)

    def unset(self, image_id, image_path, key="undo"):
        self._write(image_id, image_path, "unset", "", key)

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


# ------------------------------------------------------------------
# Resolution (read side) - fold the event log(s) into current truth
# ------------------------------------------------------------------
def _iter_session_rows(sessions_dir, param=None):
    """Yield every event row across all session files, optionally filtered by param."""
    pattern = os.path.join(sessions_dir, "session_*.csv")
    for fpath in sorted(glob.glob(pattern)):
        try:
            with open(fpath, newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    if param is None or row.get("param") == param:
                        row["_session_file"] = os.path.basename(fpath)
                        yield row
        except (OSError, csv.Error):
            # A half-written trailing row from a crash is simply ignored.
            continue


def resolve_judgments(sessions_dir, param, reviewer=None):
    """Fold session logs into {image_id: {...current judgment...}}.

    Events are applied in timestamp order so the newest keypress wins. An "unset"
    removes the image's judgment. Also tracks how many times each image flipped
    between distinct classes, which surfaces reviewer indecision / disagreement.

    reviewer: if given, only that reviewer's events are folded in - this is what
    lets each reviewer resume THEIR own progress on a feature independently.
    """
    events = [r for r in _iter_session_rows(sessions_dir, param)
              if reviewer is None or r.get("reviewer") == reviewer]
    events.sort(key=lambda r: (r.get("event_ts", ""), r.get("_session_file", "")))

    current = {}
    flips = {}
    for row in events:
        iid = row.get("image_id", "")
        if not iid:
            continue
        action = row.get("action", "label")
        if action == "unset":
            current.pop(iid, None)
            continue
        judgment = row.get("judgment", "")
        prev = current.get(iid)
        if prev is not None and prev["judgment"] != judgment:
            flips[iid] = flips.get(iid, 0) + 1
        current[iid] = {
            "image_id": iid,
            "param": param,
            "judgment": judgment,
            "reviewer": row.get("reviewer", ""),
            "event_ts": row.get("event_ts", ""),
            "image_path": row.get("image_path", ""),
            "session_file": row.get("_session_file", ""),
        }
    for iid, entry in current.items():
        entry["n_changes"] = flips.get(iid, 0)
    return current


def list_reviewers(sessions_dir):
    """Return known reviewers and their banked progress, for the startup screen.

    {reviewer: {"judgments": int, "params": sorted[str], "last_ts": str}} where
    'judgments' is the count of current (resolved) judgments across all features -
    i.e. real work saved, ignoring undos and re-judgments. Newest reviewers first
    is left to the caller; this returns a dict.
    """
    if not os.path.isdir(sessions_dir):
        return {}
    reviewers = {}
    params = set()
    for row in _iter_session_rows(sessions_dir):
        rv = row.get("reviewer", "")
        if not rv:
            continue
        params.add(row.get("param"))
        info = reviewers.setdefault(rv, {"judgments": 0, "params": set(), "last_ts": ""})
        info["params"].add(row.get("param"))
        ts = row.get("event_ts", "")
        if ts > info["last_ts"]:
            info["last_ts"] = ts
    # Count current judgments per reviewer by resolving each feature they touched.
    for pm in params:
        for rv, info in reviewers.items():
            if pm in info["params"]:
                info["judgments"] += len(resolve_judgments(sessions_dir, pm, reviewer=rv))
    for info in reviewers.values():
        info["params"] = sorted(info["params"])
    return reviewers


# ------------------------------------------------------------------
# Input imageset loaders
# ------------------------------------------------------------------
def load_items_from_folder(folder, exts=VALID_IMAGE_EXTS, recursive=False):
    """Return [{image_id, image_path}] for every image in a folder.

    image_id is the filename (basename), which is what the classifier's
    predictions should also key on for a folder-based imageset.
    """
    folder = os.path.abspath(folder)
    items = []
    if recursive:
        for root, _dirs, files in os.walk(folder):
            for f in sorted(files):
                if f.lower().endswith(exts):
                    items.append({"image_id": f, "image_path": os.path.join(root, f)})
    else:
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(exts):
                items.append({"image_id": f, "image_path": os.path.join(folder, f)})
    return items


def load_items_from_manifest(manifest_csv, path_col, id_col=None):
    """Return [{image_id, image_path}] from a manifest CSV (e.g. cut_images' output).

    path_col names the column holding the image path (e.g. "tile_path"). id_col
    is the stable join key; it defaults to path_col so ids match the classifier,
    which is trained straight off the same manifest.
    """
    id_col = id_col or path_col
    items = []
    with open(manifest_csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if path_col not in reader.fieldnames:
            raise ValueError(
                f"--path-col '{path_col}' not in {manifest_csv}. "
                f"Columns: {reader.fieldnames}"
            )
        for row in reader:
            path = row.get(path_col, "")
            if not path:
                continue
            items.append({"image_id": row.get(id_col, path), "image_path": path})
    return items


# ------------------------------------------------------------------
# Metrics (no sklearn - keep the tool dependency-free)
# ------------------------------------------------------------------
def confusion_matrix(pairs, labels):
    """pairs: iterable of (truth, pred). Returns labels x labels count matrix (list of lists)."""
    idx = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for truth, pred in pairs:
        m[idx[truth]][idx[pred]] += 1
    return m


def accuracy(pairs):
    pairs = list(pairs)
    if not pairs:
        return 0.0
    agree = sum(1 for t, p in pairs if t == p)
    return agree / len(pairs)


def per_class_prf(pairs, labels):
    """Per-class precision/recall/F1/support, truth-vs-pred. Returns {label: {...}}."""
    m = confusion_matrix(pairs, labels)
    idx = {l: i for i, l in enumerate(labels)}
    out = {}
    for l in labels:
        i = idx[l]
        tp = m[i][i]
        fp = sum(m[t][i] for t in range(len(labels)) if t != i)
        fn = sum(m[i][p] for p in range(len(labels)) if p != i)
        support = tp + fn
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out[l] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
    return out


def cohen_kappa(pairs, labels):
    """Cohen's kappa - agreement corrected for chance. 1.0=perfect, 0=chance."""
    pairs = list(pairs)
    n = len(pairs)
    if n == 0:
        return 0.0
    m = confusion_matrix(pairs, labels)
    po = sum(m[i][i] for i in range(len(labels))) / n
    row_tot = [sum(m[i]) for i in range(len(labels))]
    col_tot = [sum(m[i][j] for i in range(len(labels))) for j in range(len(labels))]
    pe = sum((row_tot[i] / n) * (col_tot[i] / n) for i in range(len(labels)))
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)
