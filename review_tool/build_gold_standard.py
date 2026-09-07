"""
build_gold_standard.py - fold every review session into ONE gold-standard CSV.

The session logs under review_data/sessions/ are the raw, append-only record of
everything the specialist did (including re-judgments and undos). This script
resolves them into the current truth - the newest judgment per image wins - and
writes a single, stable gold-standard file that the classifier is scored against
and that we keep long-term.

It is non-destructive: it only reads the session logs and (over)writes the one
gold CSV. Re-run it any time after more reviewing; the gold standard simply grows
and stays consistent.

USAGE
  python build_gold_standard.py                       # param from default profile
  python build_gold_standard.py --param slice_quality
  python build_gold_standard.py --param slice_quality --out review_data/gold/slice_quality_gold.csv
"""

import os
import csv
import json
import argparse

import review_store as store

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROFILE = os.path.join(HERE, "profiles", "slice_quality.json")
DEFAULT_SESSIONS = os.path.join(HERE, "review_data", "sessions")
DEFAULT_GOLD_DIR = os.path.join(HERE, "review_data", "gold")

GOLD_FIELDS = ["image_id", "param", "judgment", "reviewer",
               "event_ts", "n_changes", "session_file", "image_path"]


def param_from_profile(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["param"]


def main():
    p = argparse.ArgumentParser(description="Consolidate review sessions into a gold-standard CSV.")
    p.add_argument("--param", default=None,
                   help="Feature to consolidate (default: read from --profile).")
    p.add_argument("--profile", default=DEFAULT_PROFILE,
                   help="Profile to read the param from if --param is omitted.")
    p.add_argument("--sessions-dir", default=DEFAULT_SESSIONS)
    p.add_argument("--reviewer", default=None,
                   help="Only consolidate this reviewer's judgments (default: everyone).")
    p.add_argument("--out", default=None,
                   help="Output gold CSV (default: review_data/gold/<param>_gold.csv).")
    args = p.parse_args()

    param = args.param or param_from_profile(args.profile)
    out = args.out or os.path.join(DEFAULT_GOLD_DIR, f"{param}_gold.csv")

    if not os.path.isdir(args.sessions_dir):
        raise SystemExit(f"No sessions directory yet: {args.sessions_dir}\n"
                         "Run swipe_review.py first.")

    resolved = store.resolve_judgments(args.sessions_dir, param, reviewer=args.reviewer)
    if not resolved:
        raise SystemExit(f"No judgments found for param '{param}' in {args.sessions_dir}.")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    rows = sorted(resolved.values(), key=lambda r: r["image_id"])
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=GOLD_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in GOLD_FIELDS})

    # Report class balance + how many images the reviewer changed their mind on.
    counts = {}
    reviewers = set()
    revisited = 0
    for r in rows:
        counts[r["judgment"]] = counts.get(r["judgment"], 0) + 1
        reviewers.add(r["reviewer"])
        if r.get("n_changes", 0):
            revisited += 1

    print("--- Gold standard built ---")
    print(f"Param:        {param}")
    print(f"Images:       {len(rows)}")
    for label in sorted(counts):
        print(f"  {label:<12} {counts[label]}")
    print(f"Reviewers:    {', '.join(sorted(reviewers))}")
    print(f"Re-judged:    {revisited} image(s) changed class at least once during review")
    print(f"Output:       {out}")


if __name__ == "__main__":
    main()
