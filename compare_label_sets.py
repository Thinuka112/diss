"""Compare the three machine label sets over the same clean pool.

  python compare_label_sets.py

Reports class balance, pairwise agreement, and — the part that matters — per-member behaviour
inside the cascade, so a shift in the unusable rate can be attributed to a specific member rather
than guessed at. Read-only.
"""
import collections
import csv
import os

REPO = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(REPO, "models", "vlm_labels", "gemini_gemini_flash_latest_labels.csv")
V2 = os.path.join(REPO, "models", "vlm_labels", "vertex_gemini_2_5_flash_v2_labels.csv")
ENS = os.path.join(REPO, "models", "vlm_labels", "ensemble_cascade_labels.csv")
VALID = ("usable", "unusable")


def load(path, key="source_image"):
    if not os.path.exists(path):
        return {}, []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {r[key]: r["label"] for r in rows if r.get("label") in VALID}, rows


def pct(n, d):
    return f"{n / d * 100:.1f}%" if d else "n/a"


def balance(name, lab):
    c = collections.Counter(lab.values())
    n = sum(c.values())
    print(f"  {name:22s} n={n:5d}   usable {c['usable']:5d}   "
          f"unusable {c['unusable']:5d} ({pct(c['unusable'], n)})")


def agree(a, b, na, nb):
    ks = [k for k in a if k in b]
    if not ks:
        print(f"  {na} vs {nb}: no overlap")
        return
    same = sum(1 for k in ks if a[k] == b[k])
    flips = collections.Counter(f"{a[k]}->{b[k]}" for k in ks if a[k] != b[k])
    print(f"  {na:12s} vs {nb:12s}  overlap {len(ks):5d}  agreement {pct(same, len(ks)):>7s}"
          f"   flips {dict(flips)}")


v1, _ = load(V1)
v2, _ = load(V2)
ens, ens_rows = load(ENS)

print("=== class balance ===")
balance("v1 (3.7-flash, v1)", v1)
balance("v2 (2.5-flash, v2)", v2)
balance("ensemble cascade", ens)

print("\n=== pairwise agreement ===")
agree(v1, v2, "v1", "v2")
agree(v2, ens, "v2", "ensemble")
agree(v1, ens, "v1", "ensemble")

if ens_rows:
    ok = [r for r in ens_rows if r.get("label") in VALID]
    s2 = [r for r in ok if r.get("stage") == "2"]
    print(f"\n=== cascade internals  (n={len(ok)}) ===")
    print(f"  stage 1 (cheap members unanimous): {len(ok) - len(s2):5d}  {pct(len(ok) - len(s2), len(ok))}")
    print(f"  stage 2 (escalated to 7 members):  {len(s2):5d}  {pct(len(s2), len(ok))}")
    print(f"  unusable rate | stage 1: "
          f"{pct(sum(1 for r in ok if r['stage'] == '1' and r['label'] == 'unusable'), len(ok) - len(s2))}"
          f"   stage 2: {pct(sum(1 for r in s2 if r['label'] == 'unusable'), len(s2))}")
    print("\n  per-member unusable rate (stage-2 slides only — the contested ones):")
    for m in ("flash_v2", "flash_v5", "flash_v4", "pro_v2"):
        vals = [r[m] for r in s2 if r.get(m) in VALID]
        print(f"    {m:10s} {pct(sum(1 for v in vals if v == 'unusable'), len(vals)):>7s}  (n={len(vals)})")
    for m, thr in (("flash_v6_score", 3), ("pro_v6_score", 3)):
        vals = [int(r[m]) for r in s2 if str(r.get(m, "")).isdigit()]
        print(f"    {m:10s} {pct(sum(1 for v in vals if v < thr), len(vals)):>7s}  (n={len(vals)}, "
              f"median score {sorted(vals)[len(vals) // 2] if vals else 'n/a'})")
    ty = [int(r["tile_yes"]) for r in s2 if str(r.get("tile_yes", "")).isdigit()]
    if ty:
        print(f"    tiling     {pct(sum(1 for v in ty if v < 2), len(ty)):>7s}  (n={len(ty)}, "
              f"mean tiles-yes {sum(ty) / len(ty):.1f}/9)")
    votes = [int(r["unusable_votes"]) for r in s2 if str(r.get("unusable_votes", "")).isdigit()]
    if votes:
        print(f"\n  vote distribution on escalated slides (unusable if >=5): "
              f"{dict(sorted(collections.Counter(votes).items()))}")
    print(f"\n  donors: {len({r['donor'] for r in ok})}")
