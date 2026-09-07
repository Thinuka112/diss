"""
build_recode_set.py - pick a random subset of already-labelled slides for an
intra-rater re-code pass (reliability check).

Draws N image_ids (seeded, reproducible) from the review-set manifest - i.e. from
the slides the expert has already judged once - and writes them to
webapp/recode_set.txt. The web app serves ONLY these ids under the
`slice_quality_recode` profile, so the expert re-judges the same slides blind as a
second pass. Comparing pass 1 (slice_quality) with pass 2 (slice_quality_recode)
gives Cohen's kappa (self-consistency).

  python build_recode_set.py            # 50 slides, seed 42
  python build_recode_set.py --n 40 --seed 7
"""

import os
import csv
import random
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "webapp", "review_set_manifest.csv")
OUT = os.path.join(HERE, "webapp", "recode_set.txt")


def main():
    ap = argparse.ArgumentParser(description="Pick a re-code subset for intra-rater agreement.")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        ids = [row["image_id"] for row in csv.DictReader(fh)]
    if not ids:
        raise SystemExit(f"No image_ids in {MANIFEST}")
    n = min(args.n, len(ids))
    chosen = sorted(random.Random(args.seed).sample(ids, n))

    with open(OUT, "w", encoding="utf-8") as fh:
        for iid in chosen:
            fh.write(iid + "\n")

    print(f"Picked {n} of {len(ids)} slides (seed {args.seed}) for the re-code pass.")
    print(f"Written: {OUT}")
    for iid in chosen[:5]:
        print(f"  e.g. {iid}")


if __name__ == "__main__":
    main()
