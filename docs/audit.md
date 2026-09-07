# Data audit and donor accounting

This file replaces dissertation Appendix F. It backs the audit story in Chapter 3. The full narrative is in `AUDIT.md` at the repo root. The tables live in `audit/`.

## What the audit found

The first downloader selected by tissue site alone and never read the sample's morphology codes. Two issues followed.

**Contamination.** Some downloaded cores came from donors whose atlas records carry neoplasm morphology codes. Adenocarcinoma and malignant carcinoid. Sixteen such donors were identified in the export. `audit/cancer_donor_images.csv` lists every affected image with its donor, codes and whether it reached the labelled set (209 rows).

**Donor concentration.** The 500 first-pass labels came from only 22 donors. Donor 1904 alone supplied 92 images and was also among the cancer-carrying donors. `audit/donor_distribution_labelled500.csv` gives the per-donor counts.

**Split leakage.** The early image-level split let eight donors appear in both train and test. `audit/donors_spanning_splits_100set.csv` records the affected donors per split scheme. This is why every reported result uses donor-grouped cross-validation.

## What changed

All sixteen cancer-carrying donors were excluded. The exclusion is metadata, not deletion. The revised downloader checks morphology before fetching, caps acquisition at twelve images per donor and skips donors already in the set. That added 224 images from 19 cancer-free donors. The corrected dataset covers all 43 cancer-free donors the atlas holds.

## Files

| File | What it is |
|---|---|
| `AUDIT.md` | The full audit narrative |
| `audit/cancer_donor_images.csv` | Every image from a cancer-carrying donor |
| `audit/donor_distribution_labelled500.csv` | Per-donor image counts in the first 500 labels |
| `audit/donors_spanning_splits_100set.csv` | Donors that crossed train/test under the early splits |
| `audit/A6_resolution_comparison.png` | Native vs network-input resolution comparison |
