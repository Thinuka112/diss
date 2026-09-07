# Swipe review — the human gold standard for the slice-quality classifier

A Tkinter and Pillow tool with no other dependencies. A domain specialist swipes through images with the arrow keys and records one judgment per image. Those judgments are the gold standard the classifier is measured against.

Three steps. Review. Consolidate. Compare.

```
review_tool/
  swipe_review.py          # 1. the arrow-key reviewer (writes session logs)
  build_gold_standard.py   # 2. fold all sessions into one gold-standard CSV
  compare_to_classifier.py # 3. score the classifier vs the gold standard
  review_store.py          # shared storage + metrics (no GUI, unit-testable)
  profiles/
    slice_quality.json     # what to judge + which key = which class (configurable)
  review_data/             # created at runtime (git-ignored)
    sessions/  gold/  reports/
```

The web-app variant the expert actually used remotely lives in `webapp/`. FastAPI plus OpenSeadragon. Same session-log format and same profiles. See `DEPLOY.md`.

## 1. Review

Double-click `launch.bat`. No command line. It opens on the sign-in screen using the imageset at `../HPA_small_intestine`. That folder is not part of this repo. Download the dataset first (see the root README), or point the tool at any image folder by editing the one `python swipe_review.py` line in `launch.bat`. Or run it yourself:

```
python swipe_review.py --images ../HPA_small_intestine          # a folder of images
python swipe_review.py --manifest ...\manifest.csv --path-col tile_path   # or a manifest CSV
```

It is a single window. Sign-in first, then the images appear in the same window. On open it asks two things.

1. Your name. Every judgment is saved under it and it decides resume. Saved reviewers are listed with their progress. Click a name to sign in. Double-click to jump straight in. Or type a new name.
2. Which feature to review (`slice_quality`, `loc_stretch`, ...). The dialog shows how many images you have already judged per feature. You can see whether you are resuming or starting fresh.

Keys (defaults, set in the profile): → usable · ← unusable · Space skip · Backspace undo · Esc quit.

- The model's prediction is never shown. The human is not anchored.
- Each keypress is appended and fsync'd to a session log named with a microsecond timestamp (`session_<param>_<reviewer>_<YYYYMMDD_HHMMSS_ffffff>.csv`). Parallel runs and reviewers never collide.
- Resume is per reviewer and per feature. Images already judged for this feature are skipped. A different reviewer or feature starts from the first image. Pass `--review-all` to redo a pass from scratch.
- For scripted runs skip the dialog. Pass both `--reviewer <name>` and `--profile <json>`.

### Multiple features

Each JSON in `profiles/` is a separate feature. An independent labelling pass over the whole imageset with its own classes. Four ship with the repo.

- `slice_quality.json`: binary (→ usable / ← unusable). The main Stage-1 pass.
- `slice_quality_recode.json`: binary re-grade of a 50-image subset. The agreement check.
- `slice_quality_topup.json`: binary. The 250-image model-mined top-up pass.
- `loc_stretch.json`: tile-level yes/no for the Stage-2 localisation gold standard.

To add another, copy a profile and change `param` and the key→`label` `bindings` (up to four classes via ←↓↑→). It appears in the picker automatically. The `label` strings should match whatever your classifier emits. Or reconcile them later with `--label-map` (step 3).

## 2. Consolidate into the gold standard

```
python build_gold_standard.py --param slice_quality
```

Resolves every session log into `review_data/gold/slice_quality_gold.csv` and prints the class balance. Newest judgment per image wins. Undos are respected. Re-run after more reviewing. This file is the gold standard.

## 3. Compare to the classifier

Point it at your classifier's predictions CSV and name the columns:

```
python compare_to_classifier.py \
    --gold review_data/gold/slice_quality_gold.csv \
    --predictions path\to\predictions.csv \
    --pred-key-col tile_path --pred-label-col prediction
```

Prints and saves accuracy, per-class precision/recall/F1, a confusion matrix and Cohen's kappa. Also writes a timestamped `disagreements_*.csv` listing every image where the model and the human differ.

- If the model emits different label strings than the human, map them: `--label-map '{"1":"usable","0":"unusable"}'`.
- If predictions are keyed by full path but the gold uses filenames (folder mode), add `--pred-key-basename`.
- You can skip step 2 and score straight off the raw logs with `--param slice_quality` instead of `--gold ...`.

### Expected predictions format

Any CSV with a key column matching the gold `image_id` (the tile path in manifest mode, the filename in folder mode) and a label column. Example:

```csv
tile_path,prediction
C:\...\tiles\Duodenum_A1BG_..__tile_01.jpg,usable
C:\...\tiles\Duodenum_A1BG_..__tile_02.jpg,unusable
```
