"""
compare_to_classifier.py - score the classifier against the human gold standard.

Joins the specialist's gold-standard judgments to the classifier's predictions on
image_id and reports how well the model agrees with the human: coverage, overall
accuracy, per-class precision/recall/F1, a confusion matrix and Cohen's kappa. It
also writes out the exact list of disagreements so they can be eyeballed.

GOLD comes from either:
  --gold <csv>           a file built by build_gold_standard.py, OR
  --sessions-dir + --param   resolved live from the raw session logs.

PREDICTIONS is any CSV the classifier emits. Tell it which columns to use:
  --predictions <csv> --pred-key-col <col> --pred-label-col <col>
If the model labels don't match the human vocab (e.g. model says "0"/"1"), map
them with --label-map '{"1":"usable","0":"unusable"}'.

EXAMPLE
  python compare_to_classifier.py \
      --gold review_data/gold/slice_quality_gold.csv \
      --predictions C:\\...\\predictions.csv \
      --pred-key-col tile_path --pred-label-col prediction
"""

import os
import csv
import json
import argparse

import review_store as store

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PROFILE = os.path.join(HERE, "profiles", "slice_quality.json")
DEFAULT_SESSIONS = os.path.join(HERE, "review_data", "sessions")
DEFAULT_REPORTS = os.path.join(HERE, "review_data", "reports")


def load_gold(args):
    """Return {image_id: judgment} either from a gold CSV or live from session logs."""
    if args.gold:
        gold = {}
        with open(args.gold, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                gold[row["image_id"]] = {"judgment": row["judgment"],
                                         "image_path": row.get("image_path", "")}
        return gold
    if not args.param:
        raise SystemExit("Provide --gold <csv>, or --param (with --sessions-dir) to resolve live.")
    resolved = store.resolve_judgments(args.sessions_dir, args.param, reviewer=args.reviewer)
    return {iid: {"judgment": r["judgment"], "image_path": r.get("image_path", "")}
            for iid, r in resolved.items()}


def load_predictions(args):
    """Return {image_id: predicted_label}, applying key-basename + label-map options."""
    label_map = json.loads(args.label_map) if args.label_map else {}
    preds = {}
    with open(args.predictions, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for col in (args.pred_key_col, args.pred_label_col):
            if col not in reader.fieldnames:
                raise SystemExit(f"Column '{col}' not in {args.predictions}. "
                                 f"Columns: {reader.fieldnames}")
        for row in reader:
            key = row[args.pred_key_col]
            if args.pred_key_basename:
                key = os.path.basename(key)
            label = row[args.pred_label_col]
            preds[key] = label_map.get(label, label)
    return preds


def format_confusion(labels, matrix):
    w = max([len(l) for l in labels] + [5])
    head = " " * (w + 2) + "  ".join(f"{l:>{w}}" for l in labels) + "   (pred ->)"
    lines = [head]
    for i, l in enumerate(labels):
        cells = "  ".join(f"{matrix[i][j]:>{w}}" for j in range(len(labels)))
        lines.append(f"{l:>{w}}  {cells}")
    lines.append("(gold = rows)")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="Score a classifier against the human gold standard.")
    p.add_argument("--gold", default=None, help="Gold-standard CSV from build_gold_standard.py.")
    p.add_argument("--param", default=None, help="Resolve gold live from sessions for this param.")
    p.add_argument("--sessions-dir", default=DEFAULT_SESSIONS)
    p.add_argument("--reviewer", default=None,
                   help="When resolving live (--param), restrict to one reviewer.")
    p.add_argument("--predictions", required=True, help="Classifier predictions CSV.")
    p.add_argument("--pred-key-col", default="image_id", help="Prediction column to join on.")
    p.add_argument("--pred-label-col", default="prediction", help="Prediction column with the class.")
    p.add_argument("--pred-key-basename", action="store_true",
                   help="Reduce the prediction key to a filename (match folder-mode gold ids).")
    p.add_argument("--label-map", default=None,
                   help='JSON mapping model labels to gold vocab, e.g. \'{"1":"usable"}\'.')
    p.add_argument("--reports-dir", default=DEFAULT_REPORTS)
    args = p.parse_args()

    gold = load_gold(args)
    preds = load_predictions(args)
    if not gold:
        raise SystemExit("Gold standard is empty - review some images first.")

    matched_ids = [iid for iid in gold if iid in preds]
    missing = [iid for iid in gold if iid not in preds]

    pairs = [(gold[iid]["judgment"], preds[iid]) for iid in matched_ids]
    labels = sorted({t for t, _ in pairs} | {pr for _, pr in pairs})

    acc = store.accuracy(pairs)
    kappa = store.cohen_kappa(pairs, labels)
    prf = store.per_class_prf(pairs, labels)
    cm = store.confusion_matrix(pairs, labels)
    disagreements = [(iid, gold[iid]["judgment"], preds[iid], gold[iid]["image_path"])
                     for iid in matched_ids if gold[iid]["judgment"] != preds[iid]]

    # ---- build the report text ----
    lines = []
    lines.append("=== Classifier vs gold standard ===")
    lines.append(f"Gold images:        {len(gold)}")
    lines.append(f"Predictions:        {len(preds)}")
    lines.append(f"Matched (scored):   {len(matched_ids)}")
    lines.append(f"Gold w/o prediction:{len(missing)}")
    lines.append("")
    if matched_ids:
        lines.append(f"Accuracy:           {acc:.4f}  ({len(matched_ids) - len(disagreements)}/{len(matched_ids)} agree)")
        lines.append(f"Cohen's kappa:      {kappa:.4f}")
        lines.append("")
        lines.append(f"{'class':<14}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
        for l in labels:
            m = prf[l]
            lines.append(f"{l:<14}{m['precision']:>10.4f}{m['recall']:>10.4f}"
                         f"{m['f1']:>10.4f}{m['support']:>10}")
        lines.append("")
        lines.append("Confusion matrix:")
        lines.append(format_confusion(labels, cm))
        lines.append("")
        lines.append(f"Disagreements: {len(disagreements)}")
    else:
        lines.append("No overlap between gold and predictions - check --pred-key-col "
                     "(and try --pred-key-basename).")
    report = "\n".join(lines)
    print(report)

    # ---- persist report + disagreements (timestamped, collision-proof) ----
    os.makedirs(args.reports_dir, exist_ok=True)
    stamp = store._fs_stamp()
    tag = args.param or (os.path.splitext(os.path.basename(args.gold))[0] if args.gold else "compare")
    report_path = os.path.join(args.reports_dir, f"report_{tag}_{stamp}.txt")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(report + "\n")

    dis_path = os.path.join(args.reports_dir, f"disagreements_{tag}_{stamp}.csv")
    with open(dis_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "gold", "prediction", "image_path"])
        w.writerows(disagreements)

    print(f"\nReport written:        {report_path}")
    print(f"Disagreements written: {dis_path}")


if __name__ == "__main__":
    main()
