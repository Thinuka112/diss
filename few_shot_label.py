"""Few-shot calibration of the slice-quality labeller (lever 1 of 2).

Hypothesis: v2's residual error is over-strictness on BORDERLINE-usable slides (DEV usable recall
0.850 vs unusable recall 0.900). Showing the model a handful of expert-labelled worked examples --
weighted toward the borderline cases it currently gets wrong -- should pull usable recall up
without collapsing unusable recall.

LEAKAGE DISCIPLINE (the whole point of the design):
  Exemplars are drawn ONLY from DEV \\ dev_tune, so no slide the model is shown ever appears in the
  set it is scored on. Scoring is on the FULL dev_tune (200, balanced), which makes the result
  directly comparable to the logged v2 baseline (acc 0.875 / unusable recall 0.900 / usable 0.850).
  The 120-slide TEST set is never touched here.

  Forced by DEV's composition: DEV holds 103 unusable, 100 of which are already in dev_tune, so
  only 3 unusable slides exist outside it. Those 3 are all we can use on the unusable side.

  python few_shot_label.py pick          # select exemplars -> probe/fewshot_exemplars.json
  python few_shot_label.py score A       # 3 unusable + 3 borderline-usable (balanced, 6 shots)
  python few_shot_label.py score B       # 3 unusable + 5 borderline-usable (8 shots)

Env: PROBE_MODEL (google/gemini-2.5-flash), PROBE_MAXEDGE (1512), FS_WORKERS (16), FS_PICK_N (150).
"""
import concurrent.futures as cf
import csv
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe"))
import spend
from vertex_common import (CLASSES, IMG_DIR, REPO, encode, get_client, img_part, parse_label,
                           score, text_part)

MAXEDGE = int(os.environ.get("PROBE_MAXEDGE", "1512"))
WORKERS = int(os.environ.get("FS_WORKERS", "16"))
PICK_N = int(os.environ.get("FS_PICK_N", "150"))
SEED = 42

DEV = os.path.join(REPO, "probe", "dev_manifest.csv")
TUNE = os.path.join(REPO, "probe", "dev_tune.csv")
# Scored on dev_eval2 (100 FRESH usable + the 100 dev_tune unusable), not dev_tune: v2's prompt was
# written against dev_tune's disagreements, so its 0.875 there is in-sample. On fresh usable slides
# v2 scores usable recall 0.693 -- matching the 120-test's 0.700. dev_eval2 is the honest comparator.
EVAL = os.environ.get("FS_EVAL", os.path.join(REPO, "probe", "dev_eval2.csv"))
PROMPT_PATH = os.path.join(REPO, "probe", "prompts", "v2_lenient.txt")
EX_PATH = os.path.join(REPO, "probe", "fewshot_exemplars.json")
OUT_JSON = os.path.join(REPO, "probe", "fewshot_results.json")

PROMPT = open(PROMPT_PATH, encoding="utf-8").read().strip()
VARIANTS = {"base": (0, 0), "A": (3, 3), "B": (3, 5),   # (n_unusable, n_borderline_usable)
            "test_base": (0, 0), "gold847": (0, 0)}                        # zero-shot arm of the final TEST evaluation
# Repeat arms: identical config, re-run to measure run-to-run noise. Gemini 2.5 does hidden
# thinking (~490 output tokens/call) which is stochastic, so temperature=0 is NOT deterministic.
VARIANTS.update({f"{k}_rep{i}": v for k, v in list(VARIANTS.items()) for i in (2, 3)})


def rows(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def ask(client, model, parts, max_tokens=512):
    r = client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=0,
        messages=[{"role": "user", "content": parts}])
    spend.add(model, r.usage)
    return r.choices[0].message.content


def pick():
    """Find borderline-usable exemplars: DEV\\dev_tune usable slides that plain v2 calls unusable."""
    tune_ids = {r["image_id"] for r in rows(TUNE)}
    outside = [r for r in rows(DEV) if r["image_id"] not in tune_ids]
    unusable = [r["image_id"] for r in outside if r["label"] == "unusable"]
    usable = [r["image_id"] for r in outside if r["label"] == "usable"]
    print(f"DEV outside dev_tune: {len(outside)}  ({len(unusable)} unusable, {len(usable)} usable)")

    random.seed(SEED)
    probe_set = random.sample(usable, min(PICK_N, len(usable)))
    client, model = get_client()

    def run(fn):
        try:
            parts = [text_part(PROMPT), img_part(encode(os.path.join(IMG_DIR, fn), MAXEDGE))]
            return parse_label(ask(client, model, parts))
        except Exception as e:
            return f"ERR:{type(e).__name__}"

    print(f"probing {len(probe_set)} held-out USABLE slides with plain v2 to find borderline cases...")
    preds = list(cf.ThreadPoolExecutor(max_workers=WORKERS).map(run, probe_set))
    borderline = [f for f, p in zip(probe_set, preds) if p == "unusable"]   # model wrong => borderline
    clear = [f for f, p in zip(probe_set, preds) if p == "usable"]
    errs = sum(1 for p in preds if p not in CLASSES)
    print(f"  borderline-usable (v2 wrongly says unusable): {len(borderline)}/{len(probe_set)}"
          f"   clear-usable: {len(clear)}   errors: {errs}")

    ex = {"unusable": unusable, "borderline_usable": borderline, "clear_usable": clear[:5],
          "meta": {"seed": SEED, "probed": len(probe_set), "maxedge": MAXEDGE, "model": model,
                   "source": "DEV minus dev_tune", "prompt": os.path.basename(PROMPT_PATH)}}
    json.dump(ex, open(EX_PATH, "w", encoding="utf-8"), indent=2)
    print(f"wrote {EX_PATH}")
    spend.report("fewshot-pick", model)


def build_shots(ex, n_unusable, n_usable):
    """Interleaved worked examples; unusable first so the model never sees a usable-only run."""
    picks = ([(f, "unusable") for f in ex["unusable"][:n_unusable]]
             + [(f, "usable") for f in ex["borderline_usable"][:n_usable]])
    if len(picks) < n_unusable + n_usable:      # top up if too few borderline cases were found
        need = n_unusable + n_usable - len(picks)
        picks += [(f, "usable") for f in ex["clear_usable"][:need]]
    if not picks:                               # zero-shot control: the plain v2 prompt, unchanged
        return [text_part(PROMPT)], []
    parts = [text_part(
        PROMPT + "\n\nBefore you answer, study these worked examples. Each is a section from the "
                 "same collection, already graded by the expert pathologist whose rubric you are "
                 "applying. They show where the expert draws the line — note that sections which "
                 "look complex, faint or poorly oriented are still graded USABLE when one "
                 "qualifying villus stretch is present somewhere.\n")]
    for i, (fn, lab) in enumerate(picks, 1):
        parts += [text_part(f"Example {i}:"),
                  img_part(encode(os.path.join(IMG_DIR, fn), MAXEDGE)),
                  text_part(f"Expert's grade: {lab}")]
    parts.append(text_part(
        "\nNow grade the following section by the same standard. "
        "Answer with exactly one word: usable or unusable."))
    return parts, picks


def run_variant(name):
    ex = json.load(open(EX_PATH, encoding="utf-8"))
    n_un, n_us = VARIANTS[name]
    shots, picks = build_shots(ex, n_un, n_us)
    tune = rows(EVAL)
    tune_ids = {r["image_id"] for r in tune}
    assert not (tune_ids & {f for f, _ in picks}), "LEAK: an exemplar is in the scoring set"
    print(f"variant {name}: {len(picks)} shots "
          f"({sum(1 for _, l in picks if l == 'unusable')} unusable / "
          f"{sum(1 for _, l in picks if l == 'usable')} usable), "
          f"scoring {len(tune)} slides from {os.path.basename(EVAL)}")

    client, model = get_client()

    def run(fn):
        try:
            parts = shots + [img_part(encode(os.path.join(IMG_DIR, fn), MAXEDGE))]
            return parse_label(ask(client, model, parts))
        except Exception:
            return "ERR"

    fns = [r["image_id"] for r in tune]
    truth = [r["label"] for r in tune]
    preds = list(cf.ThreadPoolExecutor(max_workers=WORKERS).map(run, fns))

    m = score(truth, preds, f"few-shot {name}")
    if m:
        m["shots"] = [{"image_id": f, "label": l} for f, l in picks]
        m["maxedge"], m["model"] = MAXEDGE, model
        m["manifest"] = f"{os.path.basename(EVAL)}({len(tune)})"
        allr = json.load(open(OUT_JSON, encoding="utf-8")) if os.path.exists(OUT_JSON) else {}
        allr[name] = m
        json.dump(allr, open(OUT_JSON, "w", encoding="utf-8"), indent=2)
        print(f"wrote {OUT_JSON}")
        if "base" in allr and name != "base":
            b = allr["base"]
            print(f"  vs zero-shot base on the same set: acc {b['accuracy']:.3f} "
                  f"(delta {m['accuracy'] - b['accuracy']:+.3f})  "
                  f"unusable recall {b['unusable_recall']:.3f} "
                  f"(delta {m['unusable_recall'] - b['unusable_recall']:+.3f})  "
                  f"usable recall {b['usable_recall']:.3f} "
                  f"(delta {m['usable_recall'] - b['usable_recall']:+.3f})")
    # Full per-slide predictions (not just disagreements) so repeat runs can be diffed slide-by-slide.
    with open(os.path.join(REPO, "probe", f"fewshot_preds_{name}.csv"),
              "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "true", "pred", "match"])
        for i, fn in enumerate(fns):
            w.writerow([fn, truth[i], preds[i], "yes" if preds[i] == truth[i] else "no"])
    spend.report(f"fewshot-{name}", model)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "pick"
    if cmd == "pick":
        pick()
    elif cmd == "score":
        run_variant(sys.argv[2] if len(sys.argv) > 2 else "A")
    else:
        sys.exit(__doc__)
