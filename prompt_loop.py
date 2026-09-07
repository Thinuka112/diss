"""prompt_loop.py - the iterative prompt-optimisation loop for the slice-quality labeller.

  score:    python prompt_loop.py score probe/prompts/v3.txt probe/prompt_tune.csv v3
  explain:  python prompt_loop.py explain probe/prompts/v3.txt v3 [n]

`score` runs a prompt over a manifest and dumps every miss to probe/loop_errors_{tag}.csv.
`explain` re-asks the model about its OWN misses, demanding a one-sentence reason, so each round's
prompt revision is written against a stated failure mode instead of guesswork.

Discipline: iterate on probe/prompt_tune.csv (50/50). Check probe/prompt_val.csv (50/50, disjoint)
only to confirm a winner generalises. The 120-slide TEST set stays sealed until the very end.
Config is otherwise frozen at the validated one: 1512px, temperature 0, gemini-2.5-flash.
"""
import concurrent.futures as cf
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "probe"))
import spend
from vertex_common import (CLASSES, IMG_DIR, REPO, encode, get_client, img_part, parse_label,
                           score, text_part)

MAXEDGE = int(os.environ.get("PROBE_MAXEDGE", "1512"))
WORKERS = int(os.environ.get("LOOP_WORKERS", "24"))
# Hidden thinking (~490 tok/call) is drawn from this budget; too low truncates it and the verdict
# degrades. 512 is the validated default -- raise via env to test the thinking-budget lever.
MAXTOK = int(os.environ.get("LOOP_MAXTOK", "512"))
# Hidden thinking is billed at OUTPUT rates and is ~70% of per-call cost. Set LOOP_REASONING=none
# to switch it off and test whether the accuracy actually depends on it.
REASONING = os.environ.get("LOOP_REASONING", "")
HIST = os.path.join(REPO, "probe", "prompt_loop_history.json")


def rows(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def do_score(prompt_path, manifest, tag):
    prompt = open(prompt_path, encoding="utf-8").read().strip()
    data = rows(manifest)
    fns = [r["image_id"] for r in data]
    truth = [r["label"] for r in data]
    client, model = get_client()

    def run(fn):
        try:
            kw = dict(model=model, max_tokens=MAXTOK, temperature=0,
                      messages=[{"role": "user", "content": [
                          text_part(prompt),
                          img_part(encode(os.path.join(IMG_DIR, fn), MAXEDGE))]}])
            if REASONING:
                kw["reasoning_effort"] = REASONING
            r = client.chat.completions.create(**kw)
            spend.add(model, r.usage)
            return parse_label(r.choices[0].message.content)
        except Exception:
            return "ERR"

    preds = list(cf.ThreadPoolExecutor(max_workers=WORKERS).map(run, fns))
    m = score(truth, preds, tag)

    err_csv = os.path.join(REPO, "probe", f"loop_errors_{tag}.csv")
    with open(err_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "true", "pred"])
        for i, fn in enumerate(fns):
            if preds[i] in CLASSES and preds[i] != truth[i]:
                w.writerow([fn, truth[i], preds[i]])
    with open(os.path.join(REPO, "probe", f"loop_preds_{tag}.csv"),
              "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["image_id", "true", "pred"])
        for i, fn in enumerate(fns):
            w.writerow([fn, truth[i], preds[i]])

    if m:
        m.update({"prompt": os.path.basename(prompt_path), "manifest": os.path.basename(manifest)})
        hist = json.load(open(HIST, encoding="utf-8")) if os.path.exists(HIST) else {}
        hist[tag] = m
        json.dump(hist, open(HIST, "w", encoding="utf-8"), indent=2)
        fp = sum(1 for i in range(len(fns)) if truth[i] == "usable" and preds[i] == "unusable")
        fn_ = sum(1 for i in range(len(fns)) if truth[i] == "unusable" and preds[i] == "usable")
        print(f"  MISSES: {fp} usable->unusable (too strict) | {fn_} unusable->usable (too lenient)")
        print(f"  errors -> {err_csv}")
    spend.report(f"loop-{tag}", model)


def do_explain(prompt_path, tag, n=12):
    """Ask the model to justify its own misses -- the 'why it was wrong' the revision targets."""
    prompt = open(prompt_path, encoding="utf-8").read().strip()
    errs = rows(os.path.join(REPO, "probe", f"loop_errors_{tag}.csv"))[:n]
    if not errs:
        print("no errors to explain.")
        return
    client, model = get_client()
    ask = (prompt.replace("Answer with exactly one word: usable or unusable.", "").strip()
           + "\n\nState your verdict (usable or unusable), then in ONE sentence give the single "
             "concrete visual reason: name what epithelium you did or did not find and where. "
             "Format: VERDICT | reason")

    def run(r):
        try:
            resp = client.chat.completions.create(
                model=model, max_tokens=700, temperature=0,
                messages=[{"role": "user", "content": [
                    text_part(ask),
                    img_part(encode(os.path.join(IMG_DIR, r["image_id"]), MAXEDGE))]}])
            spend.add(model, resp.usage)
            return (resp.choices[0].message.content or "").strip().replace("\n", " ")
        except Exception as e:
            return f"ERR {e}"

    out = list(cf.ThreadPoolExecutor(max_workers=WORKERS).map(run, errs))
    print(f"\n=== why it missed ({tag}) — expert label vs model reasoning ===")
    for r, txt in zip(errs, out):
        print(f"\n[expert={r['true']:8s} model said={r['pred']:8s}] {r['image_id']}")
        print(f"   {txt[:400]}")
    spend.report(f"explain-{tag}", model)


if __name__ == "__main__":
    if sys.argv[1] == "score":
        do_score(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "explain":
        do_explain(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 12)
