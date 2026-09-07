"""Cumulative Vertex/Gemini spend tracker with a hard alert threshold.

Every experiment that calls Vertex imports this, feeds it each response's `usage`, and calls
report() at the end. State accumulates in probe/vertex_spend.json so the total survives across
runs and sessions (the point is the CUMULATIVE figure, not the per-run one).

  from probe.spend import add, report          # or: import spend (when run from probe/)
  add(MODEL, r.usage); ...; report("few-shot A")

Billing note: `completion_tokens` EXCLUDES reasoning tokens on Gemini 2.5 (verified: prompt=5,
completion=1, reasoning=28, total=34). Output is therefore billed as total-prompt, which captures
reasoning too, so thinking can never hide from the estimate.
"""
import json
import os
import threading

# USD per 1M tokens (input, output) — Vertex AI list price, checked 2026-08.
# Update here if the tariff changes; the stored history keeps the old totals intact.
PRICES = {
    "google/gemini-2.5-flash": (0.30, 2.50),
    "google/gemini-2.5-pro": (1.25, 10.00),
}
DEFAULT_PRICE = (0.30, 2.50)

ALERT_USD = 50.0                       # HARD ALERT the user at/above this cumulative estimate
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vertex_spend.json")

_lock = threading.Lock()
_run = {"calls": 0, "in": 0, "out": 0, "usd": 0.0}


def _load():
    try:
        with open(PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {"cumulative_usd": 0.0, "calls": 0, "in_tokens": 0, "out_tokens": 0, "runs": []}


def add(model, usage):
    """Accumulate one API response's token usage into the current run's tally."""
    if usage is None:
        return
    pin = getattr(usage, "prompt_tokens", 0) or 0
    tot = getattr(usage, "total_tokens", 0) or 0
    out = max(tot - pin, getattr(usage, "completion_tokens", 0) or 0)
    pi, po = PRICES.get(model, DEFAULT_PRICE)
    usd = pin / 1e6 * pi + out / 1e6 * po
    with _lock:
        _run["calls"] += 1
        _run["in"] += pin
        _run["out"] += out
        _run["usd"] += usd


def report(tag, model=""):
    """Persist this run's spend, print the run + cumulative estimate, alert past the threshold."""
    st = _load()
    st["cumulative_usd"] = round(st.get("cumulative_usd", 0.0) + _run["usd"], 6)
    st["calls"] = st.get("calls", 0) + _run["calls"]
    st["in_tokens"] = st.get("in_tokens", 0) + _run["in"]
    st["out_tokens"] = st.get("out_tokens", 0) + _run["out"]
    st.setdefault("runs", []).append({
        "tag": tag, "model": model, "calls": _run["calls"],
        "in_tokens": _run["in"], "out_tokens": _run["out"], "usd": round(_run["usd"], 6),
    })
    with open(PATH, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=2)

    cum = st["cumulative_usd"]
    print(f"\n[spend] run '{tag}': {_run['calls']} calls, "
          f"{_run['in']:,} in + {_run['out']:,} out tokens = ${_run['usd']:.4f}")
    print(f"[spend] CUMULATIVE ESTIMATE: ${cum:.4f}  ({st['calls']:,} calls) -> {PATH}")
    if cum >= ALERT_USD:
        print("\n" + "!" * 78)
        print(f"!! BUDGET ALERT: cumulative Vertex estimate ${cum:.2f} >= ${ALERT_USD:.0f}.")
        print("!! Tell the user NOW before running anything further.")
        print("!" * 78)
    return cum
