"""Shared conventions for the notebook series (imported by every notebook).

Carries the coursework's visual identity (palette, confusion-matrix style, METRICS FOR
REPORT block) and the verification contract: every notebook recomputes its headline numbers
from the raw files and checks them against the research record via verify(). The point is
that the reader can re-run everything and audit the claimed numbers themselves.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

# ---- coursework palette (identical hex values to the graded AI&ML notebook) ----
TRAIN_COLOR = "#F4A896"   # light  - train curves
VAL_COLOR   = "#8B1A1A"   # dark   - val curves / accents
CMAP        = "Reds"
USABLE_C    = "#2e7d4f"
UNUSABLE_C  = "#b02323"

# repo root = parent of notebooks/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def p(*parts):
    """Path relative to the repo root."""
    return os.path.join(ROOT, *parts)


def set_seed(seed=42):
    import random
    import numpy as _np
    random.seed(seed)
    _np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---- the verification contract ----
_CHECKS = []


def verify(name, computed, expected, tol=0, source=""):
    """Compare a recomputed value against the research-record value. Exact by default."""
    if isinstance(expected, (list, tuple)):
        ok = list(np.ravel(computed)) == list(np.ravel(expected))
    elif tol:
        ok = abs(float(computed) - float(expected)) <= tol
    else:
        ok = computed == expected
    _CHECKS.append((name, ok))
    mark = "✓" if ok else "✗ MISMATCH"
    src = f"   [{source}]" if source else ""
    print(f"  {mark}  {name}: computed {computed}  expected {expected}{src}")
    return ok


def verify_summary():
    good = sum(1 for _, ok in _CHECKS if ok)
    print("\n" + "=" * 62)
    print(f"VERIFICATION: {good}/{len(_CHECKS)} checks passed"
          + ("" if good == len(_CHECKS) else "  <-- INVESTIGATE THE MISMATCHES"))
    print("=" * 62)


# ---- coursework-style figures ----
def plot_confusion(cm, title, classes=("unusable", "usable"), ax=None):
    cm = np.asarray(cm)
    if ax is None:
        _, ax = plt.subplots(figsize=(3.8, 3.4))
    ax.imshow(cm, cmap=CMAP)
    ax.set_xticks(range(len(classes)), classes, fontsize=9)
    ax.set_yticks(range(len(classes)), classes, fontsize=9)
    thr = cm.max() / 2 if cm.max() else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if cm[i, j] > thr else VAL_COLOR)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    return ax


def metrics_report(title, **kv):
    """The coursework's copy-pasteable METRICS FOR REPORT block."""
    print("=" * 62)
    print(f"METRICS FOR REPORT - {title}")
    print("=" * 62)
    for k, v in kv.items():
        print(f"  {k:34s} {v}")
    print("=" * 62)
