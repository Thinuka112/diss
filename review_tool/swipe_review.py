"""
swipe_review.py - lightweight arrow-key image reviewer for a domain specialist.

ONE window, two views:
  1. Sign-in  - type your name (or click a returning reviewer) and pick a feature.
  2. Review   - the images appear in the SAME window; swipe with the arrow keys.

Every keypress is saved immediately to a timestamped, append-only session log
(see review_store), so nothing is ever lost and two runs can never collide.

WHAT IT IS FOR
  Building a hand-made GOLD STANDARD to measure a classifier against. Reviewing is
  blind (the model's prediction is never shown, to avoid anchoring); resume is per
  reviewer + per feature, so each person picks up where they left off and each
  feature is an independent pass over every image.

EASIEST LAUNCH
  Double-click launch.bat (Windows). No command line needed - it opens straight on
  the sign-in screen using the default imageset.

CONFIGURABLE (optional flags)
  --images <folder> | --manifest <csv> --path-col tile_path    (which imageset)
  --profile <json>                                              (skip the picker)
  --reviewer <name>                                             (skip the sign-in)
"""

import os
import glob
import json
import random
import argparse
import tkinter as tk
from tkinter import font as tkfont, messagebox

from PIL import Image, ImageTk

import review_store as store

HERE = os.path.dirname(os.path.abspath(__file__))
PROFILES_DIR = os.path.join(HERE, "profiles")
DEFAULT_PROFILE = os.path.join(PROFILES_DIR, "slice_quality.json")
DEFAULT_SESSIONS = os.path.join(HERE, "review_data", "sessions")
# Used when launched with no --images/--manifest (e.g. double-clicking launch.bat).
DEFAULT_IMAGES = os.path.abspath(os.path.join(HERE, "..", "HPA_small_intestine"))

# ------------------------------------------------------------------
# Palette + fonts - flat dark palette shared by both views.
# ------------------------------------------------------------------
BG      = "#0f1218"   # window
CARD    = "#171c25"   # panels / lists
FIELD   = "#212836"   # inputs
FG      = "#eef2f7"   # primary text
MUTED   = "#8a94a6"   # secondary text
LINE    = "#2a3342"   # hairlines / borders
ACCENT  = "#3b82f6"   # primary action / highlight
ACCENT2 = "#2f6fe0"   # accent hover
GOOD    = "#3ecf7d"   # success
FONT    = "Segoe UI"  # native, clean; Tk falls back gracefully if absent

KEY_GLYPH = {"Left": "←", "Right": "→", "Up": "↑", "Down": "↓",
             "space": "Space", "Return": "Enter", "BackSpace": "Backspace"}


def f(size, bold=False):
    return tkfont.Font(family=FONT, size=size, weight="bold" if bold else "normal")


def _hover(widget, normal, hot):
    widget.bind("<Enter>", lambda e: widget.config(bg=hot))
    widget.bind("<Leave>", lambda e: widget.config(bg=normal))


def load_profile(path):
    """Load and validate a judging profile (the param + key->class bindings)."""
    with open(path, encoding="utf-8") as fh:
        prof = json.load(fh)
    if "param" not in prof or "bindings" not in prof:
        raise ValueError(f"Profile {path} must define 'param' and 'bindings'.")
    seen = set()
    for b in prof["bindings"]:
        if "key" not in b or "label" not in b:
            raise ValueError("Each binding needs a 'key' and a 'label'.")
        if b["key"] in seen:
            raise ValueError(f"Duplicate key binding: {b['key']}")
        seen.add(b["key"])
    prof.setdefault("prompt", f"Judge: {prof['param']}")
    prof.setdefault("skip_key", "space")
    prof.setdefault("undo_key", "BackSpace")
    return prof


def discover_profiles(profiles_dir):
    """Return [(path, profile_dict)] for every valid profile JSON in profiles_dir.

    Each profile is a separate FEATURE to review. Corrupt profiles are skipped.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(profiles_dir, "*.json"))):
        try:
            out.append((path, load_profile(path)))
        except Exception:
            continue
    return out


def _styled_list(parent, height):
    return tk.Listbox(parent, height=height, exportselection=False, activestyle="none",
                      bg=CARD, fg=FG, borderwidth=0, highlightthickness=1,
                      highlightbackground=LINE, highlightcolor=ACCENT,
                      selectbackground=ACCENT, selectforeground="#ffffff", font=f(11))


# ==================================================================
#  View 1 - sign in
# ==================================================================
class StartView:
    """The sign-in screen. Calls on_start(reviewer, profile_path) when ready."""

    def __init__(self, parent, profiles, items, sessions_dir, review_all,
                 default_name, preselect_path, on_start):
        self.profiles = profiles
        self.items = items
        self.sessions_dir = sessions_dir
        self.review_all = review_all
        self.on_start = on_start
        self.reviewers = store.list_reviewers(sessions_dir)

        self.frame = tk.Frame(parent, bg=BG)
        self.frame.pack(fill="both", expand=True)
        card = tk.Frame(self.frame, bg=BG)
        card.place(relx=0.5, rely=0.5, anchor="center")

        tk.Label(card, text="Swipe Review", bg=BG, fg=FG, font=f(24, True)).pack(anchor="w")
        tk.Label(card, text="Sign in, choose a feature, then start swiping.",
                 bg=BG, fg=MUTED, font=f(11)).pack(anchor="w", pady=(2, 20))

        self.name_var = tk.StringVar(value=default_name)

        tk.Label(card, text="YOUR NAME", bg=BG, fg=MUTED, font=f(9, True)).pack(anchor="w")
        self.name_entry = tk.Entry(card, textvariable=self.name_var, width=34, bg=FIELD,
                                   fg=FG, insertbackground=FG, borderwidth=0, font=f(13),
                                   highlightthickness=1, highlightbackground=LINE,
                                   highlightcolor=ACCENT)
        self.name_entry.pack(anchor="w", fill="x", ipady=7, ipadx=6, pady=(5, 16))

        self.reviewer_names = []
        if self.reviewers:
            tk.Label(card, text="RETURNING REVIEWERS  ·  click to sign in",
                     bg=BG, fg=MUTED, font=f(9, True)).pack(anchor="w")
            self.rev_list = _styled_list(card, height=min(4, len(self.reviewers)))
            self.reviewer_names = [n for n, _ in sorted(
                self.reviewers.items(), key=lambda kv: kv[1]["last_ts"], reverse=True)]
            for n in self.reviewer_names:
                info = self.reviewers[n]
                last = (info["last_ts"] or "")[:10]
                feats = ", ".join(info["params"])
                self.rev_list.insert("end",
                    f"   {n}    {info['judgments']} judged"
                    + (f"  ·  {feats}" if feats else "")
                    + (f"  ·  last {last}" if last else ""))
            self.rev_list.pack(anchor="w", fill="x", pady=(5, 16))
            self.rev_list.bind("<<ListboxSelect>>", self._pick_reviewer)
            self.rev_list.bind("<Double-Button-1>", lambda e: self._start())
        else:
            self.rev_list = None

        tk.Label(card, text="FEATURE TO REVIEW  ·  each is a separate pass",
                 bg=BG, fg=MUTED, font=f(9, True)).pack(anchor="w")
        self.feat_list = _styled_list(card, height=min(5, max(2, len(profiles))))
        for _p, prof in profiles:
            self.feat_list.insert("end", f"   {prof['param']}")
        self.feat_list.pack(anchor="w", fill="x", pady=(5, 6))
        sel = 0
        if preselect_path:
            for i, (p, _pr) in enumerate(profiles):
                if os.path.abspath(p) == os.path.abspath(preselect_path):
                    sel = i
                    break
        self.feat_list.selection_set(sel)
        self.feat_list.bind("<<ListboxSelect>>", self._refresh)

        self.status = tk.Label(card, text="", bg=BG, fg=MUTED, anchor="w",
                               wraplength=430, justify="left", font=f(10))
        self.status.pack(anchor="w", fill="x", pady=(6, 16))

        self.start_btn = tk.Button(card, text="Start reviewing  →", command=self._start,
                                   bg=ACCENT, fg="#ffffff", activebackground=ACCENT2,
                                   activeforeground="#ffffff", borderwidth=0, font=f(13, True),
                                   padx=22, pady=10, cursor="hand2")
        self.start_btn.pack(anchor="e")
        _hover(self.start_btn, ACCENT, ACCENT2)

        self.name_var.trace_add("write", self._refresh)
        top = parent.winfo_toplevel()
        top.bind("<Return>", lambda e: self._start())
        self.name_entry.focus_set()
        self._refresh()

    def set_status(self, text, warn=False):
        self.status.config(text=text, fg=("#ffb454" if warn else MUTED))

    def _pick_reviewer(self, *_a):
        if self.rev_list:
            sel = self.rev_list.curselection()
            if sel:
                self.name_var.set(self.reviewer_names[sel[0]])

    def _current(self):
        name = self.name_var.get().strip()
        sel = self.feat_list.curselection()
        idx = sel[0] if sel else 0
        return name, self.profiles[idx]

    def _refresh(self, *_a):
        name, (_path, prof) = self._current()
        total = len(self.items)
        done = 0
        if name:
            res = store.resolve_judgments(self.sessions_dir, prof["param"], reviewer=name)
            done = sum(1 for it in self.items if it["image_id"] in res)
        left = total if self.review_all else total - done
        who = name or "enter a name"
        verb = f"re-review all {total}" if self.review_all else f"{left} left"
        tail = "   ← fresh start" if (name and done == 0 and not self.review_all) else ""
        self.set_status(f"{who}  ·  '{prof['param']}': {done}/{total} done  ·  {verb}{tail}")

    def _start(self, *_a):
        name, (path, _prof) = self._current()
        if not name:
            self.set_status("Type your name (or click a returning reviewer) first.", warn=True)
            self.name_entry.focus_set()
            return
        self.on_start(name, path)

    def destroy(self):
        self.frame.destroy()


# ==================================================================
#  View 2 - review
# ==================================================================
class ReviewView:
    def __init__(self, parent, items, profile, log, max_w, max_h, already_done, on_done):
        self.parent = parent
        self.root = parent.winfo_toplevel()
        self.items = items
        self.profile = profile
        self.log = log
        self.max_w, self.max_h = max_w, max_h
        self.on_done = on_done

        self.key_to_label = {b["key"]: b["label"] for b in profile["bindings"]}
        self.i = 0
        self.session_actions = []      # stack of {"i","type","image_id","image_path"}
        self.done_count = 0
        self.prior_done = already_done
        self.last_label = ""
        self._photo = None

        self.frame = tk.Frame(parent, bg=BG)
        self.frame.pack(fill="both", expand=True)

        head = tk.Frame(self.frame, bg=BG)
        head.pack(side="top", fill="x", padx=20, pady=(14, 0))
        tk.Label(head, text=profile["param"].upper(), bg=BG, fg=ACCENT,
                 font=f(9, True)).pack(anchor="center")
        self.header = tk.Label(head, bg=BG, fg=FG, justify="center", font=f(15, True))
        self.header.pack(anchor="center", pady=(2, 5))
        self.legend = tk.Label(head, bg=BG, fg=MUTED, justify="center", font=f(10))
        self.legend.pack(anchor="center")
        self.legend.config(text=self._legend_text())

        self._bar_w = max_w
        self.progress = tk.Canvas(self.frame, height=4, width=self._bar_w, bg=LINE,
                                  highlightthickness=0)
        self.progress.pack(side="top", pady=(12, 10))
        self._bar = self.progress.create_rectangle(0, 0, 0, 4, fill=ACCENT, width=0)

        self.stage = tk.Label(self.frame, bg="#0a0c11")
        self.stage.pack(side="top", expand=True, fill="both", padx=20)

        self.status = tk.Label(self.frame, bg=BG, fg=MUTED, anchor="w", font=f(10))
        self.status.pack(side="bottom", fill="x", padx=20, pady=(10, 12))

        for key in self.key_to_label:
            self.root.bind(f"<{key}>", self._on_label_key)
        self.root.bind(f"<{profile['skip_key']}>", self._on_skip)
        self.root.bind(f"<{profile['undo_key']}>", self._on_undo)
        for q in ("<Escape>", "<q>", "<Q>"):
            self.root.bind(q, lambda e: self._quit())
        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self.root.focus_set()
        self._show()

    # ---- rendering -------------------------------------------------------
    def _legend_text(self):
        parts = [f"{KEY_GLYPH.get(b['key'], b['key'])} {b['label']}"
                 for b in self.profile["bindings"]]
        parts.append(f"{KEY_GLYPH.get(self.profile['skip_key'], self.profile['skip_key'])} skip")
        parts.append("Backspace undo")
        parts.append("Esc quit")
        return "      ".join(parts)

    def _fit(self, img):
        """Scale to fit the stage, preserving aspect ratio (upscale tiny tiles <=3x)."""
        w, h = img.size
        scale = min(self.max_w / w, self.max_h / h)
        scale = min(scale, 3.0) if scale > 1 else scale
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resample = Image.LANCZOS if scale < 1 else Image.BICUBIC
        return img.resize((nw, nh), resample)

    def _show(self):
        if self.i >= len(self.items):
            return self._finish()
        item = self.items[self.i]
        self.header.config(text=self.profile["prompt"], fg=FG)
        try:
            with Image.open(item["image_path"]) as im:
                im = im.convert("RGB")
                disp = self._fit(im)
            self._photo = ImageTk.PhotoImage(disp)
            self.stage.config(image=self._photo, text="")
        except Exception as exc:
            self._photo = None
            self.stage.config(image="", fg="#ff6b6b", font=f(11),
                              text=f"Could not open image:\n{item['image_path']}\n\n{exc}\n\n"
                                   f"Press {self.profile['skip_key']} to skip.")
        total = len(self.items)
        self.progress.coords(self._bar, 0, 0, int(self._bar_w * self.i / total), 4)
        last = f"    ·    last: {self.last_label}" if self.last_label else ""
        self.status.config(
            text=f"{self.i + 1} / {total}    ·    you judged {self.done_count} this session"
                 f"{last}    ·    {os.path.basename(item['image_path'])}")

    # ---- key handlers ----------------------------------------------------
    def _on_label_key(self, event):
        if self.i >= len(self.items):
            return
        label = self.key_to_label.get(event.keysym)
        if label is None:
            return
        item = self.items[self.i]
        self.log.label(item["image_id"], item["image_path"], label, event.keysym)
        self.session_actions.append({"i": self.i, "type": "label",
                                     "image_id": item["image_id"],
                                     "image_path": item["image_path"]})
        self.done_count += 1
        self.last_label = label
        self.i += 1
        self._show()

    def _on_skip(self, event):
        if self.i >= len(self.items):
            return
        item = self.items[self.i]
        self.session_actions.append({"i": self.i, "type": "skip",
                                     "image_id": item["image_id"],
                                     "image_path": item["image_path"]})
        self.last_label = "skipped"
        self.i += 1
        self._show()

    def _on_undo(self, event):
        if not self.session_actions:
            return
        last = self.session_actions.pop()
        if last["type"] == "label":
            self.log.unset(last["image_id"], last["image_path"])
            self.done_count = max(0, self.done_count - 1)
        self.last_label = "undone"
        self.i = last["i"]
        self._show()

    # ---- end / teardown --------------------------------------------------
    def _finish(self):
        self.header.config(text="✓  All images reviewed", fg=GOOD)
        self.progress.coords(self._bar, 0, 0, self._bar_w, 4)
        self.stage.config(image="", fg=MUTED, font=f(12),
                          text=(f"Judged {self.done_count} image(s) this session.\n\n"
                                f"Saved to:\n{self.log.path}\n\n"
                                "Backspace to revisit the last one, or Esc to finish."))
        self._photo = None
        self.status.config(text=f"{len(self.items)} / {len(self.items)}    ·    complete")

    def _quit(self):
        for seq in list(self.key_to_label) + [self.profile["skip_key"],
                                              self.profile["undo_key"]]:
            try:
                self.root.unbind(f"<{seq}>")
            except Exception:
                pass
        self.on_done()


# ==================================================================
#  App - one window, swaps StartView -> ReviewView
# ==================================================================
class App:
    def __init__(self, root, args, items, profiles):
        self.root = root
        self.args = args
        self.items = items
        self.profiles = profiles

        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.max_w = min(args.max_width, sw - 140)
        self.max_h = min(args.max_height, sh - 300)
        w = min(self.max_w + 60, sw - 60)
        h = min(self.max_h + 230, sh - 60)
        root.title("Swipe Review")
        root.configure(bg=BG)
        root.minsize(680, 560)
        root.geometry(f"{int(w)}x{int(h)}+{(sw - int(w)) // 2}+{max(20, (sh - int(h)) // 3)}")

        self.container = tk.Frame(root, bg=BG)
        self.container.pack(fill="both", expand=True)
        self.start_view = None

        # Skip the sign-in only when BOTH name and feature came from the CLI.
        if args.reviewer and args.profile:
            self._begin(args.reviewer, args.profile)
        else:
            self._show_start()

    def _default_name(self):
        return (self.args.reviewer or os.environ.get("USERNAME")
                or os.environ.get("USER") or "")

    def _clear(self):
        for child in self.container.winfo_children():
            child.destroy()
        self.start_view = None

    def _show_start(self):
        self._clear()
        self.start_view = StartView(
            self.container, self.profiles, self.items, self.args.sessions_dir,
            self.args.review_all, default_name=self._default_name(),
            preselect_path=self.args.profile, on_start=self._begin)

    def _begin(self, reviewer, profile_path):
        profile = load_profile(profile_path)
        param = profile["param"]
        resolved = store.resolve_judgments(self.args.sessions_dir, param, reviewer=reviewer)
        already = len(resolved)
        if self.args.review_all:
            items = list(self.items)
        else:
            items = [it for it in self.items if it["image_id"] not in resolved]
        if self.args.limit and self.args.limit > 0:
            items = items[:self.args.limit]

        if not items:
            msg = f"{reviewer} has already judged every image for '{param}'."
            print(msg)
            if self.start_view:
                self.start_view.set_status(msg + "  Pick another feature, or use --review-all.",
                                           warn=True)
            else:
                messagebox.showinfo("Nothing to review", msg)
                self.root.destroy()
            return

        log = store.SessionLog(store.session_path(self.args.sessions_dir, param, reviewer),
                               reviewer, param)
        print(f"Reviewer: {reviewer}   Feature: {param}   To review: {len(items)}"
              f"   ({already} already judged by {reviewer})")
        print(f"Writing:  {log.path}")

        self._clear()
        ReviewView(self.container, items, profile, log, self.max_w, self.max_h,
                   already, on_done=self.root.destroy)


def parse_args():
    p = argparse.ArgumentParser(description="Lightweight arrow-key image reviewer.")
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--images", help="Folder of images to review (folder mode).")
    src.add_argument("--manifest", help="Manifest CSV listing images (manifest mode).")
    p.add_argument("--path-col", default="tile_path",
                   help="Manifest column with the image path (default: tile_path).")
    p.add_argument("--id-col", default=None,
                   help="Manifest column to use as the join id (default: same as --path-col).")
    p.add_argument("--recursive", action="store_true",
                   help="Folder mode: also search sub-folders.")
    p.add_argument("--profile", default=None,
                   help="Judging profile JSON. If omitted, you pick it on the sign-in screen.")
    p.add_argument("--reviewer", default=None,
                   help="Reviewer name. If omitted, you enter it on the sign-in screen.")
    p.add_argument("--sessions-dir", default=DEFAULT_SESSIONS,
                   help="Where session logs are written.")
    p.add_argument("--review-all", action="store_true",
                   help="Re-review images already judged (ignore resume).")
    p.add_argument("--shuffle", action="store_true", help="Randomise order (reduces bias).")
    p.add_argument("--seed", type=int, default=1234, help="Shuffle seed (reproducible).")
    p.add_argument("--limit", type=int, default=0, help="Only queue the first N images (0 = all).")
    p.add_argument("--max-width", type=int, default=1100, help="Max display width (px).")
    p.add_argument("--max-height", type=int, default=820, help="Max display height (px).")
    return p.parse_args()


def main():
    args = parse_args()

    # Zero-config launch: fall back to the bundled imageset if none was given.
    if not args.images and not args.manifest:
        if os.path.isdir(DEFAULT_IMAGES):
            args.images = DEFAULT_IMAGES
        else:
            raise SystemExit(
                "No imageset given and the default folder was not found:\n"
                f"  {DEFAULT_IMAGES}\n"
                "Pass --images <folder> or --manifest <csv> --path-col <col>.")

    if args.images:
        items = store.load_items_from_folder(args.images, recursive=args.recursive)
        source_desc = f"folder {args.images}"
    else:
        items = store.load_items_from_manifest(args.manifest, args.path_col, args.id_col)
        source_desc = f"manifest {args.manifest} [{args.path_col}]"
    if not items:
        raise SystemExit(f"No images found in {source_desc}.")
    if args.shuffle:
        random.Random(args.seed).shuffle(items)

    profiles = discover_profiles(PROFILES_DIR)
    if args.profile and os.path.abspath(args.profile) not in \
            [os.path.abspath(p) for p, _ in profiles]:
        profiles.insert(0, (args.profile, load_profile(args.profile)))
    if not profiles:
        raise SystemExit(f"No judging profiles found in {PROFILES_DIR}.")

    print(f"Imageset: {source_desc}  ({len(items)} images)")
    root = tk.Tk()
    App(root, args, items, profiles)
    root.mainloop()
    print("Session ended. All judgments saved.")


if __name__ == "__main__":
    main()
