"""
apply_label_correction.py - append a corrected judgment to the live reviewer.

For when the expert asks to change a specific slide's label after finishing the
batch. The UI only serves *unjudged* slides, so there is no in-app way to revisit
slide N; this appends a new label event under the expert's name, which supersedes
the old one (the store resolves newest-wins). Append-only, so the original stays
in the log and the change is fully auditable.

Run as the admin - it authenticates as the reviewer:

    # PowerShell:
    $env:REVIEW_PASSCODE="****"; python review_tool/apply_label_correction.py Duodenum_BICRA_Male_35_3219.jpg unusable
    # bash:
    REVIEW_PASSCODE=**** python review_tool/apply_label_correction.py Duodenum_BICRA_Male_35_3219.jpg unusable

Env: REVIEW_PASSCODE (required), REVIEWER (default expert), PARAM (default
slice_quality_topup), BASE_URL (default the deployed app).
"""
import os
import sys
import ssl
import json
import urllib.request
import http.cookiejar

BASE = os.environ.get("BASE_URL", "https://slice-review-3bbe85.fly.dev")
REVIEWER = os.environ.get("REVIEWER", "expert")
PARAM = os.environ.get("PARAM", "slice_quality_topup")
PASSCODE = os.environ.get("REVIEW_PASSCODE")


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: apply_label_correction.py <image_id> <new_label>")
    if not PASSCODE:
        raise SystemExit("set REVIEW_PASSCODE (the reviewer passcode) in the environment")
    image_id, new_label = sys.argv[1], sys.argv[2]

    ctx = ssl.create_default_context()
    # This machine's local Python CA bundle is stale; the server cert is valid.
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx),
                                     urllib.request.HTTPCookieProcessor(cj))

    def call(path, body):
        req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with op.open(req, timeout=60) as r:
            return r.status, r.read().decode()

    st, _ = call("/api/signin", {"passcode": PASSCODE, "name": REVIEWER, "profile": PARAM})
    if st != 200:
        raise SystemExit(f"sign-in failed ({st}) - check REVIEW_PASSCODE / REVIEWER / PARAM")
    st, resp = call("/api/judge", {"image_id": image_id, "judgment": new_label, "key": "correction"})
    print(f"signed in as {REVIEWER}; set {image_id} -> {new_label}: HTTP {st} {resp}")


if __name__ == "__main__":
    main()
