# Deploying the slice-quality review app (Fly.io)

A public link so the domain expert can review in any browser. No install. Access is
gated by a passcode and locked to two named users. The review images are baked into
the container. The judgment files live on a persistent Fly volume.

## One-time setup
1. Install the Fly CLI and sign in:
   ```
   # Windows (PowerShell):  iwr https://fly.io/install.ps1 -useb | iex
   fly auth login
   ```
2. From the `review_tool/` folder (where `fly.toml` + `Dockerfile` are):
   ```
   fly launch --no-deploy          # confirm/choose an app name + region (lhr = London)
   fly volumes create review_data --size 1 --region lhr   # 1 GB persistent disk
   ```
3. Set the secrets (choose your own values):
   ```
   fly secrets set REVIEW_PASSCODE="pick-a-passcode" \
                   ADMIN_TOKEN="pick-an-admin-token" \
                   REVIEW_SECRET="a-long-random-string" \
                   ALLOWED_REVIEWERS="YourName,ExpertName"
   ```

## Deploy / redeploy
```
fly deploy      # builds locally-bundled image (incl. the 500 images) and ships it
```
This prints a link like `https://<app>.fly.dev`. Send it and the passcode to the expert.
They sign in with the exact name from `ALLOWED_REVIEWERS`.

## Get the dataset back
- In a browser (admin):
  `https://<app>.fly.dev/api/export?token=<ADMIN_TOKEN>&param=<profile>&fmt=xlsx`
  (or `fmt=csv`) downloads the current labelled dataset. `<profile>` is any profile
  enabled in `fly.toml` `ENABLED_PROFILES` (e.g. `slice_quality_topup`, `loc_stretch`).
- Progress/agreement: `.../api/stats?token=<ADMIN_TOKEN>&param=<profile>`.

## Notes
- Cost. `auto_stop_machines="stop"` and `min_machines_running=0` make it sleep when
  idle. Cold start is about 2 s on the next visit. Set `min_machines_running=1` in
  `fly.toml` during an active review push for zero cold starts.
- Persistence. Judgments live on the `review_data` volume and survive redeploys.
  Keep `REVIEW_SECRET` stable or sign-in cookies reset.
- One machine only. The file-based store is not multi-instance safe. Two users is fine.
- Local test of the same container:
  ```
  docker build -f review_tool/Dockerfile -t slice-review review_tool
  docker run -p 8080:8000 -e REVIEW_PASSCODE=review123 -e ALLOWED_REVIEWERS="you,expert" \
             -v ${PWD}/review_tool/review_data:/app/review_tool/review_data slice-review
  # open http://localhost:8080
  ```
