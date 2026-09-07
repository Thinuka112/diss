# Running the slice-quality training on Hex (Bath CS GPU cloud)

The point of moving to Hex is **VRAM**: the local RTX 2060 (6 GB) forces whole
slides down to 448 px, which discards the fine detail (folds, focal blur, oblique
cuts) that defines many *unusable* slides. A Hex GPU with more memory lets us
train at 768-1024 px, the one hardware-gated lever on unusable recall.

Workflow is the standard one from the Hex wiki: **SSH in -> transfer files ->
run -> transfer results back**. The cloud has **no backups**, so the trained
model must be copied back down (the data + code already live in git / on the laptop).

Placeholders below: `<USER>` = your Bath username, `<NODE>` = a GPU node from the
Hex **Usage** page (e.g. `ogg`).

---

## 0. One-time: pick a node
On the Hex **Usage** page, choose a **free general node** and note its GPU + VRAM.
Paste that VRAM figure back to Claude to set INPUT_SIZE/BATCH (768 px needs ~11-16 GB;
1024 px / bigger backbones want 24 GB+).

## 1. Connect (from PowerShell on the laptop)
```
ssh <USER>@<NODE>.cs.bath.ac.uk          # password = your DDaT password
# on campus you can shorten to:  ssh <NODE>.cs
# off-campus: turn on the Bath VPN first if it hangs
```
On the node, sanity-check the GPU and make a working dir:
```
nvidia-smi
mkdir -p ~/quality
```

## 2. Transfer code + data up (from a SECOND PowerShell, in the repo root)
```
scp hex/requirements.txt hex/setup_and_run.sh train_quality_cnn.py <USER>@<NODE>.cs.bath.ac.uk:~/quality/
ssh <USER>@<NODE>.cs.bath.ac.uk "mkdir -p ~/quality/review_tool/results ~/quality/review_tool/webapp"
scp review_tool/results/final_clean_dataset_20260728.csv <USER>@<NODE>.cs.bath.ac.uk:~/quality/review_tool/results/
scp -r review_tool/webapp/review_images <USER>@<NODE>.cs.bath.ac.uk:~/quality/review_tool/webapp/review_images
```
This mirrors the paths `train_quality_cnn.py` expects (it reads
`review_tool/results/<csv>` and `review_tool/webapp/review_images/`, writes
`models/quality/`). The training only loads the 597 slides named in the CSV, so
the extra images in the folder are ignored - fine to send the whole folder.

## 3. Run (on the node)
Run inside `tmux` so a dropped SSH session does not kill the job:
```
tmux new -s train                 # start a persistent session
cd ~/quality
bash setup_and_run.sh             # set CUDA_TAG=cuXXX first if nvidia-smi shows a different CUDA
# detach any time with:  Ctrl-b  then  d
# reattach later with:   tmux attach -t train
```

## 4. Bring the results back down (from the laptop, repo root)
```
scp -r <USER>@<NODE>.cs.bath.ac.uk:~/quality/models/quality ./models/quality_hex
```
**Do this every run** - Hex keeps no backups. `models/quality_hex/` then holds the
metrics JSON + checkpoints for the write-up (kept separate from the local run).

## 5b. The PLIP experiment (vision-language backbone)
`train_quality_plip.py` runs the frozen-PLIP head-to-head vs ResNet-50 (scope:
`research/scoping/03a-vlm-backbone.md`). It uses the **same** 847-label set and
donor-grouped folds, so its metrics compare directly to `models/quality/quality_metrics.json`.

Transfer it up alongside the CNN files, then on the node (inside the same venv):
```
python train_quality_plip.py         # extracts frozen PLIP features + 5-fold CV
```
**Weights:** it fetches `vinid/plip` (~600 MB) from Hugging Face on first load.
- **If garlick has internet:** nothing to do - it downloads itself.
- **If garlick is firewalled:** on your laptop run `python hex/fetch_plip_weights.py`
  (-> `./plip_weights`), `scp -r plip_weights <USER>@garlick.cs.bath.ac.uk:~/quality/`,
  then run with `PLIP_MODEL=~/quality/plip_weights python train_quality_plip.py`.

Output: `models/quality_plip/quality_plip_metrics.json` + confusion PNG - remember to
`scp` them back down (no backups). Embeddings are cached (`plip_embeddings.npz`), so
re-runs skip the GPU pass.

## 5. Higher resolution (the whole point)
Resolution + batch are set by env vars - no code edit, and the local 448 px baseline
stays reproducible. `setup_and_run.sh` defaults to **640 px / batch 8** (safe for the
RTX 3080 'Everyone' nodes). To override:
```
QUALITY_INPUT_SIZE=768 QUALITY_BATCH=8 bash setup_and_run.sh
```
Guidance by node:
- **garlick / ogg** (RTX 3080, ~10 GB, available now): 640 px / batch 8; try 768 / batch 4-6.
- **carrot** (A100 80 GB, request via the Hex list): 1024 px / batch 12+, and CTransPath/ConvNeXt become feasible.

If you see `CUDA out of memory`, lower `QUALITY_BATCH` first, then `QUALITY_INPUT_SIZE`.
Everything else (donor-grouped 5-fold, seed 42, metrics) is unchanged, so the higher-res
numbers are directly comparable to the 448 px baseline in the write-up.
