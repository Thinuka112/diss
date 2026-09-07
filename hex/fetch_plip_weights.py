"""
fetch_plip_weights.py - download the PLIP weights to a local folder, for when the
compute node has NO outbound internet. Run this on a machine that DOES (your laptop),
then scp the folder up to the node and point PLIP_MODEL at it.

    python hex/fetch_plip_weights.py                    # -> ./plip_weights
    scp -r plip_weights <USER>@garlick.cs.bath.ac.uk:~/quality/plip_weights
    # then on the node:
    PLIP_MODEL=~/quality/plip_weights python train_quality_plip.py

If garlick DOES have internet, you don't need this - train_quality_plip.py fetches
'vinid/plip' from Hugging Face automatically on first load.
"""
import os
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    raise SystemExit("pip install huggingface_hub transformers first")

MODEL = os.environ.get("PLIP_MODEL", "vinid/plip")
default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plip_weights")
OUT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else default_out)

path = snapshot_download(repo_id=MODEL, local_dir=OUT)
print(f"downloaded {MODEL} -> {path}")
print("scp this folder to the node, then set PLIP_MODEL to it.")
