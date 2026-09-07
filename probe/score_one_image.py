"""
score_one_image.py - run the Stage-1 quality ensemble on a single image.
Same model + preprocessing as mine_unusable_candidates.py (5-fold 448px ResNet-50,
ensemble-mean softmax, P(unusable)=index 0).

    python probe/score_one_image.py "C:\\path\\to\\image.png"
"""
import sys, glob, os
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
from torchvision import transforms
import timm

MODEL_GLOB = "models/quality/quality_resnet50_fold*.pt"
INPUT_SIZE = 448
MEAN = [0.485, 0.456, 0.406]; STD = [0.229, 0.224, 0.225]
CLASSES = ["unusable", "usable"]

path = sys.argv[1]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
paths = sorted(glob.glob(MODEL_GLOB))
if not paths:
    raise SystemExit(f"no checkpoints match {MODEL_GLOB}")

im = np.asarray(Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR))
a = im.astype(np.float32); lum = a.mean(2); mx = a.max(2); mn = a.min(2)
sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
tf = float(((lum < 200) | (sat > 0.10)).mean())

t = torch.from_numpy(im.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
t = transforms.Normalize(MEAN, STD)(t).to(dev)

per_fold = []
probs = None
with torch.no_grad(), torch.autocast("cuda", enabled=(dev.type == "cuda")):
    for p in paths:
        ck = torch.load(p, map_location=dev)
        m = timm.create_model(ck.get("arch", "resnet50"), pretrained=False,
                              num_classes=len(ck.get("classes", CLASSES)))
        m.load_state_dict(ck["state_dict"]); m.eval().to(dev)
        pr = F.softmax(m(t).float(), 1)
        per_fold.append(float(pr[0, CLASSES.index("unusable")]))
        probs = pr if probs is None else probs + pr
probs = (probs / len(paths)).cpu().numpy()[0]
pu = float(probs[CLASSES.index("unusable")])

print(f"image: {os.path.basename(path)}")
print(f"per-fold P(unusable): " + ", ".join(f"{x:.3f}" for x in per_fold))
print(f"ENSEMBLE  P(unusable)={pu:.3f}   P(usable)={1-pu:.3f}   tissue_fraction={tf:.3f}")
print(f"model verdict @0.5 threshold: {'UNUSABLE' if pu >= 0.5 else 'USABLE'}")
