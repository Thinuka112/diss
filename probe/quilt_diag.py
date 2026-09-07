"""
quilt_diag.py - diagnostic: is Quilt-LLaVA actually SEEING the image, and can it
ever say 'unusable'?  Runs on 3 usable + 3 unusable held-out slides, two questions
each. Short, pasteable output. Confirms whether the all-'usable' probe result is a
genuine model limitation or a technical (vision-path / prompt) issue.

    python quilt_diag.py
"""
import os, csv, torch
from PIL import Image
from llava.constants import (IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN,
                             DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (process_images, tokenizer_image_token,
                            get_model_name_from_path, KeywordsStoppingCriteria)

MODEL_PATH = "wisdomik/Quilt-Llava-v1.5-7b"
HOME = os.path.expanduser("~")
IMG_DIR  = os.path.join(HOME, "quilt", "probe_images")
MANIFEST = os.path.join(HOME, "quilt", "probe_testset.csv")
CONV_MODE = "llava_v1"

Q_DESCRIBE = "Describe what you see in this image in one short sentence."
Q_SIMPLE   = ("Is this small-intestine tissue section usable for reading epithelium "
              "along the apical-to-basal axis? Answer 'usable' or 'unusable', then one short reason.")

disable_torch_init()
mn = get_model_name_from_path(MODEL_PATH)
tok, model, ip, _ = load_pretrained_model(MODEL_PATH, None, mn, load_4bit=True)
print("model loaded\n", flush=True)


def ask(path, q, maxtok=64):
    image = Image.open(path).convert("RGB")
    it = process_images([image], ip, model.config)
    it = ([t.to(model.device, dtype=torch.float16) for t in it]
          if isinstance(it, list) else it.to(model.device, dtype=torch.float16))
    tokstr = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
              if getattr(model.config, "mm_use_im_start_end", False) else DEFAULT_IMAGE_TOKEN)
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], tokstr + "\n" + q)
    conv.append_message(conv.roles[1], None)
    ids = tokenizer_image_token(conv.get_prompt(), tok, IMAGE_TOKEN_INDEX,
                                return_tensors="pt").unsqueeze(0).to(model.device)
    stop = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    sc = KeywordsStoppingCriteria([stop], tok, ids)
    with torch.inference_mode():
        out = model.generate(ids, images=it, do_sample=False, max_new_tokens=maxtok,
                             use_cache=True, stopping_criteria=[sc])
    return tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True)[0].strip().replace("\n", " ")


rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
pick = [r for r in rows if r["true_label"] == "usable"][:3] + \
       [r for r in rows if r["true_label"] == "unusable"][:3]
for r in pick:
    p = os.path.join(IMG_DIR, r["image_id"])
    print("=" * 72)
    print(f"{r['image_id']}   TRUE={r['true_label']}")
    print("  describe:", ask(p, Q_DESCRIBE, 48))
    print("  usable? :", ask(p, Q_SIMPLE, 64))
print("=" * 72)
print("DIAG_DONE", flush=True)
