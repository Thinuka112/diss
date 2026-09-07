"""
quilt_probe.py - Quilt-LLaVA batch probe (runs ON garlick, inside the quilt-llava venv).

Runs a FIXED prompt (the expert's rubric verbatim) over the held-out probe images and writes
predictions to a CSV. Scoring (unusable recall etc.) is done afterwards on the laptop
against the human labels. Based on the repo's own llava/serve/cli.py inference path.

    python quilt_probe.py            # all images in the manifest
    python quilt_probe.py 8          # first 8 (quick sanity)
"""
import os
import sys
import csv
import torch
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
OUT      = os.path.join(HOME, "quilt", "quilt_probe_results.csv")
CONV_MODE = "llava_v1"           # Quilt-Llava-v1.5 -> "v1" -> llava_v1 (per cli.py auto-infer)

# the expert's rubric, verbatim (Slice_Usability_Rubric.docx)
PROMPT = (
    "You are a histopathology quality reviewer assessing a Human Protein Atlas small-intestine "
    "tissue section. A slice is USABLE if it contains AT LEAST ONE 'qualifying stretch' of "
    "epithelium anywhere in the image, at any orientation. A qualifying stretch requires ALL of: "
    "(1) at least about 5 adjacent epithelial cells in a clear LINEAR arrangement; "
    "(2) BOTH surfaces visible - the apical surface facing the lumen and the basal surface facing "
    "the underlying tissue; (3) staining that can be read consistently along the stretch. "
    "Stain colour does not matter (brown signal or blue nuclear counterstain are both fine). "
    "Poor or complex regions elsewhere do NOT matter - one qualifying stretch is enough. "
    "A slice is UNUSABLE if NO qualifying stretch exists - for example crypt cross-sections "
    "(circular, tightly packed profiles), gland cells, goblet-cell-dominated regions, or "
    "crypt-to-villus transition zones. If genuinely ambiguous, answer unusable. "
    "Answer with exactly one word: usable or unusable."
)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    disable_torch_init()
    model_name = get_model_name_from_path(MODEL_PATH)
    print(f"loading {MODEL_PATH} (4-bit) ...", flush=True)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        MODEL_PATH, None, model_name, load_4bit=True)
    print("model loaded.", flush=True)

    def classify(path):
        image = Image.open(path).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)
        if isinstance(image_tensor, list):
            image_tensor = [t.to(model.device, dtype=torch.float16) for t in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + PROMPT
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + PROMPT
        conv = conv_templates[CONV_MODE].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                          return_tensors="pt").unsqueeze(0).to(model.device)
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        stopping = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
        with torch.inference_mode():
            out = model.generate(input_ids, images=image_tensor, do_sample=False,
                                 max_new_tokens=16, use_cache=True, stopping_criteria=[stopping])
        txt = tokenizer.batch_decode(out[:, input_ids.shape[1]:],
                                     skip_special_tokens=True)[0].strip().lower()
        if "unusable" in txt: return "unusable", txt
        if "usable" in txt:   return "usable", txt
        return "?", txt

    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    if limit:
        rows = rows[:limit]
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["image_id", "true_label", "pred", "raw"])
        for i, r in enumerate(rows, 1):
            try:
                pred, txt = classify(os.path.join(IMG_DIR, r["image_id"]))
            except Exception as e:
                pred, txt = "ERR", str(e)[:150]
            w.writerow([r["image_id"], r["true_label"], pred, txt]); fh.flush()
            if i % 10 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}", flush=True)
    print(f"DONE -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
