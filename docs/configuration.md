# Configuration, seeds and environment

This file replaces dissertation Appendix B. It records how every run was configured and how to rebuild the dataset from scratch.

## Config blocks

Every script keeps its parameters in a config block of constants at the top of the file. Edit those only. Nothing is passed on the command line unless the script says so. Paths in the blocks are relative to the repo root.

## Seeds

The seed is 42 everywhere randomness enters. That covers the donor-grouped fold partition, the train and validation splits, augmentation and weight initialisation. The fold partition regenerates identically on every run. One caveat. `train_quality_cnn.py` enables `cudnn.benchmark`. Runs are seed-stable on one machine but not bit-identical across GPUs.

## Environment

Python 3.11 or newer. Install with `pip install -r requirements.txt`. Training needs a CUDA build of PyTorch and `train_quality_cnn.py` refuses to train on CPU. The stat tests need scipy only. The demos need the standard library or scikit-learn only.

## Hardware

Training ran on garlick. That is a GPU node of the University of Bath's Hex cloud. Its RTX 2080 cards have 8 GB of memory. At 640 by 640 pixels that fits a batch of six images and sets the maximum workable resolution. Smaller local runs used an RTX 2060 at 448 pixels. The resolution gap measures about +0.03 macro-F1 in favour of 640.

## Rebuilding the dataset

Raw images are not shipped. The pipeline rebuilds them from the atlas export.

1. Download the HPA small-intestine XML export (`normal_expression_Small.xml` from proteinatlas.org).
2. Set `XML_FILE` in the config block of `download_small_intestine.py`, `download_bulk.py` and `download_clean_donors.py`.
3. Run the downloaders. Filenames encode `{tissue}_{gene}_{sex}_{age}_{patientId}.jpg`. That filename is the join key for every table in the pipeline.
4. Run `cut_images.py` for the 8x8 tiling and `image_metadata.py` for the metadata join.
5. The label CSVs are shipped in `review_tool/results/`. Training and evaluation run from those directly.

The pipeline never deletes or moves an image. Unusable is a label. Not a deletion.
