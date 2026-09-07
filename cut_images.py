
import os
import csv
from PIL import Image


# CONFIG - edit these only

SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER  = os.path.join(SCRIPT_DIR, "HPA_small_intestine")
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "HPA_small_intestine_tiles")
COLS = 8
ROWS = 8
MANIFEST_CSV  = os.path.join(OUTPUT_FOLDER, "manifest.csv")
# ------------------------------------------------------------------

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")
SEX_FIELDS = ("Male", "Female", "Unknown")


def gene_from_filename(filename):
    """Extract the gene symbol from a source filename.

    The gene is the token immediately before the sex field, e.g.
        Small_intestine_PDGFRA_Female_84_2256.jpg -> "PDGFRA"
        Duodenum_A1BG_Male_50_1904.jpg            -> "A1BG"
    Falls back to "Unknown" if the expected pattern is not found.
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    tokens = stem.split("_")
    for i, token in enumerate(tokens):
        if token in SEX_FIELDS and i > 0:
            return tokens[i - 1]
    return "Unknown"


def cut_image(image_path, output_root, cols, rows, manifest_writer):
    """Cut a single image into a cols x rows grid of tiles.

    Saves the tiles into a per-source subfolder, writes one manifest row per tile,
    and returns (tile_count, gene).
    """
    source_name = os.path.basename(image_path)
    stem = os.path.splitext(source_name)[0]
    gene = gene_from_filename(source_name)


    tile_folder = os.path.join(output_root, stem)
    os.makedirs(tile_folder, exist_ok=True)

    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    width, height = image.size


    tile_w = width // cols
    tile_h = height // rows

    pad = len(str(cols * rows))

    tile_index = 1
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            upper = r * tile_h
            box = (left, upper, left + tile_w, upper + tile_h)
            tile = image.crop(box)

            tile_name = f"{stem}__tile_{tile_index:0{pad}d}.jpg"
            tile_path = os.path.join(tile_folder, tile_name)
            tile.save(tile_path, "JPEG", quality=95)

            rel_path = os.path.relpath(tile_path, SCRIPT_DIR).replace(os.sep, "/")
            manifest_writer.writerow([rel_path, tile_index, gene, source_name])
            tile_index += 1

    image.close()
    return tile_index - 1, gene


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)


    images = [
        os.path.join(INPUT_FOLDER, f)
        for f in sorted(os.listdir(INPUT_FOLDER))
        if f.lower().endswith(VALID_EXTENSIONS)
    ]

    if not images:
        print(f"No images found in {INPUT_FOLDER}")
        return

    total_tiles = 0
    genes_seen = set()

    with open(MANIFEST_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["tile_path", "tile_index", "gene", "source_image"])

        for image_path in images:
            tile_count, gene = cut_image(image_path, OUTPUT_FOLDER, COLS, ROWS, writer)
            total_tiles += tile_count
            genes_seen.add(gene)
            print(f"{os.path.basename(image_path)}: {tile_count} tiles, gene={gene}")


    print("\n--- Summary ---")
    print(f"Total tiles:    {total_tiles}")
    print(f"Source images:  {len(images)}")
    print(f"Distinct genes: {len(genes_seen)}")
    print(f"Manifest:       {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
