

import xml.etree.ElementTree as ET
import requests
import os
import time
from collections import defaultdict


XML_FILE = r"C:\Users\<user>\Downloads\normal_expression_Small.xml"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FOLDER = os.path.join(SCRIPT_DIR, "HPA_small_intestine")
MAX_IMAGES = 120
MAX_PER_GENE = 2


SMALL_INTESTINE_TERMS = ("small intestine", "duodenum", "jejunum", "ileum")


MAX_RETRIES = 3
RETRY_BACKOFFS = (2, 4, 8)
ROOT_CLEAR_EVERY = 50


def download_image(url, dest_path):
    """Fetch one image to dest_path, retrying with backoff.

    Returns True on success, False if every attempt fails.
    """
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                f.write(response.content)
            return True
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFFS[min(attempt, len(RETRY_BACKOFFS) - 1)]
                print(
                    f"  request failed ({e}); retry {attempt + 1}/{MAX_RETRIES} in {wait}s...")
                time.sleep(wait)
            else:
                print(
                    f"  giving up after {MAX_RETRIES + 1} attempts: {url} ({e})")
    return False


def download(xml_file, output_folder, max_images=None, max_per_gene=None):
    """Stream the HPA XML and download Small intestine IHC sample images.

    Returns (downloaded_count, [saved_file_paths]).

    Nesting (confirmed from the file): gene <name> is one-per-<entry>; the
    per-sample unit is <patient> (sex/age/patientId, then one <sample> with
    <snomedParameters> and the sample <imageUrl>). So gene resets per entry and
    every other tracked field resets per patient. is_si is also cleared when a
    <patient> ends so the out-of-sample representative "selected" images can
    never inherit a previous sample's small-intestine match.
    """
    os.makedirs(output_folder, exist_ok=True)

    current_name = "Unknown"

    current_sex = "Unknown"
    current_age = "Unknown"
    current_patient_id = "Unknown"
    current_tissue = "Small_intestine"
    current_is_si = False

    downloaded = 0
    skipped_existing = 0
    failed_urls = []
    saved_files = []
    per_gene = defaultdict(int)
    entries_seen = 0

    context = ET.iterparse(xml_file, events=("start", "end"))
    _, root = next(context)

    stop = False
    for event, elem in context:
        if event == "start":
            if elem.tag == "entry":
                current_name = "Unknown"
            elif elem.tag == "patient":

                current_sex = "Unknown"
                current_age = "Unknown"
                current_patient_id = "Unknown"
                current_tissue = "Small_intestine"
                current_is_si = False
            continue

        tag = elem.tag

        if tag == "name" and current_name == "Unknown":

            current_name = elem.text.replace(
                " ", "_") if elem.text else "Unknown"
        elif tag == "sex":
            current_sex = elem.text if elem.text else "Unknown"
        elif tag == "age":
            current_age = elem.text if elem.text else "Unknown"
        elif tag == "patientId":
            current_patient_id = elem.text if elem.text else "Unknown"
        elif tag == "snomed" and "tissueDescription" in elem.attrib:
            td = elem.attrib["tissueDescription"]
            if any(term in td.lower() for term in SMALL_INTESTINE_TERMS):
                current_is_si = True
                current_tissue = td.replace(" ", "_")
        elif tag == "imageUrl" and elem.text:
            image_url = elem.text

            if current_is_si and "rna" not in image_url.lower():
                gene = current_name
                if max_per_gene is None or per_gene[gene] < max_per_gene:
                    base = f"{current_tissue}_{gene}_{current_sex}_{current_age}_{current_patient_id}.jpg"
                    file_path = os.path.join(output_folder, base)

                    if os.path.exists(file_path):

                        skipped_existing += 1
                        per_gene[gene] += 1
                        print(
                            f"[{downloaded + skipped_existing}] Already have: {base}")
                    else:
                        if download_image(image_url, file_path):
                            downloaded += 1
                            per_gene[gene] += 1
                            saved_files.append(file_path)
                            print(
                                f"[{downloaded + skipped_existing}] Downloaded: {base}")
                            # be polite between successful downloads
                            time.sleep(0.5)
                        else:
                            failed_urls.append(image_url)

                    if max_images is not None and (downloaded + skipped_existing) >= max_images:
                        print(
                            f"\nReached the target of {max_images} images. Stopping.")
                        stop = True
        elif tag == "patient":

            current_is_si = False
        elif tag == "entry":
            entries_seen += 1
            if entries_seen % ROOT_CLEAR_EVERY == 0:
                root.clear()

        elem.clear()

        if stop:
            break

    print("\n--- Summary ---")
    print(f"New downloads this run:    {downloaded}")
    print(f"Already present (skipped): {skipped_existing}")
    print(f"Total satisfied targets:   {downloaded + skipped_existing}")
    print(f"Output folder:             {output_folder}")
    if failed_urls:
        print(f"\nPermanently failed after retries ({len(failed_urls)}):")
        for u in failed_urls:
            print(f"  {u}")
        print("Re-run the script to retry these - files already on disk will be skipped.")
    else:
        print("\nNo permanent download failures.")

    return downloaded, saved_files


if __name__ == "__main__":
    download(XML_FILE, OUTPUT_FOLDER, MAX_IMAGES, MAX_PER_GENE)
