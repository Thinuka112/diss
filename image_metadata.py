"""
image_metadata.py - attach HPA gene-level annotation to each downloaded image.

WHY THE WEB, NOT THE LOCAL XML:
  The local normal_expression_Small.xml is a narrow export (only high-expression
  small-intestine genes; its entry list starts at A1CF and is missing A1BG,
  A2ML1, A4GNT, ...). The images, downloaded from a broader/older query, span 50
  genes - 40 of which are absent from that file. A gene's small-intestine IHC
  expression level and antibody reliability are GENE-LEVEL properties (independent
  of which donor's section you hold), so we fetch them fresh, per gene, from the
  Human Protein Atlas and join by gene. This covers all 50 genes.

WHAT WE ADD (per gene, from proteinatlas.org):
  - ensembl             : the gene's Ensembl ID (stable join key / provenance).
  - si_expression_level : HPA's small-intestine IHC call (Not detected / Low /
                          Medium / High), preferring Duodenum.
  - reliability         : trust in the antibody's IHC staining (enhanced /
                          supported / approved / uncertain).

WHAT WE DELIBERATELY DO NOT ADD - a per-image "normal vs cancer" snomed flag:
  It cannot be recovered reliably here. The images predate this HPA release, so
  their donor IDs no longer match, and HPA's per-gene data contains BOTH normal
  and cancer small-intestine samples (carcinoid, adenocarcinoma, lymphoma), so a
  gene-level morphology aggregate would mix them and mislabel a normal section.
  These images come from HPA's normal-tissue atlas and are normal by construction.

There is NO HPA "slice quality" field either; that label is produced by this
project's own pipeline (filter_blank_tiles.py). These columns are context, not
the label. Image identity (tissue/sex/age/patient) is taken from the filename,
which the downloader wrote and is authoritative. Non-destructive: writes one CSV.
"""

import os
import csv
import json
import time
import io
import xml.etree.ElementTree as ET

import requests

# ------------------------------------------------------------------
# CONFIG - edit these only
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "HPA_small_intestine")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "image_metadata.csv")
REQUEST_PAUSE = 0.3          # polite delay between HPA requests (seconds)
# ------------------------------------------------------------------

SMALL_INTESTINE_TERMS = ("small intestine", "duodenum", "jejunum", "ileum")
SEX_FIELDS = ("Male", "Female", "Unknown")
SEARCH_URL = "https://www.proteinatlas.org/api/search_download.php"
GENE_XML_URL = "https://www.proteinatlas.org/{ensembl}.xml"
MAX_RETRIES = 3
RETRY_BACKOFFS = (2, 4, 8)


def parse_filename(name):
    """Split {tissue}_{gene}_{sex}_{age}_{patientId}.jpg, anchoring on the sex
    token (tissue may hold underscores/commas; gene may hold hyphens)."""
    stem = os.path.splitext(name)[0]
    tokens = stem.split("_")
    for i, tok in enumerate(tokens):
        if tok in SEX_FIELDS and i >= 1:
            tissue = " ".join(tokens[:i - 1]) if i >= 2 else tokens[0]
            gene = tokens[i - 1]
            age = tokens[i + 1] if i + 1 < len(tokens) else ""
            patient_id = tokens[i + 2] if i + 2 < len(tokens) else ""
            return tissue, gene, tok, age, patient_id
    return "", "Unknown", "", "", ""


def http_get(url, params=None):
    """GET with retry/backoff. Returns response text, or None on final failure."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            return r.text
        except Exception as e:
            if attempt < MAX_RETRIES:
                wait = RETRY_BACKOFFS[min(attempt, len(RETRY_BACKOFFS) - 1)]
                print(f"    request failed ({e}); retry {attempt + 1}/{MAX_RETRIES} in {wait}s...")
                time.sleep(wait)
            else:
                print(f"    giving up after {MAX_RETRIES + 1} attempts: {url}")
    return None


def resolve_ensembl(gene):
    """Gene symbol -> Ensembl gene ID via the HPA search API (exact match)."""
    text = http_get(SEARCH_URL, {"search": gene, "format": "json",
                                 "columns": "g,eg", "compress": "no"})
    if not text:
        return None
    try:
        rows = json.loads(text)
    except json.JSONDecodeError:
        return None
    for row in rows:
        if row.get("Gene") == gene:
            return row.get("Ensembl")
    return rows[0].get("Ensembl") if rows else None


def pick_si_level(si_levels):
    """Best small-intestine IHC expression level from a tissue->level map."""
    if "duodenum" in si_levels:
        return si_levels["duodenum"]
    if "small intestine" in si_levels:
        return si_levels["small intestine"]
    for tissue, level in si_levels.items():
        if any(term in tissue for term in SMALL_INTESTINE_TERMS):
            return level
    return ""


def parse_gene_xml(xml_text):
    """Extract (si_expression_level, reliability) from one gene's HPA XML.

    Both come from the gene-level IHC tissue summary
    <tissueExpression technology="IHC" assayType="tissue">, NOT from the
    per-antibody raw blocks (which carry <level type="staining"> instead of
    <level type="expression">).
    """
    reliability = ""
    si_levels = {}
    cur_data_tissue = None
    in_ihc_tissue = False
    in_antibody = False
    stack = []

    for event, elem in ET.iterparse(io.BytesIO(xml_text.encode("utf-8")),
                                    events=("start", "end")):
        tag = elem.tag
        if event == "start":
            stack.append(tag)
            if tag == "antibody":
                in_antibody = True
            elif tag == "tissueExpression":
                in_ihc_tissue = (elem.attrib.get("technology") == "IHC"
                                 and elem.attrib.get("assayType") == "tissue")
            elif tag == "data" and in_ihc_tissue and not in_antibody:
                cur_data_tissue = None
            continue

        parent = stack[-2] if len(stack) >= 2 else None
        if tag == "verification" and in_ihc_tissue and not in_antibody \
                and not reliability and elem.attrib.get("type") == "reliability":
            reliability = (elem.text or "").strip()
        elif tag == "tissue" and in_ihc_tissue and not in_antibody and parent == "data":
            cur_data_tissue = (elem.text or "").strip()
        elif tag == "level" and in_ihc_tissue and not in_antibody and parent == "data" \
                and elem.attrib.get("type") == "expression":
            if cur_data_tissue:
                si_levels.setdefault(cur_data_tissue.lower(), (elem.text or "").strip())
        elif tag == "antibody":
            in_antibody = False
        elif tag == "tissueExpression":
            in_ihc_tissue = False

        if stack:
            stack.pop()
        elem.clear()

    return pick_si_level(si_levels), reliability


def gene_meta(gene, cache):
    """Resolve + fetch one gene's annotation, memoised in `cache`."""
    if gene in cache:
        return cache[gene]
    ensembl = resolve_ensembl(gene)
    time.sleep(REQUEST_PAUSE)
    result = {"ensembl": ensembl or "", "si_expression_level": "", "reliability": ""}
    if ensembl:
        xml_text = http_get(GENE_XML_URL.format(ensembl=ensembl))
        time.sleep(REQUEST_PAUSE)
        if xml_text:
            level, rel = parse_gene_xml(xml_text)
            result.update(si_expression_level=level, reliability=rel)
    cache[gene] = result
    return result


def build(image_folder, output_csv):
    files = sorted(f for f in os.listdir(image_folder)
                   if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not files:
        print(f"No images found in {image_folder}")
        return

    parsed = {f: parse_filename(f) for f in files}
    genes = sorted({p[1] for p in parsed.values()})
    print(f"{len(files)} images across {len(genes)} genes; fetching from HPA...")

    cache = {}
    for i, gene in enumerate(genes, 1):
        gm = gene_meta(gene, cache)
        print(f"  [{i}/{len(genes)}] {gene} -> {gm['ensembl'] or 'UNRESOLVED'}"
              f"  level={gm['si_expression_level'] or '-'}"
              f"  reliability={gm['reliability'] or '-'}")

    fieldnames = ["source_image", "gene", "ensembl", "tissue", "sex", "age",
                  "patient_id", "si_expression_level", "reliability"]
    unresolved = set()
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for f in files:
            tissue, gene, sex, age, patient_id = parsed[f]
            gm = cache[gene]
            if not gm["ensembl"]:
                unresolved.add(gene)
            writer.writerow({
                "source_image": f, "gene": gene, "ensembl": gm["ensembl"],
                "tissue": tissue, "sex": sex, "age": age, "patient_id": patient_id,
                "si_expression_level": gm["si_expression_level"],
                "reliability": gm["reliability"],
            })

    levels, rels = {}, {}
    for gene in genes:
        gm = cache[gene]
        lk = gm["si_expression_level"] or "(none)"
        rk = gm["reliability"] or "(none)"
        levels[lk] = levels.get(lk, 0) + 1
        rels[rk] = rels.get(rk, 0) + 1

    print("\n--- Summary ---")
    print(f"Images written:  {len(files)}")
    print(f"Genes resolved:  {len(genes) - len(unresolved)}/{len(genes)}")
    print(f"Output:          {output_csv}")
    print("\nSI expression level (per gene):")
    for k, v in sorted(levels.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14} {v}")
    print("Reliability (per gene):")
    for k, v in sorted(rels.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<14} {v}")
    if unresolved:
        print(f"\nCould not resolve {len(unresolved)} gene(s): {', '.join(sorted(unresolved))}")


if __name__ == "__main__":
    build(IMAGE_FOLDER, OUTPUT_CSV)
