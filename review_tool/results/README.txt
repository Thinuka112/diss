What each CSV in this folder is
===============================

Names follow <what>_<export date YYYYMMDD>.csv. The dates stay in the names so
every number in the write-up traces back to the exact file behind it. "expert"
means the domain expert's judgments (reviewer column pseudonymised: expert/author).

Label datasets (whole slides, usable vs unusable):

  final_clean_dataset_20260802.csv    the main dataset - all 847 expert labels.
                                      Every Stage-1 result trains and evaluates
                                      on this file.
  final_clean_dataset_20260728.csv    earlier cut, first 597 labels only. Kept
                                      because the 847 file is rebuilt from it
                                      (assemble_enriched_dataset.py).
  combined_human_confident_20260827.csv
                                      847 expert labels plus 4,834 machine labels
                                      that passed the confidence audit. Training
                                      set for the combined-data models.

The expert's raw labelling passes (inputs to the above):

  expert_original500_20260727.csv     first pass, 500 slides.
  expert_recode50_20260727.csv        50 slides re-graded blind, for the
                                      intra-rater consistency check.
  expert_topup_20260802.csv           250 slides the model flagged as likely
                                      unusable, then human-labelled (top-up pass).

Stage-2 gold standard (tiles inside slides):

  loc_stretch_export_20260827.csv     the expert's tile-level yes/no judgments,
                                      840 rows. Stage-2 localisation is scored
                                      against this file.
