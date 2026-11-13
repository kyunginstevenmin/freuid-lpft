#!/usr/bin/env bash
# Rebuild splits/{train,val,test}.csv from the raw per-dataset label
# CSVs. Run from data/:  bash build_dataset.sh
# Deterministic: all sampling/splitting is seeded (42); a full rerun
# byte-reproduces every intermediate and the final CSVs (verified 2026-08-20).
set -euo pipefail

PY=${PY:-python}
SIDT=sources/SIDT_clips_cropped
FID=sources/FantasyID
FRD=sources/FREUID

# Step 0 (not run here): pipeline/{SIDT,FantasyID}/generate_csv.py create the
# raw label csvs from the image dirs. SIDT's is deterministic (sorted walk);
# FantasyID's is filesystem-order-dependent and must NOT be rerun over the
# existing train.csv/test.csv (see its docstring). FREUID's train_labels.csv
# ships with the challenge data.

# SIDT: type -> unify -> partition -> sample -> split
$PY pipeline/SIDT/add_type_column.py
$PY pipeline/SIDT/unify_schema.py
$PY pipeline/SIDT/partition_by_type.py
$PY pipeline/SIDT/sample_frames.py --csv $SIDT/pool_allframes.csv    --n 10 --out $SIDT/pool_trainval.csv
# test is sampled 10-then-1: that order is the committed test set's provenance
# (1-from-all picks different frames) — do not collapse to a single --n 1
$PY pipeline/SIDT/sample_frames.py --csv $SIDT/holdout_allframes.csv --n 10 --out $SIDT/holdout_frames10.csv
$PY pipeline/SIDT/sample_frames.py --csv $SIDT/holdout_frames10.csv  --n 1  --out $SIDT/holdout_test.csv
$PY pipeline/SIDT/train_val_split.py

# FantasyID: combine -> type -> unify -> partition -> split
$PY pipeline/FantasyID/combine_splits.py
$PY pipeline/FantasyID/add_type_column.py
$PY pipeline/FantasyID/unify_schema.py
$PY pipeline/FantasyID/partition_by_type.py
$PY pipeline/FantasyID/train_val_split.py

# FREUID: unify -> partition -> split
$PY pipeline/FREUID/unify_schema.py
$PY pipeline/FREUID/partition_by_type.py
$PY pipeline/FREUID/train_val_split.py

# IDNet: split from the committed manifest (docs/idnet-integration-plan.md).
# splits consumes v2 (6 fraud types, Sept-2024 release); v1
# (pipeline/IDNet/, sources/IDNet/) is preserved frozen and not built here.
# Step 0 (not run here, needs network + AWS creds): make_manifest.py rebuilds
# the manifest from the remote zips (seeded, must reproduce the committed
# row-hash); fetch_images.py materializes sources/IDNet_v2/Images/ from it.
$PY pipeline/IDNet_v2/make_splits.py

# Combine per-dataset splits into splits/{train,val,test}.csv
$PY pipeline/combine_datasets.py
