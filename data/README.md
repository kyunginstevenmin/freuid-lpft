# final dataset
the final split of the dataset will consists of:
1) train
2) val
3) test.
train and val will be from the same underlying distribution.
test is constructed of purely unseen document types.


# data schema
the columns of the dataset will be unified as:
- id,image_path,label,type,source

# pipeline
One command, run from `data/`, rebuilds everything from the raw label CSVs (deterministic,
byte-reproduces the committed files): `bash build_dataset.sh`

Per dataset:
1. unify schema of rows.
2. partition the unseen-type test holdout from the train/val pool.
3. train/val split each dataset's in-distribution pool (see per-dataset sections).
4. combine the per-dataset splits into `splits/{train,val,test}.csv`
   (see **Combined** below).

# where the raw sources go

Download each source and place it under `sources/`, in the layout its release ships. The
build script, the split CSVs and training all name these paths, so the layout is fixed:

```
sources/
  FREUID/
    train_labels.csv                  ships with the Kaggle training data
    FREUID_train/*.jpeg
  SIDT_clips_cropped/                 the SIDTD clips_cropped release
    Images/reals/*.jpg
    Images/fakes/*.jpg
  FantasyID/                          as released
    train/{bonafide,attack}/...
    test/{bonafide,attack}/...
    train.csv, test.csv               committed here; do not regenerate (see FantasyID below)
  IDNet_v2/
    manifest.csv                      committed here
    Images/<country>/<folder>/*.png   written by pipeline/IDNet_v2/fetch_images.py
```

The image folders are git-ignored. Only the label and manifest CSVs listed as committed are in
the repo. With the sources in place, `bash build_dataset.sh` from this directory reproduces
`splits/`, and `python src/train.py ... --split_dir data/splits` from the repo root finds the
images, because image paths resolve against the parent of the split directory.

# dataset statistics

Rows per source in each split (`splits/{train,val,test}.csv`):

| source    |  train |    val |   test |  total |
|-----------|-------:|-------:|-------:|-------:|
| freuid    | 42,788 | 10,697 | 15,867 | 69,352 |
| sidt      |  6,540 |  1,700 |    367 |  8,607 |
| fantasyid |  2,084 |    534 |    666 |  3,284 |
| idnet     | 11,970 |  3,030 |  5,000 | 20,000 |
| **total** | 63,382 | 15,961 | 21,900 | 101,243 |

Document types per source (train/val types are disjoint from test — test types
are held out as unseen):

| source    | train/val types | test types | test type names                          |
|-----------|----------------:|-----------:|------------------------------------------|
| freuid    |               4 |          1 | EGYPT/DL                                 |
| sidt      |               7 |          3 | esp_id, fin_id, rus_internalpassport     |
| fantasyid |              11 |          2 | portugal, turkiye                        |
| idnet     |              15 |          5 | idnet_{esp, fin, rus, az, ca}            |
| **total** |          **37** |     **11** | 48 document types overall                |

Label balance per split (0 = genuine, 1 = attack):

| split | genuine | attack |     total | genuine : attack |
|-------|--------:|-------:|----------:|-----------------:|
| train |  37,769 | 25,613 |    63,382 |      59.6 : 40.4 |
| val   |   9,510 |  6,451 |    15,961 |      59.6 : 40.4 |
| test  |  10,959 | 10,941 |    21,900 |      50.0 : 50.0 |

(pre-IDNet: train 31,769/19,643 = 61.8:38.2, val 8,010/4,921, test
8,459/8,441 — IDNet adds a ~50:50 slice to every split.)

Notes: FREUID is ~68% of all rows (was ~85% before IDNet). Train/val skew
genuine (~60%); the test set is near-balanced (~50/50). SIDT test is
downsampled to 1 frame per document (367 rows), while its train/val keep up
to 10 frames per document.

## FREUID

```
train_labels.csv (69,352 rows)
schema: id, image_path, label, is_digital, type
        │  pipeline/FREUID/unify_schema.py
        │  drop is_digital, add source="freuid"
        ▼
train_labels_unified.csv (69,352 rows)
schema: id, image_path, label, type, source   ← unified schema
        │  pipeline/FREUID/partition_by_type.py
        │  holdout = {EGYPT/DL}
        ▼
   ┌────┴─────────────────────────────────────────┐
   │                                               │
pool_trainval.csv                   holdout_test.csv
   (53,485 rows)  train/val pool       (15,867 rows)  test (unseen type)
   │
   │  pipeline/FREUID/train_val_split.py  --val-frac 0.2  --seed 42
   │  split rows directly (id unique); stratified per type
   ▼
   ┌────┴───────────────┐
split_train.csv     split_val.csv
   (42,788 rows)       (10,697 rows)
```

Steps:
1. **Unify** — `unify_schema.py` drops `is_digital`, adds `source="freuid"`,
   and writes `train_labels_unified.csv` in the target schema. Inputs are
   never mutated.
2. **Partition** — `partition_by_type.py` splits by `type == 'EGYPT/DL'`:
   Egypt driving licenses become the unseen-type test split
   (`holdout_test.csv`), everything else the train/val pool
   (`pool_trainval.csv`). The original ad-hoc partition run (which had no
   script) is recorded in `split_manifest.json` (source sha256, row counts,
   git commit); the script reproduces its outputs.
3. **Train/val split** — `train_val_split.py` splits the train/val pool
   (`pool_trainval.csv`) 80/20 into `split_train.csv` and
   `split_val.csv`. Unlike SIDT/FantasyID there's no leakage grain: every `id`
   is a unique document (one image per id, no frames or variants), so rows are
   split directly — stratified per `type` to keep each type's 80/20 ratio.
   Seeded; an `id.is_unique` assert guards the assumption (fails loudly if the
   source ever gains repeated documents). The `holdout_test.csv` partition is
   the held-out test set of the unseen EGYPT/DL type and is not split further.

## SIDT_clips_cropped

```
Images/  (reals/ + fakes/ subdirs)
        │  pipeline/SIDT/generate_csv.py
        │  sorted rglob *.jpg; is_attack = (parent dir == "fakes")
        ▼
images.csv (74,037 rows)
schema: id, path, is_attack
        │  pipeline/SIDT/add_type_column.py
        │  type = first two id tokens (e.g. "alb_id", "rus_internalpassport")
        ▼
images_typed.csv (74,037 rows)   [+ type]
        │  pipeline/SIDT/unify_schema.py
        │  path→image_path, is_attack→label, bool→1/0, add source="sidt"
        ▼
images_typed_unified.csv (74,037 rows)
schema: id, image_path, label, type, source   ← unified schema
        │  pipeline/SIDT/partition_by_type.py
        │  holdout = {esp_id, rus_internalpassport, fin_id}
        ▼
   ┌────┴──────────────────────────────────────────────┐
   │                                                    │
pool_allframes.csv                     holdout_allframes.csv
   (52,023 rows)                          (22,014 rows)
   │                                                    │
   │  sample_frames.py --n 10 --seed 42                 │  sample_frames.py --n 10
   │  keep ≤10 frames/doc (doc = id before "_frame_")   ▼
   ▼                                       holdout_frames10.csv (3,669 rows)
pool_trainval.csv (8,240 rows)                          │  sample_frames.py --n 1
   │                                                    ▼
   │                                       holdout_test.csv
   │                                          (367 rows)  test (unseen types)
   │  pipeline/SIDT/train_val_split.py  --val-frac 0.2  --seed 42
   │  split by base doc {type}_{docnum}; stratified per type
   ▼
   ┌────┴───────────────────┐
split_train.csv         split_val.csv
   (6,540 rows)             (1,700 rows)
```

Steps:
1. **Generate** — `pipeline/SIDT/generate_csv.py` (formerly lived in the data
   dir) walks `Images/` for `*.jpg` (sorted, so output is deterministic) and
   emits one row per image, labelling `is_attack=True` when the file sits under
   a `fakes/` dir (`reals/` → False).
2. **Type** — `add_type_column.py` infers a `type` from the first two
   underscore-tokens of the id (e.g. `alb_id_14_...` → `alb_id`).
3. **Unify** — `unify_schema.py` renames `path→image_path` and
   `is_attack→label`, casts the label to `1/0`, adds `source="sidt"`, and
   reorders to the target schema (`images_typed_unified.csv`). Inputs are never
   mutated.
4. **Partition** — `partition_by_type.py` splits by a fixed holdout set
   `{esp_id, rus_internalpassport, fin_id}` — these types become the unseen
   test split (`holdout_allframes.csv`), everything else is train/val
   (`pool_allframes.csv`). Both keep every frame.
5. **Sample** — `sample_frames.py --n 10` keeps at most 10 frames per document
   (a doc is the id up to `_frame_`), so long clips don't dominate:
   `pool_allframes.csv → pool_trainval.csv`. Seeded; docs with <10 frames keep
   all. The holdout side is sampled the same way (`holdout_frames10.csv`) as
   the first stage of the test subsample (step 7).
6. **Split** — `train_val_split.py` splits the in-distribution pool
   (`pool_trainval.csv`) 80/20 into `split_train.csv` and
   `split_val.csv`. Split grain is the **base document** `{type}_{docnum}`
   (e.g. `alb_id_14`), not the row: the ≤5 frames of one id — and its fake
   variants, since each fake is a manipulation of that same real — all land on
   the same side, so no identity leaks across train/val. Documents are split
   (not rows), stratified per `type` so each type keeps its 80/20 ratio. Seeded
   for reproducibility; an assert guarantees no doc straddles the split. The
   identifier choice is worked out in `notebooks/SIDT_train_val_split.ipynb`.
   The holdout partition is the held-out test set of unseen types and is not
   split further.
7. **Subsample test to 1 frame/doc** — the holdout is sampled in two stages,
   `sample_frames.py --n 10` (`holdout_frames10.csv`, 3,669 rows) then `--n 1`
   (`holdout_test.csv`, 367 rows). The 10-then-1 order is the historical
   provenance of the committed test set — sampling 1 directly from all frames
   picks different (equally valid) frames and would silently change
   `splits/test.csv`, so keep the two-stage order. Unlike train/val
   (N=10), the test set keeps a single frame per doc so near-duplicate frames
   don't inflate or skew per-document test metrics — every test document weighs
   the same regardless of clip length. (FREUID is already 1 image/doc and
   FantasyID has no frames, so only SIDT needs this.)

## FantasyID

```
train/  test/  (each: bonafide/ + attack/… device subdirs)
        │  pipeline/FantasyID/generate_csv.py   (run separately per split)
        │  rglob *.jpg; is_attack = "attack" in path;
        │  attack_type from folder (digital_1|2→face_text, digital_3|…→text, facedancer→face)
        ▼
train.csv (1,899)          test.csv (1,385)
schema: path, is_attack, attack_type
        └──────────┬────────────────┘
                   │  pipeline/FantasyID/combine_splits.py
                   │  concat train + test; original split discarded
                   ▼
             combined.csv (3,284 rows)
                   │  pipeline/FantasyID/add_type_column.py
                   │  type = filename prefix before "-" (e.g. "arabic-003_03" → "arabic")
                   ▼
             combined_typed.csv (3,284 rows)   [+ type, 13 types]
                   │  pipeline/FantasyID/unify_schema.py
                   │  id = slug(full path minus ext), path→image_path,
                   │  is_attack→label bool→1/0, drop attack_type, add source="fantasyid"
                   ▼
             combined_typed_unified.csv (3,284 rows)
             schema: id, image_path, label, type, source   ← unified schema
                   │  pipeline/FantasyID/partition_by_type.py
                   │  holdout = {portugal, turkiye}
                   ▼
   ┌───────────────┴────────────────────────────────────────┐
   │                                                         │
pool_trainval.csv                               holdout_test.csv
   (2,618 rows)  train/val pool                    (666 rows)  test (unseen types)
   │
   │  pipeline/FantasyID/train_val_split.py  --val-frac 0.2  --seed 42
   │  split by doc = filename stem (e.g. arabic-003_03); stratified per type
   ▼
   ┌────┴───────────────────┐
split_train.csv         split_val.csv
   (2,084 rows)             (534 rows)
```

Steps:
1. **Generate** — `pipeline/FantasyID/generate_csv.py` (formerly lived in the
   data dir) walks each split dir for `*.jpg` and emits
   one row per image: `is_attack=True` when `"attack"` is in the path, plus an
   `attack_type` inferred from the device/method folder
   (`digital_1|digital_2`→`face_text`, `digital_3|textdiffuserft_bfei`→`text`,
   `facedancer`→`face`, bonafide→`none`). Produces `train.csv` and `test.csv`.
2. **Combine** — `combine_splits.py` concatenates `train.csv` and `test.csv`
   into `combined.csv`. The original train/test directory split is discarded on
   purpose; the test split is redefined below by holdout document type.
3. **Type** — `add_type_column.py` adds a `type` from the filename prefix before
   the first `-` (13 types total).
4. **Unify** — `unify_schema.py` converts the typed csv to the unified schema
   (`combined_typed_unified.csv`): derives
   `id` from a slugified full path (strip `.jpg`, join parts with `_` →
   `train_bonafide_huawei_arabic-003_03`), renames `path→image_path` and
   `is_attack→label` (bool→`1/0`), drops `attack_type`, and adds
   `source="fantasyid"`. The split-dir prefix is kept in the `id` so combined
   train+test rows can't collide (the bare filename stem repeats across devices
   *and* across splits). Inputs are never mutated.
5. **Partition** — `partition_by_type.py` splits the unified csv by the fixed
   holdout set `{portugal, turkiye}` — these become the unseen test types
   (`holdout_test.csv`, 666 rows), everything else is train/val
   (`pool_trainval.csv`, 2,618 rows).
6. **Split** — `train_val_split.py` splits the in-distribution pool
   (`pool_trainval.csv`) 80/20 into `split_train.csv`
   and `split_val.csv`. Split grain is the **filename stem** `doc` (e.g.
   `arabic-003_03`), not the row: one physical document appears ~9× — across the
   three capture methods (huawei / iphone15pro / scan), bonafide vs attack, and
   the original train/test dirs — all sharing nearly all their pixels. Grouping
   by stem keeps every variant of a document on the same side, so no identity
   (face/text/layout) leaks across train/val. Documents are split (not rows),
   stratified per `type` so each type keeps its 80/20 ratio. Seeded; an assert
   guarantees no doc straddles the split. The identifier choice is worked out in
   `notebooks/FantasyID_train_val_split.ipynb`. The
   `holdout_test.csv` partition is the held-out test set of
   unseen types and is not split further.

## IDNet

Two frozen versions, each with its own scripts and data dir — splits
consumes **v2**; v1 (4 fraud types, EU from the Jan-2024 release:
`pipeline/IDNet/`, `sources/IDNet/`, kept in the development repo, not here) is preserved for
comparability and is regenerable the same way (its row-hash is pinned in
tests alongside v2's). This section documents v2.

Purely digital renders (Guo et al., arXiv:2408.01690, CC0): real government
document templates content-erased via SD-2.0 inpainting, synthetic faces
(generated.photos) and generated text — no print/capture step anywhere. Added
2026-09-14 per `EVALUATION.md` §2.1/§2.3 and `docs/idnet-integration-plan.md`
(sampling decisions + provenance live there; this section is the pipeline).

```
20 country zips, ALL from the Sept-2024 Zenodo records (a regenerated
release with 6 fraud types; the Jan-2024 S3 mirror is a different person
sample and is not used — see docs/idnet-integration-plan.md)
        │  pipeline/IDNet_v2/make_manifest.py   (seeded 42, remote central
        │  directories only — asserts 5,979 positives/country and
        │  fraud keys ⊆ positive keys, then key-disjoint sampling)
        │  per country: 500 genuine + 84/84/83/83/83/83 over fraud1-6
        │  (v2 amendment 2026-09-15; doc key = stem minus _fake_* suffix)
        ▼
manifest.csv (20,000 rows; committed row-hash over id,label,type)
        │  pipeline/IDNet_v2/fetch_images.py  (ranged zip extraction only —
        │  no archive downloads; existing files trusted only on a CRC match
        │  vs the origin zip; writes per-file md5 into the manifest)
        ▼
Images/<country>/<folder>/<file>.{png,jpg}  (20,000 files, ~10GB)
        │  pipeline/IDNet_v2/make_splits.py
        │  holdout = {esp, fin, rus, az, ca}  (EVALUATION.md §2.1 dedup rule
        │  + 2 arbitrary US states); rest 80:20 stratified per
        │  (country, folder), seed 42
        ▼
   ┌────┴───────────────────┬───────────────────────┐
split_train.csv         split_val.csv         holdout_test.csv
  (11,970 rows)           (3,030 rows)           (5,000 rows)
  15 countries × 798      15 countries × 202     5 countries × 1,000
```

Notes:
1. **Leakage grain = document key** (filename stem): a fraud image is a
   manipulated copy of the positive with the same stem, so key-disjoint
   sampling means each document contributes exactly one image — no genuine /
   fraud near-duplicates can straddle any split line, by construction.
2. **Known caveat:** the same generated.photos face pool recurs across all
   countries, so faces in the 5 holdout countries also appear on
   train-country documents (weakest spot: fraud2 morphs). Accepted —
   templates/documents differ entirely; see the plan doc.
3. `type = idnet_<country>` to avoid colliding with SIDT's country types.
4. Rebuild without network: `make_splits.py` runs from the committed
   manifest. Full rebuild (network + AWS creds): `make_manifest.py` must
   reproduce the committed row-hash, then `fetch_images.py`.

## Combined

```
           in-distribution split          held-out unseen types
           split_train / split_val        (test partition)
   SIDT      6,540 / 1,700                   367   (1 frame/doc)
FantasyID    2,084 / 534                     666
  FREUID    42,788 / 10,697               15,867
   IDNet    11,970 / 3,030                 5,000
              │        │                     │
              │   pipeline/combine_datasets.py
              │   per source: rewrite image_path to a
              │   repo-root-relative real path, then concat
              ▼        ▼                     ▼
        train.csv    val.csv            test.csv
        (63,382)     (15,961)           (21,900)      → splits/
        schema: id, image_path, label, type, source
```

rows by source — train+val (in-distribution): freuid 53,485 · idnet 15,000 ·
sidt 8,240 · fantasyid 2,618 | test (unseen types): freuid 15,867 · idnet
5,000 · fantasyid 666 · sidt 367 (SIDT test is 1 frame/doc; train/val is N=10)

`combine_datasets.py` builds all three outputs: `train.csv` / `val.csv` from each
dataset's `split_train.csv` / `split_val.csv`, and `test.csv` from each dataset's
held-out unseen-type partition (SIDT `esp_rus_fin`, FantasyID `portugal_turkiye`,
FREUID `EGYPT/DL`). `INPUTS` maps the per-source input file for each output.

Path rewrite is the point: each dataset's `image_path` is relative to its own
dir with a different convention (FREUID's logical `train/` is really
`FREUID_train/`). The loader joins a **single** `base_path` with `image_path`
(`utils/load_tdata.py`), so every path is rewritten to be relative to `data/` and
real (e.g. `sources/FantasyID/train/bonafide/huawei/arabic-003_03.jpg`) —
the combined CSVs then load with `base_path` = `data/`. Two asserts guard the
result: `id` is unique across all four sources (no collision), and every
`image_path` exists on disk.

FREUID dominates (~68% of rows) and the combined set skews genuine-heavy; any
source/class rebalancing is left to the training loader, not this step.
