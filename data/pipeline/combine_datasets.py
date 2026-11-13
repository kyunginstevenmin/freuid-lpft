"""Concatenate per-dataset splits into combined train.csv / val.csv / test.csv.

Each dataset's `image_path` is relative to its own dir and the conventions
differ (FREUID's logical `train/` is really `FREUID_train/`). The loader joins a
single base_path with image_path (utils/load_tdata.py), so here every path is
rewritten to be repo-root-relative and real — the combined CSVs load with
base_path = repo root. A sample-existence assert guards the remap.

train/val come from each dataset's in-distribution split_{train,val}.csv; test
comes from each dataset's held-out unseen-type partition.
"""
import os

import pandas as pd

# source dir -> function mapping a split image_path to a repo-root-relative real path
REMAP = {
    "SIDT_clips_cropped": lambda p: f"sources/SIDT_clips_cropped/{p}",
    "FantasyID":          lambda p: f"sources/FantasyID/{p}",
    "FREUID":             lambda p: "sources/FREUID/FREUID_train/" + p[len("train/"):],
    "IDNet_v2":           lambda p: f"sources/IDNet_v2/{p}",
}

# source dir -> input filename per output split
INPUTS = {
    "train": {d: "split_train.csv" for d in REMAP},
    "val":   {d: "split_val.csv" for d in REMAP},
    "test": {
        # SIDT test uses 1 frame/doc (sample_frames.py --n 1) to avoid near-
        # duplicate frames inflating/skewing per-doc test metrics; train/val N=10
        "SIDT_clips_cropped": "holdout_test.csv",
        "FantasyID":          "holdout_test.csv",
        "FREUID":             "holdout_test.csv",
        "IDNet_v2":           "holdout_test.csv",
    },
}
OUT_DIR = "splits"


def combine(split):
    parts = []
    for d, fname in INPUTS[split].items():
        df = pd.read_csv(f"sources/{d}/{fname}")
        df["image_path"] = df["image_path"].map(REMAP[d])
        parts.append(df)
    out = pd.concat(parts, ignore_index=True)

    assert out["id"].is_unique, "id collision across datasets"
    assert out["image_path"].map(os.path.exists).all(), "some image_path does not resolve"

    path = os.path.join(OUT_DIR, f"{split}.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}  ({len(out)} rows)")
    print(out.groupby(["source", "label"]).size().unstack(fill_value=0).to_string())
    return out


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        combine(split)
