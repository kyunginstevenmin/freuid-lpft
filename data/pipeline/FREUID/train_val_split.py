"""Split the FREUID unified CSV into train/val.

Every id is a unique document (one image per id, no frames or variants), so
there's no leakage grain to group on — rows are split directly, stratified per
type to keep each type's train/val ratio.
"""
import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CSV = "sources/FREUID/pool_trainval.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    csv = Path(args.csv)
    df = pd.read_csv(csv)
    assert df["id"].is_unique, "ids are not unique — a leakage grain may exist"

    val_idx = (
        df.groupby("type")
        .apply(lambda g: g.sample(frac=args.val_frac, random_state=args.seed), include_groups=False)
        .index.get_level_values(1)
    )
    df["split"] = "train"
    df.loc[val_idx, "split"] = "val"

    for name in ("train", "val"):
        out = csv.with_name(f"split_{name}.csv")
        sub = df[df["split"] == name].drop(columns="split")
        sub.to_csv(out, index=False)
        print(f"wrote {out}  ({len(sub)} rows, {sub['label'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
