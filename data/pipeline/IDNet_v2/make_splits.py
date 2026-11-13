"""IDNet split CSVs from the manifest (docs/idnet-integration-plan.md stage 3).

Test-pool countries (whole-type holdout) -> holdout_test.csv; the rest get a
seeded 80:20 split stratified per (country, folder), so each country
contributes 400/100 genuine and 100/25 per fraud type to train/val. Document
keys are sample-disjoint within a country by manifest construction, so a
row-level split respects the leakage grain.
"""
import argparse
from pathlib import Path

import pandas as pd

TEST_COUNTRIES = {"esp", "fin", "rus", "az", "ca"}  # EVALUATION.md §2.1/§2.3
OUT_DIR = Path("sources/IDNet_v2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="sources/IDNet_v2/manifest.csv")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest, dtype=str)
    df["image_path"] = (
        "Images/" + df["country"] + "/" + df["folder"] + "/"
        + df["member"].str.rsplit("/", n=1).str[1]
    )
    df["source"] = "idnet"
    cols = ["id", "image_path", "label", "type", "source"]

    is_test = df["country"].isin(TEST_COUNTRIES)
    trainval = df[~is_test]
    val_idx = (
        trainval.groupby(["country", "folder"], group_keys=False)
        .apply(lambda g: g.sample(frac=args.val_frac, random_state=args.seed))
        .index
    )
    splits = {
        "holdout_test": df[is_test],
        "split_val": trainval.loc[trainval.index.isin(val_idx)],
        "split_train": trainval.loc[~trainval.index.isin(val_idx)],
    }
    for name, sub in splits.items():
        out = OUT_DIR / f"{name}.csv"
        sub = sub.sort_values("id")
        sub[cols].to_csv(out, index=False)
        print(f"wrote {out}  ({len(sub)} rows, "
              f"{sub['label'].value_counts().to_dict()}, "
              f"{sub['country'].nunique()} countries)")

    assert not (set(splits["split_train"]["id"]) & set(splits["split_val"]["id"]))
    assert not df[df["country"].isin(TEST_COUNTRIES)]["id"].isin(
        pd.concat([splits["split_train"], splits["split_val"]])["id"]
    ).any()


if __name__ == "__main__":
    main()
