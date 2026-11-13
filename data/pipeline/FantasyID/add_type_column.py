"""Add a `type` column to a FantasyID csv, inferred from the filename.

type = the filename prefix before the first "-", e.g.
"train/bonafide/huawei/arabic-003_03.jpg" -> "arabic". Reads raw, writes a
new file; never mutates the input.
"""

import argparse
from pathlib import PurePosixPath

import pandas as pd

DEFAULT_CSV = "sources/FantasyID/combined.csv"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=DEFAULT_CSV, help="input csv (read-only)")
    p.add_argument("--out", help="output csv (default: <csv stem>_typed.csv)")
    args = p.parse_args()

    out = args.out or args.csv.replace(".csv", "_typed.csv")

    df = pd.read_csv(args.csv)
    df["type"] = df["path"].map(lambda p: PurePosixPath(p).name.split("-")[0])
    df.to_csv(out, index=False)
    print(f"added type column ({df['type'].nunique()} types) -> {out}")
    print(df["type"].value_counts())


if __name__ == "__main__":
    main()
