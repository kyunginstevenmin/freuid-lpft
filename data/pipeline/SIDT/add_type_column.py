"""Add a `type` column to a SIDT images.csv, inferred from the id.

type = first two tokens of the id, e.g. "alb_id_14_..." -> "alb_id",
"rus_internalpassport_44_..." -> "rus_internalpassport". Reads raw, writes a
new file; never mutates the input.
"""

import argparse

import pandas as pd

DEFAULT_CSV = "sources/SIDT_clips_cropped/images.csv"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=DEFAULT_CSV, help="raw images.csv (read-only)")
    p.add_argument("--out", help="output csv (default: <csv stem>_typed.csv)")
    args = p.parse_args()

    out = args.out or args.csv.replace(".csv", "_typed.csv")

    df = pd.read_csv(args.csv)
    df["type"] = df["id"].str.split("_").str[:2].str.join("_")
    df.to_csv(out, index=False)
    print(f"added type column ({df['type'].nunique()} types) -> {out}")
    print(df["type"].value_counts())


if __name__ == "__main__":
    main()
