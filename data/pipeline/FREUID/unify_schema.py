"""Unify schema of the FREUID label csv, dropping is_digital and adding source.

Reads sources/FREUID/train_labels.csv, drops the is_digital column, and
adds a source="freuid" column. Writes a new file; never mutates the input.
"""

import argparse

import pandas as pd

CSVS = ["sources/FREUID/train_labels.csv"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csvs", nargs="+", default=CSVS)
    args = p.parse_args()

    for csv in args.csvs:
        df = pd.read_csv(csv)
        df = df.drop(columns="is_digital")
        df["source"] = "freuid"

        out = csv.replace(".csv", "_unified.csv")
        df.to_csv(out, index=False)
        print(f"{len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
