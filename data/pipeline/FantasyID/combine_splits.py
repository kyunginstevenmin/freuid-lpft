"""Combine the FantasyID train and test label csvs into one.

Concatenates sources/FantasyID/train.csv and test.csv (both schema
path,is_attack,attack_type) into a single combined.csv. The original
train/test directory split is discarded here on purpose; the test split is
redefined downstream by holdout document type. Reads raw, writes a new file;
never mutates the inputs.
"""

import argparse

import pandas as pd

TRAIN = "sources/FantasyID/train.csv"
TEST = "sources/FantasyID/test.csv"
DEFAULT_OUT = "sources/FantasyID/combined.csv"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", default=TRAIN)
    p.add_argument("--test", default=TEST)
    p.add_argument("--out", default=DEFAULT_OUT)
    args = p.parse_args()

    df = pd.concat(
        [pd.read_csv(args.train), pd.read_csv(args.test)],
        ignore_index=True,
    )
    df.to_csv(args.out, index=False)
    print(f"{len(df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
