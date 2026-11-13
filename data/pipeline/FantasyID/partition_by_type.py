"""Split the unified FantasyID csv into two by holdout type.

Writes pool_trainval.csv (everything else) and holdout_test.csv (just the
holdout types, see HOLDOUT), both already in the unified schema. The holdout
types become the unseen-type test split. Never mutates input.
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CSV = "sources/FantasyID/combined_typed_unified.csv"
HOLDOUT = {"portugal", "turkiye"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=DEFAULT_CSV, help="typed csv (read-only)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    is_holdout = df["type"].isin(HOLDOUT)

    for name, part in [("pool_trainval.csv", df[~is_holdout]),
                       ("holdout_test.csv", df[is_holdout])]:
        out = Path(args.csv).with_name(name)
        part.to_csv(out, index=False)
        print(f"{len(part)} rows -> {out}")


if __name__ == "__main__":
    main()
