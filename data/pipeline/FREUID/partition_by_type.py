"""Split the unified FREUID csv into two by holdout type.

Writes pool_trainval.csv (everything else) and holdout_test.csv (just the
holdout type, see HOLDOUT), both already in the unified schema. EGYPT/DL is
the unseen-type test split. Replaces the ad-hoc 2026-07-08 partition run
recorded in sources/FREUID/split_manifest.json. Never mutates input.
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CSV = "sources/FREUID/train_labels_unified.csv"
HOLDOUT = {"EGYPT/DL"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=DEFAULT_CSV, help="unified csv (read-only)")
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
