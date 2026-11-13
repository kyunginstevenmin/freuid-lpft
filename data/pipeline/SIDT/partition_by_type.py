"""Split the unified SIDT csv into two by holdout type.

Writes pool_allframes.csv (everything else) and holdout_allframes.csv (just
the holdout types, see HOLDOUT); both keep every frame — sample_frames.py
then thins each to its per-doc frame budget. Never mutates input.
"""

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CSV = "sources/SIDT_clips_cropped/images_typed_unified.csv"
HOLDOUT = {"esp_id", "rus_internalpassport", "fin_id"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=DEFAULT_CSV, help="typed images csv (read-only)")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    is_holdout = df["type"].isin(HOLDOUT)

    for name, part in [("pool_allframes.csv", df[~is_holdout]),
                       ("holdout_allframes.csv", df[is_holdout])]:
        out = Path(args.csv).with_name(name)
        part.to_csv(out, index=False)
        print(f"{len(part)} rows -> {out}")


if __name__ == "__main__":
    main()
