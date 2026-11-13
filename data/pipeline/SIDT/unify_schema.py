"""Unify schema of the SIDT typed csv to id,image_path,label,type,source.

Renames path -> image_path, is_attack -> label, converts label from
True/False to 1/0, and adds a source="sidt" column. Reads raw, writes a new
file; never mutates the inputs.
"""

import argparse

import pandas as pd

CSVS = ["sources/SIDT_clips_cropped/images_typed.csv"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csvs", nargs="+", default=CSVS)
    args = p.parse_args()

    for csv in args.csvs:
        df = pd.read_csv(csv)
        df = df.rename(columns={"path": "image_path", "is_attack": "label"})
        df["label"] = df["label"].astype(int)
        df["source"] = "sidt"
        df = df[["id", "image_path", "label", "type", "source"]]

        out = csv.replace(".csv", "_unified.csv")
        df.to_csv(out, index=False)
        print(f"{len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
