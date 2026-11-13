"""Unify schema of the typed FantasyID csv to id,image_path,label,type,source.

Sets id from a slugified path (strip the extension, join path parts with "_"
-> e.g. "train_bonafide_huawei_arabic-003_03"), renames path -> image_path and
is_attack -> label, converts label from True/False to 1/0, drops attack_type,
and adds a source="fantasyid" column. The split-dir prefix (train/test) is
kept in the id so combined train+test rows can't collide. Reads raw, writes a
new file; never mutates the inputs.
"""

import argparse
from pathlib import PurePosixPath

import pandas as pd

CSVS = ["sources/FantasyID/combined_typed.csv"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csvs", nargs="+", default=CSVS)
    args = p.parse_args()

    for csv in args.csvs:
        df = pd.read_csv(csv)
        df["id"] = df["path"].map(
            lambda p: PurePosixPath(p).with_suffix("").as_posix().replace("/", "_")
        )
        df = df.rename(columns={"path": "image_path", "is_attack": "label"})
        df["label"] = df["label"].astype(int)
        df["source"] = "fantasyid"
        df = df[["id", "image_path", "label", "type", "source"]]

        out = csv.replace(".csv", "_unified.csv")
        df.to_csv(out, index=False)
        print(f"{len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
