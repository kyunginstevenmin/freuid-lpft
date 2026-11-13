"""Split the FantasyID typed CSV into train/val without leaking documents.

The same physical document appears ~9x — across devices (huawei/iphone/scan),
bonafide vs attack, and the original train/test dirs — sharing nearly all its
pixels. So the split grain is the image filename stem (e.g. arabic-003_03): all
variants of one document stay on the same side. See
notebooks/FantasyID_train_val_split.ipynb for how the identifier was chosen.
"""
import argparse
from pathlib import Path

import pandas as pd

DEFAULT_CSV = "sources/FantasyID/pool_trainval.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    csv = Path(args.csv)
    df = pd.read_csv(csv)
    df["doc"] = df["image_path"].map(lambda x: Path(x).stem)

    # split documents (not rows), stratified per type
    doc_type = df[["doc", "type"]].drop_duplicates().set_index("doc")["type"]
    val_docs = set(
        doc_type.groupby(doc_type)
        .apply(lambda s: s.sample(frac=args.val_frac, random_state=args.seed))
        .index.get_level_values("doc")
    )
    df["split"] = df["doc"].isin(val_docs).map({True: "val", False: "train"})
    assert (df.groupby("doc")["split"].nunique() == 1).all(), "doc straddles split"

    for name in ("train", "val"):
        out = csv.with_name(f"split_{name}.csv")
        sub = df[df["split"] == name].drop(columns=["doc", "split"])
        sub.to_csv(out, index=False)
        print(f"wrote {out}  ({len(sub)} rows, {sub['label'].value_counts().to_dict()})")


if __name__ == "__main__":
    main()
