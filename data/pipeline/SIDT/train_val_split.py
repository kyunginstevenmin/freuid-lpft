"""Split the SIDT typed CSV into train/val without leaking documents.

Each fake is a manipulation of a base real document, so the split grain is the
base document `{type}_{docnum}` (e.g. alb_id_14) — all frames and fake variants
of one identity stay on the same side. See notebooks/SIDT_train_val_split.ipynb
for how the identifier was chosen.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

DEFAULT_CSV = "sources/SIDT_clips_cropped/pool_trainval.csv"


def base_doc(row):
    m = re.match(re.escape(row["type"]) + r"_(\d+)", row["id"])
    return f"{row['type']}_{m.group(1)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    csv = Path(args.csv)
    df = pd.read_csv(csv)
    df["doc"] = df.apply(base_doc, axis=1)
    assert df["doc"].notna().all(), "some ids did not parse"

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
