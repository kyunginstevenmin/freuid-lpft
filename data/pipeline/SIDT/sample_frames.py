"""Sample N frames per unique doc from a SIDT images csv.

id column looks like "{doc}_frame_{n}_{template|dewarped}", so the doc is
everything before "_frame_". Docs with fewer than N frames keep all of them.

Canonical invocations (run from repo root):
  --csv .../pool_allframes.csv    --n 10 --out .../pool_trainval.csv
  --csv .../holdout_allframes.csv --n 10 --out .../holdout_frames10.csv
  --csv .../holdout_frames10.csv  --n 1  --out .../holdout_test.csv
Test is sampled 10-then-1 (not 1 directly from all frames): that two-stage
order is the historical provenance of the committed test set — changing it
picks different frames and silently changes splits/test.csv. 1
frame/doc keeps near-duplicate frames from skewing per-doc test metrics.
"""

import argparse

import pandas as pd

DEFAULT_CSV = "sources/SIDT_clips_cropped/pool_allframes.csv"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default=DEFAULT_CSV, help="input images.csv")
    p.add_argument("--n", type=int, default=5, help="frames per doc")
    p.add_argument("--out", help="output csv (default: <csv stem>_sample_<n>.csv)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out = args.out or args.csv.replace(".csv", f"_sample_{args.n}.csv")

    df = pd.read_csv(args.csv)
    doc = df["id"].str.replace(r"_frame_\d+_(?:template|dewarped)$", "", regex=True)
    sampled = (
        df.groupby(doc, group_keys=False)
        .apply(lambda g: g.sample(min(args.n, len(g)), random_state=args.seed))
        .reset_index(drop=True)
    )
    sampled.to_csv(out, index=False)
    print(f"{len(sampled)} rows from {doc.nunique()} docs -> {out}")


if __name__ == "__main__":
    main()
