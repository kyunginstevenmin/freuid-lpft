"""Readout study on cached patch logits (LEDGER.md, pre-registered 2026-10-09). CPU only.

Input: preds/<sha>_<protocol>_patches_s<scale>.npz from `evaluate.py --readout patches`
(per-image per-patch forgery logits + attention weights, fp16, one file per DEPLOY scale).

1. Aggregation sweep at scale 1.0: image logit = w * mean(top-k patch logits) +
   (1-w) * sum(attn * patch logits), over the pre-registered grid FRACS x WEIGHTS.
   The training-time readout (frac 0.1, w 0.5 -- PatchHead.forward) is one grid point, so
   "raw vs best aggregation" is read off the grid. Selection = lowest gen-val source-macro.
   The whole grid is written (winner's-curse check: flat surface => knobs barely matter).
2. 2x2 ablation: {raw knobs, swept knobs} x {scale 1.0, 4-scale logit-avg TTA}, each with the
   type-stratified bootstrap CI. Scales fixed at the winner's DEPLOY_SCALES.
Outputs: results/readout_sweep.csv (grid rows), results/readout_ablation.csv (2x2 rows), and
a preds/<sha>_<protocol>_<cell>.csv score cache per cell so slicing reuses them.

  python src/readout_sweep.py --sha 913c44823e78 [--protocol genval] [--boot 1000]
"""
from __future__ import annotations
import argparse, os, sys, time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT
from evaluate import (PROTOCOLS, DEPLOY_SCALES, PREDS_DIR, patches_path, cache_path,
                      macro_scores, bootstrap_macro_ci)

FRACS = (0.02, 0.05, 0.1, 0.2)            # top-k fraction of patches
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)     # weight on the top-k term; 0 / 1 = the pure aggregators
RAW = (0.1, 0.5)                          # PatchHead.forward: img = 0.5 * (topk(10%) + attn-weighted)
SWEEP_CSV = os.path.join(ROOT, "results", "readout_sweep.csv")
ABL_CSV = os.path.join(ROOT, "results", "readout_ablation.csv")


def aggregate(plog: np.ndarray, attn: np.ndarray, frac: float, w: float) -> np.ndarray:
    """Image logit per row from (n_img, n_patch) logits + attention weights. Mirrors
    PatchHead.forward (k = max(1, int(frac * N)); mean of the k largest) and evaluate.py's
    deploy branch exactly, so frac=0.1,w=0.5 is the training readout and 0.05,0.25 the winner's."""
    plog = plog.astype(np.float32); attn = attn.astype(np.float32)
    n = plog.shape[1]; k = max(1, int(frac * n))
    topk = np.partition(plog, n - k, axis=1)[:, n - k:].mean(1)
    return w * topk + (1 - w) * (plog * attn).sum(1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def _rel(paths) -> np.ndarray:
    """Paths relative to the id-fraud-detection root: the dump was written on the box
    (/home/ubuntu/...), the sweep runs on the laptop; only the repo-relative part must agree."""
    return np.array([str(p).split("/id-fraud-detection/")[-1] for p in paths])


def load_scales(sha: str, protocol: str, paths: np.ndarray) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for s in DEPLOY_SCALES:
        z = np.load(patches_path(sha, protocol, s), allow_pickle=True)
        assert (_rel(z["path"]) == _rel(paths)).all(), f"patch dump s{s} does not match the protocol dataframe"
        out[s] = (z["plog"], z["attn"])
    return out


def scores_for(scales, frac, w, tta: bool) -> np.ndarray:
    """Probability per image: single scale 1.0, or logit-average over DEPLOY_SCALES (as deploy does)."""
    use = DEPLOY_SCALES if tta else (1.0,)
    lg = np.mean([aggregate(*scales[s], frac, w) for s in use], axis=0)
    return sigmoid(lg).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", required=True, help="12-hex ckpt sha (evaluate.py prints it)")
    ap.add_argument("--protocol", default="genval")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    df = PROTOCOLS[a.protocol]()
    y, ty, src = df.label.values.astype(np.float32), df.type.values, df.source.values
    scales = load_scales(a.sha, a.protocol, df.path.values.astype(str))
    date = time.strftime("%Y-%m-%d")

    # 1. sweep at scale 1.0
    rows = []
    for frac in FRACS:
        for w in WEIGHTS:
            _, per_src, mt, ms = macro_scores(y, scores_for(scales, frac, w, tta=False), ty, src)
            rows.append({"date": date, "sha": a.sha, "protocol": a.protocol, "frac": frac, "w": w,
                         "macro_source": ms, "macro_type": mt, **{f"src/{k}": v for k, v in per_src.items()}})
    grid = pd.DataFrame(rows)
    best = grid.loc[grid.macro_source.idxmin()]
    pd.set_option("display.width", 200)
    print("aggregation sweep (scale 1.0), gen-val source-macro:")
    print(grid.pivot(index="frac", columns="w", values="macro_source").round(4).to_string())
    print(f"raw point (frac {RAW[0]}, w {RAW[1]}): "
          f"{grid[(grid.frac == RAW[0]) & (grid.w == RAW[1])].macro_source.iloc[0]:.4f}; "
          f"best: frac {best.frac}, w {best.w} -> {best.macro_source:.4f}")
    grid.to_csv(SWEEP_CSV, mode="a", header=not os.path.exists(SWEEP_CSV), index=False)

    # 2. 2x2 at the raw and swept knobs
    cells = {"raw": (RAW, False), "sweep": ((best.frac, best.w), False),
             "raw_tta": (RAW, True), "sweep_tta": ((best.frac, best.w), True)}
    out = []
    for name, ((frac, w), tta) in cells.items():
        sc = scores_for(scales, frac, w, tta)
        _, per_src, mt, ms = macro_scores(y, sc, ty, src)
        lo, hi = bootstrap_macro_ci(y, sc, ty, src, n_boot=a.boot) if a.boot else ((None, None), (None, None))
        out.append({"date": date, "sha": a.sha, "protocol": a.protocol, "cell": name, "frac": frac, "w": w,
                    "tta": tta, "macro_source": ms, "ci_lo": lo[1], "ci_hi": hi[1], "macro_type": mt,
                    "n_boot": a.boot, **{f"src/{k}": v for k, v in per_src.items()}})
        pd.DataFrame({"path": df.path.values, "label": df.label.values, "score": sc}).to_csv(
            cache_path(a.sha, a.protocol, name), index=False)
    abl = pd.DataFrame(out)
    print("\n2x2 readout ablation (source-macro [95% CI]):")
    print(abl[["cell", "frac", "w", "tta", "macro_source", "ci_lo", "ci_hi"] + [c for c in abl if c.startswith("src/")]]
          .round(4).to_string(index=False))
    abl.to_csv(ABL_CSV, mode="a", header=not os.path.exists(ABL_CSV), index=False)
    print(f"\nrows appended -> {os.path.relpath(SWEEP_CSV, ROOT)}, {os.path.relpath(ABL_CSV, ROOT)}; "
          f"per-cell score caches in preds/")


if __name__ == "__main__":
    main()
