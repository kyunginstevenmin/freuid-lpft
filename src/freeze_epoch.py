"""Epoch freeze (EVALUATION.md §3.1) from a --split_dir run's per-epoch gen-val dumps.

For every epoch: source-macro + type-stratified bootstrap CI (evaluate.bootstrap_macro_ci).
Best = lowest source-macro. Tie rule: the FROZEN epoch is the EARLIEST epoch whose
source-macro lies inside the best epoch's tie band (less training = less template
memorization). Band (EVALUATION.md §3.1): --band ci95 (default) = the 95% bootstrap
interval; --band 1se = one bootstrap SE (the literal one-standard-error rule), the
option to switch to when the 95% band ties most epochs. Prints both picks and appends
the frozen row to results/epoch_freezes.csv (the committed record; the LEDGER.md row is
copied from it). --wandb additionally attaches a *reference* artifact (S3 URI + checksum,
no upload) for the frozen epoch's lean checkpoint to the run that trained it, so the run
record points at its weights of record. Only the frozen epoch is registered; per-epoch
checkpoints stay in S3 as scaffolding.

  python src/freeze_epoch.py --tag own_v2_ctrl_s42 --vname dataset_final [--boot 1000] [--band 1se] [--wandb]
"""
from __future__ import annotations
import argparse, glob, os, re, sys, time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import ROOT
from evaluate import macro_scores, bootstrap_macro_ci

OOF_DIR = os.path.join(ROOT, "oof")
RESULTS_CSV = os.path.join(ROOT, "results", "epoch_freezes.csv")
DEFAULT_BUCKET = "s3://<your-bucket>/freuid"   # own_run_ec2.sh ABL_BUCKET


def epoch_table(gen: pd.DataFrame, preds_by_epoch: dict[int, np.ndarray], n_boot=1000, seed=0):
    rows = []
    for ep in sorted(preds_by_epoch):
        p = preds_by_epoch[ep]
        _, per_source, mt, ms = macro_scores(gen.label.values, p, gen.type.values, gen.source.values)
        lo, hi = bootstrap_macro_ci(gen.label.values, p, gen.type.values, gen.source.values,
                                    n_boot=n_boot, seed=seed)
        rows.append({"epoch": ep, "macro_source": ms, "ci_lo": lo[1], "ci_hi": hi[1],
                     "macro_type": mt, **{f"src/{k}": v for k, v in per_source.items()}})
    return pd.DataFrame(rows)


def tie_threshold(best_row, band="ci95") -> float:
    """Upper edge of the tie band around the best epoch. 1se uses the normal approximation
    SE = (ci_hi - ci_lo) / (2 * 1.96) of the bootstrap distribution (ponytail: exact SE would
    need the raw bootstrap draws; the approximation is within a few % for these sizes)."""
    if band == "ci95":
        return float(best_row.ci_hi)
    if band == "1se":
        return float(best_row.macro_source + (best_row.ci_hi - best_row.ci_lo) / (2 * 1.96))
    raise ValueError(band)


def pick_frozen_epoch(table: pd.DataFrame, band="ci95") -> int:
    """Earliest epoch whose source-macro is <= the best epoch's tie threshold."""
    best = table.loc[table.macro_source.idxmin()]
    tied = table[table.macro_source <= tie_threshold(best, band)]
    return int(tied.epoch.min())


def freeze_row(table: pd.DataFrame, tag: str, vname: str, band: str, n_boot: int, bucket: str) -> dict:
    """The committed record of one freeze: pick, band, tie counts, the frozen epoch's numbers,
    and the S3 URI of its lean checkpoint (what --wandb registers)."""
    fz = pick_frozen_epoch(table, band)
    r = table[table.epoch == fz].iloc[0].to_dict()
    best = table.loc[table.macro_source.idxmin()]
    return {"date": time.strftime("%Y-%m-%d"), "tag": tag, "vname": vname, "band": band,
            "frozen_epoch": fz, "best_epoch": int(best.epoch),
            "n_tied_ci95": int((table.macro_source <= tie_threshold(best, "ci95")).sum()),
            "n_tied_1se": int((table.macro_source <= tie_threshold(best, "1se")).sum()),
            "n_epochs": len(table), "n_boot": n_boot,
            **{k: v for k, v in r.items() if k != "epoch"},
            "lean_ckpt": f"{bucket}/checkpoints/{tag}_{vname}_ep{fz}_lean.pt"}


def append_row(row: dict, path=RESULTS_CSV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def log_reference_artifact(row: dict, project: str):
    """Attach the frozen lean checkpoint as a reference artifact to the training run
    (resumed by id, so lineage is run -> weights). Needs wandb + boto3 + AWS creds locally."""
    import wandb
    run = wandb.init(project=project, id=f"{row['tag']}_{row['vname']}", resume="must")
    art = wandb.Artifact(f"{row['tag']}_frozen", type="model",
                         metadata={k: row[k] for k in ("frozen_epoch", "band", "macro_source", "ci_lo", "ci_hi", "n_boot")})
    art.add_reference(row["lean_ckpt"])
    run.log_artifact(art, aliases=["frozen"])
    run.summary.update({"frozen/epoch": row["frozen_epoch"], "frozen/macro_source": row["macro_source"],
                        "frozen/lean_ckpt": row["lean_ckpt"]})
    run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True); ap.add_argument("--vname", default="dataset_final")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--band", default="ci95", choices=["ci95", "1se"])
    ap.add_argument("--bucket", default=DEFAULT_BUCKET)
    ap.add_argument("--wandb", action="store_true", help="register the frozen lean ckpt on the W&B run")
    ap.add_argument("--wandb_project", default="freuid-own")
    ap.add_argument("--no_write", action="store_true", help="skip the CSV row (re-reads, what-ifs); --wandb still logs")
    a = ap.parse_args()
    gen = pd.read_csv(os.path.join(OOF_DIR, f"genval_{a.tag}_{a.vname}.csv"))
    files = glob.glob(os.path.join(OOF_DIR, f"genvalpred_{a.tag}_{a.vname}_ep*.npy"))
    preds = {int(re.search(r"_ep(\d+)\.npy$", f).group(1)): np.load(f) for f in files}
    assert preds, "no per-epoch gen-val dumps found"
    t = epoch_table(gen, preds, n_boot=a.boot)
    pd.set_option("display.width", 200)
    print(t.round(4).to_string(index=False))
    best_ep = int(t.loc[t.macro_source.idxmin(), "epoch"])
    n_tied = {b: int((t.macro_source <= tie_threshold(t.loc[t.macro_source.idxmin()], b)).sum()) for b in ("ci95", "1se")}
    print(f"\nbest epoch {best_ep}; tied under ci95: {n_tied['ci95']}/{len(t)} epochs, under 1se: {n_tied['1se']}/{len(t)}")
    print(f"FROZEN epoch under --band {a.band}: {pick_frozen_epoch(t, a.band)}"
          f"   (ci95 -> {pick_frozen_epoch(t, 'ci95')}, 1se -> {pick_frozen_epoch(t, '1se')})")
    row = freeze_row(t, a.tag, a.vname, a.band, a.boot, a.bucket)
    if not a.no_write:
        append_row(row)
        print(f"row appended to {os.path.relpath(RESULTS_CSV, ROOT)}; lean ckpt = {row['lean_ckpt']}")
    if a.wandb:
        log_reference_artifact(row, a.wandb_project)
        print(f"reference artifact {a.tag}_frozen:frozen logged to run {a.tag}_{a.vname}")


if __name__ == "__main__":
    main()
