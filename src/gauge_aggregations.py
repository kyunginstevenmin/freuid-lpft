"""Re-read the gen-val predictions of frozen arms under different aggregations of the FREUID metric
(gauge-weighting hypothesis, WRITEUP-LPFT.md §5.4 / §7). CPU; inputs = oof/genval_<tag>_<vname>.csv +
oof/genvalpred_<tag>_<vname>_ep<ep>.npy (the training readout at the frozen epoch).

Aggregations (each with a type-stratified bootstrap CI over images):
  macro         per-type metric -> mean within source -> mean across sources   (the selection gauge)
  type_imgw     per-type metric, image-weighted mean                             (per-type thresholds, no source weighting)
  source_pooled one threshold per source -> mean across sources                  (source weighting, no per-type thresholds)
  pooled        one threshold over all images                                    (what the leaderboard does)
  pooled_ext    one threshold over the three external sources (no FREUID)        (is 'pooled' just FREUID's 72 %?)

  python src/gauge_aggregations.py --tags own_v2_ctrl_s42 own_v2_L_lpft_s42 own_v2_B_s42 own_v2_B_lpft_s42
"""
import argparse, os, sys
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
from freuid_metric import freuid_score

AGGS = ("macro", "type_imgw", "source_pooled", "pooled", "pooled_ext")


def score(y, s, types, sources, agg):
    df = pd.DataFrame({"y": y, "s": s, "t": types, "src": sources})
    if agg == "pooled":
        return freuid_score(df.y.values, df.s.values)[0]
    if agg == "pooled_ext":
        e = df[df.src != "freuid"]; return freuid_score(e.y.values, e.s.values)[0]
    if agg == "source_pooled":
        return float(np.mean([freuid_score(g.y.values, g.s.values)[0] for _, g in df.groupby("src")]))
    pt = {t: (freuid_score(g.y.values, g.s.values)[0], len(g)) for t, g in df.groupby("t")}
    if agg == "type_imgw":
        return float(sum(v * n for v, n in pt.values()) / sum(n for _, n in pt.values()))
    ps = {src: np.mean([pt[t][0] for t in g.t.unique()]) for src, g in df.groupby("src")}
    return float(np.mean(list(ps.values())))


def boot(y, s, types, sources, agg, n_boot, seed=0):
    groups = [np.flatnonzero(types == t) for t in np.unique(types)]
    rng = np.random.default_rng(seed); out = []
    while len(out) < n_boot:
        idx = np.concatenate([g[rng.integers(0, len(g), len(g))] for g in groups])
        try: out.append(score(y[idx], s[idx], types[idx], sources[idx], agg))
        except AssertionError: continue
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--ep", type=int, default=4)
    ap.add_argument("--vname", default="dataset_final"); ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "gauge_aggregations.csv"))
    a = ap.parse_args()
    rows = []
    for tag in a.tags:
        g = pd.read_csv(os.path.join(ROOT, "oof", f"genval_{tag}_{a.vname}.csv"))
        s = np.load(os.path.join(ROOT, "oof", f"genvalpred_{tag}_{a.vname}_ep{a.ep}.npy"))
        y, t, src = g.label.values, g.type.values, g.source.values
        for agg in AGGS:
            v = score(y, s, t, src, agg); lo, hi = boot(y, s, t, src, agg, a.boot)
            rows.append({"tag": tag, "ep": a.ep, "agg": agg, "score": v, "ci_lo": lo, "ci_hi": hi, "n_boot": a.boot})
            print(f"{tag:22} {agg:14} {v:.4f} [{lo:.4f}, {hi:.4f}]", flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False); print(f"wrote {os.path.relpath(a.out, ROOT)}")


if __name__ == "__main__":
    main()
