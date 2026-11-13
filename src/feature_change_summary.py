"""Read results/feature_change.csv against the pre-registered predictions (program 3):
  P1  LP-FT moves features less than FT, on each backbone, on every set.
  P2  Under FT (the no-LP-FT arms), in-dist features move more than out-of-distribution ones.
Bootstrap CIs over images (1000 draws); per-source breakdown for the two labelled sets.
  python src/feature_change_summary.py [--metric rel_l2|cos|pool_l2]
"""
import argparse, os, numpy as np, pandas as pd
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
ap = argparse.ArgumentParser(); ap.add_argument("--metric", default="rel_l2"); a = ap.parse_args()
df = pd.read_csv(os.path.join(ROOT, "results", "feature_change.csv")); m = a.metric
rng = np.random.default_rng(0)
def ci(v, n=1000):
    v = np.asarray(v); b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return v.mean(), np.quantile(b, 0.025), np.quantile(b, 0.975)
print(f"metric = {m}   (mean [95% CI] over images)\n")
tab = df.groupby(["tag", "set"])[m].apply(lambda v: "%.4f [%.4f, %.4f]" % ci(v)).unstack()
print(tab[["indist", "genval", "private"]].to_string())
PAIRS = [("own_v2_B_s42", "own_v2_B_lpft_s42", "ViT-B"), ("own_v2_ctrl_s42", "own_v2_L_lpft_s42", "ViT-L")]
print("\nP1  LP-FT vs FT, same images (paired difference FT - LP-FT; > 0 means LP-FT moved less):")
for ft, lp, name in PAIRS:
    for s in ["indist", "genval", "private"]:
        x = df[(df.tag == ft) & (df.set == s)].set_index("id")[m]; y = df[(df.tag == lp) & (df.set == s)].set_index("id")[m]
        d = (x - y.loc[x.index]).values; mu, lo, hi = ci(d)
        print(f"  {name:6} {s:8} FT {x.mean():.4f}  LP-FT {y.mean():.4f}  diff {mu:+.4f} [{lo:+.4f}, {hi:+.4f}]  ratio LP-FT/FT {y.mean()/x.mean():.2f}")
print("\nP2  FT arms: in-dist vs OOD (unpaired; CI on each):")
for ft, _, name in PAIRS:
    for s in ["genval", "private"]:
        i = df[(df.tag == ft) & (df.set == "indist")][m]; o = df[(df.tag == ft) & (df.set == s)][m]
        print(f"  {name:6} indist {i.mean():.4f} vs {s:8} {o.mean():.4f}  ratio OOD/ID {o.mean()/i.mean():.2f}")
print("\nper source (labelled sets), mean:")
print(df[df.set != "private"].pivot_table(index=["set", "source"], columns="tag", values=m, aggfunc="mean").round(4).to_string())

# ---- descriptive (not pre-registered): movement vs the arm's own score at the fixed readout ----
# ponytail: scores joined from the training oof files / fixed-readout submission csv; skipped if a file is missing.
DATA = os.path.join(ROOT, "..", "..", "id-fraud-detection", "dataset_final")
try:
    gen_ids = pd.read_csv(os.path.join(DATA, "test.csv")).id
    rows = []
    for tag in df.tag.unique():
        sc = {"genval": pd.Series(np.load(os.path.join(ROOT, f"oof/genvalpred_{tag}_dataset_final_ep4.npy")), index=gen_ids)}
        vid = np.load(os.path.join(ROOT, f"oof/valid_{tag}_dataset_final.npy"), allow_pickle=True)
        sc["indist"] = pd.Series(np.load(os.path.join(ROOT, f"oof/valpred_{tag}_dataset_final.npy")), index=[str(x) for x in vid])
        sub = pd.read_csv(os.path.join(ROOT, f"submissions/sub_{tag}_ep4_f0.1_w0.5_s1.csv")); sc["private"] = pd.Series(sub.label.values, index=sub.id)
        for s in ["indist", "genval", "private"]:
            d = df[(df.tag == tag) & (df.set == s)].copy(); d["score"] = sc[s].reindex(d.id).values
            assert d.score.notna().all(), (tag, s)
            rows.append((tag, s, "corr(score, movement)", np.corrcoef(d.score, d[m])[0, 1]))
            if s == "private":
                rows.append((tag, s, "lowest score decile", d[d.score <= d.score.quantile(0.1)][m].mean()))
                rows.append((tag, s, "top score decile", d[d.score >= d.score.quantile(0.9)][m].mean()))
            else:
                g = d[(d.label == 0) & (d.source == "freuid")]; q = g.score.quantile(0.9)
                rows.append((tag, s, "FREUID genuine, top score decile", g[g.score >= q][m].mean()))
                rows.append((tag, s, "FREUID genuine, rest", g[g.score < q][m].mean()))
                rows.append((tag, s, "FREUID attack", d[(d.label == 1) & (d.source == "freuid")][m].mean()))
    print("\ndescriptive (not pre-registered): movement vs the arm's own fixed-readout score; deciles are mean", m)
    print(pd.DataFrame(rows, columns=["tag", "set", "group", m]).pivot_table(index=["set", "group"], columns="tag", values=m).round(3).to_string())
except FileNotFoundError as e:
    print("\n(descriptive section skipped:", e, ")")
