"""Feature change under fine-tuning (Kumar et al. 2022 §4.3, adapted): per-image distance between the
frozen DINOv2 backbone's patch tokens (LoRA at init, B = 0 -> exact pretrained features) and the trained
arm's patch tokens, on fixed samples (results/feature_change_samples.csv: 1,000 in-dist val, 1,000
gen-val, 1,000 private; seed 42). GPU. Pre-registered predictions in docs/LEDGER.md (program 3).

Per image, over the N patch tokens (prefix/register tokens excluded; raw backbone output, before the head):
  rel_l2  = mean_p ||t1_p - t0_p||_2 / ||t0_p||_2      (relative, comparable across widths 768 / 1024)
  cos     = mean_p cos(t1_p, t0_p)
  pool_l2 = ||mean_p t1_p - mean_p t0_p||_2 / ||mean_p t0_p||_2
Rows: results/feature_change.csv (tag, set, id, source, type, label, rel_l2, cos, pool_l2).

  python src/feature_change.py --tags own_v2_ctrl_s42 own_v2_L_lpft_s42 own_v2_B_s42 own_v2_B_lpft_s42 --ifd_root /home/ubuntu/id-fraud-detection
"""
import argparse, os, sys, time
import numpy as np, pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--ifd_root", required=True)
    ap.add_argument("--samples", default=os.path.join(ROOT, "results", "feature_change_samples.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "feature_change.csv"))
    ap.add_argument("--eval_bs", type=int, default=16); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    import torch
    from torch.utils.data import DataLoader
    from data import IDNetEvalDS
    from model import FreuidModel
    torch.backends.cuda.matmul.allow_tf32 = True
    smp = pd.read_csv(a.samples); smp["path"] = [os.path.join(a.ifd_root, p) for p in smp.path]
    smp["label"] = smp.label.clip(lower=0)              # private rows carry -1; the loader wants a float
    missing = [p for p in smp.path if not os.path.exists(p)]; assert not missing, f"{len(missing)} sample images missing, e.g. {missing[0]}"

    def frozen_model(bb, cargs):
        return FreuidModel(backbone=bb, pretrained=True, lora_r=cargs.get("lora_r", 16), head_type=cargs.get("head_type", "patch")).cuda().eval()

    rows = []
    for tag in a.tags:
        ck = torch.load(os.path.join(ROOT, "checkpoints", f"{tag}_dataset_final_ep4_lean.pt"), map_location="cpu", weights_only=False)
        cargs = ck["args"]; bb = cargs["backbone"]; H, W = (int(v) for v in cargs["res"].lower().split("x"))
        m0 = frozen_model(bb, cargs)                                   # LoRA B = 0 -> exactly the pretrained backbone
        m1 = frozen_model(bb, cargs)
        missing_k, unexpected = m1.load_state_dict(ck["model_lean"], strict=False); assert not unexpected
        dl = DataLoader(IDNetEvalDS(smp, H, W), batch_size=a.eval_bs, shuffle=False, num_workers=a.workers, pin_memory=True)
        t0 = time.time(); rel_all, cos_all, pool_all = [], [], []
        with torch.no_grad():
            for x, _ in dl:                                            # streamed: nothing but per-image scalars leaves the GPU
                x = x.cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.float16):
                    tk0 = m0.backbone.forward_features(x)[:, m0.n_prefix:]
                    tk1 = m1.backbone.forward_features(x)[:, m1.n_prefix:]
                tk0, tk1 = tk0.float(), tk1.float(); d = tk1 - tk0
                rel_all.append((d.norm(dim=-1) / tk0.norm(dim=-1)).mean(1).cpu())
                cos_all.append(torch.nn.functional.cosine_similarity(tk1, tk0, dim=-1).mean(1).cpu())
                p1, p0 = tk1.mean(1), tk0.mean(1); pool_all.append(((p1 - p0).norm(dim=-1) / p0.norm(dim=-1)).cpu())
        rel, cos, pool = torch.cat(rel_all), torch.cat(cos_all), torch.cat(pool_all)
        for i in range(len(smp)):
            r = smp.iloc[i]; rows.append({"tag": tag, "backbone": bb, "set": r.set, "id": r.id, "source": r.source, "type": r.type,
                                          "label": int(r.label), "rel_l2": float(rel[i]), "cos": float(cos[i]), "pool_l2": float(pool[i])})
        df = pd.DataFrame([x for x in rows if x["tag"] == tag])
        print(f"{tag} ({time.time()-t0:.0f}s):\n" + df.groupby("set")[["rel_l2", "cos", "pool_l2"]].mean().round(4).to_string(), flush=True)
        del m0, m1; torch.cuda.empty_cache()
    pd.DataFrame(rows).to_csv(a.out, index=False); print(f"wrote {os.path.relpath(a.out, ROOT)}")


if __name__ == "__main__":
    main()
