"""Our submission path (plan step 7; LEDGER.md final pick 2026-10-09). The winner's
src/infer_private.py stays frozen as the reference path the step-7a probe validates; this
script is the one our checkpoints go through.

Differences from his: loads lean checkpoints (LoRA + head only; backbone from timm), takes the
readout knobs as arguments (top-k fraction, top-k weight, TTA scales) and aggregates with the
SAME numpy function the readout study validated (readout_sweep.aggregate: reproduces
PatchHead.forward and his deploy branch to 1e-5), so the submitted readout is exactly the
one that scored 0.0280 on gen-val. `--verify` runs the pipeline on rows of a gen-val score
cache (preds/<sha>_genval_<cell>.csv) and prints the max abs diff -- the self-check to run on
the box before the real inference.

  python src/submit.py --ckpt checkpoints/own_v2_B_lpft_s42_dataset_final_ep4_lean.pt \
      --base submissions/sample_submission.csv --img_dirs <public images> <private images> \
      --frac 0.05 --w 0.5 --scales 0.85 0.9 1.0 1.1 --out submissions/sub_B_lpft_ep4_f05_w05_tta4.csv
  python src/submit.py --ckpt ... --frac 0.05 --w 0.5 --verify preds/e678f0809bf7_genval_sweep_tta.csv --verify_n 256
Rows whose image is not found keep the base fill (0.5). Crash-safe per scale (<out>.partial.npz).
"""
from __future__ import annotations
import argparse, os, sys, time
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from readout_sweep import aggregate, sigmoid

EXTS = (".jpeg", ".jpg", ".png", ".webp")


def find_images(dirs: list[str]) -> dict[str, str]:
    have = {}
    for d in dirs:
        for f in os.listdir(d):
            stem, ext = os.path.splitext(f)
            if ext.lower() in EXTS:
                have.setdefault(stem, os.path.join(d, f))
    return have # stem: path from dir


def merge(base: pd.DataFrame, scores: dict[str, float], fill: float) -> pd.DataFrame:
    out = base[["id"]].copy()
    out["label"] = [scores.get(i, fill) for i in out.id]
    return out


def load_model(ckpt: str):
    import torch
    from model import FreuidModel
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args") or {}
    H, W = (int(v) for v in a.get("res", "448x728").lower().split("x"))
    m = FreuidModel(backbone=a.get("backbone", "vit_large_patch14_reg4_dinov2"), pretrained="model_lean" in ck,
                    lora_r=a.get("lora_r", 16), head_type=a.get("head_type", "patch")).cuda().eval()
    if "model_lean" in ck:
        _, unexpected = m.load_state_dict(ck["model_lean"], strict=False); assert not unexpected, unexpected[:3]
    else:
        m.load_state_dict(ck["model"])
    assert a.get("head_type", "patch") == "patch", "readout knobs need PatchHead"
    return m, H, W


def predict(paths: list[str], ckpt: str, frac: float, w: float, scales: list[float],
            eval_bs: int, workers: int, partial: str | None = None) -> np.ndarray:
    """Mean over scales of the aggregated image logit, then sigmoid (= readout_sweep.scores_for)."""
    import torch
    from torch.utils.data import Dataset, DataLoader
    from data import letterbox, to_tensor_norm, load_rgb

    class DS(Dataset):
        def __init__(self, Hs, Ws): self.H, self.W = Hs, Ws
        def __len__(self): return len(paths)
        def __getitem__(self, i): return to_tensor_norm(letterbox(load_rgb(paths[i]), self.H, self.W)), i

    model, H, W = load_model(ckpt); head = model.head
    lg = np.zeros(len(paths), np.float64); done = 0
    if partial and os.path.exists(partial):
        z = np.load(partial, allow_pickle=True)
        if list(z["paths"]) == paths and list(z["scales"]) == scales:
            lg, done = z["lg"], int(z["done"]); print(f"resumed: {done} scale passes done", flush=True)
    for si, s in enumerate(scales):
        if si < done: continue
        Hs, Ws = int(H * s) // 14 * 14, int(W * s) // 14 * 14
        dl = DataLoader(DS(Hs, Ws), batch_size=eval_bs, shuffle=False, num_workers=workers, pin_memory=True)
        out = np.zeros(len(paths), np.float32); t0 = time.time()
        with torch.no_grad():
            for x, idx in dl:
                with torch.autocast("cuda", dtype=torch.float16):
                    tok = model.backbone.forward_features(x.cuda(non_blocking=True))
                    patch = head.norm(tok[:, head.n_prefix:])
                    plog = head.scorer(patch).squeeze(-1).float().cpu().numpy()
                    attn = torch.softmax(head.attn(patch).squeeze(-1), dim=1).float().cpu().numpy()
                out[idx.numpy()] = aggregate(plog, attn, frac, w)      # the validated numpy readout
        lg += out; done = si + 1
        print(f"  scale {s:g} ({Hs}x{Ws}): {len(paths)} imgs in {time.time()-t0:.0f}s", flush=True)
        if partial: np.savez(partial, paths=np.array(paths), scales=np.array(scales), lg=lg, done=done)
    return sigmoid(lg / len(scales)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--frac", type=float, required=True, help="top-k fraction (final pick: 0.05)")
    ap.add_argument("--w", type=float, required=True, help="top-k weight (final pick: 0.5)")
    ap.add_argument("--scales", type=float, nargs="+", default=[0.85, 0.9, 1.0, 1.1], help="TTA scales; '1.0' = none")
    ap.add_argument("--base", help="sample_submission.csv (ids); rows without an image keep --fill")
    ap.add_argument("--img_dirs", nargs="+", help="directories holding <id>.<ext> test images")
    ap.add_argument("--out"); ap.add_argument("--fill", type=float, default=0.5)
    ap.add_argument("--verify", help="gen-val score cache csv (path,label,score) to reproduce")
    ap.add_argument("--verify_n", type=int, default=256)
    ap.add_argument("--ifd_root", help="rebase the cache's /…/id-fraud-detection/ paths onto this root (box vs laptop)")
    ap.add_argument("--eval_bs", type=int, default=32); ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    if a.verify:
        c = pd.read_csv(a.verify).head(a.verify_n)
        paths = c.path.tolist()
        if a.ifd_root:
            paths = [os.path.join(a.ifd_root, p.split("/id-fraud-detection/")[-1]) for p in paths]
        got = predict(paths, a.ckpt, a.frac, a.w, a.scales, a.eval_bs, a.workers)
        d = np.abs(got - c.score.values)
        print(f"verify vs {os.path.basename(a.verify)} on {len(c)} rows: max|diff|={d.max():.5f} mean={d.mean():.6f}"
              f"  ({'OK' if d.max() < 5e-3 else 'MISMATCH'}; fp16 dump vs live fp16 forward)")
        return

    assert a.base and a.img_dirs and a.out, "--base, --img_dirs, --out required for a submission"
    base = pd.read_csv(a.base); have = find_images(a.img_dirs) # {stem:private_or_public_dir/stem.ext}
    ids = [i for i in base.id if i in have]
    print(f"base rows {len(base)}; images found {len(ids)}; fill {a.fill} for {len(base)-len(ids)}", flush=True)
    scores = predict([have[i] for i in ids], a.ckpt, a.frac, a.w, a.scales, a.eval_bs, a.workers,
                     partial=a.out + ".partial.npz")
    merge(base, dict(zip(ids, scores)), a.fill).to_csv(a.out, index=False)
    print(f"wrote {a.out}: readout frac={a.frac} w={a.w} scales={a.scales}; "
          f"predicted {len(ids)}, filled {len(base)-len(ids)}", flush=True)


if __name__ == "__main__":
    main()
