"""readout_sweep.aggregate must reproduce PatchHead.forward and evaluate.py's deploy branch."""
import os, sys
import numpy as np
import pytest
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from model import PatchHead
from readout_sweep import aggregate, RAW, FRACS, WEIGHTS


def _head_outputs(n_img=5, n_prefix=2, n_patch=40, dim=16, seed=0):
    torch.manual_seed(seed)
    head = PatchHead(dim, n_prefix).eval()
    tokens = torch.randn(n_img, n_prefix + n_patch, dim)
    with torch.no_grad():
        img, plog = head(tokens)
        patch = head.norm(tokens[:, n_prefix:])
        attn = torch.softmax(head.attn(patch).squeeze(-1), dim=1)
    return img.numpy(), plog.numpy(), attn.numpy()


def test_raw_point_matches_patchhead_forward():
    img, plog, attn = _head_outputs()
    np.testing.assert_allclose(aggregate(plog, attn, *RAW), img, rtol=1e-5, atol=1e-6)


def test_winner_knobs_match_deploy_formula():
    _, plog, attn = _head_outputs()
    t = torch.tensor(plog); aw = torch.tensor(attn)
    k = max(1, int(0.05 * t.shape[1]))
    ref = 0.25 * t.topk(k, dim=1).values.mean(1) + 0.75 * (t * aw).sum(1)   # evaluate.py deploy branch
    np.testing.assert_allclose(aggregate(plog, attn, 0.05, 0.25), ref.numpy(), rtol=1e-5, atol=1e-6)


def test_fp16_dump_round_trip_is_close():
    img, plog, attn = _head_outputs()
    got = aggregate(plog.astype(np.float16), attn.astype(np.float16), *RAW)
    np.testing.assert_allclose(got, img, rtol=2e-3, atol=2e-3)


def test_grid_contains_raw_and_winner_points():
    assert RAW in [(f, w) for f in FRACS for w in WEIGHTS]
    assert (0.05, 0.25) in [(f, w) for f in FRACS for w in WEIGHTS]
