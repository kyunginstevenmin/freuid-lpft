"""freeze_epoch: the pick rule on a hand-built table, and the committed row it writes."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from freeze_epoch import pick_frozen_epoch, freeze_row, append_row

# ep2 is best; ep1 is inside its ci95 band (0.040 <= 0.045) but outside the 1se band; ep0 is not tied.
T = pd.DataFrame({"epoch": [0, 1, 2], "macro_source": [0.10, 0.040, 0.030],
                  "ci_lo": [0.08, 0.030, 0.020], "ci_hi": [0.12, 0.050, 0.045],
                  "macro_type": [0.1, 0.05, 0.04], "src/a": [0.1, 0.04, 0.03]})


def test_pick_earliest_tied_epoch():
    assert pick_frozen_epoch(T, "ci95") == 1
    assert pick_frozen_epoch(T, "1se") == 2     # 1se band: 0.030 + 0.025/3.92 ≈ 0.036 < 0.040


def test_row_round_trips_through_csv(tmp_path):
    row = freeze_row(T, "own_v2_x_s42", "dataset_final", "ci95", 1000, "s3://b/freuid")
    assert row["frozen_epoch"] == 1 and row["best_epoch"] == 2
    assert (row["n_tied_ci95"], row["n_tied_1se"]) == (2, 1)
    assert row["lean_ckpt"] == "s3://b/freuid/checkpoints/own_v2_x_s42_dataset_final_ep1_lean.pt"
    p = tmp_path / "f.csv"
    append_row(row, p); append_row(row, p)
    back = pd.read_csv(p)
    assert len(back) == 2 and back.macro_source.iloc[0] == 0.040 and "src/a" in back.columns
