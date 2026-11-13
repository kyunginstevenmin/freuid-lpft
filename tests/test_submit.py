"""submit.py: merge keeps the fill for rows without an image; aggregation is the validated one."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import submit, readout_sweep


def test_merge_fills_missing_and_keeps_order():
    base = pd.DataFrame({"id": ["a", "b", "c"], "label": [0, 0, 0]})
    out = submit.merge(base, {"b": 0.9}, fill=0.5)
    assert out.id.tolist() == ["a", "b", "c"] and out.label.tolist() == [0.5, 0.9, 0.5]


def test_find_images_by_stem(tmp_path):
    for f in ("x.jpg", "y.PNG", "notes.txt"): (tmp_path / f).write_bytes(b"")
    have = submit.find_images([str(tmp_path)])
    assert set(have) == {"x", "y"}


def test_uses_the_validated_aggregator():
    assert submit.aggregate is readout_sweep.aggregate
