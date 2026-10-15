import numpy as np

from reel_predictor.brain.reference import Reference


def test_default_until_three_reels(tmp_path):
    path = tmp_path / "reference.json"
    names = ["ips", "ofc"]
    ref = Reference.load(path)
    med, mad, used = ref.stats(names, min_reels=3)
    assert not used and (med == 0).all() and (mad == 1).all()
    rng = np.random.default_rng(0)
    for i in range(3):
        ref.add(f"job{i}", names, np.column_stack([rng.normal(2, 1, 30), rng.normal(-1, 0.5, 30)]))
        ref.save()
        ref = Reference.load(path)
    assert ref.n_reels == 3
    med, mad, used = ref.stats(names, min_reels=3)
    assert used
    assert abs(med[0] - 2) < 0.5 and abs(med[1] + 1) < 0.3
    assert 0.4 < mad[0] < 1.2 and 0.15 < mad[1] < 0.6


def test_add_caps_seconds_and_ignores_nan(tmp_path):
    ref = Reference.load(tmp_path / "r.json")
    vals = np.arange(300, dtype=float).reshape(-1, 1)
    vals[5] = np.nan
    ref.add("job", ["x"], vals, max_seconds=50)
    assert len(ref.data["reels"]["job"]["rois"]["x"]) <= 50
    ref.add("job", ["x"], vals[:10], max_seconds=50)        # même job : remplacé, pas dupliqué
    assert ref.n_reels == 1 and len(ref.data["reels"]["job"]["rois"]["x"]) == 9
