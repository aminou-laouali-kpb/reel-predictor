import json
from pathlib import Path

import numpy as np
import pytest

from reel_predictor.calibrate import calibrate
from reel_predictor.config import load_config
from reel_predictor.features import cut_recent_from, since_cut_series
from reel_predictor.retention import simulate_params


def _write_result(d: Path, attention: np.ndarray, cuts: list[float], duration: float) -> None:
    d.mkdir(parents=True)
    T = attention.size
    cut, since, _ = since_cut_series(cuts, T)
    rows = [{"t": t, "indices": {"attention": float(attention[t])}, "cut": bool(cut[t]),
             "since_cut": float(since[t])} for t in range(T)]
    (d / "timeline.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (d / "summary.json").write_text(json.dumps({"video": {"duration_s": duration}}))


@pytest.fixture
def cfg():
    return load_config(calibration_file=None)


@pytest.fixture
def reels(tmp_path, cfg):
    """5 reels synthétiques dont le « vrai » ratio suit des paramètres différents des défauts."""
    rng = np.random.default_rng(1)
    truth = {"hook_drop": 0.15, "base_drop": 0.03, "k_attention": 0.08, "cut_bonus": 0.01}
    tau = cfg.section("features")["cut_tau_s"]
    pairs = []
    for i in range(5):
        T = int(rng.integers(15, 45))
        att = np.clip(rng.normal(0.5, 0.2, T), 0, 1)
        cuts = sorted(rng.uniform(0, T, int(rng.integers(1, 8))).round(2).tolist())
        d = tmp_path / f"job_{i}"
        _write_result(d, att, cuts, float(T))
        cut, since, had = since_cut_series(cuts, T)
        real = simulate_params(att, cut_recent_from(since, had, tau), float(T), truth,
                               cfg.retention.hook_seconds).avg_watch_ratio
        pairs.append((d, real))
    return pairs


def test_needs_three(reels, tmp_path, cfg):
    with pytest.raises(ValueError):
        calibrate(reels[:2], cfg, out_file=tmp_path / "cal.json")
    with pytest.raises(ValueError):
        calibrate([], cfg, out_file=None)


def test_calibration_reduces_error(reels, tmp_path, cfg):
    out = tmp_path / "calibration.json"
    res = calibrate(reels, cfg, out_file=out)
    assert res["n"] == 5
    assert res["mae_after"] < res["mae_before"]
    assert res["mae_after"] < 0.5 * res["mae_before"]
    bounds = cfg.section("retention")["calibration_bounds"]
    for k, v in res["params"].items():
        assert bounds[k][0] - 1e-9 <= v <= bounds[k][1] + 1e-9
    data = json.loads(out.read_text())
    assert data["params"] == res["params"] and len(data["reels"]) == 5
    # la calibration est relue par load_config (surcharge de weights.yaml > retention)
    cfg2 = load_config(calibration_file=out)
    assert cfg2.retention.base_drop == pytest.approx(res["params"]["base_drop"])


def test_invalid_dirs_skipped(reels, tmp_path, cfg):
    pairs = reels[:3] + [(tmp_path / "absent", 0.4)]
    res = calibrate(pairs, cfg, out_file=None)
    assert res["n"] == 3 and len(res["skipped"]) == 1
    with pytest.raises(ValueError):
        calibrate(reels[:2] + [(tmp_path / "absent", 0.4)], cfg, out_file=None)
