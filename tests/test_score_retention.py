import numpy as np
import pytest

from reel_predictor.config import INDEX_NAMES, load_config
from reel_predictor.features import build_features
from reel_predictor.indices import compute_indices
from reel_predictor.retention import simulate
from reel_predictor.score import compute_score, runs_below
from reel_predictor.synthetic import make_signals


@pytest.fixture(scope="module")
def cfg():
    return load_config(calibration_file=None)


def _params(cfg):
    r = cfg.retention
    return dict(hook_seconds=r.hook_seconds, hook_drop=r.hook_drop, base_drop=r.base_drop,
                k_attention=r.k_attention, cut_bonus=r.cut_bonus, allow_increase=r.allow_increase)


def test_retention_bounded_and_decreasing(cfg):
    rng = np.random.default_rng(0)
    att = rng.uniform(0, 1, 30)
    cr = np.zeros(30)
    res = simulate(att, cr, 30.0, **_params(cfg))
    c = res.curve
    assert c.shape == (31,) and c[0] == 1.0
    assert np.all((c >= 0) & (c <= 1))
    assert np.all(np.diff(c) <= 1e-12)                   # décroissante hors cuts
    assert 0 < res.avg_watch_ratio <= 1 and res.completion == pytest.approx(c[-1])
    assert res.avg_watch_s == pytest.approx(res.avg_watch_ratio * 30.0)


def test_retention_hook_drop_stronger(cfg):
    att = np.full(10, 0.5)
    c = simulate(att, np.zeros(10), 10.0, **_params(cfg)).curve
    drops = -np.diff(c) / c[:-1]
    assert drops[:3].min() > drops[3:].max()


def test_retention_attention_and_cuts_help(cfg):
    p = _params(cfg)
    low = simulate(np.full(20, 0.2), np.zeros(20), 20.0, **p)
    high = simulate(np.full(20, 0.8), np.zeros(20), 20.0, **p)
    cuts = simulate(np.full(20, 0.2), np.ones(20), 20.0, **p)
    assert high.avg_watch_ratio > low.avg_watch_ratio
    assert cuts.avg_watch_ratio > low.avg_watch_ratio
    assert np.all(np.diff(cuts.curve) <= 1e-12)          # même avec des cuts, jamais de remontée


def test_retention_extreme_params_bounded():
    c = simulate(np.zeros(5), np.ones(5), 5.0, hook_seconds=3, hook_drop=0.9, base_drop=0.9,
                 k_attention=0.9, cut_bonus=0.0).curve
    assert np.all((c >= 0) & (c <= 1))
    up = simulate(np.ones(5), np.ones(5), 5.0, hook_seconds=0, hook_drop=0, base_drop=0,
                  k_attention=0, cut_bonus=0.5, allow_increase=True).curve
    assert np.all(up <= 1.0)


def test_partial_last_second(cfg):
    res = simulate(np.full(3, 1.0), np.zeros(3), 2.5, **{**_params(cfg), "hook_drop": 0.0, "base_drop": 0.0,
                                                        "k_attention": 0.0})
    assert res.avg_watch_s == pytest.approx(2.5)


def test_runs_below():
    x = np.array([0.5, 0.2, 0.2, 0.2, 0.6, 0.1, 0.1])
    assert runs_below(x, 0.3, 3) == [(1, 3)]
    assert runs_below(x, 0.3, 2) == [(1, 3), (5, 6)]


def _df(cfg, **kw):
    return build_features(make_signals(15.0, **kw), cfg)


def test_score_hook_bonus_and_range(cfg):
    good = _df(cfg, cuts=[1.5, 4.0, 7.0, 10.0, 13.0], words=[("Bonjour", 0.1, 0.5), ("vite", 0.6, 0.9)],
               faces={0: (1, 0.15), 1: (1, 0.15)}, texts={0: "ATTENTION", 1: "ATTENTION"}, motion={0: 30})
    bad = _df(cfg, words=[("Bonjour", 5.0, 5.4)], silence=list(range(0, 5)))
    res_g, res_b = [], []
    for d, out in ((good, res_g), (bad, res_b)):
        idx = compute_indices(d, cfg).values
        ret = simulate(idx["attention"], d["f_cut_recent"].to_numpy(), 15.0, **_params(cfg))
        fw = 0.1 if d is good else 5.0
        out.append(compute_score(d, idx, ret, fw, cfg))
    g, b = res_g[0], res_b[0]
    assert 0 <= b.score <= g.score <= 100
    assert g.breakdown["hook_bonus"] > b.breakdown["hook_bonus"]
    assert g.breakdown["hook_bonus"] <= cfg.section("score")["hook"]["max"]
    assert set(g.breakdown["index_means"]) == set(INDEX_NAMES)
    assert b.breakdown["dip_penalty"] <= 0


def test_dip_penalty(cfg):
    d = _df(cfg)
    idx = {k: np.full(len(d), 0.6) for k in INDEX_NAMES}
    idx["attention"][4:10] = 0.1
    ret = simulate(idx["attention"], np.zeros(len(d)), 15.0, **_params(cfg))
    sc = compute_score(d, idx, ret, None, cfg)
    dp = cfg.section("score")["dips"]
    assert sc.breakdown["dip_penalty"] == pytest.approx(-min(dp["max"], 6 * dp["points_per_second"]))
    assert sc.breakdown["dips"] == [[4, 10]]
