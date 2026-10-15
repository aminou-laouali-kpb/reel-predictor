import numpy as np
import pytest

from reel_predictor.brain.atlas import roi_matrix
from reel_predictor.brain.mock import MockProvider
from reel_predictor.brain.provider import BrainRequest
from reel_predictor.config import INDEX_NAMES, load_config
from reel_predictor.features import build_features
from reel_predictor.indices import (align_predictions, compute_indices, normalize_corpus, roi_values, sigmoid,
                                    zscore_video)
from reel_predictor.synthetic import make_signals


@pytest.fixture(scope="module")
def cfg():
    return load_config(calibration_file=None)


@pytest.fixture(scope="module")
def df(cfg):
    words = [(w, 0.2 + 0.4 * i, 0.5 + 0.4 * i) for i, w in enumerate("voici trois astuces pour percer vite".split())]
    words += [("abonne-toi", 17.2, 17.8), ("lien", 18.0, 18.3), ("en", 18.3, 18.4), ("bio", 18.4, 18.8)]
    sig = make_signals(20.0, cuts=[3.0, 6.0, 15.0], words=words,
                       faces={t: (1, 0.12) for t in range(0, 6)}, motion={0: 20, 1: 20, 3: 30, 15: 30},
                       texts={0: "3 ASTUCES", 1: "3 ASTUCES", 17: "LIEN EN BIO", 18: "LIEN EN BIO"},
                       silence=list(range(9, 14)))
    return build_features(sig, cfg)


@pytest.fixture(scope="module")
def mock_rois(cfg, df, tmp_path_factory):
    T = len(df)
    req = BrainRequest(video_path=None, words_path=None, duration_s=20.0, n_seconds=T,
                       work_dir=tmp_path_factory.mktemp("mock"), features=df)
    pred = MockProvider(cfg.roi_mapping).predict(req)
    names, W = roi_matrix(cfg.roi_mapping)
    aligned = align_predictions(pred.preds, pred.seconds, T)
    return pred, names, roi_values(aligned, W)


def test_features_basic(df):
    assert list(df["t"]) == list(range(20))
    assert df["cut"].tolist()[3] and df["cut"].tolist()[6] and df["cut"].tolist()[15]
    assert df.at[14, "since_cut"] == pytest.approx(9.0)    # dernier cut à 6 s, fin de la seconde 14
    assert df.at[0, "since_cut"] == pytest.approx(1.0)     # depuis le début
    assert bool(df.at[17, "cta"]) and bool(df.at[18, "cta"])
    assert not bool(df.at[5, "cta"])
    assert df.at[0, "speech_rate"] > 0
    for col in [c for c in df.columns if c.startswith("f_")]:
        v = df[col].to_numpy(float)
        assert np.all((v >= 0) & (v <= 1)), col


def test_mock_is_structured(df, mock_rois):
    pred, names, raw = mock_rois
    assert pred.preds.shape == (20, 20484) and pred.available
    aud = raw[:, names.index("auditory")]
    vis = raw[:, names.index("early_visual")]
    assert np.corrcoef(aud, df["f_speech"])[0, 1] > 0.5
    assert np.corrcoef(vis, df["f_cut_recent"] + df["f_motion"])[0, 1] > 0.5


def test_mock_is_deterministic(cfg, df, tmp_path):
    req = BrainRequest(None, None, 20.0, len(df), tmp_path, features=df)
    a = MockProvider(cfg.roi_mapping).predict(req).preds
    b = MockProvider(cfg.roi_mapping).predict(req).preds
    np.testing.assert_array_equal(a, b)


def test_indices_with_brain(cfg, df, mock_rois):
    _, names, raw = mock_rois
    roi = normalize_corpus(raw, np.zeros(len(names)), np.ones(len(names)), 1.4826)
    res = compute_indices(df, cfg, roi, names)
    for k in INDEX_NAMES:
        v = res.values[k]
        assert v.shape == (20,) and np.all(np.isfinite(v)) and np.all((v >= 0) & (v <= 1)), k
        assert np.all(np.isfinite(res.brain[k])), k
    # l'action culmine sur le CTA
    assert res.values["action"][17:19].mean() > res.values["action"][7:13].mean()
    # l'attention est plus haute juste après un cut que dans le long plan muet
    assert res.values["attention"][3:5].mean() > res.values["attention"][10:14].mean()


def test_indices_without_brain(cfg, df):
    res = compute_indices(df, cfg)
    for k in INDEX_NAMES:
        assert np.all(np.isnan(res.brain[k]))
        assert np.all((res.values[k] >= 0) & (res.values[k] <= 1))
    assert res.values["attention"][3:5].mean() > res.values["attention"][10:14].mean()


def test_brain_weight_zero_equals_signal(cfg, df, mock_rois):
    import copy

    _, names, raw = mock_rois
    c2 = copy.copy(cfg)
    c2.weights = copy.deepcopy(cfg.weights)
    c2.weights["normalization"]["smoothing"] = []
    for k in INDEX_NAMES:
        c2.weights["indices"][k]["brain_weight"] = 0.0
    res = compute_indices(df, c2, raw, names)
    for k in INDEX_NAMES:
        np.testing.assert_allclose(res.values[k], res.signal[k])


def test_normalization_helpers():
    assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
    raw = np.array([[1.0, np.nan], [3.0, np.nan], [5.0, 2.0]])
    z = zscore_video(raw)
    assert np.nanmean(z[:, 0]) == pytest.approx(0.0, abs=1e-6)
    assert z[2, 1] == 0.0 and np.isnan(z[0, 1])
    n = normalize_corpus(raw, np.array([1.0, 0.0]), np.array([2.0, 1.0]), 1.4826)
    assert n[1, 0] == pytest.approx((3 - 1) / (1.4826 * 2))


def test_align_predictions_missing_seconds():
    preds = np.ones((3, 5), np.float32)
    al = align_predictions(preds, np.array([0, 2, 7]), T=4)
    assert np.isnan(al[1]).all() and np.isnan(al[3]).all() and not np.isnan(al[2]).any()
    al2 = align_predictions(preds, np.array([1, 2, 3]), T=4, hrf_shift_s=1)
    assert not np.isnan(al2[0]).any() and np.isnan(al2[3]).all()
