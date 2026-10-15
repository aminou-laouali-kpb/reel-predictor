"""Pipeline complet sur la vidéo synthétique (data/samples/demo_synthetic.mp4) — schéma du contrat §3.

Deux analyses par session : MockProvider (cerveau simulé) et un fournisseur TRIBE en échec
(-> repli HeuristicProvider). Aucun appel au vrai TRIBE.
"""
import json
import math
from pathlib import Path

import numpy as np
import pytest

from reel_predictor import paths
from reel_predictor.brain.mock import MockProvider
from reel_predictor.brain.provider import BrainUnavailable
from reel_predictor.config import INDEX_NAMES, load_config
from reel_predictor.media import VideoUnreadableError
from reel_predictor.pipeline import analyze
from reel_predictor.stages import STAGE_IDS
from reel_predictor.timeline import TIMELINE_KEYS, encode_brain, read_timeline

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def video() -> Path:
    from scripts.make_samples import OUT, make

    return OUT if OUT.exists() else make()


@pytest.fixture(scope="module")
def cfg():
    return load_config(calibration_file=None)


class FailingTribe:
    name = "tribe"

    def predict(self, request):
        raise BrainUnavailable("Mémoire insuffisante pour TRIBE v2", "test")


@pytest.fixture(scope="module")
def mock_run(video, cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("results") / "job_mock"
    events = []
    res = analyze(video, out, cfg, progress=lambda s, p, m: events.append((s, p, m)),
                  brain=MockProvider(cfg.roi_mapping))
    return res, events


@pytest.fixture(scope="module")
def heur_run(video, cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("results") / "job_heur"
    return analyze(video, out, cfg, brain=FailingTribe())


def _check_timeline(rows, roi_names, brain: bool):
    assert [r["t"] for r in rows] == list(range(len(rows)))
    for r in rows:
        assert tuple(r) == TIMELINE_KEYS
        assert isinstance(r["t"], int)
        assert set(r["roi"]) == set(roi_names) == set(r["roi_video_z"])
        for d in (r["roi"], r["roi_video_z"]):
            for v in d.values():
                assert (isinstance(v, float) and math.isfinite(v)) if brain else v is None
        assert set(r["indices"]) == set(INDEX_NAMES)
        for v in r["indices"].values():
            assert isinstance(v, float) and 0.0 <= v <= 1.0
        assert isinstance(r["cut"], bool) and isinstance(r["text_new"], bool) and isinstance(r["silence"], bool)
        assert isinstance(r["since_cut"], float) and r["since_cut"] > 0
        assert isinstance(r["faces"], int) and r["faces"] >= 0
        assert isinstance(r["face_area"], float) and 0 <= r["face_area"] <= 1
        assert isinstance(r["words"], list) and all(isinstance(w, str) for w in r["words"])
        assert isinstance(r["speech_rate"], float) and r["speech_rate"] >= 0
        assert isinstance(r["text"], str) and isinstance(r["text_words"], int)
        assert isinstance(r["audio_rms"], float) and 0 <= r["audio_rms"] <= 1
        assert isinstance(r["retention"], float) and 0 <= r["retention"] <= 1
        assert isinstance(r["flags"], list) and all(isinstance(f, str) for f in r["flags"])


def _check_summary(s, T):
    for k in ("version", "job_id", "created_at", "video", "score", "score_breakdown", "retention",
              "recommendations", "cuts", "transcript", "brain", "timings", "config"):
        assert k in s, k
    assert s["version"] == 1
    v = s["video"]
    assert set(v) >= {"filename", "duration_s", "fps", "width", "height", "has_audio"}
    assert v["width"] == 720 and v["height"] == 1280 and v["has_audio"] is True
    assert abs(v["duration_s"] - 20.0) < 0.1
    assert isinstance(s["score"], int) and 0 <= s["score"] <= 100
    sb = s["score_breakdown"]
    assert set(sb["index_means"]) == set(INDEX_NAMES)
    assert isinstance(sb["hook_bonus"], float) and isinstance(sb["dip_penalty"], float) and sb["dip_penalty"] <= 0
    assert isinstance(sb["weights"], dict)
    r = s["retention"]
    assert len(r["curve"]) == T + 1 and r["curve"][0] == 1.0
    assert all(0 <= x <= 1 for x in r["curve"])
    assert 0 < r["avg_watch_ratio"] <= 1 and 0 < r["avg_watch_s"] <= v["duration_s"] and 0 <= r["completion"] <= 1
    for rec in s["recommendations"]:
        assert set(rec) == {"t", "t_end", "rule", "severity", "message"}
        assert rec["severity"] in ("high", "medium", "low") and isinstance(rec["message"], str)
        assert isinstance(rec["t"], float) or isinstance(rec["t"], int)
    assert all(isinstance(c, float) for c in s["cuts"])
    assert set(s["transcript"]) == {"language", "text"}
    b = s["brain"]
    assert set(b) >= {"available", "provider", "model", "degraded", "reason", "rois", "confidence"}
    assert set(b["confidence"]) == set(INDEX_NAMES)
    assert all(c in ("medium", "low") for c in b["confidence"].values())
    assert set(s["config"]) == {"weights", "roi_mapping", "rules"}
    assert all(isinstance(x, (int, float)) for x in s["timings"].values())
    json.dumps(s)


def test_mock_outputs_match_contract(mock_run, cfg):
    res, _ = mock_run
    out = res.out_dir
    rows = read_timeline(out / "timeline.jsonl")
    assert rows == res.timeline
    T = len(rows)
    assert T == 20
    _check_timeline(rows, list(cfg.roi_mapping["rois"]), brain=True)
    s = json.loads((out / "summary.json").read_text())
    _check_summary(s, T)
    assert s["brain"]["available"] is True and s["brain"]["provider"] == "mock" and s["brain"]["degraded"] is False
    # carte cérébrale
    raw = (out / "brain.u8").read_bytes()
    assert len(raw) == T * paths.N_VERTICES
    meta = json.loads((out / "brain_meta.json").read_text())
    assert meta["T"] == T and meta["n_vertices"] == 20484 and meta["hemi_split"] == 10242
    assert meta["vmin"] == pytest.approx(-meta["vmax"]) and meta["vmax"] > 0
    assert meta["space"] == "fsaverage5" and isinstance(meta["units"], str)
    # flags
    assert "low_context" in rows[0]["flags"] and "low_context" not in rows[10]["flags"]
    assert "hook" in rows[0]["flags"]
    # words.json pour le worker
    w = json.loads((out / "words.json").read_text())
    assert set(w) == {"language", "duration_s", "words"} and w["words"]
    assert set(w["words"][0]) == {"w", "start", "end"}


def test_mock_progress(mock_run):
    _, events = mock_run
    assert events, "aucune progression"
    pcts = [p for _, p, _ in events]
    assert all(b >= a for a, b in zip(pcts, pcts[1:])), "progression non monotone"
    assert pcts[-1] == pytest.approx(1.0)
    assert {s for s, _, _ in events} <= set(STAGE_IDS)
    assert all(isinstance(m, str) and m for _, _, m in events)


def test_signals_plausible(mock_run):
    res, _ = mock_run
    s, rows = res.summary, res.timeline
    # cuts du scénario : 3, 6, 15 s
    assert len(s["cuts"]) == 3
    for got, want in zip(s["cuts"], [3.0, 6.0, 15.0]):
        assert abs(got - want) < 0.2
    assert [r["t"] for r in rows if r["cut"]] == [3, 6, 15]
    # transcription française
    assert s["transcript"]["language"] == "fr"
    txt = s["transcript"]["text"].lower()
    assert "astuce" in txt and "instagram" in txt
    assert rows[0]["words"], "la parole commence dans la 1re seconde"
    # OCR
    assert "ASTUCES" in rows[1]["text"].upper()
    assert "ABONNE" in rows[17]["text"].upper()
    assert rows[7]["text_words"] >= 20
    assert rows[12]["text"] == ""
    # silence ~9,5–14 s
    assert all(rows[t]["silence"] for t in (10, 11, 12))
    assert not rows[2]["silence"]
    # CTA
    assert "cta" in rows[17]["flags"]


def test_rules_fire_where_expected(mock_run):
    recs = mock_run[0].summary["recommendations"]
    by = {}
    for r in recs:
        by.setdefault(r["rule"], []).append(r)
    assert "silence_gap" in by and any(9 <= r["t"] <= 11 and r["t_end"] >= 13 for r in by["silence_gap"])
    assert "text_overload" in by and any(r["t"] == 7 for r in by["text_overload"])
    assert "attention_drop_no_cut" in by and all(8 <= r["t"] <= 14 for r in by["attention_drop_no_cut"])
    assert "no_face_long" in by
    assert "cta_missing" not in by and "slow_start" not in by and "weak_hook" not in by


def test_heuristic_fallback(heur_run, cfg):
    res = heur_run
    out = res.out_dir
    s = json.loads((out / "summary.json").read_text())
    T = len(res.timeline)
    _check_summary(s, T)
    _check_timeline(res.timeline, list(cfg.roi_mapping["rois"]), brain=False)
    b = s["brain"]
    assert b["available"] is False and b["degraded"] is True and b["provider"] == "heuristic"
    assert b["reason"] == "Mémoire insuffisante pour TRIBE v2"
    assert not (out / "brain.u8").exists() and not (out / "brain_meta.json").exists()
    assert all("no_brain" in r["flags"] for r in res.timeline)
    assert any("Mémoire insuffisante" in w for w in s.get("warnings", []))


def test_unreadable_video_raises(tmp_path):
    bad = tmp_path / "video.mp4"
    bad.write_bytes(b"ceci n'est pas une video" * 100)
    with pytest.raises(VideoUnreadableError):
        analyze(bad, tmp_path / "job_bad", load_config(calibration_file=None))


def test_encode_brain():
    a = np.zeros((3, paths.N_VERTICES), np.float32)
    a[0, :10] = 5.0
    a[1] = np.nan
    u8, meta = encode_brain(a)
    assert u8.shape == (3, paths.N_VERTICES) and u8.dtype == np.uint8
    assert (u8[1] == 128).all()
    assert meta["vmin"] == -meta["vmax"]
    vmin, vmax = meta["vmin"], meta["vmax"]
    decoded = vmin + u8.astype(float) / 255 * (vmax - vmin)
    assert abs(decoded[2, 0]) < (vmax - vmin) / 255 + 1e-9
