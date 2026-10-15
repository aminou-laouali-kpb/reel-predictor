"""Simulation des variantes du coach (reel_predictor/coach/simulate.py, contrat §7).

Rapides : voix de synthèse macOS réelle (`say`, ~1 s), cerveau simulé (worker TRIBE remplacé par une fonction)
ou mode heuristique. Le test `tribe` lance le vrai worker `--no-video` (pytest -m tribe).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from reel_predictor import paths
from reel_predictor.coach import simulate as sim
from reel_predictor.coach.schema import Segment, Variant
from reel_predictor.config import INDEX_NAMES, load_config

pytestmark = pytest.mark.skipif(shutil.which("say") is None or shutil.which("ffmpeg") is None,
                                reason="voix de synthèse macOS (say) ou ffmpeg absents")


@pytest.fixture(scope="module")
def cfg():
    return load_config(calibration_file=None)


@pytest.fixture()
def coach_cfg():
    c = yaml.safe_load((paths.CONFIG_DIR / "coach.yaml").read_text(encoding="utf-8"))
    c["simulation"]["timeout_s"] = 120
    return c


def S(role, vo, screen="", shot="face_cam", cut=False, edit=""):
    return Segment(role=role, voice_over=vo, on_screen_text=screen, shot=shot, cut_at_start=cut, edit=edit)


GOOD = Variant(angle="chiffré", hook_summary="montant d'emblée", rationale="test", segments=[
    S("hook", "1400 euros par mois pour étudier en Europe.", "1400 € PAR MOIS", "close_up"),
    S("fact", "Trois pays, un master, tout payé.", "3 PAYS, TOUT PAYÉ", "broll", True),
    S("body", "Il suffit d'une licence.", "UNE LICENCE", "face_cam", True),
    S("cta", "Abonne-toi et commente GUIDE.", "COMMENTE GUIDE", "close_up", True),
])
BAD = Variant(angle="intro", hook_summary="aucune", rationale="test", segments=[
    S("hook", "Bonjour à tous, aujourd'hui je voulais vous parler d'un sujet assez général.", "", "screen"),
    S("body", "Il existe plusieurs programmes en Europe pour les étudiants qui souhaitent partir à l'étranger.", "", "screen"),
    S("cta", "Voilà, merci de m'avoir écouté.", "", "screen"),
])


@pytest.fixture()
def job_dir(tmp_path):
    """Résultat d'analyse minimal (summary + timeline + words) : 12 s, 3 phrases, cut à 4,2 s, CTA à la fin."""
    job = tmp_path / "20261012-000000-test"
    job.mkdir()
    words = []
    for text, t0 in (("1400 euros par mois pour étudier en Europe.", 0.3),
                     ("C'est la bourse Erasmus Mundus, 1400 euros pendant deux ans.", 4.4),
                     ("Clique sur le lien en bio pour mon guide.", 8.5)):
        for k, w in enumerate(text.split()):
            words.append({"w": w, "start": round(t0 + 0.4 * k, 2), "end": round(t0 + 0.4 * k + 0.35, 2)})
    texts = {0: "1400 € PAR MOIS", 1: "1400 € PAR MOIS", 2: "1400 € PAR MOIS", 5: "ERASMUS MUNDUS",
             6: "ERASMUS MUNDUS", 9: "PRENDREMONGUIDE", 10: "PRENDREMONGUIDE"}
    rows = [{"t": t, "text": texts.get(t, ""), "faces": 0 if t >= 10 else 1,
             "words": [w["w"] for w in words if t <= w["start"] < t + 1]} for t in range(12)]
    (job / "timeline.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (job / "words.json").write_text(json.dumps({"language": "fr", "duration_s": 12.0, "words": words}), encoding="utf-8")
    (job / "summary.json").write_text(json.dumps({"video": {"duration_s": 12.0}, "cuts": [4.2],
                                                  "transcript": {"language": "fr", "text": ""}}), encoding="utf-8")
    return job


def fake_brain(calls):
    """Remplace le worker : activité synthétique (parole -> +, bruit déterministe) au format du worker."""
    def run(wav, words_path, out_path, cfg, no_text, timeout_s, progress):
        calls.append(Path(wav).name)
        data = json.loads(Path(words_path).read_text(encoding="utf-8"))
        n = int(np.ceil(data["duration_s"] - 1e-3))
        speech = np.zeros(n)
        for w in data["words"]:
            speech[min(n - 1, int(w["start"]))] += 1.0
        rng = np.random.default_rng(len(data["words"]))
        preds = rng.normal(0, 0.05, (n, paths.N_VERTICES)).astype(np.float32) + 0.1 * speech[:, None]
        if progress:
            progress(1.0, "faux cerveau")
        return preds, np.arange(n), {"runtime_s": 0.1, "text_feature": "disabled" if no_text else "llama",
                                     "video_feature": "disabled", "n_rows": n}
    return run


def check_contract(res, n):
    assert set(res) >= {"mode", "reason", "results", "best", "timings"}
    assert res["mode"] in ("tribe_no_video", "heuristic")
    assert len(res["results"]) == n
    for r in res["results"]:
        assert set(r) >= {"name", "duration_s", "score", "segments", "timeline", "evaluation"}
        assert isinstance(r["score"], int) and 0 <= r["score"] <= 100
        assert r["duration_s"] > 0
        assert all(set(s) == {"role", "t", "t_end"} and s["t_end"] > s["t"] for s in r["segments"])
        assert [row["t"] for row in r["timeline"]] == list(range(len(r["timeline"])))
        for row in r["timeline"]:
            assert set(row["indices"]) == set(INDEX_NAMES)
            assert all(0.0 <= v <= 1.0 for v in row["indices"].values()) and 0.0 <= row["retention"] <= 1.0
        ev = r["evaluation"]
        assert {"moments", "attainment", "seconds_below_floor", "index_means"} <= set(ev)
    json.dumps(res)                      # sérialisable tel quel (écrit par le service coach)


# ----------------------------------------------------------------------------- baseline
def test_baseline_variant(job_dir):
    v = sim.baseline_variant(job_dir)
    assert v.angle == sim.BASELINE_ANGLE
    assert [s.role for s in v.segments] == ["hook", "fact", "cta"]   # « 1400 euros » est dans l'accroche
    assert v.segments[0].voice_over.startswith("1400 euros") and v.segments[0].on_screen_text == "1400 € PAR MOIS"
    assert [s.cut_at_start for s in v.segments] == [False, True, False]
    assert v.segments[1].on_screen_text == "ERASMUS MUNDUS"
    assert v.segments[0].shot == "face_cam"


def test_baseline_cta_from_glued_ocr_only(job_dir):
    # rien à l'oral, mais « PRENDREMONGUIDE » à l'écran (OCR collé) : le dernier segment reste un CTA
    words = json.loads((job_dir / "words.json").read_text())
    last = [w for w in words["words"] if w["start"] >= 8.5]
    for w, new in zip(last, ["Merci", "et", "à", "bientôt", "les", "amis", "de", "la", "chaîne."]):
        w["w"] = new
    (job_dir / "words.json").write_text(json.dumps(words))
    v = sim.baseline_variant(job_dir)
    assert "clique" not in v.segments[-1].voice_over.lower()
    assert v.segments[-1].role == "cta" and "PRENDREMONGUIDE" in v.segments[-1].on_screen_text


def test_segment_words_timing():
    x = np.concatenate([np.zeros(1600), 0.3 * np.ones(16000), np.zeros(3200)]).astype(np.float32)
    ws = sim._segment_words("Bonjour, 1400 euros !", x, 10.0)
    assert [w["w"] for w in ws] == ["Bonjour,", "1400", "euros!"]
    assert ws[0]["start"] >= 10.0 and ws[-1]["end"] <= 10.0 + x.size / 16000
    assert all(a["end"] <= b["start"] for a, b in zip(ws, ws[1:]))
    # « 1400 » se dit plus longtemps que « euros »
    assert (ws[1]["end"] - ws[1]["start"]) > (ws[2]["end"] - ws[2]["start"])


# ----------------------------------------------------------------------------- simulate
def test_simulate_heuristic_ranks_good_first(job_dir, cfg, coach_cfg):
    coach_cfg["simulation"]["brain_mode"] = "heuristic"
    steps = []
    res = sim.simulate([sim.baseline_variant(job_dir), BAD, GOOD], job_dir, cfg, coach_cfg,
                       progress=lambda p, m: steps.append((p, m)))
    check_contract(res, 3)
    assert res["mode"] == "heuristic" and "brain_mode" in res["reason"]
    assert [r["name"] for r in res["results"]] == ["original", "variante 1", "variante 2"]
    assert res["best"] == 2
    good, bad = res["results"][2], res["results"][1]
    assert good["score"] > bad["score"]
    assert good["evaluation"]["index_means"]["attention"] > bad["evaluation"]["index_means"]["attention"]
    assert [s["role"] for s in good["segments"]] == ["hook", "fact", "body", "cta"]
    # durées issues de la voix de synthèse : segments contigus couvrant toute la variante
    segs = good["segments"]
    assert segs[0]["t"] == 0 and abs(segs[-1]["t_end"] - good["duration_s"]) < 0.05
    assert all(abs(a["t_end"] - b["t"]) < 0.02 for a, b in zip(segs, segs[1:]))
    ps = [p for p, _ in steps]
    assert ps == sorted(ps) and ps[-1] == 1.0 and all(isinstance(m, str) and m for _, m in steps)


def test_simulate_brain_normalized_on_original_and_reused(job_dir, cfg, coach_cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(sim, "_run_brain", fake_brain(calls))
    monkeypatch.setattr("reel_predictor.brain.tribe.worker_installed", lambda: True)
    ref_before = paths.REFERENCE_FILE.stat().st_mtime if paths.REFERENCE_FILE.exists() else None
    res = sim.simulate([sim.baseline_variant(job_dir), GOOD, BAD], job_dir, cfg, coach_cfg)
    check_contract(res, 3)
    assert res["mode"] == "tribe_no_video" and res["reason"] is None
    assert len(calls) == 3 and res["best"] in (1, 2)
    assert res["normalization"]["source"].startswith("original de cette")
    assert (job_dir / "coach" / sim.REFERENCE_FILE).exists()
    # la référence de corpus des Reels réels n'est jamais touchée
    assert (paths.REFERENCE_FILE.stat().st_mtime if paths.REFERENCE_FILE.exists() else None) == ref_before
    # seconde passe : une variante seule réutilise la référence enregistrée -> mêmes indices
    calls.clear()
    res2 = sim.simulate([GOOD], job_dir, cfg, coach_cfg)
    assert len(calls) == 1 and res2["best"] == 0 and res2["results"][0]["name"] == "variante 1"
    assert res2["normalization"]["source"].startswith("original enregistré")
    assert res2["results"][0]["timeline"] == res["results"][1]["timeline"]
    assert res2["results"][0]["score"] == res["results"][1]["score"]


def test_single_variant_without_saved_reference_simulates_original(job_dir, cfg, coach_cfg, monkeypatch):
    calls = []
    monkeypatch.setattr(sim, "_run_brain", fake_brain(calls))
    monkeypatch.setattr("reel_predictor.brain.tribe.worker_installed", lambda: True)
    res = sim.simulate([GOOD], job_dir, cfg, coach_cfg)
    check_contract(res, 1)
    assert len(calls) == 2                          # la variante + l'original (référence)
    assert res["normalization"]["source"].startswith("original re-simulé")


def test_tribe_failure_falls_back_to_heuristic(job_dir, cfg, coach_cfg, monkeypatch):
    def boom(*a, **k):
        raise sim._BrainFailure("Mémoire insuffisante pour TRIBE v2", "test")

    monkeypatch.setattr(sim, "_run_brain", boom)
    monkeypatch.setattr("reel_predictor.brain.tribe.worker_installed", lambda: True)
    res = sim.simulate([sim.baseline_variant(job_dir), GOOD], job_dir, cfg, coach_cfg)
    check_contract(res, 2)
    assert res["mode"] == "heuristic" and res["reason"].startswith("Mémoire insuffisante")
    assert res["best"] == 1


def test_unexpected_brain_error_never_raises(job_dir, cfg, coach_cfg, monkeypatch):
    monkeypatch.setattr(sim, "_run_brain", lambda *a, **k: (_ for _ in ()).throw(ValueError("x")))
    monkeypatch.setattr("reel_predictor.brain.tribe.worker_installed", lambda: True)
    res = sim.simulate([GOOD, BAD], job_dir, cfg, coach_cfg)
    assert res["mode"] == "heuristic" and "ValueError" in res["reason"]
    assert [r["name"] for r in res["results"]] == ["variante 1", "variante 2"] and res["best"] in (0, 1)


def test_tts_failure_estimates_durations(job_dir, cfg, coach_cfg, monkeypatch):
    def no_say(*a, **k):
        raise RuntimeError("say absent")

    monkeypatch.setattr(sim, "_say", no_say)
    res = sim.simulate([GOOD], job_dir, cfg, coach_cfg)
    check_contract(res, 1)
    # raison citée : la voix de synthèse, ou le worker TRIBE s'il n'est pas encore installé (clone neuf)
    assert res["mode"] == "heuristic"
    assert "Voix de synthèse" in res["reason"] or "Worker TRIBE non installé" in res["reason"]


@pytest.mark.tribe
def test_simulate_real_tribe_no_video(job_dir, cfg, coach_cfg):
    res = sim.simulate([sim.baseline_variant(job_dir), GOOD, BAD], job_dir, cfg, coach_cfg)
    check_contract(res, 3)
    assert res["mode"] == "tribe_no_video", res["reason"]
    assert all(b["video_feature"] == "disabled" for b in res["brain_runs"])
    assert res["best"] == 1
