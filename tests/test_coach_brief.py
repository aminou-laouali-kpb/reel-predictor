"""Diagnostic structuré du coach (`reel_predictor/coach/brief.py`) : local, sans réseau."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from reel_predictor import paths
from reel_predictor.coach.brief import (brief_json, build_brief, find_cta, find_dips, join_words, number_salience,
                                        split_sentences, text_cards)

ERASMUS = paths.RESULTS_DIR / "20261012-111457-f4e96d26"
IDX = ("attention", "action", "tension", "manipulation", "emotion", "memory")


def coach_cfg() -> dict:
    with open(paths.CONFIG_DIR / "coach.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# Script synthétique : accroche, chiffre, développement, creux (plan long), CTA collé par l'OCR
SCRIPT = [
    (0, ["Tu", "veux", "étudier"], "ETUDIER EN EUROPE"),
    (1, ["en", "Europe", "gratuitement", "?"], "ETUDIER EN EUROPE"),
    (2, ["Voici", "comment."], "ETUDIER EN EUROPE"),
    (3, ["La", "bourse", "paie"], ""),
    (4, ["1400", "euros", "par", "mois."], "1400 € / MOIS"),
    (5, ["C", "'est", "pour", "tout"], "1400 € / MOIS"),
    (6, ["le", "monde", "ou", "presque."], ""),
    (7, ["Il", "faut", "une", "licence"], ""),
    (8, ["et", "de", "l", "'anglais."], ""),
    (9, ["Les", "dossiers", "ouvrent"], ""),
    (10, ["en", "octobre."], ""),
    (11, ["Prends", "mon", "guide"], "PRENDREMONGUIDE"),
    (12, ["lien", "en", "bio."], "PRENDREMONGUIDE"),
]


def make_job(tmp: Path, *, with_words_json: bool = False) -> Path:
    d = tmp / "20260101-120000-deadbeef"
    d.mkdir(parents=True)
    T = len(SCRIPT)
    rows = []
    cuts = [3.2, 7.1]
    for t, words, text in SCRIPT:
        att = 0.62 if t < 7 else (0.44 if t in (8, 9, 10) else 0.6)
        since = t + 1 - max([0.0] + [c for c in cuts if c <= t + 1])
        rows.append({
            "t": t, "indices": {k: (att if k == "attention" else 0.5) for k in IDX},
            "cut": any(t <= c < t + 1 for c in cuts), "since_cut": round(since, 2), "faces": 1, "face_area": 0.1,
            "words": words, "speech_rate": 3.0, "text": text, "text_words": len(text.split()),
            "text_new": t == 0 or text != SCRIPT[t - 1][2], "audio_rms": 0.5, "silence": False,
            "retention": round(1 - 0.04 * (t + 1), 3), "flags": [],
        })
    with open(d / "timeline.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {
        "version": 1, "job_id": d.name, "video": {"duration_s": float(T), "fps": 30.0},
        "score": 60, "score_breakdown": {"index_means": {k: 0.5 for k in IDX}},
        "retention": {"curve": [1.0] + [r["retention"] for r in rows], "avg_watch_ratio": 0.6, "completion": 0.4},
        "recommendations": [{"t": 8.0, "t_end": 11.0, "rule": "attention_drop_no_cut", "severity": "high",
                             "message": "Couper à 8 s"}],
        "cuts": cuts, "transcript": {"language": "fr", "text": "..."},
        "brain": {"available": True, "provider": "tribe", "degraded": False, "confidence": {"attention": "medium"}},
        "config": {"weights": {"cta_keywords": ["abonne-toi", "lien en bio"]}},
    }
    (d / "summary.json").write_text(json.dumps(summary, ensure_ascii=False), encoding="utf-8")
    if with_words_json:
        words = []
        for t, ws, _ in SCRIPT:
            for i, w in enumerate(ws):
                s = t + i / len(ws)
                words.append({"w": w, "start": round(s, 3), "end": round(s + 0.8 / len(ws), 3)})
        (d / "words.json").write_text(json.dumps({"language": "fr", "duration_s": T, "words": words}), encoding="utf-8")
    return d


def test_join_and_split_words():
    assert join_words(["C", "'est", "la", "bourse", "d", "'un", "pays"]) == "C'est la bourse d'un pays"
    assert join_words(["c", "-à", "-dire", "que"]) == "c-à-dire que"
    words = [{"w": w, "start": i * 0.3, "end": i * 0.3 + 0.25} for i, w in enumerate("Bonjour à tous. Voici la suite".split())]
    sents = split_sentences(words)
    assert [s["text"] for s in sents] == ["Bonjour à tous.", "Voici la suite"]
    # pause longue => nouvelle phrase même sans ponctuation
    words2 = [{"w": "un", "start": 0.0, "end": 0.2}, {"w": "deux", "start": 2.0, "end": 2.2}]
    assert len(split_sentences(words2)) == 2


def test_cta_numbers_cards_and_dips():
    from reel_predictor.coach.brief import _cta_keywords

    kw = _cta_keywords({"config": {"weights": {"cta_keywords": ["abonne-toi"]}}})
    assert "guide" in find_cta("TMAS QUOI TU PRENDREMONGUIDE", kw)  # mots collés par l'OCR
    assert "abonne-toi" in find_cta("Abonne-toi pour la suite", kw)
    assert find_cta("Les dossiers ouvrent en octobre", kw) == []
    assert number_salience("1400 euros par mois") == 2
    assert number_salience("au moins 3 pays") == 1
    assert number_salience("aucun chiffre ici") == 0
    rows = [{"t": t, "text": txt, "text_words": len(txt.split()), "text_new": True}
            for t, txt in enumerate(["UN DEUX", "un deux trois quatre cinq six sept huit neuf dix onze douze"])]
    cards = text_cards(rows, reading_wps=3.5, max_words=8)
    assert not cards[0]["too_long"] and cards[1]["too_long"] and cards[1]["too_fast"]
    att = [0.7, 0.7, 0.7, 0.45, 0.44, 0.7, 0.7]
    dips = find_dips([{"t": t, "indices": {"attention": v}, "since_cut": 1.0} for t, v in enumerate(att)], 0.5)
    assert dips and dips[0]["t"] == 3 and dips[0]["t_end"] == 5 and dips[0]["kind"] == "sous le plancher"


@pytest.mark.parametrize("with_words_json", [False, True])
def test_build_brief_synthetic(tmp_path, with_words_json):
    d = make_job(tmp_path, with_words_json=with_words_json)
    cfg = coach_cfg()
    b = build_brief(d, cfg)
    segs = b["segments"]
    assert segs and segs[0]["t"] == 0.0 and segs[0]["role"] == "hook"
    assert segs[-1]["t_end"] == pytest.approx(len(SCRIPT))
    roles = [s["role"] for s in segs]
    assert "fact" in roles and roles[-1] == "cta"
    fact = next(s for s in segs if s["role"] == "fact")
    assert "1400" in fact["voice"]
    cta = segs[-1]
    assert "guide" in cta["cta"]
    # creux d'attention 8–11 s sous le plancher, plan long > 4 s, recommandation recopiée
    assert any(dp["t"] <= 8 and dp["t_end"] >= 10 for dp in b["dips"])
    assert any(ls["s"] >= 4 for ls in b["long_shots"])
    assert b["recommendations"][0]["rule"] == "attention_drop_no_cut"
    ev = b["evaluation"]
    assert {m["name"] for m in ev["moments"]} == {"hook", "fact", "cta"}
    hook = next(m for m in ev["moments"] if m["name"] == "hook")
    assert hook["t"] == 0 and hook["t_end"] == 3 and hook["met"] is False
    assert hook["gap"]["attention"] == pytest.approx(0.9 - 0.62, abs=0.01)
    assert 8 in ev["seconds_below_floor"]
    assert len(b["series"]["attention"]) == len(SCRIPT)
    # sérialisation compacte, déterministe, sans NaN
    s1, s2 = brief_json(b), brief_json(build_brief(d, cfg))
    assert s1 == s2 and "NaN" not in s1
    json.loads(s1)


def test_brief_without_speech(tmp_path):
    d = make_job(tmp_path)
    rows = [json.loads(line) for line in (d / "timeline.jsonl").read_text().splitlines()]
    with open(d / "timeline.jsonl", "w") as fh:
        for r in rows:
            r["words"] = []
            fh.write(json.dumps(r) + "\n")
    b = build_brief(d, coach_cfg())
    assert len(b["segments"]) == 3  # un segment par plan (2 cuts)
    assert b["segments"][0]["role"] == "hook"


@pytest.mark.skipif(not (ERASMUS / "timeline.jsonl").is_file(), reason="résultat réel Erasmus absent")
def test_brief_real_erasmus():
    b = build_brief(ERASMUS, coach_cfg())
    s = brief_json(b)
    assert 6000 < len(s) < 24000  # ≈ 3–6 k tokens
    # les deux vrais problèmes du Reel : creux 19–24 s et 38–45 s, plans de 8–10 s sans cut
    assert any(dp["t"] <= 22 < dp["t_end"] for dp in b["dips"])
    assert any(dp["t"] <= 42 < dp["t_end"] for dp in b["dips"])
    assert any(ls["t"] <= 37 and ls["t_end"] >= 46 for ls in b["long_shots"])
    assert b["segments"][0]["role"] == "hook" and "1400" in b["segments"][0]["voice"]
    assert any(s["role"] == "cta" for s in b["segments"])
    assert b["evaluation"]["attainment"] < 1.0
