"""Tests de la lecture des captures Instagram Insights (app/insights_ocr.py)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from app import insights_ocr as ocr
from app.insights_ocr import Box, extract_fields, match_label, parse_count, parse_duration, parse_percent

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _load_fixture(name: str):
    spec = importlib.util.spec_from_file_location(f"_fx_{name}", FIXTURES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------- parsing
@pytest.mark.parametrize("text,expected", [
    ("12,3 k", 12300), ("12,3k", 12300), ("12.3K", 12300), ("1,2 M", 1_200_000), ("1 234", 1234),
    ("1 234", 1234), ("1.234", 1234), ("1,234", 1234), ("8734", 8734), ("1 234 567", 1_234_567),
    ("1.234.567", 1_234_567), ("56", 56), ("0", 0),
])
def test_parse_count(text, expected):
    assert parse_count(text) == expected


@pytest.mark.parametrize("text", ["+12", "-5", "45 %", "abc", "", "2:35", "12 oct"])
def test_parse_count_rejects(text):
    assert parse_count(text) is None


@pytest.mark.parametrize("text,expected,unit", [
    ("2:35", 155, True), ("1:02:03", 3723, True), ("1 h 20 min", 4800, True), ("35 s", 35, True),
    ("4 min 12 s", 252, True), ("1 j 2 h 3 min", 93780, True), ("12,5 s", 12.5, True), ("4 sec", 4, True),
    ("1d 2h 3m", 93780, True), ("4 S", 4, True), ("12", 12, False),
])
def test_parse_duration(text, expected, unit):
    v, has_unit = parse_duration(text)
    assert v == pytest.approx(expected)
    assert has_unit is unit


@pytest.mark.parametrize("text", ["abc", "+5 s", "12 %", ""])
def test_parse_duration_rejects(text):
    assert parse_duration(text)[0] is None


@pytest.mark.parametrize("text,expected", [("12 %", 12.0), ("45,2%", 45.2), ("0 %", 0.0), ("100 %", 100.0)])
def test_parse_percent(text, expected):
    assert parse_percent(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["+5 %", "-3 %", "120 %", "45"])
def test_parse_percent_rejects(text):
    assert parse_percent(text) is None


# ---------------------------------------------------------------------------- libellés
@pytest.mark.parametrize("text,field", [
    ("Vues", "views"), ("Views", "views"), ("Comptes touchés", "reach"), ("Accounts reached", "reach"),
    ("Durée de visionnage", "watch_time_s"), ("Watch time", "watch_time_s"),
    ("Durée moyenne de visionnage", "avg_watch_s"), ("Duree moyenne de visionnage", "avg_watch_s"),
    ("Average watch time", "avg_watch_s"), ("Taux de skip", "skip_rate"), ("Skip rate", "skip_rate"),
    ("J'aime", "likes"), ("J’aime", "likes"), ("Likes", "likes"), ("Commentaires", "comments"),
    ("Partages", "shares"), ("Enregistrements", "saves"), ("Saves", "saves"), ("Abonnements", "follows"),
    ("Follows", "follows"), ("Interactions", "interactions"), ("Duréee moyenne de visionage", "avg_watch_s"),
])
def test_match_label(text, field):
    m = match_label(ocr.normalize(text))
    assert m is not None and m[0] == field


@pytest.mark.parametrize("text", ["Abonnés", "Statistiques du reel", "9:41", "Aperçu"])
def test_match_label_ignores(text):
    assert match_label(ocr.normalize(text)) is None


# ---------------------------------------------------------------------------- géométrie (sans OCR)
def _box(text, x1, y1, x2, y2, conf=0.99):
    return Box(text, conf, x1, y1, x2, y2)


def test_extract_same_line_below_inline_and_sentence():
    boxes = [
        _box("Vues", 50, 100, 150, 140), _box("12,3 k", 900, 100, 1020, 140),
        _box("Durée moyenne de visionnage", 50, 300, 600, 340), _box("4 s", 50, 360, 150, 420),
        _box("Partages 78", 50, 500, 300, 540),
        _box("Le taux de skip de ce reel est de 45,2 %", 50, 700, 1000, 740),
        _box("Comptes touchés", 50, 900, 380, 940), _box("+12 %", 900, 900, 1020, 940),  # variation ignorée
    ]
    f = extract_fields(boxes)
    assert f["views"]["value"] == 12300 and f["views"]["position"] == "right"
    assert f["avg_watch_s"]["value"] == 4 and f["avg_watch_s"]["position"] == "below"
    assert f["shares"]["value"] == 78 and f["shares"]["position"] == "inline"
    assert f["skip_rate"]["value"] == pytest.approx(45.2)
    assert "reach" not in f
    for v in f.values():
        assert 0 <= v["confidence"] <= 1 and v["level"] in ("haute", "moyenne", "faible")


def test_extract_prefers_nearest_value_on_line():
    boxes = [_box("J'aime", 50, 100, 180, 140), _box("1 234", 600, 100, 700, 140), _box("99", 950, 100, 1000, 140)]
    assert extract_fields(boxes)["likes"]["value"] == 1234


def test_merge_results_keeps_most_confident():
    a = {"fields": {"views": {"value": 100, "confidence": 0.5}}, "raw_text": "A", "boxes": [], "window_guess": None}
    b = {"fields": {"views": {"value": 120, "confidence": 0.9}, "likes": {"value": 5, "confidence": 0.8}},
         "raw_text": "B", "boxes": [], "window_guess": "7d"}
    m = ocr.merge_results([a, b])
    assert m["fields"]["views"]["value"] == 120 and 100 in m["fields"]["views"]["alternatives"]
    assert m["fields"]["likes"]["value"] == 5
    assert m["window_guess"] == "7d"
    assert "Capture 1" in m["raw_text"] and "Capture 2" in m["raw_text"]


def test_guess_window():
    assert ocr.guess_window("11 oct. · 7 derniers jours") == "7d"
    assert ocr.guess_window("Last 7 days") == "7d"
    assert ocr.guess_window("rien") is None


# ---------------------------------------------------------------------------- OCR réel (RapidOCR)
rapidocr = pytest.importorskip("rapidocr_onnxruntime")


@pytest.mark.parametrize("dark", [False, True], ids=["clair", "sombre"])
def test_ocr_synthetic_insights_screenshot(tmp_path, dark):
    fx = _load_fixture("make_fake_insights")
    if fx.find_font(fx.FONT_CANDIDATES) is None:
        pytest.skip("aucune police système pour dessiner la capture")
    png = fx.make_insights_png(tmp_path / "insights.png", dark=dark)
    result = ocr.read_insights(png.read_bytes())
    got = {k: v["value"] for k, v in result["fields"].items()}
    for key, expected in fx.EXPECTED.items():
        assert got.get(key) == pytest.approx(expected), f"{key}: attendu {expected}, lu {got.get(key)}"
    assert result["window_guess"] == "7d"
    assert "Vues" in result["raw_text"]


def _real_captures() -> list[Path]:
    samples = ROOT / "data" / "samples"
    if not samples.is_dir():
        return []
    return sorted(p for p in samples.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp"))


@pytest.mark.skipif(not _real_captures(), reason="aucune vraie capture Insights dans data/samples/")
@pytest.mark.parametrize("capture", _real_captures(), ids=lambda p: p.name)
def test_ocr_real_captures_readonly(capture):
    """Vraies captures déposées par l'utilisateur : lecture seule, on vérifie seulement que tout est lu."""
    before = capture.stat().st_mtime
    result = ocr.read_insights(capture.read_bytes())
    assert isinstance(result["fields"], dict)
    assert result["raw_text"].strip()
    assert capture.stat().st_mtime == before
