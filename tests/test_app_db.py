"""Tests de la base SQLite de l'app (app/db.py)."""
from __future__ import annotations

import sqlite3

import pytest

from app.db import Database


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / "test.sqlite")


def test_migrations_idempotent(tmp_path):
    path = tmp_path / "m.sqlite"
    d1 = Database(path)
    v1 = d1.schema_version()
    assert v1 >= 1
    assert d1.migrate() == v1  # rejouer ne change rien
    d2 = Database(path)  # réouverture
    assert d2.schema_version() == v1
    with sqlite3.connect(path) as c:
        tables = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        cols_reels = {r[1] for r in c.execute("PRAGMA table_info(reels)")}
        cols_ins = {r[1] for r in c.execute("PRAGMA table_info(insights)")}
    assert {"reels", "insights"} <= tables
    assert cols_reels >= {"id", "job_id", "filename", "sha256", "duration_s", "analyzed_at", "score",
                          "predicted_avg_watch_ratio", "result_dir"}
    assert cols_ins >= {"id", "reel_id", "window", "captured_at", "views", "reach", "likes", "comments", "shares",
                        "saves", "follows", "watch_time_s", "avg_watch_s", "skip_rate", "raw_ocr_json",
                        "validated_at"}


def test_upsert_reel_updates_in_place(db):
    rid = db.upsert_reel(job_id="20260101-120000-abcdef12", filename="a.mp4", duration_s=30.0, score=70,
                         predicted_avg_watch_ratio=0.5, result_dir="/tmp/x")
    rid2 = db.upsert_reel(job_id="20260101-120000-abcdef12", score=72)
    assert rid == rid2
    r = db.get_reel("20260101-120000-abcdef12")
    assert r["score"] == 72 and r["filename"] == "a.mp4" and r["duration_s"] == 30.0
    assert len(db.list_reels()) == 1
    assert db.get_reel("inconnu") is None


def test_insights_upsert_per_window(db):
    rid = db.upsert_reel(job_id="j1", filename="a.mp4", duration_s=20.0, score=60)
    row = db.save_insights(rid, "48h", {"views": 1000.4, "avg_watch_s": 8.5, "skip_rate": 40}, raw_ocr={"x": 1})
    assert row["views"] == 1000 and row["avg_watch_s"] == 8.5 and row["raw_ocr"] == {"x": 1}
    assert row["validated_at"]
    db.save_insights(rid, "48h", {"views": 1500})  # remplace la fenêtre 48h
    db.save_insights(rid, "7d", {"views": 5000, "reach": 3000})
    ins = db.get_insights(rid)
    assert set(ins) == {"48h", "7d"}
    assert ins["48h"]["views"] == 1500 and ins["48h"]["avg_watch_s"] is None
    assert ins["7d"]["reach"] == 3000
    joined = db.all_insights()
    assert len(joined) == 2 and all(j["job_id"] == "j1" for j in joined)


def test_insights_rejects_unknown_window(db):
    rid = db.upsert_reel(job_id="j2")
    with pytest.raises(ValueError):
        db.save_insights(rid, "30d", {"views": 1})
