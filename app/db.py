"""Base SQLite locale (stdlib) : reels analysés + statistiques Instagram validées à la main.

Une connexion par opération (sûr entre threads). Migrations idempotentes via `PRAGMA user_version`.
Unités : durées en secondes, `skip_rate` en pourcentage (0..100), compteurs en entiers.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

WINDOWS = ("48h", "7d")
COUNT_FIELDS = ("views", "reach", "likes", "comments", "shares", "saves", "follows")
FLOAT_FIELDS = ("watch_time_s", "avg_watch_s", "skip_rate")
INSIGHT_FIELDS = COUNT_FIELDS + FLOAT_FIELDS

_MIGRATIONS: list[str] = [
    # v1 : schéma initial
    """
    CREATE TABLE IF NOT EXISTS reels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        job_id TEXT NOT NULL UNIQUE,
        filename TEXT,
        sha256 TEXT,
        duration_s REAL,
        analyzed_at TEXT,
        score REAL,
        predicted_avg_watch_ratio REAL,
        result_dir TEXT
    );
    CREATE TABLE IF NOT EXISTS insights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        reel_id INTEGER NOT NULL REFERENCES reels(id) ON DELETE CASCADE,
        "window" TEXT NOT NULL CHECK ("window" IN ('48h', '7d')),
        captured_at TEXT,
        views INTEGER, reach INTEGER, likes INTEGER, comments INTEGER, shares INTEGER,
        saves INTEGER, follows INTEGER,
        watch_time_s REAL, avg_watch_s REAL, skip_rate REAL,
        raw_ocr_json TEXT,
        validated_at TEXT
    );
    CREATE UNIQUE INDEX IF NOT EXISTS ux_insights_reel_window ON insights(reel_id, "window");
    CREATE INDEX IF NOT EXISTS ix_reels_analyzed_at ON reels(analyzed_at);
    """,
]

_REEL_COLUMNS = ("id", "job_id", "filename", "sha256", "duration_s", "analyzed_at", "score",
                 "predicted_avg_watch_ratio", "result_dir")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ migrations
    def schema_version(self) -> int:
        with self.connect() as c:
            return int(c.execute("PRAGMA user_version").fetchone()[0])

    def migrate(self) -> int:
        """Applique les migrations manquantes. Idempotent (rejouable sans effet)."""
        with self.connect() as c:
            current = int(c.execute("PRAGMA user_version").fetchone()[0])
            for version, script in enumerate(_MIGRATIONS, start=1):
                if version > current:
                    c.executescript(script)
                    c.execute(f"PRAGMA user_version = {version}")
            # Filet de sécurité : colonnes attendues (si une base a été créée à la main)
            for table, cols in (("reels", _REEL_COLUMNS),):
                have = {r[1] for r in c.execute(f"PRAGMA table_info({table})")}
                for col in cols:
                    if col not in have:
                        c.execute(f"ALTER TABLE {table} ADD COLUMN {col}")
            return int(c.execute("PRAGMA user_version").fetchone()[0])

    # ------------------------------------------------------------------ reels
    def upsert_reel(self, *, job_id: str, filename: str | None = None, sha256: str | None = None,
                    duration_s: float | None = None, analyzed_at: str | None = None, score: float | None = None,
                    predicted_avg_watch_ratio: float | None = None, result_dir: str | Path | None = None) -> int:
        with self.connect() as c:
            c.execute(
                """
                INSERT INTO reels (job_id, filename, sha256, duration_s, analyzed_at, score,
                                   predicted_avg_watch_ratio, result_dir)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    filename = COALESCE(excluded.filename, reels.filename),
                    sha256 = COALESCE(excluded.sha256, reels.sha256),
                    duration_s = COALESCE(excluded.duration_s, reels.duration_s),
                    analyzed_at = COALESCE(excluded.analyzed_at, reels.analyzed_at),
                    score = COALESCE(excluded.score, reels.score),
                    predicted_avg_watch_ratio = COALESCE(excluded.predicted_avg_watch_ratio,
                                                         reels.predicted_avg_watch_ratio),
                    result_dir = COALESCE(excluded.result_dir, reels.result_dir)
                """,
                (job_id, filename, sha256, duration_s, analyzed_at or now_iso(), score,
                 predicted_avg_watch_ratio, str(result_dir) if result_dir is not None else None),
            )
            return int(c.execute("SELECT id FROM reels WHERE job_id = ?", (job_id,)).fetchone()[0])

    def get_reel(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as c:
            row = c.execute("SELECT * FROM reels WHERE job_id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def list_reels(self) -> list[dict[str, Any]]:
        with self.connect() as c:
            rows = c.execute("SELECT * FROM reels ORDER BY analyzed_at DESC, id DESC").fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ insights
    def save_insights(self, reel_id: int, window: str, values: dict[str, Any], raw_ocr: Any = None,
                      captured_at: str | None = None) -> dict[str, Any]:
        """Enregistre (ou remplace) les statistiques validées d'un reel pour une fenêtre 48h/7d."""
        if window not in WINDOWS:
            raise ValueError(f"Fenêtre inconnue : {window!r} (attendu : 48h ou 7d)")
        clean: dict[str, Any] = {}
        for f in COUNT_FIELDS:
            v = values.get(f)
            clean[f] = None if v is None else int(round(float(v)))
        for f in FLOAT_FIELDS:
            v = values.get(f)
            clean[f] = None if v is None else float(v)
        raw = json.dumps(raw_ocr, ensure_ascii=False) if raw_ocr is not None else None
        with self.connect() as c:
            c.execute(
                f"""
                INSERT INTO insights (reel_id, "window", captured_at, {", ".join(INSIGHT_FIELDS)},
                                      raw_ocr_json, validated_at)
                VALUES (?, ?, ?, {", ".join("?" for _ in INSIGHT_FIELDS)}, ?, ?)
                ON CONFLICT(reel_id, "window") DO UPDATE SET
                    captured_at = excluded.captured_at,
                    {", ".join(f"{f} = excluded.{f}" for f in INSIGHT_FIELDS)},
                    raw_ocr_json = excluded.raw_ocr_json,
                    validated_at = excluded.validated_at
                """,
                (reel_id, window, captured_at or now_iso(), *[clean[f] for f in INSIGHT_FIELDS], raw, now_iso()),
            )
            row = c.execute('SELECT * FROM insights WHERE reel_id = ? AND "window" = ?', (reel_id, window)).fetchone()
            return _insight_dict(row)

    def get_insights(self, reel_id: int) -> dict[str, dict[str, Any]]:
        with self.connect() as c:
            rows = c.execute('SELECT * FROM insights WHERE reel_id = ?', (reel_id,)).fetchall()
            return {r["window"]: _insight_dict(r) for r in rows}

    def all_insights(self) -> list[dict[str, Any]]:
        """Toutes les statistiques validées, jointes à leur reel."""
        with self.connect() as c:
            rows = c.execute(
                """
                SELECT i.*, r.job_id, r.filename, r.score, r.duration_s, r.predicted_avg_watch_ratio, r.result_dir
                FROM insights i JOIN reels r ON r.id = i.reel_id
                ORDER BY r.analyzed_at, i."window"
                """
            ).fetchall()
            return [_insight_dict(r) for r in rows]


def _insight_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    raw = d.pop("raw_ocr_json", None)
    try:
        d["raw_ocr"] = json.loads(raw) if raw else None
    except ValueError:
        d["raw_ocr"] = raw
    return d
