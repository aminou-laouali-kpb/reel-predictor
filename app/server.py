"""Serveur FastAPI local (127.0.0.1:8765) — API du contrat §5 + fichiers statiques de l'UI.

Usage : `uv run python -m app` (voir app/__main__.py). Pour les tests : `create_app(...)` avec des
chemins temporaires et des fonctions injectées (analyse, calibration, OCR, santé).
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import re
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from reel_predictor import __version__ as PIPELINE_VERSION
from reel_predictor import paths

from . import __version__
from .db import COUNT_FIELDS, FLOAT_FIELDS, WINDOWS, Database
from .fake import fake_enabled
from .jobs import APP_JOB_FILE, JobManager, is_safe_id, read_json

COACH_STAMP_RE = re.compile(r"^\d{8}-\d{6}$")

log = logging.getLogger("reel_predictor.app")

HOST = "127.0.0.1"
PORT = 8765
STATIC_DIR = Path(__file__).resolve().parent / "static"
MAX_OCR_IMAGES = 3
ALLOWED_HOSTS = {"127.0.0.1", "localhost", "[::1]", "testserver"}


# ---------------------------------------------------------------------------- utilitaires
def _sanitize(obj: Any) -> Any:
    """Remplace NaN/inf par null (JSON strict pour le navigateur)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


def _host_only(h: str) -> str:
    h = h.strip().lower().split("/", 1)[0]
    if h.startswith("["):
        return h.split("]", 1)[0] + "]"
    return h.rsplit(":", 1)[0] if ":" in h else h


def _load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_timeline(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("t", 0))
    return rows


def _num(v: Any) -> float | None:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


class InsightsIn(BaseModel):
    job_id: str
    window: Literal["48h", "7d"]
    validated: bool = Field(False, description="validation manuelle obligatoire (case cochée dans l'UI)")
    captured_at: str | None = None
    views: float | None = None
    reach: float | None = None
    likes: float | None = None
    comments: float | None = None
    shares: float | None = None
    saves: float | None = None
    follows: float | None = None
    watch_time_s: float | None = None
    avg_watch_s: float | None = None
    skip_rate: float | None = None
    raw_ocr: Any = None


# ---------------------------------------------------------------------------- application
def create_app(*, results_dir: Path | None = None, uploads_dir: Path | None = None, db_path: Path | None = None,
               analyze_fn: Callable[..., Any] | None = None, calibrate_fn: Callable[..., Any] | None = None,
               health_fn: Callable[[], dict[str, Any]] | None = None,
               ocr_fn: Callable[[bytes], dict[str, Any]] | None = None,
               use_fake: bool | None = None,
               coach_fn: Callable[..., Any] | None = None,
               coach_key_fn: Callable[[], bool] | None = None,
               coach_cfg_fn: Callable[[], dict[str, Any]] | None = None) -> FastAPI:
    results_dir = Path(results_dir or paths.RESULTS_DIR)
    uploads_dir = Path(uploads_dir or paths.UPLOADS_DIR)
    db = Database(db_path or paths.DB_FILE)
    manager = JobManager(results_dir, uploads_dir, db, analyze_fn=analyze_fn, use_fake=use_fake)
    try:
        manager.recover()
    except Exception as exc:  # le serveur doit démarrer quoi qu'il arrive
        log.warning("Resynchronisation des résultats impossible : %s", exc)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        manager.shutdown(timeout=0.5)

    app = FastAPI(title="Reel Predictor", version=__version__, docs_url="/api/docs", redoc_url=None,
                  openapi_url="/api/openapi.json", lifespan=lifespan)
    app.state.db = db
    app.state.jobs = manager
    app.state.results_dir = results_dir

    # ------------------------------------------------------------------ sécurité locale
    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = _host_only(request.headers.get("host") or "")
        if host and host not in ALLOWED_HOSTS:  # protège contre le « DNS rebinding »
            return JSONResponse({"detail": "Hôte non autorisé."}, status_code=403)
        origin = request.headers.get("origin")
        if request.method not in ("GET", "HEAD", "OPTIONS") and origin and origin != "null":
            if _host_only(origin.split("://", 1)[-1]) not in ALLOWED_HOSTS:
                return JSONResponse({"detail": "Origine non autorisée."}, status_code=403)
        response = await call_next(request)
        path = request.url.path
        if not path.startswith("/media/") and not path.endswith("/brain"):
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    # ------------------------------------------------------------------ santé
    health_cache: dict[str, Any] = {"t": 0.0, "value": None}
    health_lock = threading.Lock()

    def _pipeline_status() -> dict[str, Any]:
        if health_fn is not None:
            return {"available": True, "status": health_fn(), "error": None}
        try:
            from reel_predictor.config import load_config
            from reel_predictor.health import status
        except ImportError as exc:
            return {"available": False, "status": None, "error": f"Pipeline indisponible : {exc}"}
        try:
            return {"available": True, "status": status(load_config()), "error": None}
        except Exception as exc:
            return {"available": True, "status": None, "error": f"Statut du pipeline illisible : {exc}"}

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        with health_lock:
            if health_cache["value"] is None or time.monotonic() - health_cache["t"] > 30:
                health_cache["value"] = _pipeline_status()
                health_cache["t"] = time.monotonic()
            pipe = health_cache["value"]
        fake = fake_enabled() if use_fake is None else use_fake
        try:  # sans importer le pipeline (cv2, whisper… : lourd)
            pipeline_module = importlib.util.find_spec("reel_predictor.pipeline") is not None
        except (ImportError, ValueError):
            pipeline_module = False
        return _sanitize({
            "ok": True,
            "version": __version__,
            "pipeline_version": PIPELINE_VERSION,
            "fake_mode": bool(fake and analyze_fn is None),
            "pipeline_available": pipeline_module or analyze_fn is not None,
            "pipeline_error": pipe["error"],
            "status": pipe["status"],
            "queue": manager.active(),
            "assets": {
                "uplot": (STATIC_DIR / "vendor" / "uPlot.iife.min.js").is_file(),
                "brain_mesh": (STATIC_DIR / "brain_mesh.json").is_file(),
            },
            "license_notice": "TRIBE v2 (Meta) : licence CC BY-NC 4.0 — usage non commercial uniquement.",
            "claude_api": _has_key(),
            "coach": _coach_cost_info(),
        })

    # ------------------------------------------------------------------ coach script & montage (API Claude)
    def _has_key(refresh: bool = False) -> bool:
        try:
            if coach_key_fn is not None:
                return bool(coach_key_fn())
            from reel_predictor.coach.keys import has_key

            return has_key(refresh=refresh)
        except Exception:
            return False

    def _coach_cfg() -> dict[str, Any] | None:
        try:
            if coach_cfg_fn is not None:
                return coach_cfg_fn()
            from reel_predictor.coach.service import load_coach_config

            return load_coach_config()
        except Exception as exc:
            log.warning("config/coach.yaml illisible : %s", exc)
            return None

    def _coach_files(d: Path) -> list[Path]:
        cdir = d / "coach"
        if not cdir.is_dir():
            return []
        return sorted((p for p in cdir.glob("*.json") if COACH_STAMP_RE.match(p.stem)), reverse=True)

    def _coach_cost_info() -> dict[str, Any]:
        cfg = _coach_cfg()
        costs = []
        try:
            for d in results_dir.iterdir():
                for f in _coach_files(d)[:20]:
                    try:
                        past = _load_json(f)
                        c = _num(past.get("cost_usd"))
                        # ne moyenner que les générations faites avec le même nombre de passes
                        same = int((past.get("config") or {}).get("rounds") or 1) == int((cfg or {}).get("rounds") or 1)
                    except Exception:
                        c, same = None, False
                    if c is not None and c > 0 and same:
                        costs.append(c)
        except OSError:
            pass
        info: dict[str, Any] = {"n_generations": len(costs)}
        if cfg:
            llm_cfg = cfg.get("llm") or {}
            info.update(model=llm_cfg.get("model"), effort=llm_cfg.get("effort"),
                        max_cost_usd=_num(llm_cfg.get("max_cost_usd")), n_variants=cfg.get("n_variants"),
                        rounds=cfg.get("rounds"))
        if costs:
            info["estimated_cost_usd"] = round(sum(costs) / len(costs), 4)
            info["basis"] = f"moyenne des {len(costs)} génération(s) précédente(s)"
        elif cfg:
            try:
                from reel_predictor.coach.llm import typical_cost

                info["estimated_cost_usd"] = typical_cost(cfg)
                info["basis"] = "estimation a priori (diagnostic typique)"
            except Exception:
                info["estimated_cost_usd"] = None
        return info

    def _coach_item(f: Path, data: dict[str, Any]) -> dict[str, Any]:
        out = data.get("output") or {}
        rk = data.get("ranking") or {}
        sim = data.get("simulation") or {}
        res = sim.get("results") or []
        bv = rk.get("best_variant")
        best = None
        if isinstance(bv, int) and len(res) > bv + 1:
            r = res[bv + 1]
            best = {"index": bv, "angle": ((out.get("variants") or [{}] * (bv + 1))[bv] or {}).get("angle"),
                    "score": r.get("score"), "attainment": (r.get("evaluation") or {}).get("attainment")}
        return {
            "coach_id": data.get("coach_id") or f"{f.parent.parent.name}.{f.stem}",
            "created_at": data.get("created_at"),
            "model": data.get("model"),
            "cost_usd": data.get("cost_usd"),
            "n_variants": len(out.get("variants") or []),
            "simulation_mode": sim.get("mode") if sim else None,
            "original": ({"score": res[0].get("score"), "attainment": (res[0].get("evaluation") or {}).get("attainment")}
                         if res else None),
            "best": best,
            "n_warnings": len(data.get("warnings") or []),
        }

    def _coach_path(coach_id: str) -> tuple[str, Path]:
        job_id, _, stamp = coach_id.rpartition(".")
        if not job_id or not is_safe_id(job_id) or not COACH_STAMP_RE.match(stamp):
            raise HTTPException(404, "Identifiant de génération invalide.")
        d = (results_dir / job_id).resolve()
        if d.parent != results_dir.resolve():
            raise HTTPException(404, "Identifiant de génération invalide.")
        return job_id, d / "coach" / f"{stamp}.json"

    @app.post("/api/results/{job_id}/coach")
    def start_coach(job_id: str) -> dict[str, str]:
        d = _result_dir(job_id)
        if not (d / "summary.json").is_file() or not (d / "timeline.jsonl").is_file():
            raise HTTPException(409, "Analyse incomplète : relancez l'analyse avant de demander des suggestions.")
        if not _has_key(refresh=True):
            from reel_predictor.coach.keys import NO_KEY_MESSAGE

            raise HTTPException(400, NO_KEY_MESSAGE)
        fn = coach_fn
        if fn is None:
            from reel_predictor.coach.service import run_coach

            def fn(job_dir: Path, progress: Callable[..., None], stamp: str) -> dict[str, Any]:
                return run_coach(job_dir, progress, stamp=stamp)
        return {"coach_id": manager.submit_coach(job_id, d, fn)}

    @app.get("/api/results/{job_id}/coach")
    def coach_history(job_id: str) -> dict[str, Any]:
        d = _result_dir(job_id)
        items = []
        for f in _coach_files(d):
            try:
                items.append(_coach_item(f, _load_json(f)))
            except Exception as exc:
                log.warning("Génération coach illisible %s : %s", f, exc)
        cfg = _coach_cfg()
        estimate = None
        if cfg and (d / "timeline.jsonl").is_file():
            try:
                from reel_predictor.coach.brief import build_brief
                from reel_predictor.coach.llm import estimate_offline

                estimate = estimate_offline(build_brief(d, cfg), cfg)
            except Exception as exc:
                log.warning("Estimation du coût coach impossible : %s", exc)
        pending = [t for t in manager.coach_tasks(job_id) if t["status"] in ("queued", "running", "error")]
        return _sanitize({"items": items, "pending": pending, "claude_api": _has_key(refresh=True), "estimate": estimate,
                          "config": _coach_cost_info()})

    @app.get("/api/coach/{coach_id}")
    def coach_status(coach_id: str) -> JSONResponse:
        job_id, f = _coach_path(coach_id)
        st = manager.get_coach(coach_id)
        if st is None:
            if not f.is_file():
                raise HTTPException(404, "Génération inconnue.")
            st = {"coach_id": coach_id, "job_id": job_id, "status": "done", "stage": "write",
                  "stage_label": "Écriture du rapport", "pct": 1.0, "message": "Suggestions prêtes", "eta_s": 0.0,
                  "error": None}
        st = dict(st)
        st["result"] = None
        if st["status"] == "done" and f.is_file():
            st["result"] = _load_json(f)
        return JSONResponse(_sanitize(st))

    @app.get("/api/coach/{coach_id}/markdown")
    def coach_markdown(coach_id: str) -> Response:
        job_id, f = _coach_path(coach_id)
        md = f.with_suffix(".md")
        if md.is_file():
            text = md.read_text(encoding="utf-8")
        elif f.is_file():
            from reel_predictor.coach.service import to_markdown

            text = to_markdown(_load_json(f))
        else:
            raise HTTPException(404, "Génération inconnue.")
        name = f"coach-{job_id}-{f.stem}.md"
        return Response(text, media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    # ------------------------------------------------------------------ analyses
    @app.post("/api/analyze")
    async def analyze_endpoint(file: UploadFile = File(...)) -> dict[str, str]:
        if not file.filename:
            raise HTTPException(400, "Aucun fichier reçu.")
        try:
            job_id = await run_in_threadpool(manager.submit_upload, file.file, file.filename)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        finally:
            await file.close()
        return {"job_id": job_id}

    @app.get("/api/jobs")
    def list_jobs() -> list[dict[str, Any]]:
        return _sanitize(manager.list_jobs())

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str) -> dict[str, Any]:
        st = manager.get(job_id)
        if st is None:
            raise HTTPException(404, "Analyse inconnue.")
        return _sanitize(st)

    def _result_dir(job_id: str) -> Path:
        if not is_safe_id(job_id):
            raise HTTPException(404, "Identifiant invalide.")
        d = (results_dir / job_id).resolve()
        if d.parent != results_dir.resolve() or not d.is_dir():
            raise HTTPException(404, "Résultat introuvable.")
        return d

    def _result_file(job_id: str, name: str) -> Path:
        f = _result_dir(job_id) / name
        if not f.is_file():
            st = manager.get(job_id)
            if st and st["status"] in ("queued", "running"):
                raise HTTPException(409, "Analyse en cours : résultat pas encore disponible.")
            raise HTTPException(404, f"{name} introuvable pour cette analyse.")
        return f

    @app.get("/api/results/{job_id}/summary")
    def get_summary(job_id: str) -> JSONResponse:
        f = _result_file(job_id, "summary.json")
        data = _load_json(f)
        meta_file = f.parent / APP_JOB_FILE
        if meta_file.is_file():
            try:
                data.setdefault("app", read_json(meta_file))
            except ValueError:
                pass
        return JSONResponse(_sanitize(data))

    @app.get("/api/results/{job_id}/timeline")
    def get_timeline(job_id: str) -> JSONResponse:
        return JSONResponse(_sanitize(_load_timeline(_result_file(job_id, "timeline.jsonl"))))

    @app.get("/api/results/{job_id}/brain")
    def get_brain(job_id: str) -> FileResponse:
        return FileResponse(_result_file(job_id, "brain.u8"), media_type="application/octet-stream")

    @app.get("/api/results/{job_id}/brain_meta")
    def get_brain_meta(job_id: str) -> JSONResponse:
        return JSONResponse(_sanitize(_load_json(_result_file(job_id, "brain_meta.json"))))

    @app.get("/media/{job_id}.mp4")
    def get_media(job_id: str) -> FileResponse:
        d = _result_dir(job_id)
        video = d / "video.mp4"
        if not video.is_file():
            others = sorted(p for p in d.iterdir() if p.suffix.lower() in (".mp4", ".mov", ".m4v"))
            if not others:
                raise HTTPException(404, "Vidéo introuvable.")
            video = others[0]
        # Starlette (>= 0.39) gère nativement Range / If-Range / 206 / 416 dans FileResponse.
        return FileResponse(video, media_type="video/mp4")

    # ------------------------------------------------------------------ reels & insights
    def _summary_of(reel: dict[str, Any]) -> dict[str, Any] | None:
        d = Path(reel.get("result_dir") or results_dir / reel["job_id"])
        try:
            return _load_json(d / "summary.json")
        except Exception:
            return None

    def _reel_view(reel: dict[str, Any], insights: dict[str, Any] | None = None) -> dict[str, Any]:
        summary = _summary_of(reel)
        ins = insights if insights is not None else db.get_insights(reel["id"])
        brain = (summary or {}).get("brain") or {}
        return {
            "job_id": reel["job_id"],
            "filename": reel["filename"],
            "analyzed_at": reel["analyzed_at"],
            "duration_s": reel["duration_s"],
            "score": reel["score"],
            "predicted_avg_watch_ratio": reel["predicted_avg_watch_ratio"],
            "result_available": summary is not None,
            "brain_available": bool(brain.get("available")),
            "degraded": bool(brain.get("degraded")),
            "fake": bool((summary or {}).get("fake")),
            "insights": {w: ins.get(w) for w in WINDOWS},
        }

    def _actual_ratio(reel: dict[str, Any], ins: dict[str, Any]) -> tuple[float | None, str | None]:
        dur = _num(reel.get("duration_s"))
        for w in ("7d", "48h"):
            row = ins.get(w)
            if row and _num(row.get("avg_watch_s")) is not None and dur and dur > 0:
                return max(0.0, min(1.0, float(row["avg_watch_s"]) / dur)), w
        return None, None

    def _calibration_pairs() -> list[tuple[Path, float]]:
        pairs = []
        for reel in db.list_reels():
            ins = db.get_insights(reel["id"])
            ratio, _w = _actual_ratio(reel, ins)
            d = Path(reel.get("result_dir") or results_dir / reel["job_id"])
            if ratio is not None and (d / "summary.json").is_file():
                pairs.append((d, ratio))
        return pairs

    def _calibration_info() -> dict[str, Any]:
        current = None
        if paths.CALIBRATION_FILE.is_file():
            try:
                current = _load_json(paths.CALIBRATION_FILE)
            except Exception:
                current = None
        n = len(_calibration_pairs())
        return {"n_eligible": n, "min_required": 3, "can_calibrate": n >= 3, "current": current}

    @app.get("/api/reels")
    def list_reels() -> dict[str, Any]:
        reels = [_reel_view(r) for r in db.list_reels()]
        return _sanitize({"reels": reels, "calibration": _calibration_info()})

    @app.post("/api/insights/ocr")
    def insights_ocr(file: list[UploadFile] = File(...), window: str | None = Form(None)) -> dict[str, Any]:
        if len(file) > MAX_OCR_IMAGES:
            raise HTTPException(400, f"{MAX_OCR_IMAGES} captures maximum.")
        from . import insights_ocr as ocr

        reader = ocr_fn or ocr.read_insights
        results = []
        for f in file:
            data = f.file.read()
            if not data:
                raise HTTPException(400, f"Capture vide : {f.filename}")
            try:
                results.append(reader(data))
            except Exception as exc:
                raise HTTPException(400, f"Lecture impossible de {f.filename} : {exc}") from exc
        merged = ocr.merge_results(results)
        if window in WINDOWS:
            merged["window"] = window
        merged["validated"] = False
        merged["notice"] = "Valeurs lues automatiquement : vérifiez chaque champ avant de valider."
        return _sanitize(merged)

    @app.post("/api/insights")
    def save_insights(body: InsightsIn) -> dict[str, Any]:
        if not body.validated:
            raise HTTPException(400, "Validation manuelle requise : vérifiez les valeurs puis cochez la case.")
        reel = db.get_reel(body.job_id)
        if reel is None:
            d = results_dir / body.job_id
            if is_safe_id(body.job_id) and (d / "summary.json").is_file():
                manager.register_result(body.job_id, d)
                reel = db.get_reel(body.job_id)
        if reel is None:
            raise HTTPException(404, "Reel inconnu : analysez-le d'abord.")
        values = body.model_dump()
        errors = []
        for f in COUNT_FIELDS + FLOAT_FIELDS:
            v = values.get(f)
            if v is not None and (not math.isfinite(v) or v < 0):
                errors.append(f"{f} doit être un nombre positif")
        sr = values.get("skip_rate")
        if sr is not None and sr > 100:
            errors.append("skip_rate est un pourcentage (0 à 100)")
        aw = values.get("avg_watch_s")
        if aw is not None and reel.get("duration_s") and aw > 10 * float(reel["duration_s"]):
            errors.append("durée moyenne de visionnage incohérente avec la durée du reel")
        if all(values.get(f) is None for f in COUNT_FIELDS + FLOAT_FIELDS):
            errors.append("aucune valeur saisie")
        if errors:
            raise HTTPException(400, "Valeurs invalides : " + " ; ".join(errors))
        row = db.save_insights(reel["id"], body.window, values, raw_ocr=body.raw_ocr, captured_at=body.captured_at)
        return _sanitize({"ok": True, "insight": row, "reel": _reel_view(reel)})

    @app.get("/api/reels/{job_id}/comparison")
    def comparison(job_id: str) -> dict[str, Any]:
        reel = db.get_reel(job_id)
        if reel is None:
            raise HTTPException(404, "Reel inconnu.")
        summary = _summary_of(reel) or {}
        ins = db.get_insights(reel["id"])
        dur = _num(reel.get("duration_s")) or _num((summary.get("video") or {}).get("duration_s"))
        retention = summary.get("retention") or {}
        pred_ratio = _num(retention.get("avg_watch_ratio")) or _num(reel.get("predicted_avg_watch_ratio"))
        pred_s = _num(retention.get("avg_watch_s")) or (pred_ratio * dur if pred_ratio is not None and dur else None)
        actual: dict[str, Any] = {}
        for w in WINDOWS:
            row = ins.get(w)
            if row:
                aw = _num(row.get("avg_watch_s"))
                actual[w] = dict(row, avg_watch_ratio=(max(0.0, min(1.0, aw / dur)) if aw is not None and dur else None))
            else:
                actual[w] = None
        real_ratio, best_w = _actual_ratio(reel, ins)
        scatter = []
        for r in db.list_reels():
            r_ins = db.get_insights(r["id"])
            for w in ("7d", "48h"):
                row = r_ins.get(w)
                if row and (row.get("views") is not None or row.get("reach") is not None):
                    rr, _ = _actual_ratio(r, r_ins)
                    scatter.append({"job_id": r["job_id"], "filename": r["filename"], "score": r["score"],
                                    "views": row.get("views"), "reach": row.get("reach"), "window": w,
                                    "predicted_avg_watch_ratio": r["predicted_avg_watch_ratio"],
                                    "actual_avg_watch_ratio": rr})
                    break
        return _sanitize({
            "job_id": job_id,
            "filename": reel["filename"],
            "duration_s": dur,
            "predicted": {
                "score": reel["score"],
                "curve": retention.get("curve"),
                "avg_watch_ratio": pred_ratio,
                "avg_watch_s": pred_s,
                "completion": retention.get("completion"),
            },
            "actual": actual,
            "best_window": best_w,
            "actual_avg_watch_ratio": real_ratio,
            "ratio_pred_vs_real": (pred_ratio / real_ratio) if (pred_ratio is not None and real_ratio) else None,
            "error_pts": ((pred_ratio - real_ratio) * 100) if (pred_ratio is not None and real_ratio is not None)
            else None,
            "scatter": scatter,
            "calibration": _calibration_info(),
        })

    @app.post("/api/calibrate")
    def calibrate_endpoint() -> dict[str, Any]:
        pairs = _calibration_pairs()
        if len(pairs) < 3:
            raise HTTPException(400, f"Au moins 3 reels avec une durée moyenne de visionnage réelle sont nécessaires "
                                     f"(actuellement : {len(pairs)}).")
        fn = calibrate_fn
        if fn is None:
            try:
                from reel_predictor.calibrate import calibrate as fn  # type: ignore[no-redef]
            except ImportError as exc:
                raise HTTPException(503, f"Module de calibration indisponible : {exc}") from exc
        try:
            result = fn(pairs)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            log.exception("Calibration en échec")
            raise HTTPException(500, f"Échec de la calibration : {exc}") from exc
        return _sanitize({"ok": True, "result": result, "pairs": [{"result_dir": str(p), "actual_ratio": r}
                                                                    for p, r in pairs]})

    # ------------------------------------------------------------------ statique
    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        svg = STATIC_DIR / "favicon.svg"
        if svg.is_file():
            return FileResponse(svg, media_type="image/svg+xml")
        return Response(status_code=204)

    if STATIC_DIR.is_dir():
        app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    return app


def run(host: str = HOST, port: int = PORT) -> None:  # pragma: no cover - utilisé par __main__
    import uvicorn

    uvicorn.run(create_app(), host=host, port=port, log_level=os.environ.get("REEL_PREDICTOR_LOG", "info"))
