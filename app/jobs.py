"""File d'attente des analyses (et des générations « coach ») : UNE tâche lourde à la fois, dans un thread worker.

Cycle d'un job :
  upload -> `data/results/<job_id>/video.mp4` (+ `app_job.json` : nom d'origine, sha256)
  -> `analyze(out_dir/"video.mp4", out_dir, config, progress=cb)` (pipeline, ou factice si
  `REEL_PREDICTOR_FAKE=1`) -> enregistrement du reel en base.
`job_id = AAAAMMJJ-HHMMSS-<8 premiers hex du sha256>` (heure locale).
Au redémarrage, les résultats présents dans `data/results/` restent consultables (resynchronisés en base).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO, Callable

from reel_predictor.stages import STAGE_LABELS

from .db import Database

log = logging.getLogger("reel_predictor.app.jobs")

JOB_ID_RE = re.compile(r"^\d{8}-\d{6}-[0-9a-f]{8}$")
# Identifiants acceptés pour consulter un résultat (le CLI du pipeline peut en produire d'autres)
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$")
APP_JOB_FILE = "app_job.json"
ERROR_LOG = "app_error.log"
CHUNK = 1024 * 1024

AnalyzeFn = Callable[..., Any]
CoachFn = Callable[..., Any]   # (job_dir, progress(stage, pct, message), stamp=...) -> dict
COACH_STAGE_LABELS = {
    "brief": "Diagnostic", "llm": "Rédaction par Claude", "simulate": "Simulation des variantes",
    "round2": "Seconde passe Claude", "simulate2": "Simulation de la variante révisée", "write": "Écriture du rapport",
}


def make_job_id(sha256_hex: str, when: datetime | None = None) -> str:
    when = when or datetime.now()
    return f"{when:%Y%m%d-%H%M%S}-{sha256_hex[:8].lower()}"


def is_safe_id(job_id: str) -> bool:
    return bool(SAFE_ID_RE.match(job_id or "")) and ".." not in job_id


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class Job:
    job_id: str
    out_dir: Path
    filename: str
    sha256: str
    status: str = "queued"  # queued | running | done | error
    stage: str | None = None
    stage_label: str | None = None
    pct: float = 0.0
    message: str = "En attente"
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None  # time.monotonic()
    finished: float | None = None

    def eta_s(self) -> float | None:
        if self.status == "done":
            return 0.0
        if self.status != "running" or self.started is None:
            return None
        elapsed = time.monotonic() - self.started
        if self.pct < 0.02 or elapsed < 1.0:
            return None
        return float(max(0, round(elapsed * (1.0 - self.pct) / self.pct)))


@dataclass
class CoachTask:
    """Génération « coach » (Claude + simulation) : passe par la MÊME file que les analyses (une tâche lourde à la fois)."""
    coach_id: str
    job_id: str
    job_dir: Path
    stamp: str
    fn: CoachFn
    status: str = "queued"  # queued | running | done | error
    stage: str | None = None
    pct: float = 0.0
    message: str = "En attente"
    error: str | None = None
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None

    def eta_s(self) -> float | None:
        if self.status == "done":
            return 0.0
        if self.status != "running" or self.started is None:
            return None
        elapsed = time.monotonic() - self.started
        if self.pct < 0.05 or elapsed < 2.0:
            return None
        return float(max(0, round(elapsed * (1.0 - self.pct) / self.pct)))


class JobManager:
    def __init__(self, results_dir: Path, uploads_dir: Path, db: Database,
                 analyze_fn: AnalyzeFn | None = None, use_fake: bool | None = None):
        self.results_dir = Path(results_dir)
        self.uploads_dir = Path(uploads_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.db = db
        self._analyze_fn = analyze_fn
        self._use_fake = use_fake
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._queue: queue.Queue[Any] = queue.Queue()
        self._coach: dict[str, CoachTask] = {}
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._current: str | None = None

    # ------------------------------------------------------------------ démarrage
    def recover(self) -> int:
        """Réenregistre en base les résultats complets trouvés sur disque. Renvoie leur nombre."""
        n = 0
        for d in sorted(self.results_dir.iterdir()) if self.results_dir.exists() else []:
            if d.is_dir() and is_safe_id(d.name) and (d / "summary.json").is_file():
                try:
                    self.register_result(d.name, d)
                    n += 1
                except Exception as exc:  # résultat corrompu : on l'ignore sans bloquer le démarrage
                    log.warning("Résultat ignoré %s : %s", d.name, exc)
        return n

    def register_result(self, job_id: str, out_dir: Path) -> int:
        summary = read_json(out_dir / "summary.json")
        meta: dict[str, Any] = {}
        if (out_dir / APP_JOB_FILE).is_file():
            try:
                meta = read_json(out_dir / APP_JOB_FILE)
            except ValueError:
                meta = {}
        video = summary.get("video") or {}
        retention = summary.get("retention") or {}
        return self.db.upsert_reel(
            job_id=job_id,
            filename=meta.get("filename") or video.get("filename"),
            sha256=meta.get("sha256"),
            duration_s=_num(video.get("duration_s")),
            analyzed_at=summary.get("created_at") or meta.get("uploaded_at"),
            score=_num(summary.get("score")),
            predicted_avg_watch_ratio=_num(retention.get("avg_watch_ratio")),
            result_dir=str(out_dir),
        )

    # ------------------------------------------------------------------ soumission
    def submit_upload(self, fileobj: BinaryIO, filename: str | None) -> str:
        """Copie la vidéo envoyée, crée le dossier de résultat et met le job en file d'attente."""
        filename = os.path.basename(filename or "video.mp4") or "video.mp4"
        tmp = self.uploads_dir / f".upload-{uuid.uuid4().hex}.part"
        h = hashlib.sha256()
        size = 0
        try:
            with open(tmp, "wb") as out:
                while True:
                    chunk = fileobj.read(CHUNK)
                    if not chunk:
                        break
                    h.update(chunk)
                    out.write(chunk)
                    size += len(chunk)
            if size == 0:
                raise ValueError("Fichier vide.")
            sha = h.hexdigest()
            with self._lock:
                job_id = make_job_id(sha)
                out_dir = self.results_dir / job_id
                if job_id in self._jobs or out_dir.exists():
                    # Même fichier envoyé deux fois dans la même seconde (double clic) : même job.
                    return job_id
                out_dir.mkdir(parents=True)
                try:
                    os.replace(tmp, out_dir / "video.mp4")
                except OSError:
                    shutil.move(str(tmp), out_dir / "video.mp4")
                (out_dir / APP_JOB_FILE).write_text(json.dumps({
                    "job_id": job_id, "filename": filename, "sha256": sha, "size": size,
                    "uploaded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                }, ensure_ascii=False, indent=2), encoding="utf-8")
                job = Job(job_id=job_id, out_dir=out_dir, filename=filename, sha256=sha)
                self._jobs[job_id] = job
                self._order.append(job_id)
                self._queue.put(job_id)
                self._ensure_worker()
                return job_id
        finally:
            tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------ consultation
    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None:
                return self._job_dict(job)
        if not is_safe_id(job_id):
            return None
        d = self.results_dir / job_id
        if (d / "summary.json").is_file():
            return {"job_id": job_id, "status": "done", "stage": "write", "stage_label": STAGE_LABELS.get("write"),
                    "pct": 1.0, "message": "Analyse terminée", "eta_s": 0.0, "error": None,
                    "filename": _filename_of(d)}
        if d.is_dir():
            err = "Analyse interrompue : l'application a été arrêtée pendant l'analyse. Relancez l'analyse."
            if (d / ERROR_LOG).is_file():
                err = "Analyse en échec (voir app_error.log dans le dossier du résultat)."
            return {"job_id": job_id, "status": "error", "stage": None, "stage_label": None, "pct": 0.0,
                    "message": err, "eta_s": None, "error": err, "filename": _filename_of(d)}
        return None

    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            return [self._job_dict(self._jobs[j]) for j in self._order]

    def active(self) -> dict[str, Any]:
        with self._lock:
            queued = [j for j in self._order if self._jobs[j].status == "queued"]
            coach_q = [c for c in self._coach.values() if c.status == "queued"]
            return {"running": self._current, "queued": len(queued) + len(coach_q), "coach_queued": len(coach_q)}

    # ------------------------------------------------------------------ coach
    def submit_coach(self, job_id: str, job_dir: Path, fn: CoachFn) -> str:
        """Met une génération coach en file (même worker que les analyses). Renvoie le coach_id."""
        with self._lock:
            for c in self._coach.values():  # double clic : même tâche si une génération est déjà en attente/en cours
                if c.job_id == job_id and c.status in ("queued", "running"):
                    return c.coach_id
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            coach_id = f"{job_id}.{stamp}"
            task = CoachTask(coach_id=coach_id, job_id=job_id, job_dir=Path(job_dir), stamp=stamp, fn=fn)
            self._coach[coach_id] = task
            self._queue.put(("coach", coach_id))
            self._ensure_worker()
            return coach_id

    def get_coach(self, coach_id: str) -> dict[str, Any] | None:
        with self._lock:
            task = self._coach.get(coach_id)
            return self._coach_dict(task) if task else None

    def coach_tasks(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return [self._coach_dict(c) for c in self._coach.values() if c.job_id == job_id]

    def _coach_dict(self, c: CoachTask) -> dict[str, Any]:
        d = {"coach_id": c.coach_id, "job_id": c.job_id, "status": c.status, "stage": c.stage,
             "stage_label": COACH_STAGE_LABELS.get(c.stage or "", c.stage), "pct": round(c.pct, 4),
             "message": c.message, "eta_s": c.eta_s(), "error": c.error}
        if c.status == "queued":
            d["message"] = "En attente (une autre tâche est en cours)" if self._current else "En attente"
        return d

    def _run_coach(self, task: CoachTask) -> None:
        with self._lock:
            task.status = "running"
            task.started = time.monotonic()
            task.message = "Démarrage…"
            self._current = task.coach_id

        def progress(stage: str, pct: float, message: str = "") -> None:
            try:
                with self._lock:
                    task.stage = stage
                    p = float(pct)
                    if p == p:
                        task.pct = max(task.pct, min(0.999, max(0.0, p)))
                    task.message = str(message or COACH_STAGE_LABELS.get(stage, stage))
            except Exception:
                pass

        try:
            task.fn(task.job_dir, progress, stamp=task.stamp)
            with self._lock:
                task.status = "done"
                task.pct = 1.0
                task.stage = "write"
                task.message = "Suggestions prêtes"
        except Exception as exc:
            msg = str(exc) or exc.__class__.__name__
            log.error("Coach %s en erreur : %s", task.coach_id, msg)
            with self._lock:
                task.status = "error"
                task.error = msg
                task.message = msg
        finally:
            with self._lock:
                task.finished = time.monotonic()
                self._current = None

    def _job_dict(self, job: Job) -> dict[str, Any]:
        d = {
            "job_id": job.job_id,
            "status": job.status,
            "stage": job.stage,
            "stage_label": job.stage_label,
            "pct": round(job.pct, 4),
            "message": job.message,
            "eta_s": job.eta_s(),
            "error": job.error,
            "filename": job.filename,
        }
        if job.status == "queued":
            waiting = [j for j in self._order if self._jobs[j].status == "queued"]
            pos = waiting.index(job.job_id) + 1 if job.job_id in waiting else None
            d["queue_position"] = pos
            if pos:
                d["message"] = "En attente (une analyse est en cours)" if self._current else "En attente"
        return d

    # ------------------------------------------------------------------ worker
    def _ensure_worker(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._worker, name="reel-analysis-worker", daemon=True)
            self._thread.start()

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            if job_id is None:
                return
            if isinstance(job_id, tuple):  # ("coach", coach_id)
                task = self._coach.get(job_id[1])
                if task is not None:
                    try:
                        self._run_coach(task)
                    except Exception:
                        log.exception("Erreur inattendue du worker (coach)")
                continue
            job = self._jobs.get(job_id)
            if job is None:
                continue
            try:
                self._run(job)
            except Exception:  # jamais de crash du worker
                log.exception("Erreur inattendue du worker")

    def shutdown(self, timeout: float = 2.0) -> None:
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout)

    def _resolve(self) -> tuple[AnalyzeFn, Any]:
        if self._analyze_fn is not None:
            return self._analyze_fn, None
        from .fake import fake_enabled

        use_fake = fake_enabled() if self._use_fake is None else self._use_fake
        if use_fake:
            from .fake import fake_analyze

            return fake_analyze, None
        try:
            from reel_predictor.config import load_config
            from reel_predictor.pipeline import analyze
        except ImportError as exc:
            raise RuntimeError(
                "Le pipeline d'analyse n'est pas disponible (module reel_predictor.pipeline introuvable ou "
                f"incomplet : {exc}). Pour tester l'interface : REEL_PREDICTOR_FAKE=1 make run."
            ) from exc
        return analyze, load_config()

    def _run(self, job: Job) -> None:
        with self._lock:
            job.status = "running"
            job.started = time.monotonic()
            job.message = "Démarrage de l'analyse…"
            self._current = job.job_id

        def progress(stage: str, pct: float, message: str = "") -> None:
            try:
                with self._lock:
                    job.stage = stage
                    job.stage_label = STAGE_LABELS.get(stage, stage)
                    p = float(pct)
                    if p == p:  # pas NaN
                        job.pct = max(job.pct, min(0.999, max(0.0, p)))
                    job.message = str(message or job.stage_label or "")
            except Exception:
                pass

        try:
            analyze, config = self._resolve()
            analyze(job.out_dir / "video.mp4", job.out_dir, config=config, progress=progress)
            if not (job.out_dir / "summary.json").is_file():
                raise RuntimeError("le pipeline n'a pas écrit summary.json")
            self.register_result(job.job_id, job.out_dir)
            with self._lock:
                job.status = "done"
                job.pct = 1.0
                job.stage = "write"
                job.stage_label = STAGE_LABELS.get("write")
                job.message = "Analyse terminée"
        except Exception as exc:
            msg = str(exc) or exc.__class__.__name__
            if not msg.startswith("Le pipeline"):
                msg = f"Échec de l'analyse : {msg}"
            try:
                (job.out_dir / ERROR_LOG).write_text(traceback.format_exc(), encoding="utf-8")
            except OSError:
                pass
            log.error("Job %s en erreur : %s", job.job_id, msg)
            with self._lock:
                job.status = "error"
                job.error = msg
                job.message = msg
        finally:
            with self._lock:
                job.finished = time.monotonic()
                self._current = None


def _num(v: Any) -> float | None:
    try:
        f = float(v)
        return f if f == f else None
    except (TypeError, ValueError):
        return None


def _filename_of(d: Path) -> str | None:
    try:
        return read_json(d / APP_JOB_FILE).get("filename")
    except Exception:
        try:
            return (read_json(d / "summary.json").get("video") or {}).get("filename")
        except Exception:
            return None
