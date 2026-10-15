"""Client du worker TRIBE v2 (sous-process isolé, venv `tribe_worker/.venv`) — contrat §1."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from .. import paths
from ..config import BrainConfig
from .provider import BrainPrediction, BrainRequest, BrainUnavailable

_PROGRESS = re.compile(r"^PROGRESS\s+([0-9]*\.?[0-9]+(?:[eE][-+]?\d+)?)\s*(.*)$")
_HF = re.compile(r"ERROR_HF_ACCESS\s+(\S+)")

EXIT_OK, EXIT_OTHER, EXIT_INPUT, EXIT_HF, EXIT_OOM = 0, 1, 2, 3, 4


def worker_installed() -> bool:
    return paths.TRIBE_WORKER_PYTHON.exists() and paths.TRIBE_WORKER_SCRIPT.exists()


def _clean_env() -> dict:
    env = dict(os.environ)
    for k in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "__PYVENV_LAUNCHER__"):
        env.pop(k, None)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def build_command(req: BrainRequest, cfg: BrainConfig, out_path: Path, no_text: bool) -> list[str]:
    cmd = [str(paths.TRIBE_WORKER_PYTHON), str(paths.TRIBE_WORKER_SCRIPT),
           "--video", str(req.video_path), "--words", str(req.words_path), "--out", str(out_path),
           "--model", cfg.model, "--cache-dir", str(paths.CACHE_DIR / "tribe"), "--device", cfg.device]
    if no_text:
        cmd.append("--no-text")
    return cmd


class _Run:
    def __init__(self, returncode: int, stderr_tail: list[str], stdout_tail: list[str], elapsed: float, timed_out: bool):
        self.returncode = returncode
        self.stderr_tail = stderr_tail
        self.stdout_tail = stdout_tail
        self.elapsed = elapsed
        self.timed_out = timed_out


def _run_worker(cmd: list[str], timeout_s: float, progress, label: str) -> _Run:
    t0 = time.time()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
                            cwd=str(paths.ROOT), env=_clean_env(), start_new_session=True)
    err_tail: deque[str] = deque(maxlen=200)
    out_tail: deque[str] = deque(maxlen=50)

    def read_out() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.rstrip("\n")
            m = _PROGRESS.match(line.strip())
            if m:
                if progress:
                    try:
                        progress(max(0.0, min(1.0, float(m.group(1)))), (m.group(2) or label).strip())
                    except Exception:  # noqa: BLE001 — un callback défaillant ne doit pas tuer le worker
                        pass
            elif line.strip():
                out_tail.append(line)

    def read_err() -> None:
        assert proc.stderr is not None
        for line in proc.stderr:
            err_tail.append(line.rstrip("\n"))

    th = [threading.Thread(target=read_out, daemon=True), threading.Thread(target=read_err, daemon=True)]
    for t in th:
        t.start()
    timed_out = False
    try:
        proc.wait(timeout=max(1.0, timeout_s))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGTERM)
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:  # noqa: BLE001
                proc.kill()
            proc.wait()
    for t in th:
        t.join(timeout=5)
    return _Run(proc.returncode, list(err_tail), list(out_tail), time.time() - t0, timed_out)


def _last_error(run: _Run) -> str:
    for line in reversed(run.stderr_tail):
        s = line.strip()
        if s and not s.startswith(("Warning", "warnings.warn", "  ")):
            return s[:240]
    return (run.stderr_tail[-1].strip()[:240] if run.stderr_tail else "aucun détail")


def load_npz(out_path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    with np.load(out_path) as z:
        preds = np.asarray(z["preds"], np.float32)
        seconds = np.asarray(z["seconds"]).astype(np.int64)
    if preds.ndim != 2 or preds.shape[1] != paths.N_VERTICES or seconds.shape[0] != preds.shape[0]:
        raise BrainUnavailable(f"Sortie TRIBE invalide : forme {preds.shape}")
    meta_path = Path(str(out_path) + ".meta.json")
    meta = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            meta = {}
    return preds, seconds, meta


class TribeProvider:
    name = "tribe"

    def __init__(self, cfg: BrainConfig):
        self.cfg = cfg

    def predict(self, req: BrainRequest) -> BrainPrediction:
        if not worker_installed():
            raise BrainUnavailable("Worker TRIBE non installé",
                                   f"{paths.TRIBE_WORKER_PYTHON} ou {paths.TRIBE_WORKER_SCRIPT} absent")
        if req.words_path is None or not Path(req.words_path).exists():
            raise BrainUnavailable("Transcription indisponible pour TRIBE")
        out_path = Path(req.work_dir) / "tribe.npz"
        deadline = time.time() + self.cfg.timeout_s
        no_text = self.cfg.no_text
        warnings: list[str] = []
        reason: str | None = None
        for attempt in range(2):
            if out_path.exists():
                out_path.unlink()
            remaining = deadline - time.time()
            if remaining <= 1:
                raise BrainUnavailable(f"Délai dépassé ({int(self.cfg.timeout_s)} s)")
            cmd = build_command(req, self.cfg, out_path, no_text)
            run = _run_worker(cmd, remaining, req.progress, "Modèle cérébral TRIBE v2")
            if run.timed_out:
                raise BrainUnavailable(f"Délai dépassé ({int(self.cfg.timeout_s)} s)", _last_error(run))
            code = run.returncode
            if code == EXIT_OK:
                if not out_path.exists():
                    raise BrainUnavailable("Le worker TRIBE n'a produit aucune sortie")
                preds, seconds, meta = load_npz(out_path)
                warnings.extend(str(w) for w in meta.get("warnings", []) if w)
                return BrainPrediction(available=True, provider="tribe", model=meta.get("model", self.cfg.model),
                                       preds=preds, seconds=seconds, degraded=False, reason=reason,
                                       meta={**meta, "wall_s": round(run.elapsed, 2), "no_text": no_text},
                                       warnings=warnings)
            if code == EXIT_HF:
                m = next((_HF.search(l) for l in run.stderr_tail + run.stdout_tail if _HF.search(l)), None)
                repo = m.group(1) if m else "un modèle gated"
                msg = f"Accès Hugging Face à {repo} manquant"
                if attempt == 0 and not no_text and self.cfg.retry_without_text and "llama" in repo.lower():
                    no_text = True
                    reason = f"{msg} : TRIBE exécuté sans la composante texte"
                    warnings.append(reason)
                    if req.progress:
                        req.progress(0.0, "Accès LLaMA refusé : relance de TRIBE sans le texte")
                    continue
                raise BrainUnavailable(msg, _last_error(run))
            if code == EXIT_OOM:
                raise BrainUnavailable("Mémoire insuffisante pour TRIBE v2", _last_error(run))
            if code == EXIT_INPUT:
                raise BrainUnavailable("Entrée refusée par le worker TRIBE", _last_error(run))
            if code is not None and code < 0:
                raise BrainUnavailable(f"Worker TRIBE interrompu (signal {-code})", _last_error(run))
            raise BrainUnavailable(f"Erreur du worker TRIBE (code {code})", _last_error(run))
        raise BrainUnavailable("Erreur du worker TRIBE")
