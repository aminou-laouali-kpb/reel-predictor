"""Téléchargement idempotent des modèles légers (visage MediaPipe, atlas Destrieux, faster-whisper).

Le checkpoint TRIBE v2 et ses backbones sont gérés par le worker (`run_tribe.py --download-only`).
"""
from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
from typing import Callable

from . import paths

FACE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_detector/"
                  "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")

ProgressFn = Callable[[str, str], None]


def ensure_face_model(dest: Path = paths.FACE_MODEL, timeout: float = 60.0) -> Path:
    dest = Path(dest)
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    try:
        with urllib.request.urlopen(FACE_MODEL_URL, timeout=timeout) as r, open(tmp, "wb") as f:
            shutil.copyfileobj(r, f)
        if Path(tmp).stat().st_size < 10_000:
            raise RuntimeError("Modèle visage téléchargé invalide (taille trop faible)")
        os.chmod(tmp, 0o644)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def ensure_atlas() -> Path:
    from .brain.atlas import load_atlas

    load_atlas()
    return paths.NILEARN_DIR


def ensure_whisper(model: str = "small") -> Path:
    from faster_whisper import download_model

    return Path(download_model(model))


def whisper_cached(model: str = "small") -> bool:
    try:
        from faster_whisper import download_model

        download_model(model, local_files_only=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def ensure_all(config=None, progress: ProgressFn | None = None) -> dict:
    """Télécharge ce qui manque. Idempotent. Renvoie {"face_model", "atlas", "whisper": "ok"|"erreur ..."}."""
    from .config import load_config

    paths.ensure_dirs()
    if config is None:
        config = load_config()
    out: dict[str, str] = {}
    steps = [
        ("face_model", "Modèle de détection de visage", ensure_face_model),
        ("atlas", "Atlas cortical Destrieux", ensure_atlas),
        ("whisper", f"Modèle de transcription ({config.speech.model})", lambda: ensure_whisper(config.speech.model)),
    ]
    for key, label, fn in steps:
        if progress:
            progress(key, f"Vérification / téléchargement : {label}")
        try:
            fn()
            out[key] = "ok"
        except Exception as e:  # noqa: BLE001
            out[key] = f"erreur : {type(e).__name__}: {e}"
    return out


if __name__ == "__main__":
    print(ensure_all(progress=lambda k, m: print(m, flush=True)))
