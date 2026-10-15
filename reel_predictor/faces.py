"""Visages (MediaPipe Tasks FaceDetector, blaze_face_short_range) + mouvement image à image.

Les deux mesures sont calculées sur les mêmes images échantillonnées (`media.FrameSet`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from . import paths
from .media import FrameSet, load_rgb

ProgressFn = Callable[[float, str], None]


@dataclass
class FrameStats:
    times: np.ndarray        # (n,) instants des images (s)
    faces: np.ndarray        # (n,) nb de visages
    face_area: np.ndarray    # (n,) aire max d'un visage / aire image
    motion: np.ndarray       # (n,) différence moyenne (0..255) avec l'image précédente (0 pour la 1re)
    face_ok: bool = True     # False si le détecteur n'a pas pu être chargé


def _make_detector(model_path: Path, min_confidence: float):
    import mediapipe as mp  # noqa: F401
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    opts = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=float(min_confidence),
    )
    return vision.FaceDetector.create_from_options(opts)


def _small_gray(rgb: np.ndarray, step: int = 8) -> np.ndarray:
    g = rgb[::step, ::step].astype(np.float32)
    return 0.299 * g[..., 0] + 0.587 * g[..., 1] + 0.114 * g[..., 2]


def analyze_frames(frames: FrameSet, min_confidence: float = 0.5, model_path: Path = paths.FACE_MODEL,
                   progress: ProgressFn | None = None) -> tuple[FrameStats, str | None]:
    """Renvoie (stats par image, avertissement éventuel)."""
    n = len(frames)
    faces = np.zeros(n, np.int32)
    area = np.zeros(n, np.float32)
    motion = np.zeros(n, np.float32)
    warning = None
    detector = None
    if not Path(model_path).exists():
        try:
            from .assets import ensure_face_model

            ensure_face_model(model_path)
        except Exception as e:  # noqa: BLE001
            warning = f"Modèle visage absent et téléchargement impossible ({e}) : visages non analysés"
    if warning is None:
        try:
            detector = _make_detector(model_path, min_confidence)
        except Exception as e:  # noqa: BLE001
            warning = f"Détecteur de visages indisponible ({type(e).__name__}: {e})"
    import mediapipe as mp

    prev = None
    try:
        for i, f in enumerate(frames.files):
            rgb = load_rgb(f)
            g = _small_gray(rgb)
            if prev is not None and prev.shape == g.shape:
                motion[i] = float(np.mean(np.abs(g - prev)))
            prev = g
            if detector is not None:
                h, w = rgb.shape[:2]
                res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
                best = 0.0
                k = 0
                for d in res.detections or []:
                    score = d.categories[0].score if d.categories else 1.0
                    if score is not None and score < min_confidence:
                        continue
                    bb = d.bounding_box
                    bw = max(0, min(bb.origin_x + bb.width, w) - max(bb.origin_x, 0))
                    bh = max(0, min(bb.origin_y + bb.height, h) - max(bb.origin_y, 0))
                    best = max(best, (bw * bh) / float(w * h))
                    k += 1
                faces[i] = k
                area[i] = best
            if progress and (i % 8 == 0 or i == n - 1):
                progress((i + 1) / max(1, n), f"Visages : image {i + 1}/{n}")
    finally:
        if detector is not None:
            detector.close()
    times = np.arange(n, dtype=np.float32) / frames.fps
    return FrameStats(times=times, faces=faces, face_area=area, motion=motion, face_ok=detector is not None), warning
