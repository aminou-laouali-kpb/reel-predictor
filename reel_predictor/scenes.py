"""Détection des changements de plan (cuts).

PySceneDetect `ContentDetector` (backend OpenCV, compatible OpenCV 5) ; repli sur le filtre
ffmpeg `select='gt(scene,X)',showinfo` si PySceneDetect échoue.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable

from .media import _bin

ProgressFn = Callable[[float, str], None]


def _detect_pyscenedetect(path: Path, threshold: float, min_scene_len_s: float, fps: float,
                          duration_s: float, progress: ProgressFn | None) -> list[float]:
    from scenedetect import ContentDetector, SceneManager, open_video

    video = open_video(str(path), backend="opencv")
    rate = float(video.frame_rate or fps or 30.0)
    min_len = max(1, int(round(min_scene_len_s * rate)))
    sm = SceneManager()
    sm.add_detector(ContentDetector(threshold=threshold, min_scene_len=min_len))
    sm.detect_scenes(video, show_progress=False)
    scenes = sm.get_scene_list()
    cuts = []
    for start, _end in scenes[1:]:
        sec = start.seconds if hasattr(start, "seconds") and not callable(start.seconds) else start.get_seconds()
        cuts.append(float(sec))
    return cuts


def _detect_ffmpeg(path: Path, threshold: float) -> list[float]:
    vf = f"scale=320:-2,select='gt(scene\\,{threshold})',showinfo"
    cmd = [_bin("ffmpeg"), "-hide_banner", "-nostats", "-i", str(path), "-an", "-vf", vf, "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    times = [float(m.group(1)) for m in re.finditer(r"pts_time:([0-9.]+)", proc.stderr)]
    return times


def detect_cuts(path: Path, duration_s: float, fps: float, detector: str = "content",
                threshold: float = 27.0, min_scene_len_s: float = 0.4, ffmpeg_threshold: float = 0.3,
                progress: ProgressFn | None = None) -> tuple[list[float], str, str | None]:
    """Renvoie (temps des cuts en s triés, méthode utilisée, avertissement éventuel)."""
    warning = None
    method = detector
    cuts: list[float] = []
    if detector == "content":
        try:
            cuts = _detect_pyscenedetect(path, threshold, min_scene_len_s, fps, duration_s, progress)
        except Exception as e:  # noqa: BLE001 — repli ffmpeg
            warning = f"PySceneDetect indisponible ({type(e).__name__}: {e}) : repli ffmpeg"
            method = "ffmpeg"
    if method == "ffmpeg":
        cuts = _detect_ffmpeg(path, ffmpeg_threshold)
    # dédoublonnage + bornes
    out: list[float] = []
    for c in sorted(cuts):
        if 0.0 < c < duration_s and (not out or c - out[-1] >= max(0.1, min_scene_len_s * 0.5)):
            out.append(round(c, 3))
    return out, method, warning
