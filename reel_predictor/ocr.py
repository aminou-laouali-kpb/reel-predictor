"""Texte à l'écran (RapidOCR / onnxruntime), ~1 image par seconde.

- filtre les lignes à faible score ou trop courtes ;
- réutilise le résultat précédent si l'image n'a quasiment pas changé (vitesse) ;
- dédoublonne entre secondes voisines (texte quasi identique => même texte, `text_new = False`).
"""
from __future__ import annotations

import difflib
import re
import threading
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .media import FrameSet, load_rgb

ProgressFn = Callable[[float, str], None]

_ENGINE = {}
_LOCK = threading.Lock()
_ALNUM = re.compile(r"[^\W_]", re.UNICODE)


@dataclass
class OcrSecond:
    text: str
    words: int
    new: bool
    score: float


def _engine(det_limit_side_len: int, det_limit_type: str):
    from rapidocr_onnxruntime import RapidOCR

    key = (det_limit_side_len, det_limit_type)
    with _LOCK:
        if key not in _ENGINE:
            _ENGINE[key] = RapidOCR(det_limit_side_len=int(det_limit_side_len), det_limit_type=det_limit_type)
        return _ENGINE[key]


def clean_text(s: str) -> str:
    """Espaces aux frontières chiffre/lettre et autour des « : » (l'OCR colle souvent les mots)."""
    s = re.sub(r"(?<=\d)(?=[^\W\d_])|(?<=[^\W\d_])(?=\d)", " ", s)
    s = re.sub(r"\s*:\s*", " : ", s)
    return re.sub(r"\s+", " ", s).strip()


def count_words(s: str) -> int:
    return sum(1 for tok in s.split() if _ALNUM.search(tok))


def similarity(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _ocr_image(engine, rgb: np.ndarray, min_score: float, min_chars: int, use_cls: bool) -> tuple[str, float]:
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    result, _ = engine(bgr, use_cls=use_cls)
    if not result:
        return "", 0.0
    # tri haut -> bas puis gauche -> droite (ordre de lecture)
    lines = []
    for box, text, score in result:
        score = float(score)
        text = str(text).strip()
        if score < min_score or len(text) < min_chars or not _ALNUM.search(text):
            continue
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        lines.append((min(ys), min(xs), text, score))
    if not lines:
        return "", 0.0
    lines.sort(key=lambda r: (round(r[0] / 20), r[1]))
    text = clean_text(" ".join(r[2] for r in lines))
    return text, float(np.mean([r[3] for r in lines]))


def read_text(frames: FrameSet, n_seconds: int, ocr_fps: float = 1.0, min_score: float = 0.6,
              min_chars: int = 2, same_text_similarity: float = 0.8, det_limit_side_len: int = 960,
              det_limit_type: str = "max", use_angle_cls: bool = False, reuse_if_diff_below: float = 1.0,
              progress: ProgressFn | None = None) -> list[OcrSecond]:
    """Une entrée par seconde `t` (texte fusionné des échantillons de la seconde)."""
    out: list[OcrSecond] = []
    if len(frames) == 0:
        return [OcrSecond("", 0, False, 0.0) for _ in range(n_seconds)]
    engine = _engine(det_limit_side_len, det_limit_type)
    per_sec = max(1, int(round(ocr_fps)))
    prev_gray = None
    prev_res: tuple[str, float] = ("", 0.0)
    cache: dict[int, tuple[str, float]] = {}
    prev_text = ""
    for t in range(n_seconds):
        texts, scores = [], []
        for k in range(per_sec):
            idx = frames.index_at(t + (k + 0.5) / per_sec)
            if idx in cache:
                res = cache[idx]
            else:
                rgb = load_rgb(frames.files[idx])
                gray = rgb[::8, ::8].mean(axis=2)
                if prev_gray is not None and prev_gray.shape == gray.shape and \
                        float(np.mean(np.abs(gray - prev_gray))) < reuse_if_diff_below:
                    res = prev_res
                else:
                    res = _ocr_image(engine, rgb, min_score, min_chars, use_angle_cls)
                prev_gray, prev_res = gray, res
                cache[idx] = res
            if res[0] and all(similarity(res[0], x) < same_text_similarity for x in texts):
                texts.append(res[0])
                scores.append(res[1])
        text = " ".join(texts)
        if text and prev_text and similarity(text, prev_text) >= same_text_similarity:
            text = prev_text   # même texte que la seconde précédente (bruit OCR) -> on stabilise
            new = False
        else:
            new = bool(text)
        out.append(OcrSecond(text=text, words=count_words(text), new=new,
                             score=float(np.mean(scores)) if scores else 0.0))
        prev_text = text
        if progress:
            progress((t + 1) / n_seconds, f"Texte à l'écran : {t + 1}/{n_seconds} s")
    return out
