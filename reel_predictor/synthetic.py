"""Construction de signaux synthétiques (sans média) pour les tests et le débogage des règles."""
from __future__ import annotations

import numpy as np

from .audio import AudioStats
from .faces import FrameStats
from .features import Signals
from .ocr import OcrSecond, count_words
from .speech import Transcript, Word


def make_signals(duration_s: float, *, cuts: list[float] | None = None,
                 words: list[tuple[str, float, float]] | None = None,
                 faces: dict[int, tuple[int, float]] | None = None,
                 motion: dict[int, float] | None = None,
                 texts: dict[int, str] | None = None,
                 silence: list[int] | None = None,
                 loud_db: dict[int, float] | None = None,
                 fps: float = 4.0, language: str = "fr") -> Signals:
    """
    duration_s : durée ; cuts : instants des cuts ; words : (mot, début, fin) ;
    faces : {t: (nb, aire)} ; motion : {t: diff moyenne 0..255} ; texts : {t: texte affiché} ;
    silence : secondes silencieuses ; loud_db : {t: dBFS} (défaut -20 dB hors silence).
    """
    T = max(1, int(np.ceil(duration_s - 0.05)))
    n = int(T * fps)
    times = np.arange(n, dtype=np.float32) / fps
    sec = np.floor(times).astype(int)
    f_faces = np.zeros(n, np.int32)
    f_area = np.zeros(n, np.float32)
    f_motion = np.zeros(n, np.float32)
    for t, (k, a) in (faces or {}).items():
        f_faces[sec == t] = k
        f_area[sec == t] = a
    for t, m in (motion or {}).items():
        f_motion[sec == t] = m
    fs = FrameStats(times=times, faces=f_faces, face_area=f_area, motion=f_motion, face_ok=True)
    ocr = []
    prev = ""
    for t in range(T):
        txt = (texts or {}).get(t, "")
        ocr.append(OcrSecond(text=txt, words=count_words(txt), new=bool(txt) and txt != prev, score=0.9 if txt else 0.0))
        prev = txt
    sil = set(silence or [])
    db = np.array([(loud_db or {}).get(t, -60.0 if t in sil else -20.0) for t in range(T)], np.float32)
    rms_abs = 10 ** (db / 20.0)
    rms = rms_abs / rms_abs.max()
    audio = AudioStats(rms=rms.astype(np.float32), rms_abs=rms_abs.astype(np.float32), dbfs=db,
                       db_std=np.zeros(T, np.float32), silence=np.array([t in sil for t in range(T)]))
    ws = [Word(w, s, e) for (w, s, e) in (words or [])]
    tr = Transcript(language=language, language_probability=1.0, text=" ".join(w.w for w in ws), words=ws)
    return Signals(duration_s=duration_s, n_seconds=T, cuts=list(cuts or []), transcript=tr, frames=fs,
                   ocr=ocr, audio=audio)
