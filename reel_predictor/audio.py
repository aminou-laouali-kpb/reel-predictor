"""Énergie audio par seconde (stdlib `wave` + numpy) : RMS, loudness (dBFS), variance, sauts, silences."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AudioStats:
    rms: np.ndarray        # (T,) RMS normalisée au max de la vidéo (0..1)
    rms_abs: np.ndarray    # (T,) RMS absolue (pleine échelle = 1)
    dbfs: np.ndarray       # (T,) loudness moyenne de la seconde (dBFS)
    db_std: np.ndarray     # (T,) écart-type intra-seconde des fenêtres (dB)
    silence: np.ndarray    # (T,) bool


def _db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, 1e-6))


def analyze(samples: np.ndarray | None, sample_rate: int, n_seconds: int, window_s: float = 0.1,
            silence_dbfs: float = -45.0, silence_rel: float = 0.06) -> AudioStats:
    T = n_seconds
    if samples is None or samples.size == 0:
        z = np.zeros(T, np.float32)
        return AudioStats(rms=z.copy(), rms_abs=z.copy(), dbfs=np.full(T, -120.0, np.float32),
                          db_std=z.copy(), silence=np.ones(T, bool))
    x = samples.astype(np.float32)
    win = max(1, int(round(window_s * sample_rate)))
    rms_abs = np.zeros(T, np.float32)
    dbfs = np.full(T, -120.0, np.float32)
    db_std = np.zeros(T, np.float32)
    for t in range(T):
        seg = x[t * sample_rate:(t + 1) * sample_rate]
        if seg.size == 0:
            continue
        rms_abs[t] = float(np.sqrt(np.mean(seg ** 2)))
        n = seg.size // win
        if n >= 1:
            w = seg[: n * win].reshape(n, win)
            wdb = _db(np.sqrt(np.mean(w ** 2, axis=1)))
            dbfs[t] = float(np.mean(wdb))
            db_std[t] = float(np.std(wdb)) if n > 1 else 0.0
        else:
            dbfs[t] = float(_db(np.array([rms_abs[t]]))[0])
    peak = float(rms_abs.max()) if rms_abs.size else 0.0
    rms = rms_abs / peak if peak > 0 else np.zeros_like(rms_abs)
    silence = (_db(rms_abs) < silence_dbfs) | (rms < silence_rel)
    return AudioStats(rms=rms.astype(np.float32), rms_abs=rms_abs, dbfs=dbfs, db_std=db_std,
                      silence=silence.astype(bool))
