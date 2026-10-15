"""ROI cérébrales -> valeurs normalisées, puis 6 indices 0..1 mélangeant cerveau et signaux."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import INDEX_NAMES, Config
from .features import feature_matrix


def sigmoid(x: np.ndarray, gain: float = 1.0, center: float = 0.0) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-gain * (np.asarray(x, float) - center)))


def align_predictions(preds: np.ndarray, seconds: np.ndarray, T: int, hrf_shift_s: float = 0.0) -> np.ndarray:
    """(N, V) + secondes -> (T, V) aligné sur t (NaN si pas de prédiction). Ligne t = prédiction t + shift."""
    V = preds.shape[1]
    out = np.full((T, V), np.nan, np.float32)
    shift = int(round(hrf_shift_s))
    for row, s in enumerate(np.asarray(seconds).astype(int)):
        t = s - shift
        if 0 <= t < T:
            out[t] = preds[row]
    return out


def roi_values(aligned: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(T, V) -> (T, R) ; lignes manquantes -> NaN."""
    ok = ~np.isnan(aligned).all(axis=1)
    out = np.full((aligned.shape[0], W.shape[0]), np.nan, np.float32)
    if ok.any():
        out[ok] = np.nan_to_num(aligned[ok], nan=0.0) @ W.T
    return out


def normalize_corpus(raw: np.ndarray, median: np.ndarray, mad: np.ndarray, mad_scale: float) -> np.ndarray:
    return (raw - median[None, :]) / (mad_scale * np.maximum(mad[None, :], 1e-6))


def zscore_video(raw: np.ndarray) -> np.ndarray:
    out = np.full_like(raw, np.nan, dtype=np.float32)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        ok = np.isfinite(col)
        if ok.sum() >= 2:
            sd = float(np.std(col[ok]))
            out[ok, j] = (col[ok] - float(np.mean(col[ok]))) / sd if sd > 1e-6 else 0.0
        elif ok.sum() == 1:
            out[ok, j] = 0.0
    return out


def smooth(x: np.ndarray, kernel: list[float]) -> np.ndarray:
    if not kernel or len(kernel) == 1 or x.size < 2:
        return x
    k = np.asarray(kernel, float)
    num = np.convolve(x, k, mode="same")
    den = np.convolve(np.ones_like(x), k, mode="same")
    return num / np.where(den > 0, den, 1.0)


@dataclass
class IndexResult:
    values: dict[str, np.ndarray]          # indice -> (T,) 0..1
    brain: dict[str, np.ndarray]           # composante cérébrale (NaN si absente)
    signal: dict[str, np.ndarray]          # composante signaux


def compute_indices(df: pd.DataFrame, cfg: Config, roi_norm: np.ndarray | None = None,
                    roi_names: list[str] | None = None) -> IndexResult:
    """`roi_norm` (T, R) : ROI normalisées (source choisie par normalization.brain_roi_source)."""
    T = len(df)
    feats = feature_matrix(df)
    norm = cfg.section("normalization")
    gain, center = float(norm["sigmoid_gain"]), float(norm["sigmoid_center"])
    kernel = list(norm.get("smoothing") or [])
    roi_idx = {n: i for i, n in enumerate(roi_names or [])}
    values, brain, signal = {}, {}, {}
    for name in INDEX_NAMES:
        icfg = cfg.section("indices")[name]
        s = np.full(T, float(icfg["baseline"]))
        for feat, w in (icfg.get("boosters") or {}).items():
            s = s + float(w) * feats[feat]
        s = np.clip(s, 0.0, 1.0)
        b = np.full(T, np.nan)
        if roi_norm is not None and roi_names:
            weights = cfg.roi_mapping["indices"][name]
            num = np.zeros(T)
            den = 0.0
            for roi, w in weights.items():
                if roi in roi_idx:
                    num = num + float(w) * np.asarray(roi_norm[:, roi_idx[roi]], float)
                    den += abs(float(w))
            if den > 0:
                b = sigmoid(num / den, gain, center)
        bw = float(icfg["brain_weight"])
        mixed = np.where(np.isfinite(b), bw * np.nan_to_num(b) + (1 - bw) * s, s)
        values[name] = np.clip(smooth(mixed, kernel), 0.0, 1.0)
        brain[name] = b
        signal[name] = s
    return IndexResult(values=values, brain=brain, signal=signal)
