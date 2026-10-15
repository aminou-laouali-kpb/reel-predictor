"""Écriture des sorties d'une analyse (contrat §3) : timeline.jsonl, summary.json, brain.u8, brain_meta.json."""
from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import paths
from .config import INDEX_NAMES

TIMELINE_KEYS = ("t", "roi", "roi_video_z", "indices", "cut", "since_cut", "faces", "face_area", "words",
                 "speech_rate", "text", "text_words", "text_new", "audio_rms", "silence", "retention", "flags")


def _f(x, nd: int = 4):
    """float JSON-compatible (NaN/inf -> None)."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def build_rows(df: pd.DataFrame, indices: dict[str, np.ndarray], retention_end: np.ndarray,
               roi_names: list[str], roi_corpus: np.ndarray | None, roi_video: np.ndarray | None,
               flags: list[list[str]]) -> list[dict]:
    rows = []
    for t in range(len(df)):
        r = df.iloc[t]
        roi = {n: (_f(roi_corpus[t, i]) if roi_corpus is not None else None) for i, n in enumerate(roi_names)}
        roz = {n: (_f(roi_video[t, i]) if roi_video is not None else None) for i, n in enumerate(roi_names)}
        rows.append({
            "t": int(t),
            "roi": roi,
            "roi_video_z": roz,
            "indices": {k: _f(np.clip(indices[k][t], 0, 1)) for k in INDEX_NAMES},
            "cut": bool(r["cut"]),
            "since_cut": _f(r["since_cut"], 3),
            "faces": int(r["faces"]),
            "face_area": _f(np.clip(r["face_area"], 0, 1)),
            "words": [str(w) for w in r["words"]],
            "speech_rate": _f(r["speech_rate"], 3),
            "text": str(r["text"] or ""),
            "text_words": int(r["text_words"]),
            "text_new": bool(r["text_new"]),
            "audio_rms": _f(np.clip(r["audio_rms"], 0, 1)),
            "silence": bool(r["silence"]),
            "retention": _f(np.clip(retention_end[t], 0, 1)),
            "flags": list(flags[t]),
        })
    return rows


def _atomic_write(path: Path, data: bytes) -> None:
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def write_timeline(path: Path, rows: list[dict]) -> None:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    _atomic_write(path, data.encode("utf-8"))


def write_json(path: Path, obj: dict) -> None:
    _atomic_write(path, json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8"))


def read_timeline(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def encode_brain(aligned: np.ndarray) -> tuple[np.ndarray, dict]:
    """(T, 20484) float (NaN = pas de prédiction) -> (uint8 (T, 20484), meta)."""
    T, V = aligned.shape
    finite = np.isfinite(aligned)
    vmax = float(np.percentile(np.abs(aligned[finite]), 98)) if finite.any() else 1.0
    if not math.isfinite(vmax) or vmax <= 1e-9:
        vmax = 1.0
    vmin = -vmax
    scaled = (np.nan_to_num(aligned, nan=0.0) - vmin) / (vmax - vmin) * 255.0
    u8 = np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
    u8[~finite] = 128      # secondes (ou sommets) sans prédiction : valeur médiane
    meta = {"T": int(T), "n_vertices": int(V), "hemi_split": int(paths.HEMI_SPLIT), "vmin": round(vmin, 6),
            "vmax": round(vmax, 6), "space": "fsaverage5", "units": "z (TRIBE, niveau session)"}
    return u8, meta


def write_brain(out_dir: Path, aligned: np.ndarray) -> dict:
    u8, meta = encode_brain(aligned)
    _atomic_write(Path(out_dir) / "brain.u8", np.ascontiguousarray(u8).tobytes(order="C"))
    write_json(Path(out_dir) / "brain_meta.json", meta)
    return meta
