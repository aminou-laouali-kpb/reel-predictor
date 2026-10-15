"""Référence de corpus par ROI (médiane / MAD) dans `data/reference.json`.

Chaque reel analysé avec le vrai cerveau (TRIBE) y ajoute ses valeurs ROI par seconde (plafonnées).
Tant que moins de `reference_min_reels` reels sont enregistrés : médiane 0 / MAD 1 par défaut.
La référence est séparée par mode texte (`llama` / `disabled`) : le niveau moyen des ROI change
quand TRIBE tourne sans LLaMA (mesuré, docs/RESEARCH.md), les deux corpus ne sont pas comparables.
roi = (valeur − médiane) / (mad_scale · MAD).
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .. import paths


class Reference:
    def __init__(self, data: dict | None = None, path: Path | None = None):
        self.path = Path(path) if path is not None else paths.REFERENCE_FILE
        self.data = data or {"version": 1, "reels": {}}
        self.data.setdefault("reels", {})

    @classmethod
    def load(cls, path: Path | None = None) -> "Reference":
        path = Path(path) if path is not None else paths.REFERENCE_FILE
        if path.exists():
            try:
                return cls(json.loads(path.read_text(encoding="utf-8")), path)
            except (OSError, json.JSONDecodeError):
                pass
        return cls(None, path)

    def _reels(self, mode: str | None) -> list[dict]:
        reels = list(self.data["reels"].values())
        if mode is None:
            return reels
        return [r for r in reels if r.get("text_feature", "llama") == mode]

    @property
    def n_reels(self) -> int:
        return len(self.data["reels"])

    def count(self, mode: str | None = None) -> int:
        return len(self._reels(mode))

    def stats(self, roi_names: list[str], min_reels: int = 3,
              mode: str | None = None) -> tuple[np.ndarray, np.ndarray, bool]:
        """(médianes (R,), MAD (R,), référence réelle utilisée ?) pour le mode texte donné."""
        R = len(roi_names)
        reels = self._reels(mode)
        if len(reels) < min_reels:
            return np.zeros(R), np.ones(R), False
        med = np.zeros(R)
        mad = np.ones(R)
        for i, name in enumerate(roi_names):
            vals = [v for reel in reels for v in reel.get("rois", {}).get(name, [])]
            if len(vals) >= 3:
                arr = np.asarray(vals, float)
                med[i] = float(np.median(arr))
                m = float(np.median(np.abs(arr - med[i])))
                mad[i] = m if m > 1e-6 else 1.0
        return med, mad, True

    def add(self, key: str, roi_names: list[str], values: np.ndarray, max_seconds: int = 120,
            mode: str | None = None) -> None:
        """`values` (T, R) valeurs ROI brutes (NaN ignorés)."""
        v = np.asarray(values, float)
        if v.shape[0] > max_seconds:
            idx = np.linspace(0, v.shape[0] - 1, max_seconds).round().astype(int)
            v = v[idx]
        self.data["reels"][key] = {
            "added_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "text_feature": mode or "llama",
            "rois": {n: [round(float(x), 4) for x in v[:, i] if np.isfinite(x)] for i, n in enumerate(roi_names)},
        }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.data, f)
        os.chmod(tmp, 0o644)
        os.replace(tmp, self.path)
