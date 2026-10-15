"""Interface commune des fournisseurs « cerveau » (TRIBE v2, heuristique, mock)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np
import pandas as pd

ProgressFn = Callable[[float, str], None]


class BrainUnavailable(RuntimeError):
    """Le fournisseur ne peut pas produire de prédiction ; `reason` est affichée à l'utilisateur."""

    def __init__(self, reason: str, detail: str | None = None):
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


@dataclass
class BrainRequest:
    video_path: Path
    words_path: Path | None
    duration_s: float
    n_seconds: int
    work_dir: Path
    features: pd.DataFrame | None = None      # signaux par seconde (utilisés par le mock)
    progress: ProgressFn | None = None


@dataclass
class BrainPrediction:
    available: bool
    provider: str                              # tribe | heuristic | mock
    model: str | None = None
    preds: np.ndarray | None = None            # (N, 20484) float32, gauche puis droite
    seconds: np.ndarray | None = None          # (N,) int : seconde de stimulus de chaque ligne
    degraded: bool = False
    reason: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class BrainProvider(Protocol):
    name: str

    def predict(self, request: BrainRequest) -> BrainPrediction:
        """Renvoie une prédiction ou lève `BrainUnavailable(reason)`."""
        ...
