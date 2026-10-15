"""Repli sans cerveau : les indices sont calculés uniquement à partir des signaux vidéo/audio/texte."""
from __future__ import annotations

from .provider import BrainPrediction, BrainRequest

DEFAULT_REASON = "Modèle cérébral désactivé dans la configuration (brain.provider = heuristic)"


class HeuristicProvider:
    name = "heuristic"

    def __init__(self, reason: str | None = None, detail: str | None = None):
        self.reason = reason or DEFAULT_REASON
        self.detail = detail

    def predict(self, request: BrainRequest) -> BrainPrediction:
        meta = {"detail": self.detail} if self.detail else {}
        return BrainPrediction(available=False, provider="heuristic", model=None, preds=None, seconds=None,
                               degraded=True, reason=self.reason, meta=meta)
