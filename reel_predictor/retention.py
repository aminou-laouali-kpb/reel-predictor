"""Modèle de rétention seconde par seconde.

r(0) = 1 ; r(t+1) = clip(r(t) · (1 − base_drop(t) − k·(1 − attention(t)) + bonus·cut_recent(t)), 0, 1)
base_drop(t) = hook_drop si t < hook_seconds, sinon base_drop. Sans `allow_increase`, le facteur est
plafonné à 1 (la courbe ne remonte jamais ; un cut récent freine seulement la chute).

`curve` a T+1 points : curve[i] = part des spectateurs présents à l'instant i s (curve[0] = 1) ;
la rétention de la seconde t de la timeline (fin de t) vaut curve[t+1].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RetentionResult:
    curve: np.ndarray          # (T+1,)
    avg_watch_s: float
    avg_watch_ratio: float
    completion: float

    @property
    def per_second(self) -> np.ndarray:
        """(T,) rétention à la fin de chaque seconde."""
        return self.curve[1:]


def simulate(attention: np.ndarray, cut_recent: np.ndarray, duration_s: float, *, hook_seconds: float,
             hook_drop: float, base_drop: float, k_attention: float, cut_bonus: float,
             allow_increase: bool = False) -> RetentionResult:
    att = np.clip(np.asarray(attention, float), 0, 1)
    cr = np.clip(np.asarray(cut_recent, float), 0, 1)
    T = att.size
    curve = np.ones(T + 1)
    for t in range(T):
        drop = hook_drop if t < hook_seconds else base_drop
        factor = 1.0 - drop - k_attention * (1.0 - att[t]) + cut_bonus * cr[t]
        if not allow_increase:
            factor = min(1.0, factor)
        curve[t + 1] = float(np.clip(curve[t] * max(0.0, factor), 0.0, 1.0))
    # intégrale (trapèzes) sur [0, durée] ; la dernière seconde peut être partielle
    duration = max(1e-6, float(duration_s))
    total = 0.0
    for t in range(T):
        length = min(1.0, max(0.0, duration - t)) if t == T - 1 else 1.0
        total += length * (curve[t] + curve[t + 1]) / 2.0
    total = min(total, duration)
    return RetentionResult(curve=curve, avg_watch_s=total, avg_watch_ratio=total / duration,
                           completion=float(curve[-1]))


def simulate_params(attention: np.ndarray, cut_recent: np.ndarray, duration_s: float, params: dict,
                    hook_seconds: float, allow_increase: bool = False) -> RetentionResult:
    return simulate(attention, cut_recent, duration_s, hook_seconds=hook_seconds,
                    hook_drop=float(params["hook_drop"]), base_drop=float(params["base_drop"]),
                    k_attention=float(params["k_attention"]), cut_bonus=float(params["cut_bonus"]),
                    allow_increase=allow_increase)
