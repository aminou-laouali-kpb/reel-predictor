"""Activité cérébrale factice, déterministe et structurée (tests, démo sans TRIBE).

Chaque ROI suit un signal plausible : auditif <- parole, visuel/MT <- cuts + mouvement,
IPS/FEF <- cuts + nouveaux textes, insula/dACC <- variations de volume, prémoteur <- CTA,
OFC <- visage proche, parahippocampe/IFG/fusiforme <- texte + répétitions, etc.
Valeurs en « z » (moyenne ~0, écart-type ~1) comme TRIBE.
"""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from .. import paths
from .atlas import roi_masks
from .provider import BrainPrediction, BrainRequest

# ROI -> {feature: poids} (features f_* de features.py)
_DRIVERS: dict[str, dict[str, float]] = {
    "auditory": {"speech": 1.6, "loudness": 0.6},
    "early_visual": {"cut_recent": 1.2, "motion": 1.0},
    "mt": {"motion": 1.4, "cut_recent": 0.5},
    "ips": {"cut_recent": 1.0, "text_new": 0.8, "motion": 0.4},
    "fef": {"cut_recent": 0.9, "text_new": 0.6},
    "insula_ant": {"loud_jump": 1.2, "loud_var": 0.8},
    "dacc": {"loud_jump": 1.0, "loud_var": 0.8},
    "premotor_ventral": {"cta": 1.4, "speech_rate": 0.4},
    "precentral": {"cta": 1.0, "motion": 0.3},
    "sma": {"cta": 1.0},
    "supramarginal": {"cta": 0.6, "speech": 0.4},
    "ofc": {"face_close": 1.4},
    "subcallosal": {"face_close": 0.8},
    "temporal_pole": {"speech": 0.6, "face_close": 0.6},
    "angular": {"speech": 0.7, "persuasion": 0.6},
    "psts": {"speech": 0.9, "face_present": 0.5},
    "dmpfc": {"persuasion": 0.9, "speech": 0.3},
    "frontopolar": {"persuasion": 0.7},
    "rectus": {"face_close": 0.5, "persuasion": 0.4},
    "parahippocampal": {"text_present": 0.9, "keyword_repeat": 0.8},
    "ifg": {"keyword_repeat": 0.8, "speech": 0.5},
    "fusiform": {"text_present": 1.0, "face_present": 0.6},
    "mfg": {"text_density": 0.8},
    "retrosplenial": {"cut_recent": 0.5},
}


def _zscore(x: np.ndarray) -> np.ndarray:
    s = float(np.std(x))
    return (x - float(np.mean(x))) / s if s > 1e-6 else np.zeros_like(x)


class MockProvider:
    name = "mock"

    def __init__(self, roi_mapping: dict, noise: float = 0.35, seed: int | None = None):
        self.roi_mapping = roi_mapping
        self.noise = noise
        self.seed = seed

    def predict(self, request: BrainRequest) -> BrainPrediction:
        T = request.n_seconds
        df = request.features if request.features is not None else pd.DataFrame({"t": np.arange(T)})
        seed = self.seed if self.seed is not None else zlib.crc32(f"{T}:{request.duration_s:.2f}".encode())
        rng = np.random.default_rng(seed)
        masks = roi_masks(self.roi_mapping)
        preds = np.zeros((T, paths.N_VERTICES), np.float32)
        # fond : bruit spatial fixe + dérive lente
        preds += rng.normal(0, self.noise, size=(1, paths.N_VERTICES)).astype(np.float32)
        drift = np.sin(np.linspace(0, 3 * np.pi, T))[:, None] * 0.2
        preds += drift.astype(np.float32)
        for roi, drivers in _DRIVERS.items():
            if roi not in masks:
                continue
            sig = np.zeros(T)
            for feat, w in drivers.items():
                col = f"f_{feat}"
                if col in df.columns:
                    sig += w * df[col].to_numpy(float)[:T]
            z = _zscore(sig) if np.any(sig) else np.zeros(T)
            preds[:, masks[roi]] += z[:, None].astype(np.float32)
        preds += rng.normal(0, self.noise, size=preds.shape).astype(np.float32)
        if request.progress:
            request.progress(1.0, "Activité simulée (mock)")
        return BrainPrediction(available=True, provider="mock", model="mock", preds=preds,
                               seconds=np.arange(T, dtype=np.int64), degraded=False, reason=None,
                               meta={"seed": int(seed), "text_feature": "simulated"})
