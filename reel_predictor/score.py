"""Score global 0..100 : moyennes d'indices pondérées (+ rétention), bonus d'accroche, pénalité de creux."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import INDEX_NAMES, Config
from .retention import RetentionResult


def runs_below(x: np.ndarray, threshold: float, min_len: int) -> list[tuple[int, int]]:
    """Intervalles [début, fin] (inclus) où x < seuil pendant au moins `min_len` secondes."""
    out, start = [], None
    for i, v in enumerate(list(x) + [np.inf]):
        if v < threshold:
            start = i if start is None else start
        else:
            if start is not None and i - start >= min_len:
                out.append((start, i - 1))
            start = None
    return out


def hook_elements(df: pd.DataFrame, first_word_s: float | None, window_s: float, speech_start_max_s: float,
                  motion_min: float) -> dict[str, bool]:
    w = df[df["t"] < window_s]
    return {
        "text": bool((w["text_words"] > 0).any()),
        "face": bool((w["faces"] > 0).any()),
        "speech_start": first_word_s is not None and first_word_s < speech_start_max_s,
        "motion_or_cut": bool(w["cut"].any() or (w["f_motion"] >= motion_min).any()),
    }


@dataclass
class ScoreResult:
    score: int
    breakdown: dict


def compute_score(df: pd.DataFrame, indices: dict[str, np.ndarray], retention: RetentionResult,
                  first_word_s: float | None, cfg: Config) -> ScoreResult:
    sc = cfg.section("score")
    weights = {k: float(v) for k, v in sc["weights"].items()}
    means = {k: float(np.mean(indices[k])) if len(indices[k]) else 0.0 for k in INDEX_NAMES}
    comp = dict(means)
    comp["retention"] = float(retention.avg_watch_ratio)
    total_w = sum(weights.values())
    base_raw = sum(weights[k] * comp[k] for k in weights) / total_w
    lo, hi = float(sc["floor"]), float(sc["ceil"])
    base = 100.0 * float(np.clip((base_raw - lo) / (hi - lo), 0.0, 1.0))

    h = sc["hook"]
    elems = hook_elements(df, first_word_s, float(h["window_s"]), float(h["speech_start_max_s"]),
                          float(h["motion_min"]))
    hook = sum(float(h[k]) for k, ok in elems.items() if ok)
    hook = min(hook, float(h["max"]))

    d = sc["dips"]
    dips = runs_below(indices[d["index"]], float(d["threshold"]), int(d["min_duration_s"]))
    dip_secs = sum(e - s + 1 for s, e in dips)
    dip = -min(float(d["max"]), dip_secs * float(d["points_per_second"]))

    score = int(round(float(np.clip(base + hook + dip, 0, 100))))
    breakdown = {
        "index_means": {k: round(v, 4) for k, v in means.items()},
        "hook_bonus": round(hook, 2),
        "dip_penalty": round(dip, 2),
        "weights": weights,
        "base": round(base, 2),
        "retention_mean": round(comp["retention"], 4),
        "hook_elements": elems,
        "dips": [[int(s), int(e + 1)] for s, e in dips],
    }
    return ScoreResult(score=score, breakdown=breakdown)
