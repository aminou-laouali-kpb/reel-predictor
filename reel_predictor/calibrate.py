"""Calibration du modèle de rétention sur des ratios de visionnage réels (contrat §6).

`calibrate(pairs)` : `pairs` = [(dossier de résultat, ratio réel = durée moyenne de visionnage / durée)].
Ajuste hook_drop, base_drop, k_attention, cut_bonus (moindres carrés bornés + léger rappel vers
weights.yaml), écrit data/calibration.json (surcharge `weights.yaml > retention` aux analyses suivantes).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import paths
from .config import RETENTION_PARAMS, Config, load_config
from .features import cut_recent_from
from .retention import simulate_params
from .timeline import read_timeline, write_json


def _load_reel(result_dir: Path, tau: float) -> tuple[np.ndarray, np.ndarray, float]:
    rows = read_timeline(Path(result_dir) / "timeline.jsonl")
    if not rows:
        raise ValueError("timeline vide")
    duration = float(len(rows))
    sp = Path(result_dir) / "summary.json"
    if sp.exists():
        duration = float(json.loads(sp.read_text(encoding="utf-8"))["video"]["duration_s"])
    att = np.array([float(r["indices"]["attention"]) for r in rows])
    since = np.array([float(r["since_cut"]) for r in rows])
    had = np.cumsum([bool(r["cut"]) for r in rows]) > 0
    return att, cut_recent_from(since, had, tau), duration


def calibrate(pairs: list[tuple[Path, float]], config: Config | None = None,
              out_file: Path | None | str = "default") -> dict:
    """out_file : défaut `paths.CALIBRATION_FILE` ; None = ne rien écrire."""
    if isinstance(out_file, str) and out_file == "default":
        out_file = paths.CALIBRATION_FILE
    cfg = config or load_config()
    if len(pairs) < 3:
        raise ValueError(f"Calibration impossible : au moins 3 reels nécessaires (reçu {len(pairs)})")
    tau = float(cfg.section("features")["cut_tau_s"])
    reels, skipped, warnings = [], [], []
    for result_dir, ratio in pairs:
        try:
            att, cr, dur = _load_reel(Path(result_dir), tau)
        except Exception as e:  # noqa: BLE001
            skipped.append({"result_dir": str(result_dir), "error": f"{type(e).__name__}: {e}"})
            continue
        r = float(ratio)
        if not np.isfinite(r) or r < 0:
            skipped.append({"result_dir": str(result_dir), "error": f"ratio invalide {ratio!r}"})
            continue
        if r > 1.0:
            warnings.append(f"{Path(result_dir).name} : ratio {r:.2f} > 1 (revisionnages) ramené à 1")
            r = 1.0
        reels.append((Path(result_dir), att, cr, dur, r))
    if len(reels) < 3:
        raise ValueError(f"Calibration impossible : {len(reels)} reel(s) exploitable(s), 3 nécessaires")

    rc = cfg.section("retention")
    hook_s = float(rc["hook_seconds"])
    allow = bool(rc["allow_increase"])
    names = list(RETENTION_PARAMS)
    bounds = [tuple(float(x) for x in rc["calibration_bounds"][k]) for k in names]
    prior = np.array([float(cfg.raw["weights"]["retention"][k]) for k in names])
    current = np.array([float(cfg.retention.as_dict()[k]) for k in names])
    span = np.array([max(1e-6, hi - lo) for lo, hi in bounds])
    ridge = float(rc["calibration_ridge"])
    real = np.array([r[4] for r in reels])

    def preds(x: np.ndarray) -> np.ndarray:
        p = dict(zip(names, x))
        return np.array([simulate_params(att, cr, dur, p, hook_s, allow).avg_watch_ratio
                         for _, att, cr, dur, _ in reels])

    def objective(x: np.ndarray) -> float:
        err = preds(x) - real
        return float(np.mean(err ** 2) + ridge * np.mean(((x - prior) / span) ** 2))

    from scipy.optimize import minimize

    clip = lambda x: np.array([np.clip(v, lo, hi) for v, (lo, hi) in zip(x, bounds)])  # noqa: E731
    best_x, best_f = clip(current), objective(clip(current))
    starts = [clip(current), clip(prior)] + [np.array([lo + f * (hi - lo) for lo, hi in bounds])
                                             for f in (0.15, 0.5)]
    for x0 in starts:
        for method in ("L-BFGS-B", "Powell"):
            try:
                r = minimize(objective, x0, method=method, bounds=bounds)
                x = clip(r.x)
                f = objective(x)
                if f < best_f:
                    best_x, best_f = x, f
            except Exception:  # noqa: BLE001
                continue
    before = preds(current)
    after = preds(best_x)
    mae_before = float(np.mean(np.abs(before - real)))
    mae_after = float(np.mean(np.abs(after - real)))
    if mae_after > mae_before:          # aucune amélioration : on garde les paramètres actuels
        best_x, after, mae_after = current, before, mae_before
    params = {k: round(float(v), 6) for k, v in zip(names, best_x)}
    result = {"n": len(reels), "params": params, "mae_before": round(mae_before, 5), "mae_after": round(mae_after, 5)}
    if out_file is not None:
        out_file = Path(out_file)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        write_json(out_file, {
            "version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **result,
            "reels": [{"result_dir": str(d), "real": round(float(x), 4), "pred_before": round(float(b), 4),
                       "pred_after": round(float(a), 4)} for (d, *_), x, b, a in zip(reels, real, before, after)],
            "skipped": skipped,
            "warnings": warnings,
        })
    if skipped:
        result["skipped"] = skipped
    if warnings:
        result["warnings"] = warnings
    return result
