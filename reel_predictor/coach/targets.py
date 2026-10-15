"""Évaluation des cibles « pics aux moments clés + plancher » (contrat partagé coach / simulation / UI).

Un moment clé = une fenêtre temporelle + les indices visés. Il est « atteint » quand le maximum de chacun
de ses indices sur la fenêtre atteint sa cible (`peak_by_index[k]`, sinon `peak`). Le plancher compte les
secondes où `floor_index` < `floor`.

Cibles relatives (`mode: relative`) : la cible d'un indice = son `peak_percentile`ᵉ percentile sur les vrais
Reels déjà analysés (« aussi haut que vos meilleurs moments »), le plancher = le `floor_percentile`ᵉ percentile
de `floor_index`. La simulation (voix de synthèse, sans image) n'a pas la même échelle que les analyses réelles :
`to_sim_scale` convertit les cibles par correspondance de quantiles entre l'original analysé et l'original re-voicé.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

INDEX_NAMES = ("attention", "action", "tension", "manipulation", "emotion", "memory")


def peak_for(tcfg: dict, k: str) -> float:
    return float((tcfg.get("peak_by_index") or {}).get(k, tcfg["peak"]))


def moments_from_roles(spans: list[tuple[str, float, float]], duration_s: float, tcfg: dict) -> list[dict]:
    """`spans` = [(role, t0, t1)] des segments ; renvoie [{name, t, t_end, indices}] (un par moment configuré)."""
    out = []
    for name, m in tcfg["moments"].items():
        roles = set(m.get("roles", []))
        win = m.get("window_s")
        if win is not None:
            t0, t1 = float(win[0]), min(float(win[1]), duration_s)
        else:
            # premier bloc CONTIGU de segments de ce rôle : un segment étiqueté loin plus tard ne doit pas
            # étirer la fenêtre (sinon le max sur la fenêtre gonfle artificiellement le moment)
            hits = sorted((a, b) for r, a, b in spans if r in roles)
            if not hits:
                continue
            t0, t1 = hits[0]
            for a, b in hits[1:]:
                if a > t1 + 0.5:
                    break
                t1 = max(t1, b)
        if t1 > t0:
            out.append({"name": name, "t": round(t0, 2), "t_end": round(t1, 2), "indices": list(m["indices"])})
    return out


def evaluate(rows: list[dict], moments: list[dict], tcfg: dict) -> dict:
    """`rows` = lignes de timeline ({t, indices{...}}). Renvoie le détail par moment + synthèse."""
    floor = float(tcfg["floor"])
    fidx = tcfg.get("floor_index", "attention")
    res_m, met_pairs, n_pairs = [], 0, 0
    for m in moments:
        m = {k: m[k] for k in ("name", "t", "t_end", "indices")}
        sel = [r for r in rows if m["t"] - 0.5 < r["t"] + 0.5 and r["t"] < m["t_end"]] or rows[:1]
        vals = {k: round(max(float(r["indices"][k]) for r in sel), 3) for k in m["indices"]}
        tgt = {k: round(peak_for(tcfg, k), 3) for k in m["indices"]}
        ok = {k: v >= tgt[k] for k, v in vals.items()}
        met_pairs += sum(ok.values())
        n_pairs += len(ok)
        res_m.append({**m, "values": vals, "targets": tgt, "met": all(ok.values()), "met_by_index": ok})
    below = [int(r["t"]) for r in rows if float(r["indices"][fidx]) < floor]
    means = {k: round(sum(float(r["indices"][k]) for r in rows) / max(1, len(rows)), 3) for k in INDEX_NAMES}
    return {
        "peak": float(tcfg["peak"]), "peak_by_index": tcfg.get("peak_by_index"), "floor": floor, "floor_index": fidx,
        "moments": res_m,
        "attainment": round(met_pairs / n_pairs, 3) if n_pairs else 0.0,
        "seconds_below_floor": below,
        "index_means": means,
    }


def rank_key(ev: dict, score: float) -> tuple:
    """Clé de tri (décroissante) : moments atteints, puis moins de secondes sous le plancher, puis score."""
    return (ev["attainment"], -len(ev["seconds_below_floor"]), score)


# ----------------------------------------------------------------------------- cibles relatives
def _real_timelines(results_dir: Path) -> list[list[dict]]:
    """Timelines des vraies analyses avec cerveau (pas les simulations du coach)."""
    out = []
    for d in sorted(Path(results_dir).glob("*/")):
        try:
            summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            if not (summary.get("brain") or {}).get("available"):
                continue
            rows = [json.loads(line) for line in (d / "timeline.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        except (OSError, ValueError):
            continue
        if rows:
            out.append(rows)
    return out


def resolve_targets(tcfg: dict, results_dir: Path | None = None) -> dict:
    """Cibles effectives (échelle des analyses réelles) + `basis` (comment elles ont été obtenues)."""
    out = {k: v for k, v in tcfg.items() if k not in ("peak_by_index", "basis")}
    if tcfg.get("mode", "absolute") != "relative":
        out["basis"] = {"mode": "absolute"}
        return out
    if results_dir is None:
        from .. import paths
        results_dir = paths.RESULTS_DIR
    tls = _real_timelines(results_dir)
    min_reels = int(tcfg.get("min_reels", 2))
    if len(tls) < min_reels:
        out["basis"] = {"mode": "absolute", "n_reels": len(tls),
                        "reason": f"cible relative indisponible : {len(tls)} Reel(s) analysé(s) avec le modèle "
                                  f"cérébral, il en faut {min_reels} — cibles fixes utilisées"}
        return out
    rows = [r for tl in tls for r in tl]
    pp, fp = float(tcfg.get("peak_percentile", 90)), float(tcfg.get("floor_percentile", 25))
    lo, hi = float(tcfg.get("peak_min", 0.5)), float(tcfg.get("peak_max", 0.99))
    out["peak_by_index"] = {k: round(float(np.clip(np.percentile([float(r["indices"][k]) for r in rows], pp), lo, hi)), 3)
                            for k in INDEX_NAMES}
    fidx = tcfg.get("floor_index", "attention")
    out["floor"] = round(float(np.percentile([float(r["indices"][fidx]) for r in rows], fp)), 3)
    out["basis"] = {"mode": "relative", "n_reels": len(tls), "n_seconds": len(rows),
                    "peak_percentile": pp, "floor_percentile": fp, "scale": "analyses réelles"}
    return out


def _map_value(x: float, real: np.ndarray, sim: np.ndarray) -> float:
    """Correspondance de quantiles real -> sim, extrapolation linéaire (rapport des écarts-types) hors plage."""
    real, sim = np.sort(np.asarray(real, float)), np.sort(np.asarray(sim, float))
    if real.size < 2 or sim.size < 2:
        return x
    if real[0] <= x <= real[-1]:
        q = float(np.interp(x, real, np.linspace(0.0, 1.0, real.size)))
        return float(np.quantile(sim, q))
    scale = float(sim.std()) / float(real.std()) if real.std() > 1e-6 else 1.0
    edge_r, edge_s = (real[-1], sim[-1]) if x > real[-1] else (real[0], sim[0])
    return float(edge_s + (x - edge_r) * scale)


def to_sim_scale(tcfg: dict, real_rows: list[dict], sim_rows: list[dict]) -> dict:
    """Cibles converties dans l'échelle de la simulation, via l'original (analyse réelle vs re-voicé)."""
    if not real_rows or not sim_rows:
        return {**tcfg, "basis": {**(tcfg.get("basis") or {}), "scale": "non convertie (original manquant)"}}
    col = lambda rows, k: [float(r["indices"][k]) for r in rows]  # noqa: E731
    lo, hi = 0.05, float(tcfg.get("peak_max", 0.99))
    peaks = {k: round(float(np.clip(_map_value(peak_for(tcfg, k), col(real_rows, k), col(sim_rows, k)), lo, hi)), 3)
             for k in INDEX_NAMES}
    fidx = tcfg.get("floor_index", "attention")
    floor = round(float(np.clip(_map_value(float(tcfg["floor"]), col(real_rows, fidx), col(sim_rows, fidx)), lo, hi)), 3)
    return {**tcfg, "peak_by_index": peaks, "floor": floor,
            "basis": {**(tcfg.get("basis") or {}), "scale": "simulation (quantiles de l'original re-voicé)",
                      "real_scale": {"peak_by_index": {k: round(peak_for(tcfg, k), 3) for k in INDEX_NAMES},
                                     "floor": float(tcfg["floor"])}}}


# ----------------------------------------------------------------------------- comparaison à l'original
def compare_to_original(ev: dict, ev_orig: dict, tol: float = 0.02) -> dict:
    """Gains / régressions d'une variante vs l'original (moments clés + moyennes), tolérance `tol`."""
    om = {m["name"]: m for m in ev_orig.get("moments") or []}
    items = []
    for m in ev.get("moments") or []:
        o = om.get(m["name"])
        if not o:
            continue
        for k, v in m["values"].items():
            if k in o["values"]:
                items.append({"moment": m["name"], "index": k, "original": o["values"][k], "variant": v,
                              "delta": round(v - o["values"][k], 3)})
    means = {k: round(ev["index_means"][k] - ev_orig["index_means"][k], 3)
             for k in INDEX_NAMES if k in ev.get("index_means", {}) and k in ev_orig.get("index_means", {})}
    return {
        "moments": items,
        "means_delta": means,
        "regressions": [i for i in items if i["delta"] < -tol],
        "gains": [i for i in items if i["delta"] > tol],
        "below_floor_delta": len(ev.get("seconds_below_floor") or []) - len(ev_orig.get("seconds_below_floor") or []),
    }
