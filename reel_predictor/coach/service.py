"""Service coach : brief -> Claude -> simulation locale des variantes -> (2ᵉ passe) -> export JSON + Markdown.

`run_coach(job_dir, progress)` écrit `data/results/<job>/coach/<AAAAMMJJ-HHMMSS>.json` (+ `.md`) et renvoie le dict
du contrat §7 : `{"created_at", "model", "usage", "cost_usd", "output", "brief", "simulation", "warnings"}`
(+ `coach_id`, `job_id`, `stamp`, `estimate`, `ranking`, `rounds`, `config`).
Si la simulation échoue, la sortie de Claude est quand même renvoyée (avec un avertissement).
"""
from __future__ import annotations

import json
import logging
import os
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import yaml

from .. import paths
from .brief import build_brief
from .schema import CoachOutput
from .targets import compare_to_original, evaluate, rank_key, resolve_targets, to_sim_scale

log = logging.getLogger("reel_predictor.coach.service")

COACH_DIR = "coach"
STAMP_FMT = "%Y%m%d-%H%M%S"
ProgressFn = Callable[[str, float, str], None]

STAGE_LABELS = {
    "brief": "Diagnostic",
    "llm": "Rédaction par Claude",
    "simulate": "Simulation des variantes",
    "round2": "Seconde passe Claude",
    "simulate2": "Simulation de la variante révisée",
    "write": "Écriture du rapport",
}
ROLE_LABELS = {"hook": "Accroche", "fact": "Chiffre clé", "body": "Développement", "cta": "Appel à l'action"}
SHOT_LABELS = {"face_cam": "Face caméra", "close_up": "Plan serré", "screen": "Écran", "broll": "B-roll",
               "text_card": "Carton texte"}
ACTION_LABELS = {"cut": "Cut", "trim": "Raccourcir", "zoom": "Zoom", "text": "Texte", "broll": "B-roll",
                 "sfx": "Effet sonore", "music": "Musique", "reorder": "Réordonner"}
INDEX_LABELS = {"attention": "Attention", "action": "Action", "tension": "Tension", "manipulation": "Manipulation",
                "emotion": "Émotion", "memory": "Mémoire"}
MOMENT_LABELS = {"hook": "Accroche", "fact": "Chiffre clé", "cta": "Appel à l'action"}


def load_coach_config(path: Path | None = None) -> dict:
    with open(path or paths.CONFIG_DIR / "coach.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def make_coach_id(job_id: str, stamp: str) -> str:
    return f"{job_id}.{stamp}"


def split_coach_id(coach_id: str) -> tuple[str, str]:
    job_id, _, stamp = coach_id.rpartition(".")
    return job_id, stamp


def _now_stamp() -> str:
    return datetime.now().strftime(STAMP_FMT)


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _sum_usage(a: dict, b: dict) -> dict:
    return {k: int(a.get(k, 0)) + int(b.get(k, 0)) for k in set(a) | set(b)}


def ranking(sim: dict | None) -> dict:
    """Ordre des résultats simulés (meilleur d'abord) et meilleure VARIANTE (index dans output.variants)."""
    if not sim or not sim.get("results"):
        return {"order": [], "best_variant": None, "original_rank": None}
    res = sim["results"]
    order = sorted(range(len(res)), key=lambda i: rank_key(res[i]["evaluation"], float(res[i].get("score") or 0)),
                   reverse=True)
    best = next((i for i in order if i > 0), None)
    return {"order": order, "best_variant": (best - 1) if best is not None else None,
            "original_rank": order.index(0) + 1 if 0 in order else None}


def apply_sim_targets(sim: dict | None, job_dir: Path, tcfg_real: dict) -> None:
    """Réévalue les résultats simulés avec les cibles converties dans l'échelle de la simulation, et ajoute
    la comparaison de chaque variante à l'original re-voicé (`vs_original` : gains / régressions)."""
    if not sim or not sim.get("results"):
        return
    try:
        real_rows = [json.loads(line) for line in (Path(job_dir) / "timeline.jsonl").read_text(encoding="utf-8").splitlines()
                     if line.strip()]
    except OSError:
        real_rows = []
    res = sim["results"]
    tcfg_sim = to_sim_scale(tcfg_real, real_rows, res[0].get("timeline") or [])
    for r in res:
        moments = (r.get("evaluation") or {}).get("moments") or []
        if r.get("timeline"):
            r["evaluation"] = evaluate(r["timeline"], moments, tcfg_sim)
    for r in res[1:]:
        if r.get("evaluation") and res[0].get("evaluation"):
            r["vs_original"] = compare_to_original(r["evaluation"], res[0]["evaluation"])
    sim["targets"] = tcfg_sim


def _default_sim_fns() -> tuple[Callable, Callable]:
    from .simulate import baseline_variant, simulate  # import paresseux : module de l'agent SIMULATION

    return simulate, baseline_variant


def _load_pipeline_config() -> Any:
    from ..config import load_config

    return load_config()


def run_coach(job_dir: Path, progress: ProgressFn | None = None, *, stamp: str | None = None,
              coach_cfg: dict | None = None, client: Any | None = None, config: Any | None = None,
              simulate_fn: Callable | None = None, baseline_fn: Callable | None = None,
              write: bool = True) -> dict:
    from . import llm

    job_dir = Path(job_dir)
    coach_cfg = coach_cfg if coach_cfg is not None else load_coach_config()
    # cibles effectives (relatives à vos Reels si `targets.mode: relative`), échelle des analyses réelles
    coach_cfg = {**coach_cfg, "targets": resolve_targets(coach_cfg["targets"])}
    stamp = stamp or _now_stamp()
    job_id = job_dir.name
    rounds_cfg = int(coach_cfg.get("rounds", 1))
    n_variants = int(coach_cfg.get("n_variants", 3))
    warnings: list[str] = []

    sim_span = (0.35, 0.95) if rounds_cfg < 2 else (0.35, 0.65)

    def report(stage: str, pct: float, msg: str) -> None:
        if progress:
            try:
                progress(stage, max(0.0, min(0.999, pct)), msg)
            except Exception:
                pass

    # 1. diagnostic local
    report("brief", 0.01, "Diagnostic de la vidéo (local)…")
    brief = build_brief(job_dir, coach_cfg)

    # 2. Claude
    report("llm", 0.03, "Préparation de la requête Claude…")
    try:
        r1 = llm.generate(brief, coach_cfg, client=client, n_variants=n_variants,
                          progress=lambda f, m: report("llm", 0.03 + 0.32 * f, m))
    except llm.CoachLLMError as exc:
        if exc.cost_usd > 0:
            raise llm.CoachLLMError(exc.code, f"{exc} (coût déjà engagé : {exc.cost_usd:.2f} $)",
                                    cost_usd=exc.cost_usd, usage=exc.usage) from exc
        raise
    output: CoachOutput = r1.output
    usage = dict(r1.usage)
    cost = r1.cost_usd
    warnings.extend(r1.warnings)
    rounds = [{"round": 1, "model": r1.model, "usage": r1.usage, "cost_usd": r1.cost_usd, "estimate": r1.estimate,
               "stop_reason": r1.stop_reason, "fallback_used": r1.fallback_used, "request_id": r1.request_id}]

    # 3. simulation locale (original re-voicé + variantes)
    sim: dict | None = None
    try:
        sim_fn, base_fn = _default_sim_fns() if (simulate_fn is None or baseline_fn is None) else (simulate_fn, baseline_fn)
        simulate_fn, baseline_fn = simulate_fn or sim_fn, baseline_fn or base_fn
        cfg = config if config is not None else _load_pipeline_config()
        report("simulate", sim_span[0], "Simulation locale : voix de synthèse + modèle cérébral…")
        a, b = sim_span
        sim = simulate_fn([baseline_fn(job_dir), *output.variants], job_dir, cfg, coach_cfg,
                          progress=lambda f, m: report("simulate", a + (b - a) * float(f), m))
        if not sim or len(sim.get("results") or []) != 1 + len(output.variants):
            raise ValueError("résultats de simulation incomplets")
        apply_sim_targets(sim, job_dir, coach_cfg["targets"])
    except Exception as exc:  # la sortie de Claude reste utile sans simulation
        log.warning("Simulation en échec : %s\n%s", exc, traceback.format_exc())
        warnings.append(f"Simulation des variantes impossible ({exc.__class__.__name__} : {exc}). "
                        "Les suggestions de Claude sont affichées sans indices simulés.")
        sim = None

    # 4. seconde passe (optionnelle) sur la meilleure variante simulée
    if rounds_cfg >= 2 and sim is not None:
        rk = ranking(sim)
        vi = rk["best_variant"]
        if vi is not None:
            report("round2", 0.66, "Seconde passe : Claude corrige les segments faibles…")
            try:
                revised, r2 = llm.revise_variant(brief, coach_cfg, output, vi, sim["results"][vi + 1], sim["results"][0],
                                                 client=client, budget_spent_usd=cost,
                                                 progress=lambda f, m: report("round2", 0.66 + 0.09 * f, m))
                usage = _sum_usage(usage, r2.usage)
                cost = round(cost + r2.cost_usd, 4)
                warnings.extend(f"Passe 2 : {w}" for w in r2.warnings)
                rounds.append({"round": 2, "model": r2.model, "usage": r2.usage, "cost_usd": r2.cost_usd,
                               "estimate": r2.estimate, "stop_reason": r2.stop_reason,
                               "fallback_used": r2.fallback_used, "request_id": r2.request_id, "revised_from": vi})
                revised = revised.model_copy(update={"angle": f"{revised.angle} (révisée)"})
                output = output.model_copy(update={"variants": [*output.variants, revised]})
                try:
                    report("simulate2", 0.76, "Simulation de la variante révisée…")
                    s2 = simulate_fn([revised], job_dir, cfg, coach_cfg,
                                     progress=lambda f, m: report("simulate2", 0.76 + 0.2 * float(f), m))
                    res2 = (s2 or {}).get("results") or []
                    if len(res2) != 1:
                        raise ValueError("résultat de simulation manquant")
                    res2[0]["name"] = f"variante {len(output.variants)} (révisée)"
                    sim["results"].append(res2[0])
                    apply_sim_targets(sim, job_dir, coach_cfg["targets"])
                except Exception as exc:
                    output = output.model_copy(update={"variants": output.variants[:-1]})
                    warnings.append(f"Variante révisée non simulée ({exc}) : elle n'est pas affichée.")
            except llm.CostLimitError as exc:
                warnings.append(f"Seconde passe annulée : {exc}")
            except llm.CoachLLMError as exc:
                cost = round(cost + exc.cost_usd, 4)
                warnings.append(f"Seconde passe en échec : {exc}")

    rk = ranking(sim)
    if sim is not None:
        sim["best"] = rk["order"][0] if rk["order"] else sim.get("best")
        if rk["original_rank"] == 1:
            warnings.append("Dans la simulation, l'original re-voicé reste le mieux classé : "
                            "privilégiez le plan de montage plutôt qu'une réécriture.")

    llm_cfg = coach_cfg.get("llm") or {}
    result = {
        "version": 1,
        "coach_id": make_coach_id(job_id, stamp),
        "job_id": job_id,
        "stamp": stamp,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model": r1.model,
        "usage": usage,
        "cost_usd": round(cost, 4),
        "estimate": r1.estimate,
        "output": output.model_dump(),
        "brief": brief,
        "simulation": sim,
        "ranking": rk,
        "rounds": rounds,
        "config": {"model": llm_cfg.get("model"), "effort": llm_cfg.get("effort"), "n_variants": n_variants,
                   "rounds": rounds_cfg, "targets": coach_cfg.get("targets"),
                   "targets_sim": (sim or {}).get("targets"),
                   "max_words_per_card": coach_cfg.get("max_words_per_card"),
                   "reading_wps": coach_cfg.get("reading_wps")},
        "warnings": warnings,
    }
    if write:
        report("write", 0.98, "Écriture du rapport…")
        out_dir = job_dir / COACH_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_atomic(out_dir / f"{stamp}.json", json.dumps(result, ensure_ascii=False, indent=1))
        _write_atomic(out_dir / f"{stamp}.md", to_markdown(result))
    return result


# ----------------------------------------------------------------------------- export Markdown
def _ft(s: Any) -> str:
    try:
        s = float(s)
    except (TypeError, ValueError):
        return "–"
    return f"{int(s // 60)}:{int(s % 60):02d}"


def _cell(s: Any) -> str:
    return " ".join(str(s or "").split()).replace("|", "\\|") or "—"


def _num(v: Any) -> str:
    try:
        return f"{float(v):.2f}".replace(".", ",")
    except (TypeError, ValueError):
        return "–"


def variant_script_text(v: dict, spans: list[dict] | None = None, title: str = "") -> str:
    """Script prêt à tourner (texte brut) — même format que le bouton « Copier le script » de l'UI."""
    lines = [title or f"Variante — {v.get('angle', '')}", ""]
    for i, s in enumerate(v.get("segments") or []):
        span = (spans or [])[i] if spans and i < len(spans) else None
        when = f"[{_ft(span['t'])}–{_ft(span['t_end'])}] " if span else ""
        lines.append(f"{when}{ROLE_LABELS.get(s['role'], s['role']).upper()}")
        lines.append(f"Voix off : {s['voice_over']}")
        if s.get("on_screen_text"):
            lines.append(f"Texte à l'écran : {s['on_screen_text']}")
        shot = SHOT_LABELS.get(s["shot"], s["shot"]) + (" (cut)" if s.get("cut_at_start") else "")
        lines.append(f"Plan : {shot}")
        if s.get("edit"):
            lines.append(f"Montage : {s['edit']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def to_markdown(result: dict) -> str:
    out = result.get("output") or {}
    sim = result.get("simulation")
    res = (sim or {}).get("results") or []
    rk = result.get("ranking") or {}
    brief = result.get("brief") or {}
    tg = (result.get("config") or {}).get("targets") or {}
    peak, floor = tg.get("peak", 0.9), tg.get("floor", 0.5)
    L: list[str] = []
    L.append(f"# Coach script & montage — {result.get('job_id')}")
    L.append("")
    L.append(f"- Généré le {result.get('created_at')} · modèle {result.get('model')} · "
             f"coût réel {_num(result.get('cost_usd', 0))} $")
    if sim:
        L.append(f"- Simulation : mode `{sim.get('mode')}`" + (f" ({sim.get('reason')})" if sim.get("reason") else ""))
    basis = tg.get("basis") or {}
    if basis.get("mode") == "relative":
        pbi = tg.get("peak_by_index") or {}
        L.append(f"- Cibles relatives à vos {basis.get('n_reels')} Reels analysés : pics ≥ "
                 f"{int(basis.get('peak_percentile', 90))}ᵉ percentile aux moments clés ("
                 + ", ".join(f"{INDEX_LABELS.get(k, k)} {_num(v)}" for k, v in pbi.items())
                 + f"), plancher {INDEX_LABELS.get(tg.get('floor_index', 'attention'), 'attention')} ≥ {_num(floor)}")
        L.append("- Dans les tableaux de simulation, ces cibles sont converties à l'échelle de la simulation "
                 "(correspondance de quantiles avec l'original re-voicé).")
    else:
        L.append(f"- Cibles : pics ≥ {_num(peak)} aux moments clés, plancher ≥ {_num(floor)} ailleurs"
                 + (f" ({basis['reason']})" if basis.get("reason") else ""))
    L.append("")
    L.append("> Simulation sans image et avec une voix de synthèse ; indices spéculatifs et non calibrés. "
             "Les gains simulés ne garantissent pas les vues.")
    L.append("")
    if result.get("warnings"):
        L.append("## Avertissements")
        L += [f"- {w}" for w in result["warnings"]]
        L.append("")
    L.append("## Diagnostic")
    L.append(out.get("diagnosis_summary", ""))
    L.append("")
    ev = brief.get("evaluation") or {}
    if ev.get("moments"):
        L.append("| Moment clé (original) | Fenêtre | Valeurs | Cible atteinte |")
        L.append("|---|---|---|---|")
        for m in ev["moments"]:
            vals = ", ".join(f"{INDEX_LABELS.get(k, k)} {_num(v)}" for k, v in m["values"].items())
            L.append(f"| {MOMENT_LABELS.get(m['name'], m['name'])} | {_ft(m['t'])}–{_ft(m['t_end'])} | {vals} | "
                     f"{'oui' if m['met'] else 'non'} |")
        L.append(f"\nSecondes sous le plancher (original) : {len(ev.get('seconds_below_floor') or [])}")
        L.append("")
    L.append("## Plan de montage (vidéo actuelle, sans retourner)")
    plan = out.get("montage_plan") or []
    if plan:
        L.append("| Temps | Action | Consigne | Texte à l'écran | Effet attendu |")
        L.append("|---|---|---|---|---|")
        for a in plan:
            L.append(f"| {_ft(a['t'])}–{_ft(a['t_end'])} | {ACTION_LABELS.get(a['action'], a['action'])} | "
                     f"{_cell(a['instruction'])} | {_cell(a.get('on_screen_text'))} | {_cell(a.get('expected_effect'))} |")
    else:
        L.append("_Aucune action proposée._")
    L.append("")
    order = [i - 1 for i in rk.get("order") or [] if i > 0] or list(range(len(out.get("variants") or [])))
    base = res[0] if res else None
    L.append("## Variantes de script" + (" (meilleure d'abord)" if res else ""))
    for vi in order:
        variants = out.get("variants") or []
        if vi >= len(variants):
            continue
        v = variants[vi]
        r = res[vi + 1] if len(res) > vi + 1 else None
        badge = " — **recommandée**" if vi == rk.get("best_variant") else ""
        L.append("")
        L.append(f"### Variante {vi + 1} — {v['angle']}{badge}")
        L.append("")
        L.append(f"**Accroche** : {v['hook_summary']}")
        L.append("")
        if r:
            e = r.get("evaluation") or {}
            be = (base or {}).get("evaluation") or {}
            L.append(f"Score simulé {r.get('score')} (original re-voicé {base.get('score') if base else '–'}) · "
                     f"durée {_num(r.get('duration_s'))} s · moments clés atteints {round(100 * e.get('attainment', 0))} % "
                     f"(original {round(100 * be.get('attainment', 0))} %) · secondes sous le plancher "
                     f"{len(e.get('seconds_below_floor') or [])} (original {len(be.get('seconds_below_floor') or [])})")
            L.append("")
            bm = {m["name"]: m for m in be.get("moments") or []}
            if e.get("moments"):
                L.append("| Moment clé | Indice | Original | Variante | Cible |")
                L.append("|---|---|---|---|---|")
                for m in e["moments"]:
                    for k, val in m["values"].items():
                        ov = (bm.get(m["name"]) or {}).get("values", {}).get(k)
                        tgt = (m.get("targets") or {}).get(k)
                        trend = ""
                        if ov is not None:
                            d = float(val) - float(ov)
                            trend = " ▲" if d > 0.02 else (" ▼" if d < -0.02 else "")
                        L.append(f"| {MOMENT_LABELS.get(m['name'], m['name'])} | {INDEX_LABELS.get(k, k)} | {_num(ov)} | "
                                 f"{_num(val)}{trend} | {('≥ ' + _num(tgt) + ' : ') if tgt is not None else ''}"
                                 f"{'atteinte' if m['met_by_index'].get(k) else 'non'} |")
                L.append("")
                regs = ((r or {}).get("vs_original") or {}).get("regressions") or []
                if regs:
                    L.append("**Régressions vs original** : " + " ; ".join(
                        f"{INDEX_LABELS.get(g['index'], g['index'])} ({MOMENT_LABELS.get(g['moment'], g['moment']).lower()}) "
                        f"{_num(g['original'])} → {_num(g['variant'])}" for g in regs))
                    L.append("")
        spans = (r or {}).get("segments")
        L.append("| Temps | Voix off | Texte à l'écran | Plan | Montage |")
        L.append("|---|---|---|---|---|")
        for i, s in enumerate(v["segments"]):
            sp = spans[i] if spans and i < len(spans) else None
            when = f"{_ft(sp['t'])}–{_ft(sp['t_end'])}" if sp else f"#{i + 1}"
            shot = SHOT_LABELS.get(s["shot"], s["shot"]) + (" ✂" if s["cut_at_start"] else "")
            L.append(f"| {when} {ROLE_LABELS.get(s['role'], s['role'])} | {_cell(s['voice_over'])} | "
                     f"{_cell(s['on_screen_text'])} | {shot} | {_cell(s['edit'])} |")
        L.append("")
        L.append(f"_Pourquoi_ : {v['rationale']}")
    if out.get("notes"):
        L.append("")
        L.append("## Notes et limites")
        L += [f"- {n}" for n in out["notes"]]
    L.append("")
    return "\n".join(L)
