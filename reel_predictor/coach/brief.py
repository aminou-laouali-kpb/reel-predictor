"""Diagnostic structuré compact d'une analyse (entrée du coach Claude et base du plan de montage).

`build_brief(job_dir, coach_cfg)` lit `summary.json`, `timeline.jsonl` (et `words.json` s'il existe) et renvoie
un dict JSON compact (≈ 3–6 k tokens) :
- `video` : durée, score, moyennes, rétention, confiance par indice ;
- `segments` : phrases horodatées (depuis les mots Whisper) avec indices moyens/min/max, cuts, plan le plus long,
  texte à l'écran (mots vs temps de lecture), visages, rétention, rôle supposé (hook / fact / body / cta) ;
- `text_issues` : cartons de texte trop chargés ou affichés trop peu de temps (les pires) ;
- `key_moments` + `targets` : évaluation des cibles (`targets.evaluate`) sur la vidéo ORIGINALE ;
- `dips` : creux d'attention (sous le plancher, baisses relatives) et plans longs ;
- `series` : indices et rétention seconde par seconde (2 décimales) ;
- `recommendations` : règles déjà déclenchées par le pipeline.
Aucun appel réseau, aucune dépendance lourde.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from ..textutils import keyword_tokens, strip_accents, tokens
from .targets import INDEX_NAMES, evaluate, moments_from_roles, peak_for

# Découpage des phrases
SENTENCE_END = re.compile(r"[.!?…]+[»\"']?$")
PAUSE_SPLIT_S = 0.8          # silence entre deux mots qui force une nouvelle phrase
MAX_SEGMENT_S = 7.0          # phrase trop longue : coupée au mot suivant
MAX_SEGMENT_WORDS = 24
LONG_SHOT_S = 4.0            # plan sans cut au-delà duquel l'attention décroche en général
REL_DROP = 0.12              # baisse relative d'attention vs moyenne des 5 s précédentes
TEXT_CHARS = 90              # troncature des textes recopiés dans le brief
MAX_TEXT_ISSUES = 8          # cartons de texte problématiques recopiés (les pires)

# Mots-clés d'appel à l'action (complètent `weights.yaml > cta_keywords`) ; recherche aussi « collée » (OCR).
CTA_EXTRA = ["clique", "cliquez", "clic", "lien", "bio", "guide", "abonne", "commente", "partage", "enregistre",
             "telecharge", "inscris", "bouton", "dm", "ecris-moi", "prends", "suis-moi", "follow"]
NUMBER_WORDS = {"mille", "million", "millions", "milliard", "milliards", "cent", "cents", "pourcent", "pourcents"}
UNIT_WORDS = {"euros", "euro", "eur", "dollars", "dollar", "francs", "fcfa", "%", "pourcent", "ans", "mois",
              "jours", "heures", "fois", "km", "kg"}


# ----------------------------------------------------------------------------- lecture
def _read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_timeline(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r.get("t", 0))
    return rows


def _f(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        return x if math.isfinite(x) else default
    except (TypeError, ValueError):
        return default


def _r2(x: float) -> float:
    return round(float(x), 2)


def _clip(s: str, n: int = TEXT_CHARS) -> str:
    s = " ".join(str(s or "").split())
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def load_words(job_dir: Path, rows: list[dict]) -> list[dict]:
    """Mots horodatés : `words.json` (Whisper, précis) sinon les mots de la timeline (début = seconde)."""
    wf = job_dir / "words.json"
    if wf.is_file():
        try:
            data = _read_json(wf)
            words = [{"w": str(w["w"]), "start": _f(w.get("start")), "end": _f(w.get("end"), _f(w.get("start")))}
                     for w in data.get("words", []) if str(w.get("w", "")).strip()]
            if words:
                return sorted(words, key=lambda w: w["start"])
        except (ValueError, KeyError, TypeError, OSError):
            pass
    out = []
    for r in rows:
        ws = [w for w in (r.get("words") or []) if str(w).strip()]
        for i, w in enumerate(ws):
            s = int(r["t"]) + i / max(1, len(ws))
            out.append({"w": str(w), "start": s, "end": s + 1 / max(1, len(ws))})
    return out


def join_words(words: list[str]) -> str:
    """Recolle les mots Whisper (« d » + « 'un » -> « d'un », « c » + « -à » -> « c-à »)."""
    out = ""
    for w in words:
        w = w.strip()
        if not w:
            continue
        if not out or w[0] in "'’-,.)" or out.endswith(("'", "’", "-", "(")):
            out += w
        else:
            out += " " + w
    return out.strip()


def split_sentences(words: list[dict]) -> list[dict]:
    """Phrases horodatées : coupure sur ponctuation finale, pause > PAUSE_SPLIT_S ou longueur excessive."""
    sents: list[list[dict]] = []
    cur: list[dict] = []
    for i, w in enumerate(words):
        if cur:
            gap = w["start"] - cur[-1]["end"]
            too_long = (w["start"] - cur[0]["start"] > MAX_SEGMENT_S) or len(cur) >= MAX_SEGMENT_WORDS
            if gap > PAUSE_SPLIT_S or too_long:
                sents.append(cur)
                cur = []
        cur.append(w)
        if SENTENCE_END.search(w["w"].strip()):
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return [{"t": s[0]["start"], "t_end": s[-1]["end"], "text": join_words([w["w"] for w in s]),
             "n_words": len(s)} for s in sents]


# ----------------------------------------------------------------------------- signaux
def _cta_keywords(summary: dict) -> list[tuple[str, tuple[str, ...]]]:
    cfg = ((summary.get("config") or {}).get("weights") or {}).get("cta_keywords") or []
    return keyword_tokens([*cfg, *CTA_EXTRA])


def _compact(s: str) -> str:
    return "".join(tokens(s))


def find_cta(text: str, keywords: list[tuple[str, tuple[str, ...]]]) -> list[str]:
    """Mots-clés CTA présents (mots séparés OU collés par l'OCR, ex. « PRENDREMONGUIDE »)."""
    toks = tokens(text)
    comp = _compact(text)
    hits = set()
    for label, kt in keywords:
        m = len(kt)
        if any(tuple(toks[i:i + m]) == kt for i in range(len(toks) - m + 1)):
            hits.add(label)
        elif len("".join(kt)) >= 5 and "".join(kt) in comp:
            hits.add(label)
    return sorted(hits)


def number_salience(text: str) -> int:
    """0 = aucun chiffre ; 1 = petit nombre ; 2 = nombre marquant (≥ 10, %, montant, « mille »...)."""
    raw = strip_accents(text.lower())
    toks = tokens(raw)
    best = 0
    for i, t in enumerate(toks):
        if t.isdigit():
            nxt = toks[i + 1] if i + 1 < len(toks) else ""
            best = max(best, 2 if (int(t) >= 10 or nxt in UNIT_WORDS) else 1)
        elif t in NUMBER_WORDS:
            best = 2
    if "%" in raw or "€" in raw or "$" in raw:
        best = max(best, 2) if re.search(r"\d", raw) else best
    return best


def text_cards(rows: list[dict], reading_wps: float, max_words: int) -> list[dict]:
    """Textes à l'écran successifs : un « carton » = suite de secondes au même texte OCR."""
    cards: list[dict] = []
    for r in rows:
        txt = str(r.get("text") or "").strip()
        t = int(r["t"])
        if not txt:
            continue
        nw = int(_f(r.get("text_words"), len(txt.split())))
        if cards and not r.get("text_new") and cards[-1]["t_end"] == t:
            c = cards[-1]
            c["t_end"] = t + 1
            c["words"] = max(c["words"], nw)
            if len(txt) > len(c["text"]):
                c["text"] = txt
            continue
        cards.append({"t": t, "t_end": t + 1, "text": txt, "words": nw})
    for c in cards:
        dur = c["t_end"] - c["t"]
        need = c["words"] / max(0.1, reading_wps)
        c["shown_s"] = dur
        c["need_s"] = round(need, 1)
        c["too_long"] = c["words"] > max_words
        # L'OCR mélange titres et sous-titres incrustés (qui changent chaque seconde) : seul un manque net compte.
        c["too_fast"] = need - dur >= 1.5
        c["text"] = _clip(c["text"])
    return cards


def shots_from_cuts(cuts: list[float], duration: float) -> list[tuple[float, float]]:
    bounds = [0.0, *sorted(c for c in cuts if 0 < c < duration), duration]
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def _rows_in(rows: list[dict], t0: float, t1: float) -> list[dict]:
    sel = [r for r in rows if r["t"] + 1 > t0 and r["t"] < t1]
    if not sel and rows:
        sel = [min(rows, key=lambda r: abs(r["t"] + 0.5 - t0))]
    return sel


def find_dips(rows: list[dict], floor: float, idx: str = "attention") -> list[dict]:
    """Creux : secondes consécutives sous le plancher OU en baisse relative (≥ REL_DROP vs 5 s précédentes)."""
    vals = [_f(r["indices"].get(idx)) for r in rows]
    flags = []
    for i, v in enumerate(vals):
        prev = vals[max(0, i - 5):i]
        ref = max(prev) if prev else v
        rel = (ref - v) / ref if ref > 0 else 0.0
        flags.append((v < floor, rel >= REL_DROP))
    dips, cur = [], None
    for i, (below, drop) in enumerate(flags):
        if below or drop:
            if cur is None:
                cur = {"t": int(rows[i]["t"]), "t_end": int(rows[i]["t"]) + 1, "below_floor": below, "rel_drop": drop}
            else:
                cur["t_end"] = int(rows[i]["t"]) + 1
                cur["below_floor"] |= below
                cur["rel_drop"] |= drop
        elif cur is not None:
            dips.append(cur)
            cur = None
    if cur is not None:
        dips.append(cur)
    for d in dips:
        seg = [v for r, v in zip(rows, vals) if d["t"] <= r["t"] < d["t_end"]]
        d["min"] = _r2(min(seg))
        d["since_cut_max"] = _r2(max(_f(r.get("since_cut")) for r in rows if d["t"] <= r["t"] < d["t_end"]))
        d["kind"] = "sous le plancher" if d.pop("below_floor") else "baisse relative"
        d.pop("rel_drop")
    return [d for d in dips if d["t_end"] - d["t"] >= 2 or d["kind"] == "sous le plancher"]


# ----------------------------------------------------------------------------- brief
def build_brief(job_dir: Path, coach_cfg: dict) -> dict:
    job_dir = Path(job_dir)
    summary = _read_json(job_dir / "summary.json")
    rows = load_timeline(job_dir / "timeline.jsonl")
    if not rows:
        raise ValueError("timeline.jsonl vide : analyse incomplète.")
    tcfg = coach_cfg["targets"]
    reading_wps = float(coach_cfg.get("reading_wps", 3.5))
    max_words = int(coach_cfg.get("max_words_per_card", 8))
    video = summary.get("video") or {}
    duration = _f(video.get("duration_s"), float(rows[-1]["t"] + 1)) or float(rows[-1]["t"] + 1)
    cuts = [float(c) for c in (summary.get("cuts") or []) if _f(c, -1) >= 0]
    shots = shots_from_cuts(cuts, duration)
    cards = text_cards(rows, reading_wps, max_words)
    cta_kw = _cta_keywords(summary)

    # --- segments (phrases) ; chaque segment couvre jusqu'au début du suivant
    sents = split_sentences(load_words(job_dir, rows))
    if not sents:  # pas de parole : un segment par plan
        sents = [{"t": a, "t_end": b, "text": "", "n_words": 0} for a, b in shots]
    if sents[0]["t"] < 1.0:  # la 1re phrase couvre le tout début de la vidéo
        sents[0]["t"] = 0.0
    segs = []
    for i, s in enumerate(sents):
        t0 = s["t"]
        t1 = sents[i + 1]["t"] if i + 1 < len(sents) else duration
        t1 = max(t1, s["t_end"], t0 + 0.5)
        sel = _rows_in(rows, t0, t1)
        ind = {k: [_f(r["indices"].get(k)) for r in sel] for k in INDEX_NAMES}
        seg_cuts = [round(c, 1) for c in cuts if t0 <= c < t1]
        shot_len = max((b - a) for a, b in shots if a < t1 and b > t0) if shots else duration
        seg_cards = [c for c in cards if c["t"] < t1 and c["t_end"] > t0]
        ocr = " / ".join(dict.fromkeys(c["text"] for c in seg_cards))
        ret0 = _f(sel[0].get("retention"), 1.0)
        ret1 = _f(sel[-1].get("retention"), ret0)
        dur = t1 - t0
        segs.append({
            "i": i,
            "t": _r2(t0), "t_end": _r2(t1),
            "voice": _clip(s["text"], 220),
            "wps": _r2(s["n_words"] / dur) if dur > 0 else 0.0,
            "mean": {k: _r2(sum(v) / len(v)) for k, v in ind.items()},
            "min": {k: _r2(min(v)) for k, v in ind.items() if k in ("attention", "tension")},
            "max": {k: _r2(max(v)) for k, v in ind.items() if k in ("attention", "tension", "memory", "emotion", "action")},
            "cuts": seg_cuts,
            "shot_s": _r2(shot_len),
            "since_cut_max": _r2(max(_f(r.get("since_cut")) for r in sel)),
            "faces": max(int(_f(r.get("faces"))) for r in sel),
            "face_area": _r2(max(_f(r.get("face_area")) for r in sel)),
            "on_screen": _clip(ocr, 140),
            "on_screen_words": max((c["words"] for c in seg_cards), default=0),
            "text_issue": ("trop de mots" if any(c["too_long"] for c in seg_cards) else "") or
                          ("pas le temps de lire" if any(c["too_fast"] for c in seg_cards) else ""),
            "retention": [_r2(ret0), _r2(ret1)],
            "cta": find_cta(f"{s['text']} {ocr}", cta_kw),
            "_num": number_salience(s["text"]),
        })

    # --- rôles supposés de l'original
    for s in segs:
        s["role"] = "body"
    for s in segs:
        if s["t"] < 3.0:
            s["role"] = "hook"
    fact = None
    for want in (2, 1):
        cands = [s for s in segs if s["_num"] >= want and s["role"] != "hook"]
        if cands:
            fact = cands[0]
            break
    if fact is None:
        cands = [s for s in segs if s["_num"] > 0]
        fact = cands[0] if cands else None
    if fact is not None and fact["role"] != "hook":
        fact["role"] = "fact"
    cta_segs = [s for s in segs if s["cta"] and s["t"] >= 0.5 * duration]
    for s in cta_segs:
        s["role"] = "cta"
    for s in segs:
        s.pop("_num")

    spans = [(s["role"], s["t"], s["t_end"]) for s in segs]
    if fact is not None and fact["role"] == "hook":  # chiffre clé dans l'accroche : le moment « fact » = ce segment
        spans.append(("fact", fact["t"], fact["t_end"]))
    moments = moments_from_roles(spans, duration, tcfg)
    ev = evaluate(rows, moments, tcfg)
    peak = float(tcfg["peak"])
    for m in ev["moments"]:
        m["gap"] = {k: _r2(max(0.0, peak_for(tcfg, k) - v)) for k, v in m["values"].items()}
    missing = [name for name in tcfg["moments"] if name not in {m["name"] for m in moments}]

    # --- creux et plans longs
    floor = float(tcfg["floor"])
    fidx = tcfg.get("floor_index", "attention")
    dips = find_dips(rows, floor, fidx)
    long_shots = [{"t": _r2(a), "t_end": _r2(b), "s": _r2(b - a)} for a, b in shots if b - a >= LONG_SHOT_S]
    text_issues = sorted(({k: c[k] for k in ("t", "t_end", "text", "words", "need_s", "shown_s")}
                          for c in cards if c["too_long"] or c["too_fast"]),
                         key=lambda c: c["shown_s"] - c["need_s"])[:MAX_TEXT_ISSUES]
    text_issues.sort(key=lambda c: c["t"])

    retention = summary.get("retention") or {}
    curve = retention.get("curve") or []
    brain = summary.get("brain") or {}
    bd = summary.get("score_breakdown") or {}
    return {
        "version": 1,
        "job_id": summary.get("job_id") or job_dir.name,
        "video": {
            "duration_s": _r2(duration),
            "score": summary.get("score"),
            "index_means": {k: _r2(_f(v)) for k, v in (bd.get("index_means") or ev["index_means"]).items()},
            "avg_watch_ratio": _r2(_f(retention.get("avg_watch_ratio"))),
            "completion": _r2(_f(retention.get("completion"))),
            "retention_3s": _r2(_f(curve[3])) if len(curve) > 3 else None,
            "n_cuts": len(cuts),
            "avg_shot_s": _r2(duration / max(1, len(shots))),
            "language": (summary.get("transcript") or {}).get("language"),
            "brain": {"provider": brain.get("provider"), "degraded": bool(brain.get("degraded")),
                      "confidence": brain.get("confidence") or {}},
        },
        "constraints": {"reading_wps": reading_wps, "max_words_per_card": max_words},
        "targets": {"peak": peak, "peak_by_index": tcfg.get("peak_by_index"), "basis": tcfg.get("basis"),
                    "floor": floor, "floor_index": fidx,
                    "moments": {k: {"indices": list(v["indices"]), "roles": list(v.get("roles", []))}
                                for k, v in tcfg["moments"].items()}},
        "evaluation": {
            "moments": ev["moments"],
            "missing_moments": missing,
            "attainment": ev["attainment"],
            "seconds_below_floor": ev["seconds_below_floor"],
        },
        "segments": segs,
        "dips": dips,
        "long_shots": long_shots,
        "text_issues": text_issues,
        "cuts": [round(c, 1) for c in cuts],
        "recommendations": [{"t": _r2(_f(r.get("t"))), "t_end": _r2(_f(r.get("t_end"), _f(r.get("t")))),
                             "rule": r.get("rule"), "severity": r.get("severity"),
                             "message": _clip(r.get("message", ""), 200)}
                            for r in (summary.get("recommendations") or [])],
        "series": {
            **{k: [_r2(_f(r["indices"].get(k))) for r in rows] for k in INDEX_NAMES},
            "retention": [_r2(_f(r.get("retention"))) for r in rows],
        },
    }


def brief_json(brief: dict) -> str:
    """Sérialisation compacte et DÉTERMINISTE (clés triées : préfixe stable pour le cache de prompt)."""
    return json.dumps(brief, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
