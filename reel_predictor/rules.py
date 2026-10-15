"""Recommandations « quoi changer », horodatées.

Chaque règle de `config/rules.yaml` correspond à une fonction enregistrée par son nom
(`@register("nom")`). Une règle renvoie des `Hit(t, t_end, vars)` ; les déclenchements consécutifs
ou qui se chevauchent sont fusionnés en un intervalle, puis le message FR est formaté.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from .score import hook_elements, runs_below
from .textutils import tokens

SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2}


class Num(float):
    """Nombre affiché sans décimales inutiles dans les messages (« 7 » plutôt que « 7.0 »)."""

    def __format__(self, spec: str) -> str:
        if spec:
            return format(float(self), spec)
        v = round(float(self), 1)
        return str(int(v)) if v == int(v) else f"{v:.1f}".replace(".", ",")


@dataclass
class Hit:
    t: float
    t_end: float
    vars: dict = field(default_factory=dict)


@dataclass
class RuleContext:
    df: pd.DataFrame                  # features + colonnes idx_<indice>
    duration_s: float
    first_word_s: float | None
    has_speech: bool
    face_ok: bool = True
    # Rétention prédite à la fin de chaque seconde (T,). Si absente (le pipeline ne la passe pas), elle est
    # recalculée à la demande depuis idx_attention / f_cut_recent avec les paramètres de rétention effectifs.
    retention: np.ndarray | None = None

    @property
    def T(self) -> int:
        return len(self.df)

    def index(self, name: str) -> np.ndarray:
        return self.df[f"idx_{name}"].to_numpy(float)

    def retention_end(self) -> np.ndarray:
        """Rétention prédite à la fin de chaque seconde (même modèle que le pipeline)."""
        if self.retention is None:
            from .config import load_config
            from .retention import simulate

            rp = load_config().retention
            cr = self.df["f_cut_recent"].to_numpy(float) if "f_cut_recent" in self.df else np.zeros(self.T)
            self.retention = simulate(self.index("attention"), cr, self.duration_s,
                                      hook_seconds=rp.hook_seconds, hook_drop=rp.hook_drop,
                                      base_drop=rp.base_drop, k_attention=rp.k_attention,
                                      cut_bonus=rp.cut_bonus, allow_increase=rp.allow_increase).per_second
        return np.asarray(self.retention, float)


RuleFn = Callable[[RuleContext, dict], list[Hit]]
REGISTRY: dict[str, RuleFn] = {}


def register(name: str) -> Callable[[RuleFn], RuleFn]:
    def deco(fn: RuleFn) -> RuleFn:
        REGISTRY[name] = fn
        return fn
    return deco


def _runs(mask: np.ndarray, min_len: int = 1) -> list[tuple[int, int]]:
    """Intervalles [début, fin] (inclus) où mask est vrai pendant au moins min_len secondes."""
    return runs_below(np.where(np.asarray(mask, bool), 0.0, 1.0), 0.5, max(1, int(min_len)))


# ----------------------------------------------------------------------------- règles
def _dec(x: float) -> str:
    """0.6123 -> « 0,61 » (valeurs d'indice dans les messages)."""
    return f"{float(x):.2f}".replace(".", ",")


@register("attention_drop_no_cut")
def attention_drop_no_cut(ctx: RuleContext, p: dict) -> list[Hit]:
    """Creux d'attention dans un plan qui dure. Deux déclencheurs (fusionnés) :

    - absolu : attention < `attention_below` pendant ≥ `min_duration_s` s, plan de ≥ `since_cut_min_s` s ;
    - relatif : attention ≥ `rel_drop` (ex. 15 %) sous son maximum des `rel_window_s` secondes précédentes,
      pendant ≥ `rel_min_duration_s` s, plan de ≥ `since_cut_min_s` s. Un Reel dont l'attention reste
      moyenne (0,5–0,6) n'atteint jamais le seuil absolu mais décroche quand même : la baisse relative le
      voit. L'intervalle signalé commence au début de la baisse (seconde qui suit le pic), sans remonter
      avant le dernier cut.
    """
    att = ctx.index("attention")
    since = ctx.df["since_cut"].to_numpy(float)
    cut = ctx.df["cut"].to_numpy(bool) if "cut" in ctx.df else np.zeros(ctx.T, bool)
    long_plan = since >= float(p["since_cut_min_s"])
    T = ctx.T
    raw: list[tuple[int, int, int]] = []        # (début signalé, fin incluse, début de la détection)
    for s, e in _runs((att < float(p["attention_below"])) & long_plan, int(p["min_duration_s"])):
        raw.append((s, e, s))
    rel_drop = float(p.get("rel_drop", 0) or 0)
    if rel_drop > 0:
        w = max(1, int(p.get("rel_window_s", 5)))
        rel = np.zeros(T, bool)
        for t in range(1, T):
            peak = float(np.max(att[max(0, t - w):t]))
            rel[t] = peak > 0 and att[t] <= (1.0 - rel_drop) * peak
        ignore_end = int(p.get("ignore_last_s", 0) or 0)
        onset_drop = float(p.get("onset_drop", 0.05))
        for s, e in _runs(rel & long_plan, int(p.get("rel_min_duration_s", 2))):
            if s >= T - ignore_end:          # bord de fin (prédiction TRIBE peu fiable, cf. low_context_end)
                continue
            # début du creux : on remonte la pente jusqu'au pic local, puis 1re seconde ≥ onset_drop sous ce pic
            peak = s
            while peak - 1 >= max(0, s - 2 * w) and att[peak - 1] >= att[peak]:
                peak -= 1
            onset = next((t for t in range(peak + 1, s + 1) if att[t] <= (1.0 - onset_drop) * att[peak]), s)
            last_cut = max([t for t in range(0, s + 1) if cut[t]], default=0)
            raw.append((max(onset, last_cut), e, s))
    if not raw:
        return []
    raw.sort()
    merged: list[list[int]] = []
    for a, e, d in raw:
        if merged and a <= merged[-1][1] + 1:
            merged[-1][1] = max(merged[-1][1], e)
            merged[-1][2] = min(merged[-1][2], d)
        else:
            merged.append([a, e, d])
    hits = []
    for a, e, d in merged:
        before = att[max(0, a - 2):a + 1]       # pic juste avant le creux (le début signalé suit le pic)
        hits.append(Hit(a, min(e + 1, ctx.duration_s), {
            "since_cut": Num(math.floor(since[a])),
            "plan_s": Num(round(float(since[a:e + 1].max()))),
            "att_from": _dec(float(before.max())),
            "att_to": _dec(float(att[a:e + 1].min())),
        }))
    return hits


@register("hook_dropoff")
def hook_dropoff(ctx: RuleContext, p: dict) -> list[Hit]:
    """Rétention prédite à `at_s` secondes sous `retention_below` : l'accroche fait fuir trop de monde."""
    at = float(p.get("at_s", 3))
    k = int(math.ceil(at)) - 1                 # rétention à la fin de la seconde k = instant k+1 = at
    ret = ctx.retention_end()
    if ret.size == 0:
        return []
    k = int(np.clip(k, 0, ret.size - 1))
    r = float(ret[k])
    if r >= float(p["retention_below"]):
        return []
    att = ctx.index("attention")[: k + 1]
    return [Hit(0.0, min(at, ctx.duration_s), {
        "retention": f"{round(100 * r)} %", "lost": f"{round(100 * (1 - r))} %",
        "attention": _dec(float(np.mean(att))) if att.size else "–",
    })]


_HOOK_LABELS = {
    "text": "un texte à l'écran",
    "face": "un visage",
    "speech_start": "une parole dès la 1re seconde",
    "motion_or_cut": "du mouvement ou un cut",
}


@register("weak_hook")
def weak_hook(ctx: RuleContext, p: dict) -> list[Hit]:
    w = float(p["window_s"])
    elems = hook_elements(ctx.df, ctx.first_word_s, w, float(p["speech_start_max_s"]), float(p["motion_min"]))
    present = sum(elems.values())
    att = ctx.index("attention")[: max(1, int(math.ceil(w)))]
    weak_att = float(np.mean(att)) < float(p["attention_below"])
    if present >= int(p["min_elements"]) and not weak_att:
        return []
    missing = [lab for k, lab in _HOOK_LABELS.items() if not elems[k]]
    if not missing:
        missing = ["de l'intensité (attention prédite faible)"]
    txt = ", ".join(missing[:-1]) + (" et " if len(missing) > 1 else "") + missing[-1]
    return [Hit(0.0, min(w, ctx.duration_s), {"missing": txt, "present": present})]


@register("no_face_long")
def no_face_long(ctx: RuleContext, p: dict) -> list[Hit]:
    if not ctx.face_ok:
        return []
    mask = ctx.df["faces"].to_numpy(int) == 0
    return [Hit(s, min(e + 1, ctx.duration_s)) for s, e in _runs(mask, int(p["min_duration_s"]))]


@register("silence_gap")
def silence_gap(ctx: RuleContext, p: dict) -> list[Hit]:
    mask = ctx.df["silence"].to_numpy(bool)
    out = []
    for s, e in _runs(mask, 1):
        length = e - s + 1
        if e == ctx.T - 1:
            length -= float(p.get("ignore_last_s", 0))
        if length >= float(p["min_duration_s"]):
            out.append(Hit(s, min(e + 1, ctx.duration_s)))
    return out


@register("text_overload")
def text_overload(ctx: RuleContext, p: dict) -> list[Hit]:
    """Texte à lire (nouveaux mots non prononcés) supérieur au temps d'affichage.

    Un « événement » texte = secondes consécutives affichant le même texte. Les mots déjà présents à
    l'événement précédent (texte de décor persistant) et les mots prononcés à ± `speech_window_s`
    (sous-titres) ne comptent pas : ils ne demandent pas de lecture supplémentaire.
    """
    df = ctx.df
    T = ctx.T
    wps = float(p["reading_wps"])
    win = int(p.get("speech_window_s", 2))
    events: list[list] = []
    for t in range(T):
        text = df.at[t, "text"]
        if not text:
            continue
        if events and events[-1][1] == t and not df.at[t, "text_new"]:
            events[-1][1] = t + 1
        else:
            events.append([t, t + 1, text])
    hits: list[Hit] = []
    prev_concat = ""
    prev_end = -1
    for s, e, text in events:
        toks = tokens(text)
        persistent = prev_concat if prev_end == s else ""
        said_seq = "".join(tokens(" ".join(w for t in range(max(0, s - win), min(T, e + win))
                                            for w in df.at[t, "words"])))
        # l'OCR colle souvent les mots (« TUPEUXFAIRE ») : un mot couvert par le texte dit (ou déjà
        # affiché juste avant) en tant que sous-chaîne ne demande pas de lecture supplémentaire
        n = sum(1 for tk in toks if not (tk in persistent or tk in said_seq))
        if n >= int(p["min_words"]) and n / wps > (e - s):
            if hits and hits[-1].t_end >= s:        # événements surchargés consécutifs : on cumule
                h = hits[-1]
                h.t_end = e
                h.vars["words"] += n
            else:
                hits.append(Hit(s, e, {"words": n}))
        prev_concat, prev_end = "".join(toks), e
    for h in hits:
        h.vars["shown"] = Num(h.t_end - h.t)
        h.vars["needed"] = Num(round(h.vars["words"] / wps))
    return hits


@register("cta_missing")
def cta_missing(ctx: RuleContext, p: dict) -> list[Hit]:
    start = ctx.duration_s * (1.0 - float(p["last_fraction"]))
    t0 = int(math.floor(start))
    if bool(ctx.df["cta"].to_numpy(bool)[t0:].any()):
        return []
    return [Hit(round(start, 1), ctx.duration_s)]


@register("tension_flat")
def tension_flat(ctx: RuleContext, p: dict) -> list[Hit]:
    x = ctx.index("tension")
    w = int(p["window_s"])
    if ctx.T < w:
        return []
    flat = np.zeros(ctx.T, bool)
    for t in range(ctx.T - w + 1):
        seg = x[t:t + w]
        if float(seg.max() - seg.min()) < float(p["min_range"]):
            flat[t:t + w] = True
    return [Hit(s, min(e + 1, ctx.duration_s)) for s, e in _runs(flat, int(p["min_duration_s"]))]


@register("slow_start")
def slow_start(ctx: RuleContext, p: dict) -> list[Hit]:
    if ctx.first_word_s is None:
        return []
    if ctx.first_word_s >= float(p["speech_start_max_s"]):
        return [Hit(0.0, round(ctx.first_word_s, 1), {"start": Num(round(ctx.first_word_s, 1))})]
    return []


# ----------------------------------------------------------------------------- moteur
def merge_hits(hits: list[Hit]) -> list[Hit]:
    hits = sorted(hits, key=lambda h: (h.t, h.t_end))
    out: list[Hit] = []
    for h in hits:
        if out and h.t <= out[-1].t_end:
            out[-1].t_end = max(out[-1].t_end, h.t_end)
        else:
            out.append(Hit(h.t, h.t_end, dict(h.vars)))
    return out


def format_message(template: str, hit: Hit, duration_s: float, params: dict) -> str:
    values = {k: (Num(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in params.items()}
    values.update({"t": Num(hit.t), "t_end": Num(hit.t_end), "duration": Num(duration_s)})
    values.update(hit.vars)
    try:
        return template.format(**values)
    except (KeyError, IndexError, ValueError):
        return template.replace("{t}", format(Num(hit.t)))


def evaluate(ctx: RuleContext, rules: list[dict]) -> tuple[list[dict], list[str]]:
    """Renvoie (recommandations triées, avertissements)."""
    recs, warns = [], []
    for rule in rules:
        if not rule.get("enabled", True):
            continue
        name = rule["name"]
        fn = REGISTRY.get(name)
        if fn is None:
            warns.append(f"Règle inconnue ignorée : {name}")
            continue
        params = rule.get("params") or {}
        try:
            hits = merge_hits(fn(ctx, params))
        except Exception as e:  # noqa: BLE001 — une règle défaillante ne bloque pas l'analyse
            warns.append(f"Règle {name} en erreur : {type(e).__name__}: {e}")
            continue
        for h in hits:
            recs.append({"t": round(float(h.t), 2), "t_end": round(float(h.t_end), 2), "rule": name,
                         "severity": rule["severity"],
                         "message": format_message(rule["message"], h, ctx.duration_s, params)})
    recs.sort(key=lambda r: (r["t"], SEVERITY_RANK.get(r["severity"], 9), r["rule"]))
    return recs, warns
