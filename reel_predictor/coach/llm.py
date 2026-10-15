"""Réécriture du script et plan de montage par l'API Claude (SDK officiel `anthropic`).

Seul du TEXTE est envoyé (brief : transcription, textes OCR, indices), jamais la vidéo.

Forme de l'appel (voir `build_request` / `_call`) :
    client.beta.messages.stream(
        model=<coach.yaml llm.model>, max_tokens=<llm.max_tokens>,
        thinking={"type": "adaptive"},                     # jamais de budget_tokens / disabled
        output_config={"effort": <llm.effort>, "format": {"type": "json_schema", "schema": <CoachOutput>}},
        system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        messages=[...], betas=["server-side-fallback-2026-07-01"], fallbacks="default",
    ) -> get_final_message() -> stop_reason (refusal / max_tokens) -> CoachOutput.model_validate_json(texte)
Garde-fou : coût estimé (messages.count_tokens + hypothèse de sortie) > `max_cost_usd` => refus avant l'appel.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import ValidationError

from .brief import brief_json
from .schema import CoachOutput, Variant

log = logging.getLogger("reel_predictor.coach.llm")

FALLBACK_BETA = "server-side-fallback-2026-07-01"
CACHE_WRITE_MULT = 1.25          # écriture en cache (TTL 5 min) = 1,25 × prix d'entrée
CHARS_PER_TOKEN = 3.2            # approximation hors ligne (français + JSON) quand count_tokens est impossible
OUTPUT_BASE_TOKENS = 6000        # hypothèse de sortie : réflexion + diagnostic + plan de montage...
OUTPUT_PER_VARIANT = 2500        # ... + une variante complète
EFFORT_MULT = {"low": 0.6, "medium": 0.8, "high": 1.0, "xhigh": 1.3, "max": 1.6}

ProgressFn = Callable[[float, str], None]


SYSTEM_PROMPT = """Tu es un script doctor et chef monteur spécialisé dans les Reels Instagram / TikTok francophones \
(format vertical, 20 à 60 s, face caméra, contenus éducatifs et informatifs). Tu sais ce qui fait décrocher \
un spectateur seconde par seconde : accroche lente, plan fixe trop long, texte illisible, information diluée, \
appel à l'action flou ou multiple.

On te fournit un DIAGNOSTIC STRUCTURÉ (JSON) d'un Reel déjà tourné, produit localement par un outil d'analyse :
- `segments` : les phrases réellement dites, horodatées, avec les indices moyens/min/max sur la phrase, les cuts, \
la durée du plan, le texte affiché à l'écran (lu par OCR : il mélange titres et sous-titres incrustés, mots parfois \
collés), les visages et la rétention prédite au début et à la fin du segment ;
- `series` : six indices seconde par seconde, entre 0 et 1 — attention, action (envie d'agir), tension, \
manipulation (pression persuasive), emotion, memory (mémorisation) — et la rétention prédite ;
- `evaluation` : l'écart aux CIBLES sur la vidéo actuelle, `dips` (creux d'attention), `long_shots` (plans sans \
cut), `text_issues` (cartons trop chargés), `recommendations` (règles déjà déclenchées).
Ces indices sont des ESTIMATIONS exploratoires (modèle cérébral prédictif + signaux vidéo), non calibrées, \
relatives à la vidéo ; certains ont une confiance faible (`video.brain.confidence`). Raisonne sur les tendances \
et les écarts, pas sur la troisième décimale.

OBJECTIF : des PICS aux moments clés et un PLANCHER ailleurs (pas « tout au maximum ») ; les valeurs exactes sont dans CIBLES : `peak_by_index` (cible par indice ; à défaut `peak`) et `floor`. Si `basis.mode` vaut `relative`, ces cibles valent le niveau de vos meilleurs moments sur les Reels déjà analysés (percentile indiqué) :
- accroche (0–3 s) : attention + tension ;
- chiffre / révélation clé (rôle `fact`) : memory + emotion ;
- appel à l'action (rôle `cta`) : action ;
- partout ailleurs : l'indice de plancher (`floor_index`) ne doit pas passer sous `floor`.
Une variante ne doit pas faire BAISSER un indice clé par rapport à l'original sans raison explicite (le noter dans `notes`) : l'émotion sur le chiffre clé et l'action sur l'appel à l'action se perdent facilement en raccourcissant. Un appel à l'action peut être direct et motivant sans fausse urgence ni pression.

CE QUE TU PRODUIS (sortie JSON imposée par le schéma) :
1. `diagnosis_summary` : 3 à 5 phrases, en français, sur ce qui fait décrocher et pourquoi, en citant les secondes.
2. `montage_plan` : modifications de la vidéo ACTUELLE réalisables au montage SANS retourner (couper/raccourcir \
un passage, ajouter un cut ou un zoom punch-in sur un plan long, remplacer un carton par un texte court, \
ajouter un b-roll, un effet sonore, réordonner deux passages). Une action par ligne, ordre chronologique, \
temps `t`/`t_end` en secondes dans la vidéo actuelle (0 ≤ t < t_end ≤ durée), consigne concrète pour le monteur. \
Priorité aux creux (`dips`), aux plans longs et aux cartons illisibles.
3. `variants` : exactement le nombre de variantes de script demandé, chacune avec un ANGLE différent \
(ex. bénéfice concret, question/curiosité, erreur à éviter, preuve chiffrée). Chaque variante = suite ordonnée \
de segments (`hook` en premier, éventuellement `fact`, des `body`, un `cta` en dernier), chaque segment = une \
ou deux phrases courtes à dire face caméra, en français oral naturel (tutoiement si l'original tutoie).
4. `notes` : limites, hypothèses, indices qui ne peuvent pas monter de façon fiable.

CONTRAINTES (obligatoires) :
- Accroche : le premier segment a le rôle `hook`, il se dit en moins de 3 secondes (8 à 10 mots maximum) et \
pose la promesse ou le chiffre le plus fort dès le premier mot ; pas d'introduction ni de salutation.
- Texte à l'écran : au plus N mots par carton (N = `constraints.max_words_per_card`), lisible au rythme de \
`constraints.reading_wps` mots par seconde ; chaîne vide si aucun texte. Jamais de phrase entière recopiée.
- Un seul appel à l'action, clair, à la fin (rôle `cta`, un seul segment `cta`), qui reprend l'action réellement \
proposée dans l'original (lien, guide, commentaire…) ; ne pas en inventer une autre.
- Fond et faits : garde le sujet, les informations et les chiffres du script original. N'invente AUCUN chiffre, \
date, montant, nom, condition d'éligibilité ou résultat qui n'y figure pas ; si une information manque, \
reformule sans la préciser. La transcription automatique contient des erreurs : corrige-les seulement quand le sens \
est évident.
- Honnêteté : pas de promesse mensongère, pas de fausse urgence, pas de pression trompeuse ni de « secret que \
personne ne connaît ». L'indice `manipulation` n'est PAS un objectif à maximiser.
- Rythme : un changement de plan (`cut_at_start`) ou un nouvel élément visuel au moins toutes les 3 à 4 secondes ; \
alterne les types de plan (`shot`) ; consignes de montage concrètes dans `edit` (zoom, effet sonore, b-roll…).
- Durée totale de chaque variante proche de l'original ou plus courte (vise 25–45 s), jamais plus longue de 20 %.
- Les durées ne te sont pas demandées : elles seront mesurées par une voix de synthèse lors d'une simulation locale.
- Écris tout en français."""


# ----------------------------------------------------------------------------- erreurs
class CoachLLMError(RuntimeError):
    """Erreur du coach Claude (message en français, prêt pour l'UI). `cost_usd` = dépense déjà engagée."""

    def __init__(self, code: str, message: str, *, cost_usd: float = 0.0, usage: dict | None = None):
        super().__init__(message)
        self.code = code
        self.cost_usd = cost_usd
        self.usage = usage or {}


class CostLimitError(CoachLLMError):
    pass


@dataclass
class LLMResult:
    output: CoachOutput
    model: str
    usage: dict
    cost_usd: float
    estimate: dict
    stop_reason: str | None
    fallback_used: bool = False
    request_id: str | None = None
    warnings: list[str] = field(default_factory=list)


# ----------------------------------------------------------------------------- requête
def output_schema() -> dict:
    """JSON schema de `CoachOutput` au format accepté par les sorties structurées (additionalProperties: false)."""
    from anthropic import transform_schema

    return transform_schema(CoachOutput)


def _llm_cfg(coach_cfg: dict) -> dict:
    llm = dict(coach_cfg.get("llm") or {})
    llm.setdefault("model", "claude-opus-5-5")
    llm.setdefault("effort", "high")
    llm.setdefault("max_tokens", 32000)
    llm.setdefault("max_cost_usd", 1.0)
    llm.setdefault("price_per_mtok", {"input": 4.0, "output": 20.0, "cache_read": 0.2})
    return llm


def _first_message_text(brief: dict, coach_cfg: dict, n_variants: int) -> str:
    targets = coach_cfg.get("targets") or {}
    return (
        "DIAGNOSTIC DU REEL (JSON) :\n" + brief_json(brief) + "\n\n"
        "CIBLES (JSON) :\n" + json.dumps(targets, ensure_ascii=False, sort_keys=True) + "\n\n"
        f"Produis le diagnostic, le plan de montage de la vidéo actuelle et exactement {n_variants} variante(s) "
        f"de script (angles différents), en respectant toutes les contraintes. Texte à l'écran : "
        f"{int(coach_cfg.get('max_words_per_card', 8))} mots maximum par carton."
    )


def build_request(brief: dict, coach_cfg: dict, n_variants: int, *, history: list[dict] | None = None,
                  prefix_n: int | None = None) -> dict:
    """Paramètres de `messages.stream` (hors `betas`/`fallbacks`). `history` = tours supplémentaires (passe 2) ;
    `prefix_n` = nb de variantes du 1er message (identique à la passe 1 pour réutiliser le cache)."""
    llm = _llm_cfg(coach_cfg)
    first_text = _first_message_text(brief, coach_cfg, prefix_n if prefix_n is not None else n_variants)
    first = {"role": "user", "content": [
        # Bloc mis en cache : système + diagnostic = préfixe stable réutilisé par la passe 2 (et les relances < 5 min)
        {"type": "text", "text": first_text, "cache_control": {"type": "ephemeral"}},
    ]}
    return {
        "model": llm["model"],
        "max_tokens": int(llm["max_tokens"]),
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": llm["effort"], "format": {"type": "json_schema", "schema": output_schema()}},
        "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
        "messages": [first, *(history or [])],
    }


def round2_history(round1: CoachOutput, variant_index: int, sim_result: dict, original_result: dict | None,
                   coach_cfg: dict) -> list[dict]:
    """Tours ajoutés pour la 2ᵉ passe : réponse de la passe 1 + indices simulés de la meilleure variante."""
    tcfg = (sim_result.get("evaluation") or {}) or coach_cfg.get("targets") or {}  # échelle de la simulation
    floor = float(tcfg.get("floor", 0.5))
    fidx = tcfg.get("floor_index", "attention")
    segs = sim_result.get("segments") or []
    tl = sim_result.get("timeline") or []
    seg_stats = []
    variant = round1.variants[variant_index]
    for i, seg in enumerate(variant.segments):
        span = segs[i] if i < len(segs) else {}
        t0, t1 = float(span.get("t", 0.0)), float(span.get("t_end", 0.0))
        rows = [r for r in tl if r["t"] + 1 > t0 and r["t"] < t1] if t1 > t0 else []
        stats = {"i": i, "role": seg.role, "t": round(t0, 2), "t_end": round(t1, 2)}
        if rows:
            for k in ("attention", "tension", "emotion", "memory", "action"):
                vals = [float(r["indices"][k]) for r in rows]
                stats[k] = [round(min(vals), 2), round(max(vals), 2)]
            stats["below_floor_s"] = sum(1 for r in rows if float(r["indices"][fidx]) < floor)
        seg_stats.append(stats)
    payload = {
        "variant_index": variant_index,
        "angle": variant.angle,
        "score": sim_result.get("score"),
        "evaluation": sim_result.get("evaluation"),
        "vs_original": sim_result.get("vs_original"),
        "original_evaluation": (original_result or {}).get("evaluation"),
        "segments": seg_stats,
    }
    return [
        {"role": "assistant", "content": round1.model_dump_json()},
        {"role": "user", "content": (
            "SIMULATION LOCALE de la variante " + str(variant_index + 1) + " (voix de synthèse, sans image) : "
            "indices min/max par segment, évaluation des moments clés et secondes sous le plancher (JSON) :\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n\n"
            "Réécris UNIQUEMENT cette variante en corrigeant, par ordre de priorité : (1) les RÉGRESSIONS par "
            "rapport à l'original (`vs_original.regressions` : indices clés qui baissent, souvent l'émotion sur le "
            "chiffre clé et l'action sur l'appel à l'action), (2) les moments clés sous leur cible "
            "(`evaluation.moments[].targets`, échelle de la simulation), (3) les secondes sous le plancher. "
            "Garde ce qui marche, les mêmes faits et le même appel à l'action. "
            "Réponds avec le même schéma : `variants` contient exactement 1 variante (la version corrigée) ; "
            "`montage_plan` peut être une liste vide ; `diagnosis_summary` explique en 2-3 phrases ce que tu as changé."
        )},
    ]


# ----------------------------------------------------------------------------- coût
def _price(coach_cfg: dict) -> dict:
    p = _llm_cfg(coach_cfg)["price_per_mtok"]
    return {"input": float(p.get("input", 4.0)), "output": float(p.get("output", 20.0)),
            "cache_read": float(p.get("cache_read", 0.2)),
            "cache_write": float(p.get("cache_write", float(p.get("input", 4.0)) * CACHE_WRITE_MULT))}


def assumed_output_tokens(coach_cfg: dict, n_variants: int) -> int:
    llm = _llm_cfg(coach_cfg)
    est = (OUTPUT_BASE_TOKENS + OUTPUT_PER_VARIANT * max(1, n_variants)) * EFFORT_MULT.get(str(llm["effort"]), 1.0)
    return int(min(int(llm["max_tokens"]), est))


def approx_tokens(request: dict) -> int:
    """Approximation hors ligne des tokens d'entrée (système + messages + schéma)."""
    chars = len(SYSTEM_PROMPT) + len(json.dumps(request.get("output_config", {}), ensure_ascii=False))
    for m in request.get("messages", []):
        c = m["content"]
        chars += len(c) if isinstance(c, str) else sum(len(b.get("text", "")) for b in c)
    return int(chars / CHARS_PER_TOKEN) + 50


def cost_from_usage(usage: dict, coach_cfg: dict) -> float:
    p = _price(coach_cfg)
    usd = (usage.get("input_tokens", 0) * p["input"] + usage.get("output_tokens", 0) * p["output"]
           + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
           + usage.get("cache_creation_input_tokens", 0) * p["cache_write"]) / 1e6
    return round(usd, 4)


def estimate_cost(request: dict, coach_cfg: dict, n_variants: int, client: Any | None = None) -> dict:
    """Estimation AVANT appel. Avec `client` : tokens d'entrée exacts via `messages.count_tokens` (gratuit) ;
    sinon approximation locale. Sortie supposée : `assumed_output_tokens`. Pire cas : `max_tokens` en sortie."""
    method = "approx"
    input_tokens = approx_tokens(request)
    if client is not None:
        try:
            ct = client.messages.count_tokens(
                model=request["model"], system=request["system"], messages=request["messages"],
                thinking=request["thinking"], output_config=request["output_config"],
            )
            input_tokens = int(ct.input_tokens)
            method = "count_tokens"
        except Exception as exc:  # estimation seulement : on garde l'approximation
            log.warning("count_tokens indisponible (%s) : estimation approximative", exc.__class__.__name__)
    out_tokens = assumed_output_tokens(coach_cfg, n_variants)
    p = _price(coach_cfg)
    llm = _llm_cfg(coach_cfg)
    usd = (input_tokens * p["input"] + out_tokens * p["output"]) / 1e6
    worst = (input_tokens * p["cache_write"] + int(llm["max_tokens"]) * p["output"]) / 1e6
    return {"input_tokens": input_tokens, "output_tokens_assumed": out_tokens, "usd": round(usd, 4),
            "usd_max": round(worst, 4), "method": method, "max_cost_usd": float(llm["max_cost_usd"])}


def typical_cost(coach_cfg: dict, input_tokens: int = 7500) -> float:
    """Coût a priori d'une génération (brief typique ≈ 7,5 k tokens), sans brief ni réseau."""
    p = _price(coach_cfg)
    n = int(coach_cfg.get("n_variants", 3))
    usd = (input_tokens * p["input"] + assumed_output_tokens(coach_cfg, n) * p["output"]) / 1e6
    if int(coach_cfg.get("rounds", 1)) >= 2:
        usd += (input_tokens * p["cache_read"] + 3000 * p["input"] + assumed_output_tokens(coach_cfg, 1) * p["output"]) / 1e6
    return round(usd, 4)


def estimate_offline(brief: dict, coach_cfg: dict) -> dict:
    n = int(coach_cfg.get("n_variants", 3))
    est = estimate_cost(build_request(brief, coach_cfg, n), coach_cfg, n, client=None)
    if int(coach_cfg.get("rounds", 1)) >= 2:  # passe 2 : 1 variante, préfixe en cache
        est["usd_round2"] = round((est["input_tokens"] * _price(coach_cfg)["cache_read"]
                                   + 3000 * _price(coach_cfg)["input"]
                                   + assumed_output_tokens(coach_cfg, 1) * _price(coach_cfg)["output"]) / 1e6, 4)
        est["usd"] = round(est["usd"] + est["usd_round2"], 4)
    return est


# ----------------------------------------------------------------------------- appel
def make_client() -> Any:
    import anthropic

    from .keys import NO_KEY_MESSAGE, get_api_key

    key = get_api_key()
    if not key:
        raise CoachLLMError("no_key", NO_KEY_MESSAGE)
    return anthropic.Anthropic(api_key=key, max_retries=2,
                               timeout=anthropic.Timeout(900.0, connect=15.0))


def _usage_dict(usage: Any) -> dict:
    if usage is None:
        return {}
    out = {}
    for k in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
        v = getattr(usage, k, None)
        out[k] = int(v) if isinstance(v, (int, float)) else 0
    return out


def _fallback_used(msg: Any) -> bool:
    if any(getattr(b, "type", None) == "fallback" for b in (getattr(msg, "content", None) or [])):
        return True
    iters = getattr(getattr(msg, "usage", None), "iterations", None) or []
    return any(getattr(it, "type", None) == "fallback_message" for it in iters)


def _translate_api_error(exc: Exception, model: str) -> CoachLLMError:
    import anthropic

    if isinstance(exc, anthropic.AuthenticationError):
        return CoachLLMError("auth", "Clé API Anthropic refusée (invalide ou révoquée). Relancez « make api-key ».")
    if isinstance(exc, anthropic.PermissionDeniedError):
        return CoachLLMError("permission", f"La clé API n'a pas accès au modèle {model} (droits du compte).")
    if isinstance(exc, anthropic.NotFoundError):
        return CoachLLMError("not_found", f"Modèle introuvable : {model}. Vérifiez llm.model dans config/coach.yaml.")
    if isinstance(exc, anthropic.RateLimitError):
        retry = ""
        try:
            ra = exc.response.headers.get("retry-after")
            retry = f" (réessayez dans {int(float(ra))} s)" if ra else ""
        except Exception:
            pass
        return CoachLLMError("rate_limit", f"Limite de débit ou de quota de l'API Claude atteinte{retry}.")
    if isinstance(exc, anthropic.BadRequestError):
        msg = str(getattr(exc, "message", "") or exc)
        if "credit" in msg.lower() or "billing" in msg.lower():
            return CoachLLMError("billing", "Crédit API Anthropic insuffisant : rechargez le compte (console Anthropic).")
        return CoachLLMError("bad_request", f"Requête refusée par l'API Claude : {msg[:300]}")
    if isinstance(exc, anthropic.APIStatusError):
        code = getattr(exc, "status_code", "?")
        return CoachLLMError("api_status", f"Service Claude indisponible ou surchargé (HTTP {code}). Réessayez plus tard.")
    if isinstance(exc, anthropic.APITimeoutError):
        return CoachLLMError("timeout", "Délai dépassé en attendant l'API Claude. Réessayez.")
    if isinstance(exc, anthropic.APIConnectionError):
        return CoachLLMError("connection", "Connexion impossible à l'API Claude : vérifiez la connexion Internet.")
    return CoachLLMError("unknown", f"Erreur inattendue de l'API Claude : {exc.__class__.__name__}")


def _call(client: Any, request: dict, coach_cfg: dict, progress: ProgressFn | None,
          expected_chars: int) -> tuple[Any, str]:
    import anthropic

    llm = _llm_cfg(coach_cfg)
    chars = 0
    try:
        with client.beta.messages.stream(**request, betas=[FALLBACK_BETA], fallbacks="default") as stream:
            last = -1.0
            for event in stream:
                if getattr(event, "type", None) == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", None) == "text_delta":
                        chars += len(getattr(delta, "text", "") or "")
                        frac = min(0.95, chars / max(1, expected_chars))
                        if progress and frac - last >= 0.02:
                            last = frac
                            progress(frac, f"Rédaction par Claude… {chars // 1000} k caractères reçus")
                elif getattr(event, "type", None) == "content_block_start" and progress and chars == 0:
                    blk = getattr(getattr(event, "content_block", None), "type", "")
                    if blk == "thinking":
                        progress(0.02, "Claude réfléchit…")
            final = stream.get_final_message()
    except anthropic.APIError as exc:
        raise _translate_api_error(exc, llm["model"]) from exc

    usage = _usage_dict(getattr(final, "usage", None))
    cost = cost_from_usage(usage, coach_cfg)
    stop = getattr(final, "stop_reason", None)
    if stop == "refusal":
        det = getattr(final, "stop_details", None)
        cat = getattr(det, "category", None) if det is not None else None
        raise CoachLLMError("refusal", "Claude a refusé de traiter cette demande"
                            + (f" (catégorie : {cat})" if cat else "") + ", y compris après repli sur un autre modèle.",
                            cost_usd=cost, usage=usage)
    if stop == "max_tokens":
        raise CoachLLMError("max_tokens", f"Réponse tronquée (limite de {llm['max_tokens']} tokens atteinte) : "
                            "augmentez llm.max_tokens ou baissez n_variants / effort dans config/coach.yaml.",
                            cost_usd=cost, usage=usage)
    text = "".join(getattr(b, "text", "") for b in (getattr(final, "content", None) or [])
                   if getattr(b, "type", None) == "text")
    if not text.strip():
        raise CoachLLMError("empty", f"Réponse vide de Claude (stop_reason = {stop}).", cost_usd=cost, usage=usage)
    return final, text


def _check_output(out: CoachOutput, coach_cfg: dict, n_variants: int, duration_s: float | None) -> list[str]:
    """Contrôles des contraintes (avertissements, sans modifier le contenu)."""
    warns = []
    maxw = int(coach_cfg.get("max_words_per_card", 8))
    if len(out.variants) != n_variants:
        warns.append(f"Claude a renvoyé {len(out.variants)} variante(s) au lieu de {n_variants}.")
    for i, v in enumerate(out.variants, 1):
        if not v.segments:
            warns.append(f"Variante {i} : aucun segment.")
            continue
        if v.segments[0].role != "hook":
            warns.append(f"Variante {i} : le premier segment n'est pas une accroche.")
        n_cta = sum(1 for s in v.segments if s.role == "cta")
        if n_cta != 1:
            warns.append(f"Variante {i} : {n_cta} segment(s) d'appel à l'action (1 attendu).")
        long_cards = [s.on_screen_text for s in v.segments if len(s.on_screen_text.split()) > maxw]
        if long_cards:
            warns.append(f"Variante {i} : {len(long_cards)} texte(s) à l'écran dépassent {maxw} mots.")
        hook_words = len(v.segments[0].voice_over.split())
        if hook_words > 12:
            warns.append(f"Variante {i} : accroche de {hook_words} mots (risque de dépasser 3 s).")
    if duration_s:
        bad = [a for a in out.montage_plan if a.t < 0 or a.t_end <= a.t or a.t > duration_s + 0.5]
        if bad:
            warns.append(f"{len(bad)} action(s) du plan de montage ont des temps incohérents.")
    return warns


def generate(brief: dict, coach_cfg: dict, *, client: Any | None = None, n_variants: int | None = None,
             history: list[dict] | None = None, prefix_n: int | None = None, progress: ProgressFn | None = None,
             budget_spent_usd: float = 0.0) -> LLMResult:
    """Appel complet : estimation + garde-fou, streaming, vérifications, validation Pydantic, coût réel."""
    n = int(n_variants if n_variants is not None else coach_cfg.get("n_variants", 3))
    llm = _llm_cfg(coach_cfg)
    client = client if client is not None else make_client()
    request = build_request(brief, coach_cfg, n, history=history, prefix_n=prefix_n)
    if progress:
        progress(0.0, "Estimation du coût…")
    est = estimate_cost(request, coach_cfg, n, client=client)
    limit = float(llm["max_cost_usd"])
    if budget_spent_usd + est["usd"] > limit:
        raise CostLimitError("cost_limit", f"Coût estimé {est['usd'] + budget_spent_usd:.2f} $ > plafond {limit:.2f} $ "
                             "(llm.max_cost_usd dans config/coach.yaml) : appel annulé, rien n'a été dépensé.")
    if progress:
        progress(0.01, f"Envoi du diagnostic à Claude (≈ {est['input_tokens']} tokens, ≈ {est['usd']:.2f} $)…")
    final, text = _call(client, request, coach_cfg, progress, expected_chars=3000 + 5000 * n)
    usage = _usage_dict(getattr(final, "usage", None))
    cost = cost_from_usage(usage, coach_cfg)
    try:
        out = CoachOutput.model_validate_json(text)
    except ValidationError as exc:
        raise CoachLLMError("invalid_output", f"Réponse de Claude non conforme au schéma ({exc.error_count()} erreur(s)).",
                            cost_usd=cost, usage=usage) from exc
    warnings = _check_output(out, coach_cfg, n, (brief.get("video") or {}).get("duration_s"))
    fb = _fallback_used(final)
    if fb:
        warnings.append(f"Repli serveur : la réponse a été produite par {getattr(final, 'model', '?')} "
                        "(coût calculé au tarif du modèle principal : approximatif).")
    if progress:
        progress(1.0, "Réponse de Claude reçue")
    return LLMResult(output=out, model=str(getattr(final, "model", llm["model"])), usage=usage, cost_usd=cost,
                     estimate=est, stop_reason=getattr(final, "stop_reason", None), fallback_used=fb,
                     request_id=getattr(final, "_request_id", None), warnings=warnings)


def revise_variant(brief: dict, coach_cfg: dict, round1: CoachOutput, variant_index: int, sim_result: dict,
                   original_result: dict | None, *, client: Any | None = None, progress: ProgressFn | None = None,
                   budget_spent_usd: float = 0.0) -> tuple[Variant, LLMResult]:
    """2ᵉ passe : Claude corrige les segments faibles de la variante simulée (même schéma, 1 variante)."""
    hist = round2_history(round1, variant_index, sim_result, original_result, coach_cfg)
    res = generate(brief, coach_cfg, client=client, n_variants=1, history=hist, progress=progress,
                   prefix_n=int(coach_cfg.get("n_variants", 3)), budget_spent_usd=budget_spent_usd)
    if not res.output.variants:
        raise CoachLLMError("invalid_output", "Seconde passe : aucune variante renvoyée.", cost_usd=res.cost_usd,
                            usage=res.usage)
    return res.output.variants[0], res
