"""Règles v2 (12/10/2026) : CTA collés par l'OCR, creux d'attention relatifs, accroche qui décroche."""
import numpy as np
import pytest

from reel_predictor.config import INDEX_NAMES, load_config
from reel_predictor.features import build_features
from reel_predictor.rules import RuleContext, evaluate
from reel_predictor.synthetic import make_signals
from reel_predictor.textutils import find_glued, find_in_text, keyword_tokens, squash


@pytest.fixture(scope="module")
def cfg():
    return load_config(calibration_file=None)


@pytest.fixture(scope="module")
def cta_kw(cfg):
    return keyword_tokens(cfg.weights["cta_keywords"])


def rule(cfg, name):
    return next(r for r in cfg.rules if r["name"] == name)


def ctx_for(cfg, duration, attention=None, retention=None, **kw):
    df = build_features(make_signals(duration, **kw), cfg)
    for k in INDEX_NAMES:
        df[f"idx_{k}"] = 0.6
    if attention is not None:
        df["idx_attention"] = attention
    words = kw.get("words") or []
    return RuleContext(df=df, duration_s=duration, first_word_s=words[0][1] if words else None,
                       has_speech=bool(words), retention=retention)


def run(cfg, ctx, name):
    recs, warns = evaluate(ctx, [rule(cfg, name)])
    assert not warns
    return recs


# ----------------------------------------------------------------------------- CTA
def test_squash():
    assert squash("Lien en bio !") == "lienenbio"


@pytest.mark.parametrize("text,expected", [
    ("PRENDREMONGUIDE", {"prendre", "guide"}),
    ("PRENDS LEGUIDEULTIME DUBOURS", {"prends", "guide"}),
    ("CLIQUESURLEBOUTON", {"clique", "bouton"}),
    ("LIENENBIO", {"lien en bio"}),
    ("COPIEETCOLLE", {"copie"}),
])
def test_glued_ocr_cta_found(cta_kw, text, expected):
    assert expected <= set(find_in_text(text, cta_kw))


@pytest.mark.parametrize("text", ["TU VAS APPRENDRE", "COMPRENDSBIEN", "ABONNEMENT GRATUIT", "ADMIN",
                                  "ALIENS", "LIKELY", "SURPRENDRE"])
def test_glued_no_false_positive(cta_kw, text):
    # mots courants contenant un mot-clé, et mots-clés courts (< 5 lettres) jamais cherchés « collés »
    assert find_glued(text, cta_kw) == []


def test_short_keywords_exact_token_only(cta_kw):
    assert "lien" in find_in_text("le lien officiel", cta_kw)
    assert "dm" in find_in_text("Envoie un DM", cta_kw)
    assert "lien" not in find_in_text("LELIENOFFICIEL", cta_kw)       # « lien » (4 lettres) non collé


def test_cta_missing_not_fired_by_glued_ocr_or_spoken_copy(cfg):
    # fin de vidéo : « Copie et colle » dit + « PRENDREMONGUIDE » à l'écran (cas Erasmus Mundus)
    words = [("Copie", 17.2, 17.5), ("et", 17.5, 17.6), ("colle", 17.6, 18.0)]
    assert run(cfg, ctx_for(cfg, 20.0, words=words), "cta_missing") == []
    assert run(cfg, ctx_for(cfg, 20.0, texts={18: "PRENDREMONGUIDE"}), "cta_missing") == []
    # sans rien : la règle se déclenche toujours
    assert len(run(cfg, ctx_for(cfg, 20.0, texts={18: "MERCI"}), "cta_missing")) == 1


# ----------------------------------------------------------------------------- creux relatif
def erasmus_like():
    """Attention 0,62 -> 0,42 sur 8 s sans cut (valeurs du Reel Erasmus 36–46 s, décalées à 6–16 s)."""
    att = np.full(24, 0.62)
    att[6:16] = [0.612, 0.6105, 0.5522, 0.5111, 0.494, 0.4703, 0.4431, 0.4231, 0.4194, 0.4843]
    att[16:] = 0.61
    return att


def test_relative_drop_detected(cfg):
    recs = run(cfg, ctx_for(cfg, 24.0, attention=erasmus_like(), cuts=[6.8, 16.5]),
               "attention_drop_no_cut")
    assert len(recs) == 1
    r = recs[0]
    assert (r["t"], r["t_end"], r["severity"]) == (8, 15, "high")
    assert "0,61 → 0,42" in r["message"] and "plan de 8 s" in r["message"]


def test_relative_drop_needs_long_plan(cfg):
    # un cut au milieu du creux : plan < 4 s partout -> rien
    assert run(cfg, ctx_for(cfg, 24.0, attention=erasmus_like(), cuts=[6.8, 9.5, 12.5, 16.5]),
               "attention_drop_no_cut") == []


def test_relative_drop_small_variation_ignored(cfg):
    att = np.full(24, 0.6)
    att[8:14] = 0.55                       # −8 % seulement
    assert run(cfg, ctx_for(cfg, 24.0, attention=att, cuts=[2.0]), "attention_drop_no_cut") == []


def test_relative_drop_ignored_at_very_end(cfg):
    att = np.full(20, 0.62)
    att[18:] = 0.45                        # creux relatif (−27 %) dans les 3 dernières secondes (bord TRIBE)
    assert run(cfg, ctx_for(cfg, 20.0, attention=att, cuts=[10.0]), "attention_drop_no_cut") == []


def test_relative_can_be_disabled(cfg):
    r = {**rule(cfg, "attention_drop_no_cut")}
    r["params"] = {**r["params"], "rel_drop": 0}
    recs, _ = evaluate(ctx_for(cfg, 24.0, attention=erasmus_like(), cuts=[6.8, 16.5]), [r])
    assert recs == []                      # le minimum (0,419) n'est pas sous le seuil absolu 0,42


# ----------------------------------------------------------------------------- accroche
def test_hook_dropoff_from_given_retention(cfg):
    ret = np.linspace(0.9, 0.5, 20)
    ret[2] = 0.65
    recs = run(cfg, ctx_for(cfg, 20.0, retention=ret), "hook_dropoff")
    assert len(recs) == 1 and (recs[0]["t"], recs[0]["t_end"]) == (0, 3)
    assert "65 %" in recs[0]["message"]
    ret[2] = 0.80
    assert run(cfg, ctx_for(cfg, 20.0, retention=ret), "hook_dropoff") == []


def test_hook_dropoff_recomputes_retention_when_missing(cfg):
    weak = run(cfg, ctx_for(cfg, 20.0, attention=np.full(20, 0.25)), "hook_dropoff")
    strong = run(cfg, ctx_for(cfg, 20.0, attention=np.full(20, 0.95), cuts=[1.0, 2.0]), "hook_dropoff")
    assert len(weak) == 1 and strong == []
