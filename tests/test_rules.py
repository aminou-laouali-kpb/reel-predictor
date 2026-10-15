import numpy as np
import pytest

from reel_predictor.config import INDEX_NAMES, load_config
from reel_predictor.features import build_features
from reel_predictor.rules import REGISTRY, Hit, Num, RuleContext, evaluate, merge_hits
from reel_predictor.synthetic import make_signals


@pytest.fixture(scope="module")
def cfg():
    return load_config(calibration_file=None)


def ctx_for(cfg, duration=20.0, attention=None, tension=None, first_word=None, **kw):
    df = build_features(make_signals(duration, **kw), cfg)
    T = len(df)
    for k in INDEX_NAMES:
        df[f"idx_{k}"] = 0.6
    if attention is not None:
        df["idx_attention"] = attention
    if tension is not None:
        df["idx_tension"] = tension
    else:
        df["idx_tension"] = 0.4 + 0.3 * np.sin(np.arange(T))
    words = kw.get("words") or []
    fw = first_word if first_word is not None else (words[0][1] if words else None)
    return RuleContext(df=df, duration_s=duration, first_word_s=fw, has_speech=bool(words))


def rule_cfg(cfg, name):
    return next(r for r in cfg.rules if r["name"] == name)


def run(cfg, ctx, name):
    recs, warns = evaluate(ctx, [rule_cfg(cfg, name)])
    assert not warns
    return recs


def test_all_configured_rules_registered(cfg):
    for r in cfg.rules:
        assert r["name"] in REGISTRY


def test_attention_drop_no_cut(cfg):
    att = np.full(20, 0.7)
    att[8:12] = 0.2
    recs = run(cfg, ctx_for(cfg, attention=att, cuts=[3.0]), "attention_drop_no_cut")
    assert len(recs) == 1
    r = recs[0]
    assert (r["t"], r["t_end"], r["severity"]) == (8, 12, "high")
    assert r["message"].startswith("Chute d'attention à 8s sans changement de plan depuis 6 s")
    # un cut récent empêche la règle
    assert run(cfg, ctx_for(cfg, attention=att, cuts=[3.0, 8.5]), "attention_drop_no_cut") == []
    # une seule seconde basse ne suffit pas
    att2 = np.full(20, 0.7)
    att2[10] = 0.1
    assert run(cfg, ctx_for(cfg, attention=att2, cuts=[3.0]), "attention_drop_no_cut") == []


def test_weak_hook(cfg):
    recs = run(cfg, ctx_for(cfg, attention=np.full(20, 0.7), words=[("Bonjour", 2.2, 2.6)]), "weak_hook")
    assert len(recs) == 1 and recs[0]["t"] == 0 and recs[0]["t_end"] == 3
    assert "un texte à l'écran" in recs[0]["message"] and "un visage" in recs[0]["message"]
    good = ctx_for(cfg, attention=np.full(20, 0.7), words=[("Bonjour", 0.2, 0.6)],
                   texts={0: "3 ASTUCES", 1: "3 ASTUCES"}, faces={0: (1, 0.2)})
    assert run(cfg, good, "weak_hook") == []
    # éléments présents mais attention trop faible
    weak = ctx_for(cfg, attention=np.full(20, 0.2), words=[("Bonjour", 0.2, 0.6)],
                   texts={0: "3 ASTUCES"}, faces={0: (1, 0.2)})
    assert len(run(cfg, weak, "weak_hook")) == 1


def test_no_face_long(cfg):
    recs = run(cfg, ctx_for(cfg, faces={t: (1, 0.1) for t in range(4)}), "no_face_long")
    assert [(r["t"], r["t_end"]) for r in recs] == [(4, 20)]
    assert run(cfg, ctx_for(cfg, faces={t: (1, 0.1) for t in range(0, 20, 3)}), "no_face_long") == []


def test_silence_gap(cfg):
    recs = run(cfg, ctx_for(cfg, silence=[9, 10, 11, 12, 13]), "silence_gap")
    assert [(r["t"], r["t_end"]) for r in recs] == [(9, 14)]
    assert "Silence de 9s à 14s" in recs[0]["message"]
    assert run(cfg, ctx_for(cfg, silence=[5]), "silence_gap") == []
    assert run(cfg, ctx_for(cfg, silence=[18, 19]), "silence_gap") == []     # silence final toléré


def test_text_overload(cfg):
    long_text = " ".join(["beaucoup"] * 30)
    recs = run(cfg, ctx_for(cfg, texts={7: long_text, 8: long_text}), "text_overload")
    assert len(recs) == 1 and (recs[0]["t"], recs[0]["t_end"]) == (7, 9)
    assert "30 mots" in recs[0]["message"] and "10 s" in recs[0]["message"]
    assert run(cfg, ctx_for(cfg, texts={7: "ABONNE-TOI VITE", 8: "ABONNE-TOI VITE"}), "text_overload") == []


def test_cta_missing(cfg):
    recs = run(cfg, ctx_for(cfg), "cta_missing")
    assert len(recs) == 1 and recs[0]["t"] == 16 and recs[0]["t_end"] == 20
    assert run(cfg, ctx_for(cfg, texts={17: "LIEN EN BIO"}), "cta_missing") == []
    assert run(cfg, ctx_for(cfg, words=[("Abonne-toi", 18.0, 18.6)]), "cta_missing") == []
    assert len(run(cfg, ctx_for(cfg, words=[("Abonne-toi", 5.0, 5.6)]), "cta_missing")) == 1


def test_tension_flat(cfg):
    recs = run(cfg, ctx_for(cfg, tension=np.full(20, 0.5)), "tension_flat")
    assert [(r["t"], r["t_end"]) for r in recs] == [(0, 20)]
    assert run(cfg, ctx_for(cfg), "tension_flat") == []


def test_slow_start(cfg):
    recs = run(cfg, ctx_for(cfg, words=[("Bonjour", 2.3, 2.7)]), "slow_start")
    assert len(recs) == 1 and "2,3s" in recs[0]["message"]
    assert run(cfg, ctx_for(cfg, words=[("Bonjour", 0.4, 0.7)]), "slow_start") == []
    assert run(cfg, ctx_for(cfg), "slow_start") == []


def test_merge_and_format():
    hits = merge_hits([Hit(5, 6), Hit(6, 7), Hit(9, 10), Hit(2, 3)])
    assert [(h.t, h.t_end) for h in hits] == [(2, 3), (5, 7), (9, 10)]
    assert f"{Num(7.0)}" == "7" and f"{Num(7.5)}" == "7,5"
    assert f"{Num(3.0):.1f}" == "3.0"


def test_disabled_and_unknown_rules(cfg):
    ctx = ctx_for(cfg)
    rules = [dict(rule_cfg(cfg, "cta_missing"), enabled=False),
             {"name": "regle_inconnue", "enabled": True, "severity": "low", "params": {}, "message": "{t}"}]
    recs, warns = evaluate(ctx, rules)
    assert recs == [] and len(warns) == 1


def test_recommendations_sorted(cfg):
    att = np.full(20, 0.7)
    att[8:12] = 0.2
    ctx = ctx_for(cfg, attention=att, cuts=[3.0], silence=[9, 10, 11, 12], words=[("Bonjour", 2.0, 2.5)])
    recs, _ = evaluate(ctx, cfg.rules)
    ts = [r["t"] for r in recs]
    assert ts == sorted(ts)
    assert {"attention_drop_no_cut", "silence_gap", "cta_missing", "slow_start", "weak_hook"} <= {r["rule"] for r in recs}
    for r in recs:
        assert set(r) == {"t", "t_end", "rule", "severity", "message"}
        assert r["severity"] in ("high", "medium", "low") and r["t_end"] >= r["t"]
        assert "{" not in r["message"]
