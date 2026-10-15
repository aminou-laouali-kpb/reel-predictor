"""Client Claude du coach (`llm.py`), clé API (`keys.py`) et service (`service.py`) — client Claude MOCKÉ.

Aucun appel réseau réel : le client Anthropic est remplacé par une doublure qui enregistre les paramètres.
"""
from __future__ import annotations

import copy
import importlib.util
import inspect
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from reel_predictor import paths
from reel_predictor.coach import keys, llm, service
from reel_predictor.coach.schema import CoachOutput
from reel_predictor.coach.targets import evaluate, moments_from_roles

FAKE_KEY = "sk-ant-test-NE-PAS-AFFICHER-0123456789"
HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_t_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


brief_t = _load("test_coach_brief")


def coach_cfg(**over) -> dict:
    """Config de test : coach.yaml, mais 1 passe et cibles FIXES (indépendant de config/ et de data/results)."""
    with open(paths.CONFIG_DIR / "coach.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["rounds"] = 1
    cfg["targets"] = {**cfg["targets"], "mode": "absolute", "peak": 0.9, "floor": 0.5}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k] = {**cfg[k], **v}
        else:
            cfg[k] = v
    return cfg


def sample_output(n_variants: int = 3) -> dict:
    def seg(role, voice, txt="", shot="face_cam", cut=True, edit=""):
        return {"role": role, "voice_over": voice, "on_screen_text": txt, "shot": shot, "cut_at_start": cut, "edit": edit}

    variants = []
    for i in range(n_variants):
        variants.append({
            "angle": f"Angle {i + 1}",
            "hook_summary": "Le chiffre tombe dès le premier mot.",
            "segments": [
                seg("hook", "1400 euros par mois pour étudier en Europe.", "1400 €/MOIS", "close_up"),
                seg("fact", "Pendant tout le master, frais et voyage compris.", "TOUT LE MASTER", "text_card"),
                seg("body", "Il faut une licence et un bon niveau d'anglais.", "LICENCE + ANGLAIS", "face_cam", True, "zoom"),
                seg("cta", "Prends mon guide, lien en bio.", "LIEN EN BIO", "face_cam", True, "sfx"),
            ],
            "rationale": "Accroche chiffrée, rythme rapide.",
        })
    return {
        "diagnosis_summary": "L'attention chute à 39–46 s sur un plan fixe de 10 s.",
        "montage_plan": [
            {"t": 39.0, "t_end": 46.0, "action": "zoom", "instruction": "Zoom punch-in toutes les 3 s",
             "on_screen_text": "", "expected_effect": "Relance l'attention", "target_indices": ["attention"]},
            {"t": 48.0, "t_end": 50.0, "action": "text", "instruction": "Remplacer le carton de 33 mots",
             "on_screen_text": "PRENDS LE GUIDE", "expected_effect": "Lisible", "target_indices": ["action"]},
        ],
        "variants": variants,
        "notes": ["Indices mémoire et émotion à faible confiance."],
    }


# ----------------------------------------------------------------------------- doublure du client Anthropic
class FakeStream:
    def __init__(self, final, chunks, raise_exc=None):
        self.final, self.chunks, self.raise_exc = final, chunks, raise_exc

    def __enter__(self):
        if self.raise_exc is not None:
            raise self.raise_exc
        return self

    def __exit__(self, *a):
        return False

    def __iter__(self):
        yield SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="thinking"))
        for c in self.chunks:
            yield SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(type="text_delta", text=c))

    def get_final_message(self):
        return self.final


class FakeClient:
    def __init__(self, text=None, *, stop_reason="end_turn", usage=None, content=None, raise_exc=None,
                 count_tokens=5000, count_exc=None, iterations=None, texts=None):
        self.calls, self.count_calls = [], []
        self._texts = list(texts) if texts else None
        self.text = text if text is not None else json.dumps(sample_output(), ensure_ascii=False)
        self.stop_reason, self.content, self.raise_exc = stop_reason, content, raise_exc
        self.usage = usage or {"input_tokens": 6000, "output_tokens": 10000, "cache_read_input_tokens": 0,
                               "cache_creation_input_tokens": 0}
        self.iterations = iterations
        self._count, self._count_exc = count_tokens, count_exc
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))
        self.messages = SimpleNamespace(count_tokens=self._count_tokens)

    def _count_tokens(self, **kw):
        self.count_calls.append(kw)
        if self._count_exc:
            raise self._count_exc
        return SimpleNamespace(input_tokens=self._count)

    def _stream(self, **kw):
        self.calls.append(kw)
        text = self._texts.pop(0) if self._texts else self.text
        content = self.content if self.content is not None else [SimpleNamespace(type="text", text=text)]
        final = SimpleNamespace(content=content, stop_reason=self.stop_reason, model=kw.get("model"),
                                usage=SimpleNamespace(**self.usage, iterations=self.iterations),
                                stop_details=SimpleNamespace(category="cyber", explanation="x"), _request_id="req_test")
        return FakeStream(final, [text[i:i + 500] for i in range(0, len(text), 500)], self.raise_exc)


def tiny_brief(duration=50.0) -> dict:
    return {"video": {"duration_s": duration}, "segments": [], "evaluation": {}}


def _api_error(cls, status, headers=None):
    import httpx2

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    resp = httpx2.Response(status, request=req, headers=headers or {}, json={"error": {"message": "x"}})
    return cls("erreur simulée", response=resp, body=None)


# ----------------------------------------------------------------------------- SDK & requête
def test_sdk_supports_fallbacks_and_structured_output():
    from anthropic.resources.beta.messages import Messages as BetaMessages
    from anthropic.resources.messages import Messages

    params = inspect.signature(BetaMessages.stream).parameters
    assert {"fallbacks", "betas", "output_config", "thinking", "system"} <= set(params)
    assert "output_config" in inspect.signature(Messages.count_tokens).parameters


def _all_objects(schema):
    if isinstance(schema, dict):
        if schema.get("type") == "object":
            yield schema
        for v in schema.values():
            yield from _all_objects(v)
    elif isinstance(schema, list):
        for v in schema:
            yield from _all_objects(v)


def test_request_shape_and_schema():
    cfg = coach_cfg()
    client = FakeClient()
    res = llm.generate(tiny_brief(), cfg, client=client)
    assert len(client.calls) == 1
    kw = client.calls[0]
    assert kw["model"] == cfg["llm"]["model"] == "claude-opus-5-5"
    assert kw["max_tokens"] == cfg["llm"]["max_tokens"]
    assert kw["thinking"] == {"type": "adaptive"}
    assert "budget_tokens" not in json.dumps(kw["thinking"]) and "temperature" not in kw
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["fallbacks"] == "default"
    oc = kw["output_config"]
    assert oc["effort"] == cfg["llm"]["effort"]
    assert oc["format"]["type"] == "json_schema"
    objs = list(_all_objects(oc["format"]["schema"]))
    assert len(objs) >= 4 and all(o.get("additionalProperties") is False for o in objs)
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "N'invente AUCUN chiffre" in kw["system"][0]["text"]
    user = kw["messages"][0]["content"][0]
    assert user["cache_control"] == {"type": "ephemeral"} and "exactement 3 variante" in user["text"]
    # count_tokens appelé avec les mêmes paramètres (estimation exacte)
    assert client.count_calls and client.count_calls[0]["output_config"] == oc
    assert isinstance(res.output, CoachOutput) and len(res.output.variants) == 3
    assert res.estimate["method"] == "count_tokens" and res.estimate["input_tokens"] == 5000
    # coût réel : 6000 × 4 $ + 10000 × 20 $ par million
    assert res.cost_usd == pytest.approx(0.224)
    assert res.request_id == "req_test" and res.fallback_used is False and res.warnings == []


def test_cost_from_usage_with_cache():
    cfg = coach_cfg()
    usd = llm.cost_from_usage({"input_tokens": 1000, "output_tokens": 2000, "cache_read_input_tokens": 10000,
                               "cache_creation_input_tokens": 4000}, cfg)
    assert usd == pytest.approx((1000 * 4 + 2000 * 20 + 10000 * 0.2 + 4000 * 5.0) / 1e6, abs=1e-4)


def test_cost_guard_blocks_before_call():
    cfg = coach_cfg(llm={"max_cost_usd": 0.05})
    client = FakeClient()
    with pytest.raises(llm.CostLimitError) as ei:
        llm.generate(tiny_brief(), cfg, client=client)
    assert client.calls == [] and "plafond" in str(ei.value) and ei.value.cost_usd == 0


def test_count_tokens_failure_falls_back_to_approx():
    client = FakeClient(count_exc=RuntimeError("réseau"))
    res = llm.generate(tiny_brief(), coach_cfg(), client=client)
    assert res.estimate["method"] == "approx" and res.estimate["input_tokens"] > 1000


def test_offline_estimate_reasonable():
    est = llm.estimate_offline({"video": {"duration_s": 50}, "segments": []}, coach_cfg())
    assert 0.05 < est["usd"] < 1.0 and est["usd_max"] >= est["usd"] and est["method"] == "approx"
    assert 0.1 < llm.typical_cost(coach_cfg()) < 0.6


# ----------------------------------------------------------------------------- réponses anormales
def test_refusal_is_reported_with_cost():
    client = FakeClient(stop_reason="refusal", usage={"input_tokens": 6000, "output_tokens": 10})
    with pytest.raises(llm.CoachLLMError) as ei:
        llm.generate(tiny_brief(), coach_cfg(), client=client)
    assert ei.value.code == "refusal" and "cyber" in str(ei.value) and ei.value.cost_usd > 0


def test_max_tokens_is_reported():
    client = FakeClient(stop_reason="max_tokens", text='{"diagnosis_summary": "tronq')
    with pytest.raises(llm.CoachLLMError) as ei:
        llm.generate(tiny_brief(), coach_cfg(), client=client)
    assert ei.value.code == "max_tokens" and "max_tokens" in str(ei.value)


@pytest.mark.parametrize("text", ['{"diagnosis_summary": "x"}', "pas du json", json.dumps(
    {**sample_output(), "extra": 1})])
def test_invalid_output_rejected(text):
    with pytest.raises(llm.CoachLLMError) as ei:
        llm.generate(tiny_brief(), coach_cfg(), client=FakeClient(text=text))
    assert ei.value.code == "invalid_output"


def test_empty_answer():
    with pytest.raises(llm.CoachLLMError) as ei:
        llm.generate(tiny_brief(), coach_cfg(), client=FakeClient(content=[]))
    assert ei.value.code == "empty"


def test_fallback_detection_and_constraint_warnings():
    out = sample_output()
    out["variants"][0]["segments"][1]["on_screen_text"] = "un deux trois quatre cinq six sept huit neuf dix"
    out["variants"][1]["segments"][1]["role"] = "cta"
    out["montage_plan"][0]["t_end"] = 10.0  # t_end < t
    content = [SimpleNamespace(type="fallback", from_=SimpleNamespace(model="claude-opus-5-5")),
               SimpleNamespace(type="text", text=json.dumps(out, ensure_ascii=False))]
    res = llm.generate(tiny_brief(), coach_cfg(), client=FakeClient(content=content))
    w = " | ".join(res.warnings)
    assert res.fallback_used and "Repli serveur" in w
    assert "dépassent 8 mots" in w and "2 segment(s) d'appel" in w and "temps incohérents" in w


@pytest.mark.parametrize("cls_name,status,code,fragment", [
    ("AuthenticationError", 401, "auth", "make api-key"),
    ("PermissionDeniedError", 403, "permission", "accès"),
    ("NotFoundError", 404, "not_found", "Modèle introuvable"),
    ("RateLimitError", 429, "rate_limit", "réessayez dans 30 s"),
    ("BadRequestError", 400, "bad_request", "Requête refusée"),
    ("InternalServerError", 500, "api_status", "HTTP 500"),
])
def test_api_errors_translated(cls_name, status, code, fragment):
    import anthropic

    exc = _api_error(getattr(anthropic, cls_name), status, {"retry-after": "30"})
    with pytest.raises(llm.CoachLLMError) as ei:
        llm.generate(tiny_brief(), coach_cfg(), client=FakeClient(raise_exc=exc))
    assert ei.value.code == code and fragment in str(ei.value)
    assert FAKE_KEY not in str(ei.value)


def test_connection_errors_translated():
    import anthropic
    import httpx2

    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    with pytest.raises(llm.CoachLLMError) as ei:
        llm.generate(tiny_brief(), coach_cfg(), client=FakeClient(raise_exc=anthropic.APIConnectionError(request=req)))
    assert ei.value.code == "connection"
    with pytest.raises(llm.CoachLLMError) as ei:
        llm.generate(tiny_brief(), coach_cfg(), client=FakeClient(raise_exc=anthropic.APITimeoutError(request=req)))
    assert ei.value.code == "timeout"


def test_make_client_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv(keys.NO_KEYCHAIN_ENV, "1")
    with pytest.raises(llm.CoachLLMError) as ei:
        llm.make_client()
    assert ei.value.code == "no_key" and "make api-key" in str(ei.value)


# ----------------------------------------------------------------------------- clé API
def test_keys_env_and_keychain(monkeypatch):
    keys.reset_cache()
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    assert keys.has_key(refresh=True) and keys.key_source() == "env" and keys.get_api_key() == FAKE_KEY
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    monkeypatch.setenv(keys.NO_KEYCHAIN_ENV, "1")
    assert keys.has_key(refresh=True) is False and keys.get_api_key() is None
    # trousseau macOS simulé
    monkeypatch.delenv(keys.NO_KEYCHAIN_ENV)
    monkeypatch.setattr(keys, "_keychain_enabled", lambda: True)
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout=FAKE_KEY + "\n", stderr="")

    monkeypatch.setattr(keys.subprocess, "run", fake_run)
    assert keys.get_api_key() == FAKE_KEY and keys.key_source() == "keychain"
    assert seen["cmd"] == ["security", "find-generic-password", "-s", "reel-predictor-anthropic", "-w"]
    monkeypatch.setattr(keys.subprocess, "run",
                        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 44, stdout="", stderr="not found"))
    assert keys.has_key(refresh=True) is False
    keys.reset_cache()


def test_makefile_api_key_target():
    mk = (paths.ROOT / "Makefile").read_text(encoding="utf-8")
    assert "api-key:" in mk
    assert "security add-generic-password -U -s reel-predictor-anthropic" in mk
    # la clé ne doit jamais passer en argument (visible dans `ps`) : -w termine la commande
    line = next(l for l in mk.splitlines() if "add-generic-password" in l)
    assert line.rstrip(" \\;").endswith("-w") or "-w;" in line


# ----------------------------------------------------------------------------- service (simulation doublée)
def fake_sim_result(name, segments, base_level, cfg):
    t, spans, rows = 0.0, [], []
    for s in segments:
        dur = max(1.0, len(s["voice_over"].split()) / 3.2)
        spans.append({"role": s["role"], "t": round(t, 2), "t_end": round(t + dur, 2)})
        t += dur
    T = int(t) + 1
    for i in range(T):
        lvl = base_level + (0.3 if i < 3 else 0.0)
        rows.append({"t": i, "indices": {k: min(1.0, lvl) for k in ("attention", "action", "tension", "manipulation",
                                                                         "emotion", "memory")}, "retention": 0.9})
    moments = moments_from_roles([(s["role"], s["t"], s["t_end"]) for s in spans], t, cfg["targets"])
    return {"name": name, "duration_s": round(t, 2), "score": int(base_level * 100), "segments": spans,
            "timeline": rows, "evaluation": evaluate(rows, moments, cfg["targets"])}


def make_sim(levels):
    calls = []

    def baseline_variant(job_dir):
        v = CoachOutput.model_validate(sample_output(1)).variants[0]
        return v.model_copy(update={"angle": "original"})

    def simulate(variants, job_dir, config, coach_cfg, progress=None):
        calls.append(len(variants))
        if progress:
            progress(0.5, "simulation")
        results = []
        for i, v in enumerate(variants):
            lvl = levels.pop(0)
            name = "original" if (len(calls) == 1 and i == 0) else f"variante {i}"
            results.append(fake_sim_result(name, [s.model_dump() for s in v.segments], lvl, coach_cfg))
        return {"mode": "heuristic", "reason": "test", "results": results, "best": 0, "timings": {}}

    return simulate, baseline_variant, calls


def test_run_coach_end_to_end(tmp_path):
    job = brief_t.make_job(tmp_path)
    cfg = coach_cfg()
    simulate, baseline, calls = make_sim([0.45, 0.55, 0.7, 0.5])
    progress = []
    res = service.run_coach(job, lambda s, p, m: progress.append((s, p, m)), stamp="20261012-150000",
                            coach_cfg=cfg, client=FakeClient(), config=object(), simulate_fn=simulate,
                            baseline_fn=baseline)
    assert {"created_at", "model", "usage", "cost_usd", "output", "brief", "simulation", "warnings"} <= set(res)
    assert res["coach_id"] == f"{job.name}.20261012-150000"
    assert res["cost_usd"] == pytest.approx(0.224) and calls == [4]
    assert res["ranking"]["best_variant"] == 1  # variante 2 (niveau 0,7)
    assert res["ranking"]["order"][0] == 2
    assert (job / "coach" / "20261012-150000.json").is_file()
    md = (job / "coach" / "20261012-150000.md").read_text(encoding="utf-8")
    assert "Plan de montage" in md and "**recommandée**" in md and "| Temps | Voix off |" in md
    assert "ne garantissent pas les vues" in md
    saved = json.loads((job / "coach" / "20261012-150000.json").read_text())
    assert saved["output"]["variants"][0]["segments"][0]["role"] == "hook"
    stages = [s for s, _, _ in progress]
    assert stages[0] == "brief" and "llm" in stages and "simulate" in stages and stages[-1] == "write"
    assert all(0 <= p < 1 for _, p, _ in progress)


def test_run_coach_simulation_failure_still_returns_claude_output(tmp_path):
    job = brief_t.make_job(tmp_path)

    def broken_sim(*a, **k):
        raise RuntimeError("TRIBE introuvable")

    res = service.run_coach(job, None, stamp="20261012-150001", coach_cfg=coach_cfg(), client=FakeClient(),
                            config=object(), simulate_fn=broken_sim, baseline_fn=lambda d: None)
    assert res["simulation"] is None and len(res["output"]["variants"]) == 3
    assert any("Simulation des variantes impossible" in w for w in res["warnings"])
    assert (job / "coach" / "20261012-150001.md").is_file()


def test_run_coach_round2(tmp_path):
    job = brief_t.make_job(tmp_path)
    cfg = coach_cfg(rounds=2)
    simulate, baseline, calls = make_sim([0.45, 0.55, 0.7, 0.5, 0.8])
    out2 = sample_output(1)
    out2["variants"][0]["angle"] = "Angle 2 corrigé"
    client = FakeClient(texts=[json.dumps(sample_output()), json.dumps(out2)])
    res = service.run_coach(job, None, stamp="20261012-150002", coach_cfg=cfg, client=client, config=object(),
                            simulate_fn=simulate, baseline_fn=baseline)
    assert len(client.calls) == 2 and calls == [4, 1]
    # passe 2 : même préfixe (système + diagnostic en cache) + réponse 1 + indices simulés
    m2 = client.calls[1]["messages"]
    assert m2[0] == client.calls[0]["messages"][0] and m2[1]["role"] == "assistant" and "SIMULATION" in m2[2]["content"]
    assert len(res["output"]["variants"]) == 4 and res["output"]["variants"][3]["angle"].endswith("(révisée)")
    assert res["simulation"]["results"][4]["name"].endswith("(révisée)")
    assert res["ranking"]["best_variant"] == 3
    assert res["cost_usd"] == pytest.approx(0.448) and len(res["rounds"]) == 2


def test_run_coach_propagates_llm_errors(tmp_path):
    job = brief_t.make_job(tmp_path)
    with pytest.raises(llm.CoachLLMError) as ei:
        service.run_coach(job, None, stamp="20261012-150003", coach_cfg=coach_cfg(),
                          client=FakeClient(stop_reason="refusal"), config=object(),
                          simulate_fn=lambda *a, **k: None, baseline_fn=lambda d: None)
    assert "coût déjà engagé" in str(ei.value)
    assert not (job / "coach").exists()


def test_markdown_without_simulation():
    res = {"job_id": "x", "created_at": "2026", "model": "m", "cost_usd": 0.2, "output": sample_output(2),
           "simulation": None, "ranking": service.ranking(None), "brief": {}, "config": {}, "warnings": []}
    md = service.to_markdown(copy.deepcopy(res))
    assert "Variante 1" in md and "Variante 2" in md and "#1" in md
    txt = service.variant_script_text(sample_output(1)["variants"][0])
    assert "ACCROCHE" in txt and "Voix off : 1400 euros" in txt


# ----------------------------------------------------------------------------- cibles relatives / régressions
def _write_real_result(d, values):
    import json as _json
    d.mkdir(parents=True)
    (d / "summary.json").write_text(_json.dumps({"brain": {"available": True}}), encoding="utf-8")
    rows = [{"t": t, "indices": {k: v for k in ("attention", "action", "tension", "manipulation", "emotion", "memory")}}
            for t, v in enumerate(values)]
    (d / "timeline.jsonl").write_text("\n".join(_json.dumps(r) for r in rows), encoding="utf-8")


def test_resolve_targets_relative_and_fallback(tmp_path):
    from reel_predictor.coach import targets as T
    tcfg = coach_cfg()["targets"] | {"mode": "relative", "min_reels": 2, "peak_percentile": 90, "floor_percentile": 25}
    one = T.resolve_targets(tcfg, tmp_path)
    assert one["basis"]["mode"] == "absolute" and "reason" in one["basis"] and "peak_by_index" not in one
    _write_real_result(tmp_path / "a", [0.4, 0.5, 0.6, 0.7])
    _write_real_result(tmp_path / "b", [0.3, 0.5, 0.55, 0.65])
    rel = T.resolve_targets(tcfg, tmp_path)
    assert rel["basis"]["mode"] == "relative" and rel["basis"]["n_reels"] == 2
    assert 0.6 < rel["peak_by_index"]["attention"] < 0.71 and rel["floor"] < 0.55


def test_to_sim_scale_maps_through_original_quantiles():
    from reel_predictor.coach import targets as T
    ks = ("attention", "action", "tension", "manipulation", "emotion", "memory")
    real = [{"t": i, "indices": {k: 0.4 + 0.02 * i for k in ks}} for i in range(11)]   # 0.40 .. 0.60
    sim = [{"t": i, "indices": {k: 0.2 + 0.04 * i for k in ks}} for i in range(11)]    # 0.20 .. 0.60
    tc = {"peak": 0.9, "peak_by_index": {k: 0.5 for k in ks}, "floor": 0.45, "floor_index": "attention", "moments": {}}
    out = T.to_sim_scale(tc, real, sim)
    assert out["peak_by_index"]["attention"] == pytest.approx(0.4, abs=0.01)    # médiane réelle -> médiane simulée
    assert out["floor"] == pytest.approx(0.3, abs=0.01)
    over = T.to_sim_scale({**tc, "peak_by_index": {k: 0.7 for k in ks}}, real, sim)  # au-delà du max : extrapolation
    assert over["peak_by_index"]["attention"] > 0.6


def test_compare_to_original_flags_regressions():
    from reel_predictor.coach import targets as T
    ev_o = {"moments": [{"name": "fact", "values": {"emotion": 0.52, "memory": 0.57}}],
            "index_means": {"emotion": 0.5}, "seconds_below_floor": [1, 2, 3]}
    ev_v = {"moments": [{"name": "fact", "values": {"emotion": 0.48, "memory": 0.69}}],
            "index_means": {"emotion": 0.47}, "seconds_below_floor": [1]}
    c = T.compare_to_original(ev_v, ev_o)
    assert [g["index"] for g in c["regressions"]] == ["emotion"] and [g["index"] for g in c["gains"]] == ["memory"]
    assert c["below_floor_delta"] == -2 and c["means_delta"]["emotion"] == pytest.approx(-0.03)


def test_moments_from_roles_uses_first_contiguous_block():
    """Un segment « fact » étiqueté loin plus tard ne doit pas étirer la fenêtre du moment clé."""
    from reel_predictor.coach import targets as T
    tcfg = {"moments": {"fact": {"roles": ["fact"], "indices": ["memory"]}}}
    spans = [("hook", 0, 3), ("fact", 3, 6), ("fact", 6, 9), ("body", 9, 20), ("fact", 20, 24), ("cta", 24, 27)]
    (m,) = T.moments_from_roles(spans, 27, tcfg)
    assert (m["t"], m["t_end"]) == (3, 9)
