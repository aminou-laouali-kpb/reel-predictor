"""Endpoints du coach « script & montage » (service coach mocké, aucun appel à l'API Claude)."""
from __future__ import annotations

import importlib.util
import json
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.server import create_app
from reel_predictor.coach import service

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_tac_{name}", HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


brief_t = _load("test_coach_brief")
llm_t = _load("test_coach_llm")


def fake_coach_result(job_dir: Path, stamp: str) -> dict:
    """Résultat coach factice complet (sortie Claude + simulation doublée), écrit comme le vrai service."""
    simulate, baseline, _ = llm_t.make_sim([0.45, 0.55, 0.72, 0.5])
    return service.run_coach(job_dir, None, stamp=stamp, coach_cfg=llm_t.coach_cfg(), client=llm_t.FakeClient(),
                             config=object(), simulate_fn=simulate, baseline_fn=baseline)


def make_client(tmp_path, *, has_key=True, coach_fn=None, analyze_fn=None):
    results = tmp_path / "results"
    job = brief_t.make_job(results)
    app = create_app(results_dir=results, uploads_dir=tmp_path / "uploads", db_path=tmp_path / "db.sqlite",
                     analyze_fn=analyze_fn, health_fn=lambda: {"brain_expected": "heuristic"},
                     coach_fn=coach_fn, coach_key_fn=lambda: has_key, coach_cfg_fn=llm_t.coach_cfg)
    return TestClient(app), job


def wait_coach(c: TestClient, coach_id: str, timeout: float = 20.0) -> dict:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        st = c.get(f"/api/coach/{coach_id}").json()
        if st["status"] in ("done", "error"):
            return st
        time.sleep(0.03)
    raise AssertionError("génération coach pas terminée")


def test_no_key_refuses_and_health(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", llm_t.FAKE_KEY)  # même présente dans l'env, jamais renvoyée
    c, job = make_client(tmp_path, has_key=False, coach_fn=lambda *a, **k: pytest.fail("ne doit pas tourner"))
    r = c.post(f"/api/results/{job.name}/coach")
    assert r.status_code == 400 and "make api-key" in r.json()["detail"]
    h = c.get("/api/health")
    body = h.json()
    assert body["claude_api"] is False
    assert isinstance(body["coach"]["estimated_cost_usd"], float) and body["coach"]["model"] == "claude-opus-5-5"
    assert llm_t.FAKE_KEY not in h.text
    hist = c.get(f"/api/results/{job.name}/coach").json()
    assert hist["claude_api"] is False and hist["items"] == [] and hist["pending"] == []
    assert 0 < hist["estimate"]["usd"] < 1 and hist["estimate"]["method"] == "approx"
    assert llm_t.FAKE_KEY not in json.dumps(hist)


def test_full_flow_with_mocked_service(tmp_path):
    seen = {}

    def coach_fn(job_dir, progress, stamp):
        seen["stamp"] = stamp
        progress("llm", 0.2, "Rédaction par Claude…")
        return fake_coach_result(job_dir, stamp)

    c, job = make_client(tmp_path, coach_fn=coach_fn)
    assert c.get("/api/health").json()["claude_api"] is True
    r = c.post(f"/api/results/{job.name}/coach")
    assert r.status_code == 200
    coach_id = r.json()["coach_id"]
    assert coach_id == f"{job.name}.{seen.get('stamp') or coach_id.rsplit('.', 1)[1]}"
    st = wait_coach(c, coach_id)
    assert st["status"] == "done", st
    res = st["result"]
    assert res["coach_id"] == coach_id and len(res["output"]["variants"]) == 3
    assert res["simulation"]["results"][0]["name"] == "original"
    assert {"stage", "pct", "message", "eta_s"} <= set(st)

    hist = c.get(f"/api/results/{job.name}/coach").json()
    assert [i["coach_id"] for i in hist["items"]] == [coach_id]
    item = hist["items"][0]
    assert item["cost_usd"] == pytest.approx(0.224) and item["best"]["index"] == 1 and item["n_variants"] == 3
    assert item["original"]["score"] == 45

    md = c.get(f"/api/coach/{coach_id}/markdown")
    assert md.status_code == 200 and md.headers["content-type"].startswith("text/markdown")
    assert "attachment" in md.headers["content-disposition"] and "# Coach script & montage" in md.text
    # coût moyen des générations précédentes dans /api/health
    h = c.get("/api/health").json()["coach"]
    assert h["n_generations"] == 1 and h["estimated_cost_usd"] == pytest.approx(0.224)


def test_persisted_result_readable_after_restart(tmp_path):
    c, job = make_client(tmp_path)
    fake_coach_result(job, "20261012-160000")
    c2 = TestClient(create_app(results_dir=job.parent, uploads_dir=tmp_path / "u2", db_path=tmp_path / "db2.sqlite",
                               health_fn=lambda: {}, coach_key_fn=lambda: True))
    st = c2.get(f"/api/coach/{job.name}.20261012-160000").json()
    assert st["status"] == "done" and st["result"]["stamp"] == "20261012-160000"


def test_errors_and_invalid_ids(tmp_path):
    def failing(job_dir, progress, stamp):
        raise RuntimeError("Clé API Anthropic refusée (invalide ou révoquée).")

    c, job = make_client(tmp_path, coach_fn=failing)
    st = wait_coach(c, c.post(f"/api/results/{job.name}/coach").json()["coach_id"])
    assert st["status"] == "error" and "refusée" in st["error"] and st["result"] is None
    pend = c.get(f"/api/results/{job.name}/coach").json()["pending"]
    assert pend and pend[0]["status"] == "error"
    assert c.get(f"/api/coach/{job.name}.20990101-000000").status_code == 404
    assert c.get("/api/coach/..%2F..%2Fetc.20990101-000000").status_code == 404
    assert c.get("/api/coach/sans-horodatage").status_code == 404
    assert c.get(f"/api/coach/{job.name}.20990101-000000/markdown").status_code == 404
    assert c.post("/api/results/inconnu-123/coach").status_code == 404
    # analyse incomplète
    (job / "timeline.jsonl").unlink()
    assert c.post(f"/api/results/{job.name}/coach").status_code == 409


def test_coach_shares_the_analysis_queue(tmp_path):
    """Une seule tâche lourde à la fois : la génération attend la fin de l'analyse en cours."""
    release = threading.Event()
    started = threading.Event()

    def slow_analyze(video_path, out_dir, config=None, progress=None, brain=None):
        started.set()
        release.wait(10)
        raise RuntimeError("analyse factice interrompue")

    ran = []

    def coach_fn(job_dir, progress, stamp):
        ran.append(stamp)
        return {}

    c, job = make_client(tmp_path, coach_fn=coach_fn, analyze_fn=slow_analyze)
    r = c.post("/api/analyze", files={"file": ("x.mp4", b"0" * 64, "video/mp4")})
    assert r.status_code == 200
    assert started.wait(5)
    coach_id = c.post(f"/api/results/{job.name}/coach").json()["coach_id"]
    # double clic : même génération
    assert c.post(f"/api/results/{job.name}/coach").json()["coach_id"] == coach_id
    time.sleep(0.2)
    st = c.get(f"/api/coach/{coach_id}").json()
    assert st["status"] == "queued" and not ran
    assert c.get("/api/health").json()["queue"]["coach_queued"] == 1
    release.set()
    st = wait_coach(c, coach_id)
    assert st["status"] == "done" and ran
