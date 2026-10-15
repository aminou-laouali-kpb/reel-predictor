"""Tests de l'API HTTP de l'app (contrat §5) avec un pipeline factice injecté."""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.jobs import JOB_ID_RE
from app.server import create_app
from reel_predictor.stages import global_progress

FIXTURES = Path(__file__).resolve().parent / "fixtures"
N_VERTICES = 20484
JOB_KEYS = {"status", "stage", "stage_label", "pct", "message", "eta_s", "error"}


def _load_fixture(name: str):
    spec = importlib.util.spec_from_file_location(f"_fx_{name}", FIXTURES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fake = _load_fixture("make_fake_result")


@pytest.fixture(scope="module")
def small_video(tmp_path_factory) -> Path:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg absent")
    p = tmp_path_factory.mktemp("video") / "reel.mp4"
    return fake.make_fake_video(p, duration_s=6.0, width=180, height=320)


def fake_analyze(video_path, out_dir, config=None, progress=None, brain=None):
    for stage in ("probe", "speech", "brain", "write"):
        if progress:
            progress(stage, global_progress(stage, 0.5), f"étape {stage}")
    fake.make_fake_result(Path(out_dir), job_id=Path(out_dir).name)


calls: list = []


def fake_calibrate(pairs):
    calls.append(pairs)
    if len(pairs) < 3:
        raise ValueError("n < 3")
    return {"n": len(pairs), "params": {"base_drop": 0.03}, "mae_before": 0.2, "mae_after": 0.1}


def make_client(tmp_path, **kw) -> tuple[TestClient, dict]:
    dirs = {"results_dir": tmp_path / "results", "uploads_dir": tmp_path / "uploads", "db_path": tmp_path / "db.sqlite"}
    kw.setdefault("analyze_fn", fake_analyze)
    kw.setdefault("calibrate_fn", fake_calibrate)
    kw.setdefault("health_fn", lambda: {"brain_expected": "heuristic", "tribe_model": "x", "worker_installed": False,
                                        "hf_token": False, "llama_access": None, "reason": "test", "versions": {}})
    app = create_app(**dirs, **kw)
    return TestClient(app), dirs


def wait_done(client: TestClient, job_id: str, timeout: float = 60.0) -> dict:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        st = client.get(f"/api/jobs/{job_id}").json()
        if st["status"] in ("done", "error"):
            return st
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} pas terminé en {timeout} s")


def upload(client: TestClient, path: Path, name: str = "mon reel.mp4") -> str:
    with open(path, "rb") as fh:
        r = client.post("/api/analyze", files={"file": (name, fh, "video/mp4")})
    assert r.status_code == 200, r.text
    return r.json()["job_id"]


# ---------------------------------------------------------------------------- santé & statique
def test_health_and_static(tmp_path):
    c, _ = make_client(tmp_path)
    h = c.get("/api/health").json()
    assert h["ok"] is True and h["status"]["brain_expected"] == "heuristic"
    assert "queue" in h and "assets" in h
    r = c.get("/")
    assert r.status_code == 200 and "Reel Predictor" in r.text
    assert c.get("/app.js").status_code == 200


def test_local_only_protections(tmp_path):
    c, _ = make_client(tmp_path)
    assert c.get("/api/health", headers={"host": "evil.example"}).status_code == 403
    r = c.post("/api/calibrate", headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    assert c.get("/api/health", headers={"host": "127.0.0.1:8765"}).status_code == 200


# ---------------------------------------------------------------------------- analyse complète
def test_upload_job_results_media(tmp_path, small_video):
    c, dirs = make_client(tmp_path)
    job_id = upload(c, small_video)
    assert JOB_ID_RE.match(job_id), job_id
    st = wait_done(c, job_id)
    assert st["status"] == "done", st
    assert JOB_KEYS <= set(st) and st["pct"] == 1.0
    out = dirs["results_dir"] / job_id
    assert (out / "video.mp4").read_bytes() == small_video.read_bytes()
    assert json.loads((out / "app_job.json").read_text())["filename"] == "mon reel.mp4"

    summary = c.get(f"/api/results/{job_id}/summary").json()
    assert summary["version"] == 1 and 0 <= summary["score"] <= 100
    timeline = c.get(f"/api/results/{job_id}/timeline").json()
    T = len(timeline)
    assert isinstance(timeline, list) and T == 6
    assert [r["t"] for r in timeline] == list(range(T))
    meta = c.get(f"/api/results/{job_id}/brain_meta").json()
    assert meta["T"] == T and meta["n_vertices"] == N_VERTICES and meta["vmin"] == -meta["vmax"]
    brain = c.get(f"/api/results/{job_id}/brain")
    assert brain.status_code == 200 and len(brain.content) == T * N_VERTICES
    assert brain.headers["content-type"] == "application/octet-stream"

    # Vidéo + requêtes Range (seek du lecteur)
    size = (out / "video.mp4").stat().st_size
    full = c.get(f"/media/{job_id}.mp4")
    assert full.status_code == 200 and full.headers["accept-ranges"] == "bytes"
    assert full.headers["content-type"] == "video/mp4" and len(full.content) == size
    part = c.get(f"/media/{job_id}.mp4", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 100-199/{size}"
    assert part.content == full.content[100:200]
    tail = c.get(f"/media/{job_id}.mp4", headers={"Range": "bytes=-50"})
    assert tail.status_code == 206 and tail.content == full.content[-50:]
    assert c.get(f"/media/{job_id}.mp4", headers={"Range": f"bytes={size + 10}-"}).status_code == 416

    reels = c.get("/api/reels").json()
    assert [r["job_id"] for r in reels["reels"]] == [job_id]
    assert reels["reels"][0]["filename"] == "mon reel.mp4" and reels["reels"][0]["result_available"]


def test_unknown_and_invalid_ids(tmp_path):
    c, _ = make_client(tmp_path)
    assert c.get("/api/jobs/20260101-000000-deadbeef").status_code == 404
    assert c.get("/api/results/inconnu/summary").status_code == 404
    assert c.get("/api/results/..%2F..%2Fetc/summary").status_code == 404
    assert c.get("/media/..%2Fdb.mp4").status_code == 404
    r = c.post("/api/analyze", files={"file": ("vide.mp4", b"", "video/mp4")})
    assert r.status_code == 400


def test_analysis_error_is_reported(tmp_path):
    def boom(video_path, out_dir, config=None, progress=None, brain=None):
        progress("probe", 0.01, "Lecture")
        raise RuntimeError("Vidéo illisible : flux vidéo absent")

    c, dirs = make_client(tmp_path, analyze_fn=boom)
    job_id = upload(c, Path(__file__))  # contenu quelconque
    st = wait_done(c, job_id)
    assert st["status"] == "error" and "illisible" in st["error"]
    assert (dirs["results_dir"] / job_id / "app_error.log").is_file()
    assert c.get(f"/api/results/{job_id}/summary").status_code == 404


def test_single_worker_runs_jobs_one_at_a_time(tmp_path):
    lock = threading.Lock()
    active = {"now": 0, "max": 0}

    def slow(video_path, out_dir, config=None, progress=None, brain=None):
        with lock:
            active["now"] += 1
            active["max"] = max(active["max"], active["now"])
        try:
            progress("brain", 0.5, "…")
            time.sleep(0.4)
            fake.make_fake_result(Path(out_dir), duration_s=3, brain=False, job_id=Path(out_dir).name)
        finally:
            with lock:
                active["now"] -= 1

    c, dirs = make_client(tmp_path, analyze_fn=slow)
    src = tmp_path / "videos"
    src.mkdir()
    a, b = src / "a.mp4", src / "b.mp4"
    fake.make_fake_video(a, 3, 160, 288)
    shutil.copyfile(a, b)
    with open(b, "ab") as fh:
        fh.write(b"\0")  # contenu différent -> sha différent
    ja, jb = upload(c, a, "a.mp4"), upload(c, b, "b.mp4")
    assert ja != jb
    st_b = c.get(f"/api/jobs/{jb}").json()
    assert st_b["status"] in ("queued", "running", "done")
    jobs = c.get("/api/jobs").json()
    assert {j["job_id"] for j in jobs} == {ja, jb}
    assert wait_done(c, ja)["status"] == "done" and wait_done(c, jb)["status"] == "done"
    assert active["max"] == 1


def test_fake_mode_env(tmp_path, small_video, monkeypatch):
    monkeypatch.setenv("REEL_PREDICTOR_FAKE", "1")
    monkeypatch.setenv("REEL_PREDICTOR_FAKE_DELAY", "0")
    c, _ = make_client(tmp_path, analyze_fn=None)
    assert c.get("/api/health").json()["fake_mode"] is True
    job_id = upload(c, small_video)
    st = wait_done(c, job_id)
    assert st["status"] == "done", st
    summary = c.get(f"/api/results/{job_id}/summary").json()
    assert summary["fake"] is True and summary["brain"]["available"] is True


def test_without_pipeline_or_fake_reports_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("REEL_PREDICTOR_FAKE", raising=False)
    import app.jobs as jobs_mod

    def no_pipeline(self):
        raise RuntimeError("Le pipeline d'analyse n'est pas disponible (test).")

    monkeypatch.setattr(jobs_mod.JobManager, "_resolve", no_pipeline)
    c, _ = make_client(tmp_path, analyze_fn=None)
    st = wait_done(c, upload(c, Path(__file__)))
    assert st["status"] == "error" and st["error"].startswith("Le pipeline")


# ---------------------------------------------------------------------------- redémarrage
def test_restart_keeps_results(tmp_path):
    results = tmp_path / "results"
    done_id = "20260101-120000-aaaaaaaa"
    fake.make_fake_result(results / done_id, duration_s=4, job_id=done_id)
    broken_id = "20260101-120500-bbbbbbbb"
    (results / broken_id).mkdir(parents=True)
    (results / broken_id / "video.mp4").write_bytes(b"x")
    c, _ = make_client(tmp_path)
    reels = c.get("/api/reels").json()["reels"]
    assert [r["job_id"] for r in reels] == [done_id]
    assert c.get(f"/api/jobs/{done_id}").json()["status"] == "done"
    st = c.get(f"/api/jobs/{broken_id}").json()
    assert st["status"] == "error" and "interrompue" in st["error"]
    # Un second démarrage ne duplique rien
    c2, _ = make_client(tmp_path)
    assert len(c2.get("/api/reels").json()["reels"]) == 1


# ---------------------------------------------------------------------------- insights & calibration
def _seed_results(tmp_path, n: int) -> list[str]:
    ids = []
    for i in range(n):
        jid = f"20260102-12000{i}-{i:08x}"
        fake.make_fake_result(tmp_path / "results" / jid, duration_s=4 + i, job_id=jid, seed=i)
        ids.append(jid)
    return ids


def test_insights_validation_comparison_and_calibration(tmp_path):
    ids = _seed_results(tmp_path, 3)
    c, _ = make_client(tmp_path)
    j0 = ids[0]
    base = {"job_id": j0, "window": "48h", "views": 1200, "reach": 900, "avg_watch_s": 2.0, "skip_rate": 40}
    r = c.post("/api/insights", json=base)
    assert r.status_code == 400 and "Validation manuelle" in r.json()["detail"]
    assert c.post("/api/insights", json={**base, "validated": True, "views": -3}).status_code == 400
    assert c.post("/api/insights", json={**base, "validated": True, "skip_rate": 140}).status_code == 400
    assert c.post("/api/insights", json={**base, "validated": True, "job_id": "inconnu"}).status_code == 404
    assert c.post("/api/insights", json={**base, "validated": True, "window": "30d"}).status_code == 422
    r = c.post("/api/insights", json={**base, "validated": True, "raw_ocr": {"raw_text": "Vues 1 200"}})
    assert r.status_code == 200, r.text
    assert r.json()["insight"]["views"] == 1200

    cmp = c.get(f"/api/reels/{j0}/comparison").json()
    assert cmp["best_window"] == "48h"
    assert cmp["actual_avg_watch_ratio"] == pytest.approx(2.0 / 4.0)
    pred = cmp["predicted"]["avg_watch_ratio"]
    assert cmp["ratio_pred_vs_real"] == pytest.approx(pred / 0.5)
    assert cmp["actual"]["7d"] is None and cmp["actual"]["48h"]["avg_watch_ratio"] == pytest.approx(0.5)
    assert len(cmp["predicted"]["curve"]) >= 4
    assert [p["job_id"] for p in cmp["scatter"]] == [j0]
    assert cmp["calibration"]["can_calibrate"] is False

    calls.clear()
    r = c.post("/api/calibrate")
    assert r.status_code == 400 and "3" in r.json()["detail"] and not calls

    for i, jid in enumerate(ids[1:], start=1):
        body = {"job_id": jid, "window": "7d", "validated": True, "views": 5000 * i, "avg_watch_s": 1.5 + i}
        assert c.post("/api/insights", json=body).status_code == 200
    reels = c.get("/api/reels").json()
    assert reels["calibration"]["n_eligible"] == 3 and reels["calibration"]["can_calibrate"] is True
    r = c.post("/api/calibrate")
    assert r.status_code == 200, r.text
    assert r.json()["result"]["n"] == 3
    pairs = calls[-1]
    assert len(pairs) == 3 and all(isinstance(p, Path) and 0 <= ratio <= 1 for p, ratio in pairs)
    assert c.get(f"/api/reels/{ids[2]}/comparison").json()["best_window"] == "7d"


def test_calibrate_missing_module_returns_503(tmp_path, monkeypatch):
    ids = _seed_results(tmp_path, 3)
    c, _ = make_client(tmp_path, calibrate_fn=None)
    for jid in ids:
        assert c.post("/api/insights", json={"job_id": jid, "window": "48h", "validated": True,
                                             "avg_watch_s": 2}).status_code == 200
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "reel_predictor.calibrate":
            raise ImportError("absent (test)")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    r = c.post("/api/calibrate")
    assert r.status_code == 503


def test_ocr_endpoint_merges_and_limits(tmp_path):
    seen = []

    def reader(data: bytes):
        seen.append(len(data))
        i = len(seen)
        return {"fields": {"views": {"value": 100 * i, "confidence": 0.5 + 0.1 * i, "level": "moyenne"}},
                "raw_text": f"capture {i}", "boxes": [], "window_guess": None}

    c, _ = make_client(tmp_path, ocr_fn=reader)
    files = [("file", (f"c{i}.png", b"img%d" % i, "image/png")) for i in range(2)]
    r = c.post("/api/insights/ocr", files=files, data={"window": "7d"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["fields"]["views"]["value"] == 200 and body["window"] == "7d" and body["validated"] is False
    assert "capture 1" in body["raw_text"] and "capture 2" in body["raw_text"]
    too_many = [("file", (f"c{i}.png", b"x", "image/png")) for i in range(4)]
    assert c.post("/api/insights/ocr", files=too_many).status_code == 400


def test_ocr_endpoint_real_engine(tmp_path):
    pytest.importorskip("rapidocr_onnxruntime")
    fx = _load_fixture("make_fake_insights")
    if fx.find_font(fx.FONT_CANDIDATES) is None:
        pytest.skip("aucune police système")
    png = fx.make_insights_png(tmp_path / "ins.png")
    c, _ = make_client(tmp_path)
    r = c.post("/api/insights/ocr", files=[("file", ("ins.png", png.read_bytes(), "image/png"))])
    assert r.status_code == 200, r.text
    fields = r.json()["fields"]
    assert fields["views"]["value"] == 12300 and fields["avg_watch_s"]["value"] == 4
    assert re.search(r"Vues", r.json()["raw_text"])
