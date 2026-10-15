import time

from reel_predictor import paths
from reel_predictor.config import load_config
from reel_predictor.health import status


def test_status_fast_and_complete():
    t0 = time.time()
    s = status(load_config(calibration_file=None))
    assert time.time() - t0 < 2.0
    for k in ("brain_expected", "tribe_model", "worker_installed", "hf_token", "llama_access", "reason", "versions"):
        assert k in s
    assert s["brain_expected"] in ("tribe", "heuristic")
    assert isinstance(s["worker_installed"], bool) and isinstance(s["hf_token"], bool)
    assert s["llama_access"] in (True, False, None)
    assert "python" in s["versions"]


def test_status_without_worker(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "TRIBE_WORKER_SCRIPT", tmp_path / "absent.py")
    cfg = load_config(calibration_file=None).with_brain_provider("tribe")
    s = status(cfg)
    assert s["brain_expected"] == "heuristic" and s["worker_installed"] is False
    assert s["reason"] == "Worker TRIBE non installé"


def test_status_never_raises():
    class Broken:
        @property
        def brain(self):
            raise RuntimeError("boum")

    s = status(Broken())
    assert s["brain_expected"] == "heuristic" and s["reason"]
