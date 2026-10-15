"""Client du worker TRIBE testé avec un faux worker (aucun appel au vrai modèle)."""
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest

from reel_predictor import paths
from reel_predictor.brain.provider import BrainRequest, BrainUnavailable
from reel_predictor.brain.tribe import TribeProvider, worker_installed
from reel_predictor.config import load_config

FAKE = textwrap.dedent('''
    import argparse, json, os, sys, time
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument("--video"); ap.add_argument("--words"); ap.add_argument("--out")
    ap.add_argument("--model"); ap.add_argument("--cache-dir"); ap.add_argument("--device")
    ap.add_argument("--no-text", action="store_true")
    a = ap.parse_args()
    mode = os.environ.get("FAKE_TRIBE_MODE", "ok")
    if mode == "hf" and not a.no_text:
        print("ERROR_HF_ACCESS meta-llama/Llama-3.2-1B", file=sys.stderr); sys.exit(3)
    if mode == "hf_all":
        print("ERROR_HF_ACCESS facebook/tribev2-mini", file=sys.stderr); sys.exit(3)
    if mode == "oom":
        print("MemoryError", file=sys.stderr); sys.exit(4)
    if mode == "bad_input":
        sys.exit(2)
    if mode == "crash":
        print("RuntimeError: boum", file=sys.stderr); sys.exit(1)
    if mode == "sleep":
        print("PROGRESS 0.1 Chargement", flush=True); time.sleep(30)
    words = json.load(open(a.words))
    n = int(words["duration_s"])
    for i in range(5):
        print(f"PROGRESS {i / 4:.2f} Étape {i}", flush=True)
    print("log libre")
    secs = np.array([s for s in range(n) if s != 2], dtype=np.int32)
    preds = np.random.default_rng(0).normal(size=(len(secs), 20484)).astype(np.float16)
    np.savez(a.out, preds=preds, seconds=secs)
    json.dump({"model": a.model, "device": "cpu", "n_rows": len(secs), "runtime_s": 0.1, "stage_times": {},
               "peak_rss_mb": 1, "text_feature": "disabled" if a.no_text else "llama",
               "versions": {}, "warnings": ["avertissement de test"]}, open(a.out + ".meta.json", "w"))
''')


@pytest.fixture
def fake_worker(tmp_path, monkeypatch):
    script = tmp_path / "run_tribe.py"
    script.write_text(FAKE)
    monkeypatch.setattr(paths, "TRIBE_WORKER_PYTHON", Path(sys.executable))
    monkeypatch.setattr(paths, "TRIBE_WORKER_SCRIPT", script)
    return script


def _req(tmp_path, progress=None):
    words = tmp_path / "words.json"
    words.write_text(json.dumps({"language": "fr", "duration_s": 6.0, "words": []}))
    return BrainRequest(video_path=tmp_path / "video.mp4", words_path=words, duration_s=6.0, n_seconds=6,
                        work_dir=tmp_path, progress=progress)


def _cfg(**kw):
    import dataclasses

    return dataclasses.replace(load_config(calibration_file=None).brain, **kw)


def test_ok_with_progress(fake_worker, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_TRIBE_MODE", "ok")
    seen = []
    pred = TribeProvider(_cfg()).predict(_req(tmp_path, lambda p, m: seen.append((p, m))))
    assert pred.available and pred.provider == "tribe" and not pred.degraded
    assert pred.preds.shape == (5, 20484) and pred.preds.dtype == np.float32
    assert 2 not in pred.seconds.tolist()
    assert [p for p, _ in seen] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert seen[1][1] == "Étape 1"
    assert pred.meta["text_feature"] == "llama"
    assert "avertissement de test" in pred.warnings


def test_hf_llama_retry_without_text(fake_worker, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_TRIBE_MODE", "hf")
    pred = TribeProvider(_cfg(retry_without_text=True)).predict(_req(tmp_path))
    assert pred.available and pred.meta["text_feature"] == "disabled"
    assert "meta-llama/Llama-3.2-1B" in pred.reason
    with pytest.raises(BrainUnavailable) as e:
        TribeProvider(_cfg(retry_without_text=False)).predict(_req(tmp_path))
    assert e.value.reason == "Accès Hugging Face à meta-llama/Llama-3.2-1B manquant"


@pytest.mark.parametrize("mode,expected", [
    ("hf_all", "Accès Hugging Face à facebook/tribev2-mini manquant"),
    ("oom", "Mémoire insuffisante"),
    ("bad_input", "Entrée refusée"),
    ("crash", "Erreur du worker TRIBE (code 1)"),
])
def test_exit_codes(fake_worker, tmp_path, monkeypatch, mode, expected):
    monkeypatch.setenv("FAKE_TRIBE_MODE", mode)
    with pytest.raises(BrainUnavailable) as e:
        TribeProvider(_cfg()).predict(_req(tmp_path))
    assert expected in e.value.reason


def test_timeout(fake_worker, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_TRIBE_MODE", "sleep")
    with pytest.raises(BrainUnavailable) as e:
        TribeProvider(_cfg(timeout_s=2)).predict(_req(tmp_path))
    assert e.value.reason.startswith("Délai dépassé")


def test_worker_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "TRIBE_WORKER_SCRIPT", tmp_path / "absent.py")
    assert not worker_installed()
    with pytest.raises(BrainUnavailable) as e:
        TribeProvider(_cfg()).predict(_req(tmp_path))
    assert e.value.reason == "Worker TRIBE non installé"


@pytest.mark.tribe
def test_real_tribe_worker(tmp_path, monkeypatch):
    """Vrai worker TRIBE (long, réseau au 1er lancement) : `uv run pytest -m tribe`."""
    monkeypatch.setattr(paths, "REFERENCE_FILE", tmp_path / "reference.json")   # ne pas polluer la référence
    from scripts.make_samples import OUT, make

    from reel_predictor.pipeline import analyze

    if not worker_installed():
        pytest.skip("worker TRIBE non installé")
    video = OUT if OUT.exists() else make()
    cfg = load_config(calibration_file=None).with_brain_provider("tribe")
    out = tmp_path / "job_tribe"
    res = analyze(video, out, cfg)
    b = res.summary["brain"]
    assert b["provider"] in ("tribe", "heuristic")
    if b["available"]:
        assert (out / "brain.u8").stat().st_size == len(res.timeline) * 20484
    else:
        assert b["degraded"] and b["reason"]
