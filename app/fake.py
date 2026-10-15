"""Mode factice (`REEL_PREDICTOR_FAKE=1`) : simule le pipeline pour développer l'UI sans TRIBE.

Parcourt les étapes de `reel_predictor.stages` avec un délai (`REEL_PREDICTOR_FAKE_DELAY`, secondes
par étape, défaut 0.4), puis écrit un résultat conforme au contrat §3 via
`tests/fixtures/make_fake_result.py` (la vidéo déposée est conservée et sa durée respectée).
"""
from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from types import ModuleType
from typing import Callable

from reel_predictor.stages import STAGES, STAGE_LABELS, global_progress

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "make_fake_result.py"
_module: ModuleType | None = None


def fake_enabled() -> bool:
    return os.environ.get("REEL_PREDICTOR_FAKE", "").strip().lower() in ("1", "true", "yes", "oui", "on")


def _fixture_module() -> ModuleType:
    global _module
    if _module is None:
        spec = importlib.util.spec_from_file_location("reel_predictor_fake_result", FIXTURE)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Fixture introuvable : {FIXTURE}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _module = mod
    return _module


def fake_analyze(video_path: Path, out_dir: Path, config=None,
                 progress: Callable[[str, float, str], None] | None = None, brain=None):
    """Même signature que `reel_predictor.pipeline.analyze` (contrat §2)."""
    delay = float(os.environ.get("REEL_PREDICTOR_FAKE_DELAY", "0.4"))
    no_brain = os.environ.get("REEL_PREDICTOR_FAKE_NO_BRAIN", "") == "1"
    for stage, _w, label in STAGES:
        steps = 3 if delay > 0 else 1
        for k in range(steps):
            if progress:
                progress(stage, global_progress(stage, k / steps), f"{label} (simulation)…")
            if delay > 0:
                time.sleep(delay / steps)
    if progress:
        progress("write", global_progress("write", 0.9), STAGE_LABELS["write"])
    mod = _fixture_module()
    out = mod.make_fake_result(Path(out_dir), brain=not no_brain, job_id=Path(out_dir).name)
    if progress:
        progress("write", 1.0, "Terminé")
    return out
