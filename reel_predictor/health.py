"""État de l'installation (rapide, < 2 s, sans réseau, ne lève jamais d'exception)."""
from __future__ import annotations

import os
import platform
from importlib import metadata
from pathlib import Path

from . import paths

_PACKAGES = ("numpy", "pandas", "faster-whisper", "ctranslate2", "mediapipe", "scenedetect",
             "opencv-contrib-python", "rapidocr-onnxruntime", "onnxruntime", "nilearn", "fastapi")
_WORKER_PACKAGES = ("torch", "transformers", "tribev2")


def _hf_home() -> Path:
    return Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")


def _hf_hub_cache() -> Path:
    env = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
    return Path(env) if env else _hf_home() / "hub"


def _hf_token() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    tok = Path(os.environ.get("HF_TOKEN_PATH") or _hf_home() / "token")
    try:
        return tok.exists() and tok.read_text().strip() != ""
    except OSError:
        return False


def _repo_cached(repo: str, required: str | None = None) -> bool:
    d = _hf_hub_cache() / ("models--" + repo.replace("/", "--")) / "snapshots"
    try:
        for snap in d.iterdir():
            if required is None and any(snap.iterdir()):
                return True
            if required is not None and (snap / required).exists():
                return True
    except OSError:
        return False
    return False


def _worker_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    try:
        for sp in (paths.TRIBE_WORKER_DIR / ".venv" / "lib").glob("python*/site-packages"):
            for dist in sp.glob("*.dist-info"):
                name, _, ver = dist.name[: -len(".dist-info")].partition("-")
                if name.lower().replace("_", "-") in _WORKER_PACKAGES:
                    out[name.lower()] = ver
    except OSError:
        pass
    return out


def status(config=None) -> dict:
    res: dict = {"brain_expected": "heuristic", "tribe_model": None, "worker_installed": False,
                 "hf_token": False, "llama_access": None, "reason": None, "versions": {}}
    try:
        if config is None:
            from .config import load_config

            config = load_config()
        b = config.brain
        res["provider"] = b.provider
        res["tribe_model"] = b.model
        worker = paths.TRIBE_WORKER_PYTHON.exists() and paths.TRIBE_WORKER_SCRIPT.exists()
        res["worker_installed"] = worker
        res["hf_token"] = _hf_token()
        llama = _repo_cached("meta-llama/Llama-3.2-1B", "config.json")
        # True : poids LLaMA déjà en cache ; False : aucun jeton (repo gated inaccessible) ; None : inconnu
        res["llama_access"] = True if llama else (False if not res["hf_token"] else None)
        res["tribe_cached"] = _repo_cached(b.model)
        if b.provider == "tribe":
            if worker:
                res["brain_expected"] = "tribe"
                if not b.no_text and not llama and not res["hf_token"]:
                    res["reason"] = ("Jeton Hugging Face absent : la composante texte (LLaMA) de TRIBE sera "
                                     "probablement inaccessible" + (" — relance automatique sans texte"
                                                                    if b.retry_without_text else ""))
            else:
                res["reason"] = "Worker TRIBE non installé"
        elif b.provider == "mock":
            res["brain_expected"] = "heuristic"
            res["reason"] = "Cerveau simulé (mock) : réservé aux tests"
        else:
            res["reason"] = "Modèle cérébral désactivé dans la configuration (brain.provider = heuristic)"
        res["assets"] = {
            "face_model": paths.FACE_MODEL.exists(),
            "atlas": (paths.NILEARN_DIR / "destrieux_surface" / "left.aparc.a2009s.annot").exists(),
            "whisper": _repo_cached(f"Systran/faster-whisper-{config.speech.model}", "model.bin"),
        }
    except Exception as e:  # noqa: BLE001
        res["reason"] = res.get("reason") or f"Configuration illisible : {type(e).__name__}: {e}"
    versions: dict[str, str] = {"python": platform.python_version()}
    for p in _PACKAGES:
        try:
            versions[p] = metadata.version(p)
        except Exception:  # noqa: BLE001
            pass
    try:
        from . import __version__

        versions["reel_predictor"] = __version__
    except Exception:  # noqa: BLE001
        pass
    for k, v in _worker_versions().items():
        versions[f"worker:{k}"] = v
    res["versions"] = versions
    return res
