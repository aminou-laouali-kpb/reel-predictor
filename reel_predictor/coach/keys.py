"""Clé API Anthropic : variable `ANTHROPIC_API_KEY`, sinon trousseau macOS (service `reel-predictor-anthropic`).

La clé n'est JAMAIS journalisée, affichée ni renvoyée par l'API HTTP : seul `has_key()` (booléen) sort de ce module,
et `get_api_key()` n'est appelé qu'au moment de créer le client Anthropic.
Enregistrement : `make api-key` (saisie masquée, c'est l'utilisateur qui tape la clé).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time

KEYCHAIN_SERVICE = "reel-predictor-anthropic"
ENV_VAR = "ANTHROPIC_API_KEY"
# Désactive la lecture du trousseau (tests, ou machine partagée) : REEL_PREDICTOR_NO_KEYCHAIN=1
NO_KEYCHAIN_ENV = "REEL_PREDICTOR_NO_KEYCHAIN"
_CACHE_S = 30.0

_lock = threading.Lock()
_cache: dict[str, float | bool] = {"t": -1e9, "value": False}


def _keychain_enabled() -> bool:
    return sys.platform == "darwin" and os.environ.get(NO_KEYCHAIN_ENV, "") != "1" and shutil.which("security") is not None


def _from_keychain() -> str | None:
    if not _keychain_enabled():
        return None
    try:
        proc = subprocess.run(
            ["security", "find-generic-password", "-s", KEYCHAIN_SERVICE, "-w"],
            capture_output=True, text=True, timeout=5, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    key = proc.stdout.strip()
    return key or None


def get_api_key() -> str | None:
    """La clé (ou None). Ne pas journaliser le résultat."""
    key = os.environ.get(ENV_VAR, "").strip()
    if key:
        return key
    return _from_keychain()


def key_source() -> str | None:
    """« env », « keychain » ou None (sans révéler la clé)."""
    if os.environ.get(ENV_VAR, "").strip():
        return "env"
    return "keychain" if _from_keychain() else None


def has_key(*, refresh: bool = False) -> bool:
    """Clé disponible ? Résultat mis en cache 30 s (évite d'interroger le trousseau à chaque requête)."""
    with _lock:
        now = time.monotonic()
        if refresh or now - float(_cache["t"]) > _CACHE_S:
            _cache["value"] = get_api_key() is not None
            _cache["t"] = now
        return bool(_cache["value"])


def reset_cache() -> None:
    with _lock:
        _cache["t"] = -1e9
        _cache["value"] = False


NO_KEY_MESSAGE = ("Aucune clé API Anthropic configurée. Dans le Terminal, depuis le dossier du projet, lancez "
                  "« make api-key » (saisie masquée, la clé est rangée dans le trousseau macOS), puis réessayez. "
                  "Alternative : variable d'environnement ANTHROPIC_API_KEY avant « make run ».")
