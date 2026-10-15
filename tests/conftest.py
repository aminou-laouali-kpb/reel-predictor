"""Configuration pytest légère (aucune fixture globale lourde : les tests de l'APP doivent rester rapides).

- marqueur `tribe` : tests qui appellent le vrai worker TRIBE ; ignorés sauf `pytest -m tribe`.
- marqueur `slow` : tests qui lancent le pipeline complet (transcription, OCR...).
"""
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "tribe: appelle le vrai worker TRIBE v2 (lancer avec -m tribe)")
    config.addinivalue_line("markers", "slow: lance le pipeline complet sur la vidéo synthétique")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    markexpr = getattr(config.option, "markexpr", "") or ""
    if "tribe" in markexpr:
        return
    skip = pytest.mark.skip(reason="test TRIBE réel : lancer avec `pytest -m tribe`")
    for item in items:
        if "tribe" in item.keywords:
            item.add_marker(skip)
