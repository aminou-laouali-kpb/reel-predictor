"""Étapes du pipeline, partagées entre le pipeline (progression) et l'app (libellés)."""

# (identifiant, poids relatif dans la progression globale, libellé FR affiché dans l'UI)
STAGES: list[tuple[str, float, str]] = [
    ("probe", 0.02, "Lecture de la vidéo"),
    ("scenes", 0.05, "Détection des plans"),
    ("speech", 0.15, "Transcription"),
    ("faces", 0.08, "Visages"),
    ("ocr", 0.10, "Texte à l'écran"),
    ("audio", 0.02, "Audio"),
    ("brain", 0.50, "Modèle cérébral TRIBE v2"),
    ("indices", 0.03, "Indices"),
    ("rules", 0.02, "Recommandations"),
    ("write", 0.03, "Écriture des résultats"),
]

STAGE_IDS = [s[0] for s in STAGES]
STAGE_LABELS = {s[0]: s[2] for s in STAGES}


def global_progress(stage: str, stage_pct: float) -> float:
    """Convertit (étape, avancement 0..1 dans l'étape) en avancement global 0..1."""
    total = sum(w for _, w, _ in STAGES)
    done = 0.0
    for sid, w, _ in STAGES:
        if sid == stage:
            return min(1.0, (done + w * max(0.0, min(1.0, stage_pct))) / total)
        done += w
    raise KeyError(stage)
