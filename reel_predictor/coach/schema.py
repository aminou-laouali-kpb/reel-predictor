"""Schéma de la sortie structurée demandée à Claude (contrat partagé coach / simulation / UI).

Les durées ne sont PAS demandées au modèle : elles viennent de la voix de synthèse lors de la simulation.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

IndexName = Literal["attention", "action", "tension", "manipulation", "emotion", "memory"]
Role = Literal["hook", "fact", "body", "cta"]           # rôle narratif -> moments clés (config/coach.yaml)
Shot = Literal["face_cam", "close_up", "screen", "broll", "text_card"]
EditAction = Literal["cut", "trim", "zoom", "text", "broll", "sfx", "music", "reorder"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Segment(_Strict):
    role: Role = Field(description="hook = accroche (0-3 s), fact = chiffre/révélation clé, body = développement, cta = appel à l'action")
    voice_over: str = Field(description="Texte dit face caméra, en français oral, une ou deux phrases courtes")
    on_screen_text: str = Field(description="Texte affiché pendant ce segment (court, lisible) ; chaîne vide si aucun")
    shot: Shot = Field(description="Type de plan")
    cut_at_start: bool = Field(description="Changement de plan au début du segment")
    edit: str = Field(description="Consigne de montage pour ce segment (zoom, effet sonore, b-roll...) ; chaîne vide si aucune")


class Variant(_Strict):
    angle: str = Field(description="Angle de la variante en quelques mots")
    hook_summary: str = Field(description="Pourquoi l'accroche retient dans les 3 premières secondes")
    segments: list[Segment]
    rationale: str = Field(description="Quels indices cette version cherche à faire monter, et comment")


class MontageAction(_Strict):
    t: float = Field(description="Début (s) dans la vidéo ACTUELLE")
    t_end: float = Field(description="Fin (s) dans la vidéo ACTUELLE")
    action: EditAction
    instruction: str = Field(description="Consigne concrète pour le monteur, en français")
    on_screen_text: str = Field(description="Texte à afficher si action = text, sinon chaîne vide")
    expected_effect: str = Field(description="Effet attendu sur la rétention / les indices")
    target_indices: list[IndexName]


class CoachOutput(_Strict):
    diagnosis_summary: str = Field(description="Diagnostic en 3-5 phrases : ce qui fait décrocher et pourquoi")
    montage_plan: list[MontageAction] = Field(description="Modifications de montage de la vidéo actuelle, sans retourner, par ordre chronologique")
    variants: list[Variant] = Field(description="Variantes de script complètes, angles différents")
    notes: list[str] = Field(description="Limites et points d'attention (indices à faible confiance, etc.)")
