"""Réduction des 20 484 sommets fsaverage5 en ROI (atlas Destrieux, labels résolus par nom)."""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

from .. import paths

_LOCK = threading.Lock()
_ATLAS: "Atlas | None" = None


@dataclass
class Atlas:
    labels: list[str]          # nom de chaque valeur de carte (index = valeur)
    map_left: np.ndarray       # (10242,) int
    map_right: np.ndarray      # (10242,) int

    @property
    def vertex_labels(self) -> np.ndarray:
        """(20484,) valeur de label par sommet, gauche puis droite."""
        return np.concatenate([self.map_left, self.map_right]).astype(np.int64)

    def label_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.labels)}


def _decode(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def load_atlas() -> Atlas:
    """Charge l'atlas Destrieux (téléchargé une fois dans data/assets/nilearn)."""
    global _ATLAS
    with _LOCK:
        if _ATLAS is not None:
            return _ATLAS
        from nilearn.datasets import fetch_atlas_surf_destrieux

        paths.NILEARN_DIR.mkdir(parents=True, exist_ok=True)
        b = fetch_atlas_surf_destrieux(data_dir=str(paths.NILEARN_DIR), verbose=0)
        lut = b.get("lut") if hasattr(b, "get") else None
        if lut is not None and hasattr(lut, "columns") and {"index", "name"} <= set(lut.columns):
            n = int(lut["index"].max()) + 1
            labels = [""] * n
            for idx, name in zip(lut["index"], lut["name"]):
                labels[int(idx)] = _decode(name)
        else:
            labels = [_decode(x) for x in b["labels"]]
        ml = np.asarray(b["map_left"]).astype(np.int64)
        mr = np.asarray(b["map_right"]).astype(np.int64)
        if ml.size + mr.size != paths.N_VERTICES:
            raise ValueError(f"Atlas inattendu : {ml.size + mr.size} sommets (attendu {paths.N_VERTICES})")
        _ATLAS = Atlas(labels=labels, map_left=ml, map_right=mr)
        return _ATLAS


def resolve_labels(roi_mapping: dict, atlas: Atlas | None = None) -> dict[str, list[int]]:
    """{roi: [index de label]} ; lève ValueError si un label n'existe pas dans l'atlas."""
    atlas = atlas or load_atlas()
    idx = atlas.label_index()
    out: dict[str, list[int]] = {}
    missing: list[str] = []
    for roi, labels in roi_mapping["rois"].items():
        ids = []
        for lab in labels:
            if lab in idx:
                ids.append(idx[lab])
            else:
                missing.append(f"{roi}:{lab}")
        out[roi] = ids
    if missing:
        raise ValueError("Labels Destrieux introuvables : " + ", ".join(missing))
    return out


def roi_masks(roi_mapping: dict, atlas: Atlas | None = None) -> dict[str, np.ndarray]:
    """{roi: masque bool (20484,)} selon `hemispheres` (both | left | right)."""
    atlas = atlas or load_atlas()
    vl = atlas.vertex_labels
    hemi = roi_mapping.get("hemispheres", "both")
    hemi_mask = np.ones(paths.N_VERTICES, bool)
    if hemi == "left":
        hemi_mask[paths.HEMI_SPLIT:] = False
    elif hemi == "right":
        hemi_mask[: paths.HEMI_SPLIT] = False
    masks = {}
    for roi, ids in resolve_labels(roi_mapping, atlas).items():
        m = np.isin(vl, ids) & hemi_mask
        if not m.any():
            raise ValueError(f"ROI {roi!r} : aucun sommet")
        masks[roi] = m
    return masks


def roi_matrix(roi_mapping: dict, atlas: Atlas | None = None) -> tuple[list[str], np.ndarray]:
    """(noms, W) avec W (R, 20484) : moyenne des sommets de chaque ROI."""
    masks = roi_masks(roi_mapping, atlas)
    names = list(masks)
    W = np.zeros((len(names), paths.N_VERTICES), np.float32)
    for i, n in enumerate(names):
        W[i, masks[n]] = 1.0 / masks[n].sum()
    return names, W


def reduce(preds: np.ndarray, W: np.ndarray) -> np.ndarray:
    """(N, 20484) -> (N, R) en ignorant les NaN."""
    p = np.nan_to_num(np.asarray(preds, np.float32), nan=0.0)
    return p @ W.T
