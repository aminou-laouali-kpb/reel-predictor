"""Prépare les ressources de l'app (idempotent) :

1. uPlot (graphiques) -> app/static/vendor/ : tarball npm vérifié par son empreinte d'intégrité
   (sha512 publiée par le registre npm), seuls 2 fichiers en sont lus (aucune extraction globale).
2. app/static/brain_mesh.json : maillage 2D de la carte cérébrale, à partir de
   nilearn fsaverage5 (surfaces gonflées), 4 vues, faces visibles triées par profondeur.
3. app/static/brain3d.json : maillage 3D (surface piale fsaverage5, sillons, région de l'atlas par sommet)
   pour le cerveau interactif WebGL.
4. reel_predictor.assets.ensure_all(load_config()) si le pipeline est installé (modèles légers).

Usage : uv run python scripts/fetch_assets.py [--force]
Code de sortie 1 seulement si uPlot ou le maillage manquent à la fin.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STATIC = ROOT / "app" / "static"
VENDOR = STATIC / "vendor"
MESH_FILE = STATIC / "brain_mesh.json"
MESH3D_FILE = STATIC / "brain3d.json"
MESH3D_VERSION = 1

UPLOT_VERSION = "1.6.32"
UPLOT_FILES = {"package/dist/uPlot.iife.min.js": "uPlot.iife.min.js",
               "package/dist/uPlot.min.css": "uPlot.min.css"}
MESH_VERSION = 1
GRID = 1000  # coordonnées quantifiées sur [0, GRID]


# ---------------------------------------------------------------------------- uPlot
def _get(url: str, timeout: float = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "reel-predictor-setup"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch_uplot(force: bool = False) -> bool:
    targets = [VENDOR / name for name in UPLOT_FILES.values()]
    if not force and all(t.is_file() and t.stat().st_size > 1000 for t in targets):
        print(f"✓ uPlot déjà présent ({VENDOR.relative_to(ROOT)})")
        return True
    VENDOR.mkdir(parents=True, exist_ok=True)
    try:
        meta = json.loads(_get(f"https://registry.npmjs.org/uplot/{UPLOT_VERSION}"))
        tarball_url = meta["dist"]["tarball"]
        integrity = meta["dist"]["integrity"]
        algo, b64 = integrity.split("-", 1)
        if algo != "sha512":
            raise ValueError(f"algorithme d'intégrité inattendu : {algo}")
        data = _get(tarball_url, timeout=60)
        digest = base64.b64encode(hashlib.sha512(data).digest()).decode()
        if digest != b64:
            raise ValueError("empreinte sha512 du paquet uPlot invalide (téléchargement corrompu ?)")
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            for member_name, out_name in UPLOT_FILES.items():
                member = tar.getmember(member_name)
                if not member.isfile():
                    raise ValueError(f"{member_name} n'est pas un fichier")
                fh = tar.extractfile(member)
                if fh is None:
                    raise ValueError(f"{member_name} illisible")
                (VENDOR / out_name).write_bytes(fh.read())
        (VENDOR / "UPLOT_VERSION.txt").write_text(f"uPlot {UPLOT_VERSION} (MIT) — {tarball_url}\n{integrity}\n",
                                                  encoding="utf-8")
        print(f"✓ uPlot {UPLOT_VERSION} téléchargé et vérifié (sha512) -> {VENDOR.relative_to(ROOT)}")
        return True
    except Exception as exc:
        print(f"✗ uPlot : téléchargement impossible ({exc}). Vérifiez la connexion puis relancez `make setup`.")
        return False


# ---------------------------------------------------------------------------- maillage cérébral
def _load_hemi(fs, hemi: str):
    import numpy as np
    from nilearn import surface

    mesh = surface.load_surf_mesh(fs[f"infl_{hemi}"])
    coords = np.asarray(mesh.coordinates if hasattr(mesh, "coordinates") else mesh[0], dtype=np.float64)
    faces = np.asarray(mesh.faces if hasattr(mesh, "faces") else mesh[1], dtype=np.int64)
    try:
        sulc = np.asarray(surface.load_surf_data(fs[f"sulc_{hemi}"]), dtype=np.float64).ravel()
    except Exception:
        sulc = np.zeros(len(coords))
    v0, v1, v2 = coords[faces[:, 0]], coords[faces[:, 1]], coords[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12
    centers = (v0 + v1 + v2) / 3
    # Orientation : normales vers l'extérieur
    if np.sum(np.einsum("ij,ij->i", normals, centers - coords.mean(axis=0))) < 0:
        normals = -normals
    return coords, faces, sulc, normals, centers


def build_brain_mesh() -> dict:
    import numpy as np
    from nilearn import datasets

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    hemis = {h: _load_hemi(fs, h) for h in ("left", "right")}
    n_left = len(hemis["left"][0])
    up = np.array([0.0, 0.0, 1.0])
    # forward = direction du regard (caméra -> cerveau). Latérale gauche : on regarde depuis -x.
    view_defs = [
        ("left_lateral", "Gauche · latérale", "left", (1.0, 0.0, 0.0)),
        ("right_lateral", "Droite · latérale", "right", (-1.0, 0.0, 0.0)),
        ("left_medial", "Gauche · médiale", "left", (-1.0, 0.0, 0.0)),
        ("right_medial", "Droite · médiale", "right", (1.0, 0.0, 0.0)),
    ]
    raw = []
    for vid, label, hemi, fwd in view_defs:
        coords, faces, sulc, normals, centers = hemis[hemi]
        f = np.array(fwd)
        right = np.cross(f, up)
        sx = coords @ right
        sy = -(coords @ up)  # y écran vers le bas
        depth = centers @ f
        facing = normals @ (-f)
        visible = np.nonzero(facing > 0.0)[0]
        visible = visible[np.argsort(-depth[visible], kind="stable")]  # du plus loin au plus proche
        light = -f * 0.8 + up * 0.45 - right * 0.25
        light /= np.linalg.norm(light)
        lam = np.clip(normals[visible] @ light, 0, 1)
        sulc_face = sulc[faces[visible]].mean(axis=1)
        sulc_factor = 1.0 - 0.18 * np.clip(sulc_face, 0, 1.5) / 1.5 + 0.04 * np.clip(-sulc_face, 0, 1)
        shade = np.clip((0.5 + 0.5 * lam) * sulc_factor, 0, 1)
        raw.append((vid, label, hemi, sx, sy, faces[visible], shade))
    extent = max(max(sx.max() - sx.min(), sy.max() - sy.min()) for _, _, _, sx, sy, _, _ in raw)
    scale = GRID / extent
    views = []
    for vid, label, hemi, sx, sy, vfaces, shade in raw:
        x = np.round((sx - sx.min()) * scale).astype(int)
        y = np.round((sy - sy.min()) * scale).astype(int)
        xy = np.empty(2 * len(x), dtype=int)
        xy[0::2], xy[1::2] = x, y
        views.append({
            "id": vid, "label": label, "hemi": hemi, "offset": 0 if hemi == "left" else n_left,
            "w": int(x.max()), "h": int(y.max()),
            "xy": xy.tolist(),
            "faces": vfaces.astype(int).ravel().tolist(),
            "shade": np.round(shade * 255).astype(int).tolist(),
        })
    return {
        "version": MESH_VERSION,
        "space": "fsaverage5",
        "surface": "inflated",
        "n_vertices": 2 * n_left,
        "hemi_split": n_left,
        "grid": GRID,
        "layout": [["left_lateral", "right_lateral"], ["left_medial", "right_medial"]],
        "views": views,
        "source": "nilearn.datasets.fetch_surf_fsaverage('fsaverage5') — infl_left/right, sulc_left/right",
    }


def fetch_brain_mesh(force: bool = False) -> bool:
    if not force and MESH_FILE.is_file():
        try:
            if json.loads(MESH_FILE.read_text(encoding="utf-8")).get("version") == MESH_VERSION:
                print(f"✓ Maillage cérébral déjà présent ({MESH_FILE.relative_to(ROOT)})")
                return True
        except ValueError:
            pass
    try:
        mesh = build_brain_mesh()
    except Exception as exc:
        print(f"✗ Maillage cérébral : génération impossible ({exc}).")
        return False
    STATIC.mkdir(parents=True, exist_ok=True)
    tmp = MESH_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(mesh, separators=(",", ":")), encoding="utf-8")
    tmp.replace(MESH_FILE)
    n_faces = sum(len(v["faces"]) // 3 for v in mesh["views"])
    print(f"✓ Maillage cérébral écrit ({MESH_FILE.relative_to(ROOT)}, {MESH_FILE.stat().st_size / 1e6:.1f} Mo, "
          f"{n_faces} faces visibles sur 4 vues)")
    return True


# ---------------------------------------------------------------------------- cerveau 3D (WebGL)
ROI_LABELS_FR = {
    "ips": "Sillon intrapariétal", "fef": "Champs oculomoteurs frontaux", "early_visual": "Cortex visuel primaire",
    "mt": "Aire du mouvement (MT)", "precentral": "Cortex moteur", "premotor_ventral": "Prémoteur ventral",
    "sma": "Aire motrice supplémentaire", "supramarginal": "Gyrus supramarginal", "subcentral": "Région subcentrale",
    "insula_ant": "Insula antérieure", "dacc": "Cingulaire antérieur", "angular": "Gyrus angulaire",
    "psts": "Sillon temporal supérieur", "dmpfc": "Préfrontal dorsomédian", "frontopolar": "Pôle frontal",
    "rectus": "Gyrus droit", "temporal_pole": "Pôle temporal", "ofc": "Cortex orbitofrontal",
    "subcallosal": "Région subcalleuse", "parahippocampal": "Gyrus parahippocampique",
    "retrosplenial": "Cortex rétrosplénial", "pcc": "Cingulaire postérieur", "precuneus": "Précunéus",
    "ifg": "Gyrus frontal inférieur", "fusiform": "Gyrus fusiforme", "mfg": "Gyrus frontal moyen",
    "auditory": "Cortex auditif",
}


def _b64(a) -> str:
    return base64.b64encode(a.tobytes()).decode("ascii")


def build_brain3d() -> dict:
    """Surface piale des 2 hémisphères (gauche puis droite, même ordre que la sortie TRIBE) + ROI par sommet."""
    import numpy as np
    import yaml
    from nilearn import datasets, surface

    fs = datasets.fetch_surf_fsaverage("fsaverage5")
    coords, faces, sulcs = [], [], []
    off = 0
    for hemi in ("left", "right"):
        m = surface.load_surf_mesh(fs[f"pial_{hemi}"])
        c = np.asarray(m.coordinates if hasattr(m, "coordinates") else m[0], np.float64)
        f = np.asarray(m.faces if hasattr(m, "faces") else m[1], np.int64)
        try:
            s = np.asarray(surface.load_surf_data(fs[f"sulc_{hemi}"]), np.float64).ravel()
        except Exception:
            s = np.zeros(len(c))
        c[:, 0] += -2.0 if hemi == "left" else 2.0  # léger écart interhémisphérique
        coords.append(c)
        faces.append(f + off)
        sulcs.append(s)
        off += len(c)
    c = np.concatenate(coords)
    c -= (c.max(0) + c.min(0)) / 2
    c /= np.abs(c).max()
    f = np.concatenate(faces)
    s = np.concatenate(sulcs)
    s = np.clip((s - s.min()) / (np.ptp(s) + 1e-9), 0, 1)

    from reel_predictor.brain.atlas import roi_masks

    rm = yaml.safe_load((ROOT / "config" / "roi_mapping.yaml").read_text(encoding="utf-8"))
    masks = roi_masks({**rm, "hemispheres": "both"})
    names = list(masks)
    roi = np.zeros(len(c), np.uint8)  # 0 = aucune ROI
    for i, n in enumerate(names):
        roi[masks[n] & (roi == 0)] = i + 1
    rois = []
    for n in names:
        idx = {k: float(w) for k, ws in (rm.get("indices") or {}).items() for r, w in ws.items() if r == n}
        rois.append({"id": n, "label": ROI_LABELS_FR.get(n, n), "indices": idx})
    return {
        "version": MESH3D_VERSION, "space": "fsaverage5", "surface": "pial",
        "n_vertices": int(len(c)), "hemi_split": int(len(coords[0])), "n_faces": int(len(f)),
        "positions": _b64(c.astype("<f4")), "faces": _b64(f.astype("<u2")),
        "sulc": _b64(np.round(s * 255).astype(np.uint8)), "roi": _b64(roi),
        "rois": rois,
        "source": "nilearn fsaverage5 pial_left/right + sulc ; atlas Destrieux via config/roi_mapping.yaml",
    }


def fetch_brain3d(force: bool = False) -> bool:
    if not force and MESH3D_FILE.is_file():
        try:
            if json.loads(MESH3D_FILE.read_text(encoding="utf-8")).get("version") == MESH3D_VERSION:
                print(f"✓ Cerveau 3D déjà présent ({MESH3D_FILE.relative_to(ROOT)})")
                return True
        except ValueError:
            pass
    try:
        mesh = build_brain3d()
    except Exception as exc:
        print(f"✗ Cerveau 3D : génération impossible ({exc}). L'app utilisera la carte à plat.")
        return False
    tmp = MESH3D_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(mesh, separators=(",", ":")), encoding="utf-8")
    tmp.replace(MESH3D_FILE)
    print(f"✓ Cerveau 3D écrit ({MESH3D_FILE.relative_to(ROOT)}, {MESH3D_FILE.stat().st_size / 1e6:.1f} Mo, "
          f"{mesh['n_faces']} faces)")
    return True


# ---------------------------------------------------------------------------- modèles du pipeline
def fetch_pipeline_assets() -> None:
    try:
        from reel_predictor.assets import ensure_all
        from reel_predictor.config import load_config
    except ImportError as exc:
        print(f"• Modèles du pipeline : module pas encore disponible ({exc}). Étape ignorée — relancez "
              "`make setup` quand le pipeline sera installé.")
        return
    try:
        print("• Modèles légers du pipeline (visage, atlas, transcription) : vérification / téléchargement…",
              flush=True)
        try:
            out = ensure_all(load_config(), progress=lambda _k, msg: print(f"  {msg}", flush=True))
        except TypeError:  # ancienne signature sans `progress`
            out = ensure_all(load_config())
        if isinstance(out, dict):
            for key, val in out.items():
                mark = "✓" if str(val) == "ok" else "✗"
                print(f"  {mark} {key} : {val}")
            if all(str(v) == "ok" for v in out.values()):
                print("✓ Modèles légers du pipeline prêts")
            else:
                print("• Certains modèles manquent (non bloquant) : relancez `make setup` avec une connexion.")
        else:
            print("✓ Modèles légers du pipeline prêts")
    except Exception as exc:
        print(f"• Modèles du pipeline : échec non bloquant ({exc}). Ils seront retentés au prochain `make setup`.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--force", action="store_true", help="re-télécharger / régénérer même si présent")
    p.add_argument("--skip-pipeline", action="store_true", help="ne pas appeler reel_predictor.assets.ensure_all")
    a = p.parse_args(argv)
    ok_uplot = fetch_uplot(a.force)
    ok_mesh = fetch_brain_mesh(a.force)
    fetch_brain3d(a.force)  # non bloquant : repli sur la carte à plat
    if not a.skip_pipeline:
        fetch_pipeline_assets()
    return 0 if (ok_uplot and ok_mesh) else 1


if __name__ == "__main__":
    sys.exit(main())
