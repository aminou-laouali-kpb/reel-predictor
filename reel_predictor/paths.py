"""Chemins du projet. Tout est relatif à la racine du dépôt : aucune donnée hors de `data/`
(sauf les caches Hugging Face, partagés avec les autres outils de la machine)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
DATA_DIR = ROOT / "data"
UPLOADS_DIR = DATA_DIR / "uploads"
RESULTS_DIR = DATA_DIR / "results"
CACHE_DIR = DATA_DIR / "cache"            # caches de features (exca/TRIBE), frames temporaires
SAMPLES_DIR = DATA_DIR / "samples"
ASSETS_DIR = DATA_DIR / "assets"          # modèles légers téléchargés au setup
NILEARN_DIR = ASSETS_DIR / "nilearn"      # atlas Destrieux (+ éventuels fichiers nilearn)
FACE_MODEL = ASSETS_DIR / "blaze_face_short_range.tflite"
REFERENCE_FILE = DATA_DIR / "reference.json"      # référence de corpus par ROI (médiane/MAD)
CALIBRATION_FILE = DATA_DIR / "calibration.json"  # paramètres de rétention recalibrés
DB_FILE = DATA_DIR / "reel_predictor.sqlite"

TRIBE_WORKER_DIR = ROOT / "tribe_worker"
TRIBE_WORKER_PYTHON = TRIBE_WORKER_DIR / ".venv" / "bin" / "python"
TRIBE_WORKER_SCRIPT = TRIBE_WORKER_DIR / "run_tribe.py"

N_VERTICES = 20484          # fsaverage5, gauche (0..10241) puis droite (10242..20483)
HEMI_SPLIT = 10242


def ensure_dirs() -> None:
    for d in (DATA_DIR, UPLOADS_DIR, RESULTS_DIR, CACHE_DIR, SAMPLES_DIR, ASSETS_DIR, NILEARN_DIR):
        d.mkdir(parents=True, exist_ok=True)
