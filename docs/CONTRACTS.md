# Contrats partagés — Reel Predictor

Ce document fige les interfaces entre les 3 chantiers (RECHERCHE / ALGORITHME / APP) pour qu'ils
avancent en parallèle. **Ne pas changer un contrat sans le signaler dans son rapport final.**

Racine : la racine du dépôt. Machine de développement : Apple M5, 16 Go, macOS 27,
Python 3.12 via `uv`, ffmpeg 8.1, GNU make 3.81. Langue de l'UI et des messages : **français**.

## 0. Propriété des fichiers (chacun n'écrit QUE dans ses fichiers)

| Propriétaire | Fichiers |
|---|---|
| déjà écrits (lecture seule) | `pyproject.toml`, `uv.lock`, `reel_predictor/{__init__,paths,stages}.py`, `docs/CONTRACTS.md` |
| RECHERCHE | `docs/RESEARCH.md`, `config/roi_mapping.yaml` (valeurs/labels, **schéma figé**), `tribe_worker/**` (dont `pyproject.toml`) |
| ALGORITHME | `reel_predictor/**` (sauf fichiers ci-dessus), `config/weights.yaml`, `config/rules.yaml`, `tests/conftest.py`, `tests/test_*.py` **sauf** `tests/test_app_*.py`, `scripts/make_samples.py` |
| APP | `app/**`, `scripts/fetch_assets.py`, `scripts/check_hf_access.py`, `Makefile`, `README.md`, `tests/test_app_*.py`, `tests/fixtures/**` |

Dépendances : deux venvs.
- App + pipeline : `.venv/` à la racine (déjà synchronisé). Lancer avec `uv run ...` depuis la racine.
  Ajout de dépendance : seulement si indispensable, via `uv add <pkg>` (le signaler dans le rapport).
  Installés : fastapi, uvicorn, python-multipart, faster-whisper, scenedetect, mediapipe 1.1,
  rapidocr-onnxruntime 1.4.4 (modèles embarqués), nilearn, numpy, pandas, pyyaml, pillow,
  huggingface_hub, pytest, httpx. **OpenCV = `opencv-contrib-python` 5.0 uniquement** (override uv).
  ⚠️ `cv2` et `av` (PyAV, via faster-whisper) embarquent chacun `libavdevice` → avertissements objc
  au chargement ; si crash, isoler l'étape concernée dans un sous-process.
- Worker TRIBE : `tribe_worker/.venv/` (déjà synchronisé : tribev2 @ af58661, torch 2.6.0 avec MPS,
  numpy 2.2.6, **transformers 5.17 non pinné par tribev2 — peut casser**).

Les données vivent dans `data/` (voir `reel_predictor/paths.py`). Aucun appel réseau à
l'exécution d'une analyse, sauf le premier téléchargement des modèles (HF / mediapipe / nilearn).

## 1. Worker TRIBE (RECHERCHE implémente, ALGORITHME appelle)

```
tribe_worker/.venv/bin/python tribe_worker/run_tribe.py \
    --video PATH.mp4 --words PATH.json --out PATH.npz \
    --model facebook/tribev2-mini --cache-dir data/cache/tribe \
    [--device auto|cpu|mps] [--no-text]
```
- `--words` (écrit par le pipeline à partir de faster-whisper) :
  `{"language": "fr", "duration_s": 31.2, "words": [{"w": "Bonjour", "start": 0.32, "end": 0.71}, ...]}`
  (`language` = code ISO 639-1 renvoyé par faster-whisper ; le worker convertit si TRIBE veut autre chose.)
- `--no-text` : exécuter sans le backbone texte (LLaMA, modèle gated) si TRIBE le tolère.
- stdout : lignes `PROGRESS <float 0..1> <message FR>` (les autres lignes = logs libres).
- Codes de sortie : `0` ok · `2` entrée invalide · `3` accès HF refusé/modèle gated (stderr contient
  une ligne `ERROR_HF_ACCESS <repo>`) · `4` mémoire insuffisante · `1` autre erreur.
- Sortie `PATH.npz` : `preds` float16 `(N, 20484)` (gauche 0..10241 puis droite), `seconds` int32 `(N,)`
  = seconde de stimulus de chaque ligne (TRIBE applique déjà son décalage HRF de 5 s ; des secondes
  peuvent manquer car TRIBE supprime les segments vides).
- Sidecar `PATH.npz.meta.json` : `{"model", "device", "n_rows", "runtime_s", "stage_times": {..},
  "peak_rss_mb", "text_feature": "llama"|"disabled", "versions": {"torch","transformers","tribev2"},
  "warnings": [..]}`.
- Le process se termine après la prédiction (libère la RAM).
- Mode `--download-only --model REPO [--no-text]` : télécharge checkpoint + backbones dans le cache HF
  sans inférence (utilisé par `make models`), mêmes codes de sortie, lignes `PROGRESS`.

## 2. API Python du pipeline (ALGORITHME implémente, APP appelle)

```python
from reel_predictor.config import load_config, Config     # load_config(config_dir=paths.CONFIG_DIR) -> Config
from reel_predictor.pipeline import analyze, AnalysisResult
from reel_predictor.health import status                   # status(config) -> dict, rapide (< 2 s), jamais d'exception
from reel_predictor.assets import ensure_all               # ensure_all(config) : télécharge modèles légers (idempotent)
from reel_predictor.calibrate import calibrate             # calibrate(pairs) -> dict ; voir §6
from reel_predictor.stages import STAGES, STAGE_LABELS, global_progress

result: AnalysisResult = analyze(
    video_path: Path, out_dir: Path, config: Config | None = None,
    progress: Callable[[str, float, str], None] | None = None,   # (stage_id, pct_global 0..1, message FR)
    brain: BrainProvider | None = None,                          # None = selon config (tribe -> repli heuristique)
)
# AnalysisResult : out_dir: Path, summary: dict, timeline: list[dict]
```
- `analyze` ne lève une exception que si la vidéo est illisible. Un échec TRIBE => repli
  `HeuristicProvider`, `summary.brain.degraded = true` + `reason`.
- `config.raw` : dict `{"weights": {...}, "roi_mapping": {...}, "rules": {...}}` (YAML bruts).
- `status(config)` : `{"brain_expected": "tribe"|"heuristic", "tribe_model": str, "worker_installed": bool,
  "hf_token": bool, "llama_access": bool|None, "reason": str|None, "versions": {..}}`.

## 3. Sorties d'une analyse : `data/results/<job_id>/`

L'APP crée `out_dir`, y copie la vidéo sous `video.mp4`, puis appelle `analyze(out_dir/"video.mp4", out_dir)`.

### `timeline.jsonl` — une ligne JSON par seconde `t` (couvre `[t, t+1)`), `t = 0..ceil(durée)-1`
| clé | type | sens |
|---|---|---|
| `t` | int | seconde |
| `roi` | {nom: float\|null} | valeur ROI relative à la référence corpus (z robuste) ; null si pas de cerveau |
| `roi_video_z` | {nom: float\|null} | z-score intra-vidéo (dynamique) |
| `indices` | {attention, action, tension, manipulation, emotion, memory: float 0..1} | toujours présents |
| `cut` | bool | un cut tombe dans `[t, t+1)` |
| `since_cut` | float | secondes écoulées depuis le dernier cut (ou le début) à la fin de `t` |
| `faces` | int | max de visages détectés dans la seconde |
| `face_area` | float 0..1 | aire max d'un visage / aire image |
| `words` | [str] | mots dont le début tombe dans la seconde |
| `speech_rate` | float | mots/s (fenêtre glissante 3 s) |
| `text` | str | texte à l'écran (OCR), `""` si aucun |
| `text_words` | int | nb de mots OCR |
| `text_new` | bool | texte différent de la seconde précédente |
| `audio_rms` | float 0..1 | RMS normalisée au max de la vidéo |
| `silence` | bool | RMS sous le seuil config |
| `retention` | float 0..1 | part prédite des spectateurs encore présents à la fin de `t` |
| `flags` | [str] | ex. `low_context`, `no_brain`, `hook`, `cta` |

### `summary.json`
```json
{
  "version": 1, "job_id": "...", "created_at": "ISO-8601",
  "video": {"filename": "...", "duration_s": 31.2, "fps": 30.0, "width": 1080, "height": 1920, "has_audio": true},
  "score": 72,
  "score_breakdown": {"index_means": {"attention": 0.61, "...": 0}, "hook_bonus": 4.0, "dip_penalty": -6.0, "weights": {"...": 0}},
  "retention": {"curve": [1.0, 0.93, "..."], "avg_watch_ratio": 0.54, "avg_watch_s": 16.8, "completion": 0.31},
  "recommendations": [{"t": 7.0, "t_end": 11.0, "rule": "attention_drop_no_cut", "severity": "high", "message": "..."}],
  "cuts": [2.4, 5.1],
  "transcript": {"language": "fr", "text": "..."},
  "brain": {"available": true, "provider": "tribe", "model": "facebook/tribev2-mini", "degraded": false,
            "reason": null, "rois": ["ips", "..."], "confidence": {"attention": "medium", "...": "low"}},
  "timings": {"probe": 0.4, "...": 0},
  "config": {"weights": {}, "roi_mapping": {}, "rules": {}}
}
```
`severity` ∈ `high|medium|low`. Messages en français, actionnables (« Couper / ajouter un texte à 7 s »).

### Carte cérébrale (uniquement si `brain.available`)
- `brain.u8` : octets bruts uint8, ordre C, forme `(T, 20484)` (gauche puis droite) ; ligne `t` = seconde `t`.
  `0..255` ↦ linéaire `[vmin, vmax]`. Secondes sans prédiction : remplies avec la valeur médiane (128).
- `brain_meta.json` : `{"T": int, "n_vertices": 20484, "hemi_split": 10242, "vmin": float, "vmax": float,
  "space": "fsaverage5", "units": "z (TRIBE, niveau session)"}` ; `vmin = -vmax` (échelle divergente
  centrée sur 0, `vmax` = 98ᵉ percentile de |v|).
- Le maillage 2D pour l'affichage (`app/static/brain_mesh.json`) est produit par l'APP (`scripts/fetch_assets.py`)
  à partir de `nilearn.datasets.fetch_surf_fsaverage("fsaverage5")` (surfaces `infl_left/right`).

## 4. Configuration YAML (`config/`)
- `roi_mapping.yaml` (RECHERCHE) — schéma figé : `atlas`, `space`, `hemispheres`, `rois: {nom: [labels Destrieux]}`,
  `indices: {indice: {roi: poids signé}}`, `confidence: {indice: medium|low}`.
- `weights.yaml` (ALGORITHME) — boosters non cérébraux, mélange cerveau/signaux par indice, poids du score,
  bonus hook / pénalité creux, modèle de rétention, normalisation, `hrf_shift_s` (0), section `brain:`
  (`provider: tribe|heuristic`, `model: facebook/tribev2-mini`, `device`, `timeout_s`, `no_text`), section `speech:`
  (`model: small`, `language: auto`), fps d'échantillonnage visages/OCR. Commenté en français.
- `rules.yaml` (ALGORITHME) — liste `{name, enabled, severity, params: {...}, message: "texte FR avec {t}"}`.

## 5. API HTTP (APP implémente) — `http://127.0.0.1:8765`
- `POST /api/analyze` (multipart `file`) → `{"job_id"}` ; une seule analyse à la fois (file d'attente).
- `GET /api/jobs/{job_id}` → `{"status": "queued|running|done|error", "stage", "stage_label", "pct", "message", "eta_s", "error"}`.
- `GET /api/results/{job_id}/summary` · `/timeline` (tableau JSON) · `/brain` (octets) · `/brain_meta`.
- `GET /media/{job_id}.mp4` (support des requêtes Range pour le seek).
- `GET /api/reels` · `POST /api/insights/ocr` (image) → champs pré-remplis + texte brut · `POST /api/insights` (validé)
  · `GET /api/reels/{job_id}/comparison` · `POST /api/calibrate` · `GET /api/health`.

## 6. Calibration (ALGORITHME implémente, APP déclenche)
`calibrate(pairs: list[tuple[Path, float]]) -> dict` : `pairs` = (`result_dir`, ratio réel de visionnage moyen
= durée moyenne de visionnage / durée). Ajuste les paramètres de rétention (moindres carrés, bornés), écrit
`data/calibration.json` (surcharge `weights.yaml > retention` aux analyses suivantes) et renvoie
`{"n": int, "params": {...}, "mae_before": float, "mae_after": float}`. Lève `ValueError` si `n < 3`.

## 7. Coach « script & montage » (ajout du 12/10/2026)

Déjà écrits (lecture seule) : `reel_predictor/coach/{__init__,schema,targets}.py`, `config/coach.yaml`.
Plan détaillé : `~/.claude/plans/glowing-sniffing-sutton.md`.

| Propriétaire | Fichiers |
|---|---|
| SIMULATION | `tribe_worker/run_tribe.py` (+ `--no-video`), `reel_predictor/coach/simulate.py`, `reel_predictor/rules.py`, `reel_predictor/textutils.py`, `config/rules.yaml`, `config/weights.yaml` (section CTA uniquement), `tests/test_coach_simulate.py`, `tests/test_rules_v2.py` |
| COACH+UI | `reel_predictor/coach/{brief,llm,keys,service}.py`, `app/**`, `Makefile` (cible `api-key`), `tests/test_coach_brief.py`, `tests/test_coach_llm.py`, `tests/test_app_coach.py` |

### Worker
`run_tribe.py --no-video` : features `["audio","text"]` (+ `--no-text` combinable). La vidéo d'entrée peut alors être
un simple `.wav` passé via `--video` OU un nouveau `--audio PATH.wav` (au choix de SIMULATION, à documenter dans
`tribe_worker/README.md`). Mêmes sorties `.npz` / `.meta.json` (`meta.video_feature = "disabled"`).

### Simulation (SIMULATION implémente, COACH appelle)
```python
from reel_predictor.coach.simulate import simulate, baseline_variant
baseline_variant(job_dir: Path) -> Variant          # script ORIGINAL (transcription + OCR + cuts) en segments
simulate(variants: list[Variant], job_dir: Path, config: Config, coach_cfg: dict,
         progress: Callable[[float, str], None] | None = None) -> dict
```
`simulate` renvoie :
```json
{"mode": "tribe_no_video" | "heuristic", "reason": null,
 "results": [  // même ordre que `variants` (le 1er est normalement baseline_variant)
   {"name": "original" | "variante 1", "duration_s": 41.2, "score": 71,
    "segments": [{"role": "hook", "t": 0.0, "t_end": 2.8}],
    "timeline": [{"t": 0, "indices": {"attention": 0.8, "...": 0}, "retention": 0.93}],
    "evaluation": { "...": "sortie de targets.evaluate" }}
 ],
 "best": 2, "timings": {"tts": 4.1, "brain": 55.0}}
```
Durées issues de la voix de synthèse (`say -v Thomas`). Signaux non cérébraux reconstruits depuis les segments
(`cut_at_start`, `on_screen_text`, `shot`), puis `indices` / `retention` / `score` du pipeline. Jamais d'exception
pour un échec TRIBE : repli heuristique (`mode`, `reason`).

### Service coach (COACH implémente, APP appelle)
`reel_predictor.coach.service.run_coach(job_dir, progress) -> dict` écrit
`data/results/<job>/coach/<AAAAMMJJ-HHMMSS>.json` (+ `.md`) : `{"created_at", "model", "usage": {...}, "cost_usd",
"output": CoachOutput (dict), "brief": {...}, "simulation": <sortie simulate>, "warnings": []}`.
