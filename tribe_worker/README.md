# tribe_worker — TRIBE v2 en sous-process (venv isolé)

Exécute `facebook/tribev2-mini` (LLaMA-3.2-1B + V-JEPA 2 ViT-L + w2v-BERT 2.0) sur une vidéo et écrit la
prédiction d'activité corticale fsaverage5 seconde par seconde. Contrat : `docs/CONTRACTS.md` §1.
Recherche, sources et limites : `docs/RESEARCH.md`.

## Lancer

```bash
# installation (une fois) : venv séparé, pins de tribev2 (torch<2.7, numpy==2.2.6)
uv sync --project tribe_worker

# téléchargement des modèles (≈ 6,8 Go ; ≈ 4,3 Go avec --no-text)
tribe_worker/.venv/bin/python tribe_worker/run_tribe.py --download-only --model facebook/tribev2-mini [--no-text]

# prédiction
tribe_worker/.venv/bin/python tribe_worker/run_tribe.py \
    --video data/results/<job>/video.mp4 --words data/results/<job>/words.json \
    --out data/results/<job>/tribe.npz --model facebook/tribev2-mini --cache-dir data/cache/tribe \
    [--device auto|cpu|mps] [--no-text]
```

```bash
# sans image (simulation de scripts du coach) : audio + texte, V-JEPA 2 jamais chargé
tribe_worker/.venv/bin/python tribe_worker/run_tribe.py --no-video \
    --audio voix.wav --words words.json --out sim.npz [--no-text]
# (équivalent accepté : --no-video --video voix.wav)
```

- stdout : `PROGRESS <0..1> <message FR>` ; tout le reste (logs, tqdm) va sur stderr.
- Codes : `0` ok · `1` autre · `2` entrée invalide (vidéo/JSON illisible, `--out` sans `.npz`,
  `facebook/tribev2` complet refusé) · `3` accès HF refusé + ligne stderr `ERROR_HF_ACCESS <repo>` ·
  `4` mémoire insuffisante. Un arrêt par le système (SIGKILL, mémoire) donne un code négatif côté appelant.
- Sorties : `tribe.npz` (`preds` float16 `(N, 20484)`, `seconds` int32) + `tribe.npz.meta.json`
  (champs du contrat + champs informatifs : `peak_footprint_mb`, `devices`, `n_words`, `text_dtype`,
  `video_dtype`, `video_duration_s`, `media_digest`).
- `seconds[i]` = seconde de stimulus `[t, t+1)` ; TRIBE a déjà intégré le délai hémodynamique de 5 s :
  ne pas re-décaler. N = ceil(durée) lignes.
- Le premier lancement télécharge les modèles manquants (Hugging Face) ; ensuite le worker tourne
  hors ligne (`HF_HUB_OFFLINE=1`).
- Variables optionnelles : `TRIBE_TEXT_DTYPE` (bfloat16 sur MPS / float32 sur CPU par défaut),
  `TRIBE_VIDEO_DTYPE` (float16 par défaut sur MPS), `TRIBE_ATTN_CHUNK` (1024), `TRIBE_ALLOW_FULL=1`
  (autorise `facebook/tribev2`, déconseillé sur 16 Go).

## Mode `--no-video` (ajout 12/10/2026, coach « script & montage »)

- Entrée : `--audio PATH` (tout format lu par ffmpeg : wav, aiff, m4a… ; converti en WAV mono 16 kHz dans
  `<cache-dir>/media/<sha1>.audio.wav`, donc le cache de features audio est réutilisé si la même voix off
  revient). Un fichier audio passé par `--video` est accepté si `--no-video` est présent. `--audio` sans
  `--no-video` = code 2. `--words` reste obligatoire (liste vide possible).
- Features `["audio", "text"]` (`["audio"]` avec `--no-text`) : `data.features_to_use` ne contient pas
  `video`, l'extracteur V-JEPA 2 n'est ni instancié ni patché ; la tête reçoit des zéros pour la vidéo
  (`FmriEncoderModel.aggregate_features`), exactement l'état vu à l'entraînement par *modality dropout*
  (p = 0,3). TRIBE prévoit d'ailleurs nativement une entrée audio seule (`get_events_dataframe(audio_path=…)`).
- Événements : `Audio` (découpé 30–60 s comme d'habitude) + `Word` ; durée = durée de l'audio ;
  N = ceil(durée) lignes. `--download-only --no-video` ne télécharge pas V-JEPA 2.
- Sorties identiques ; `meta.video_feature = "disabled"` (`"vjepa2"` sinon), `video_dtype = null`.
- **Mesures** (M5 16 Go, MPS, voix `say -v Thomas`, 25,1 s dont 11 s de parole, 36 mots) [MESURÉ] :
  avec texte 22,8 s au total (chargement TRIBE 10,2 s, audio 5,1 s, LLaMA 6,7 s, tête 0,4 s), soit
  **≈ 0,5 s de calcul par seconde d'audio** hors chargement (vs ≈ 10 s/s avec la vidéo) ; empreinte pic
  **5,4 Go** (RSS 1,0 Go). `--no-text` avec features audio en cache : 9,3 s, 2,3 Go.
- **Vérification** [MESURÉ] : la ROI `auditory` suit la parole de synthèse (moyenne +0,24 pendant la parole
  vs −0,06 en silence, corrélation 0,70 avec l'indicatrice de parole, pic +0,51 ; montée ~1–2 s avant le
  début, comme en mode vidéo §1.10) ; `psts` aussi (corr. 0,84). `early_visual`/`ifg` ne suivent pas la
  parole (corr. ≈ 0), cohérent avec l'absence d'image.
- Limite : sans vidéo, les ROI visuelles/attentionnelles (`early_visual`, `mt`, `ips`, `fef`) ne voient ni
  cuts ni texte à l'écran ; leur niveau n'est **pas** comparable à celui d'un Reel analysé avec l'image.
  Ne pas mélanger ces prédictions avec la référence de corpus des Reels réels.

## Fichiers

- `run_tribe.py` — CLI, étapes, progression, codes de sortie, écriture npz/meta.
- `tribe_events.py` — DataFrame d'événements Video/Audio/Word construit depuis `--words` (remplace
  `get_events_dataframe` : pas de WhisperX, pas de spaCy) ; `ffprobe_audio` / `prepare_audio` pour
  `--no-video`.
- `tribe_compat.py` — contournements Mac (devices, V-JEPA 2 sur MPS, cache HF hors ligne, mémoire).

## Contournements appliqués (détails et preuves : `docs/RESEARCH.md` §1.6)

1. Extracteurs `device: cuda` codés en dur → `config_update` cpu puis réassignation (MPS) après validation.
2. WhisperX (`uvx`, float16, anglais, bug > 60 s) et spaCy (téléchargement à la volée) **contournés** :
   événements construits depuis la transcription faster-whisper du pipeline, contexte LLaMA identique à
   `AddText`/`AddContextToWords` (casse `capitalize()` par phrase, fenêtre 1024 mots).
3. Médias adressés par contenu dans `<cache-dir>/media/` (lien `.mp4` + `.wav` 16 kHz) : rien n'est écrit à
   côté de la vidéo ; le cache de features (`<cache-dir>/features/`, ~300 Ko par vidéo et modalité) est
   réutilisé si la même vidéo est ré-analysée.
4. `data.num_workers = 0` (pas de `fork` nécessaire sur macOS).
5. V-JEPA 2 : `skip_predictor=True`, attention exacte par blocs (sinon « Invalid buffer size: 12 GB » sur
   MPS), fp16 sur MPS (repli fp32 automatique si débordement), moyenne des tokens sur GPU en fp32, cache des
   images prétraitées. Prédictions identiques à la version non optimisée (corrélation 0,99999998).
6. LLaMA : bf16 sur MPS (écart négligeable mesuré vs fp32, moitié de mémoire et de temps), fp32 sur CPU ;
   chargé mmappé puis transféré paramètre par paramètre (le chargement d'origine faisait un pic à 12 Go).
   w2v-BERT : biais de position relatif calculé sans le tenseur (L, L, 64) (résultat identique, −1,6 Go).
7. Repli automatique MPS → CPU pour l'audio, le texte et la tête en cas d'erreur MPS (contourne le « gel »
   exca de l'extracteur) ; pour la vidéo, une erreur mémoire sur MPS sort en code 4 (sur CPU, V-JEPA 2 est
   5 à 7 fois plus lent : ~44 s de calcul par seconde de vidéo).
9. LLaMA sur MPS : la SDPA avec GQA native (`enable_gqa=True`, transformers 5) plante au-delà d'environ
   200 tokens de contexte (« LLVM ERROR … mps.matmul incompatible dimensions », SIGABRT, constaté le 12/10/2026
   sur une voix off de 190 mots) → `patch_sdpa_gqa_mps()` force `repeat_kv` sur MPS ; prédictions identiques
   (corrélation 0,99999999, écart max 5e-4 = arrondi fp16 de la sortie).
8. `--no-text` : LLaMA non chargé, la modalité texte est mise à zéro — exactement le *modality dropout*
   (p = 0,3) vu à l'entraînement.

## Mesures (Apple M5, 16 Go, `--device auto` → MPS)

Détails, conditions et autres mesures : `docs/RESEARCH.md` §1.7. Temps variables de ±30 % selon la
charge de la machine. « Empreinte » = mémoire physique totale du process, Metal/MPS compris (c'est elle qui
compte sur 16 Go ; `peak_rss_mb` ne voit pas la mémoire MPS).

| Cas | Durée vidéo | Total | Calcul / s de vidéo | Empreinte pic |
|---|---|---|---|---|
| Reel réel 1080×1920 60 i/s, **avec texte** (167 mots), défauts | 40,9 s | ≈ 416 s | **≈ 10 s** | **≈ 9,0 Go** |
| Reel réel 1080×1920 60 i/s, `--no-text` | 32,1 s | 307 – 312 s | ≈ 9,6 s | 8,7 – 9,7 Go |
| Vidéo synthétique 720×1280 30 i/s, `--no-text` | 30 s | 196 s | ≈ 6,5 s | 6,6 Go |
| Même vidéo, `--device cpu` (tout CPU) | 30 s | 1 321 s | ≈ 44 s | 2,6 Go (RSS 4,2 Go) |
| Ré-analyse d'une vidéo déjà vue (features en cache) | 30 s | ≈ 10 – 15 s | – | ≈ 2,2 Go |

Par étape (MPS) : chargement TRIBE 8 s ; audio w2v-BERT 4 – 7 s ; **vidéo V-JEPA 2 ≈ 93 % du temps**
(3,0 s par clip de 0,5 s en 720p, 4,6 s en 1080p60) ; texte LLaMA bf16 11 – 16 s pour 116 – 167 mots ;
tête 0,5 s. Le texte ajoute donc ~5 % de temps : `--no-text` fait surtout gagner 2,5 Go de
téléchargement et l'accès gated, au prix d'indices langagiers dégradés (RESEARCH §1.8).

Modèles : 698 Mo (TRIBE mini) + 1,30 Go (V-JEPA 2 ViT-L) + 2,32 Go (w2v-BERT) + 2,48 Go (LLaMA-3.2-1B,
gated) ≈ 6,8 Go dans le cache Hugging Face (`~/.cache/huggingface/hub`).

Vérifications : prédictions MPS (fp16) vs CPU (fp32) corrélées à 0,9999994 ; LLaMA bf16 vs fp32 :
0,9999976 ; ROI auditive +1,0 à +1,2 z pendant la parole (vidéo de test) ; codes 2/3 testés
(`ERROR_HF_ACCESS meta-llama/Llama-3.2-1B` sans token) ; repli MPS → CPU testé (échec simulé).
