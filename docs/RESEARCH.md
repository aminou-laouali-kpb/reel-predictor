# RECHERCHE — TRIBE v2 sur Mac, cartographie fsaverage5 → ROI → indices

Document de référence du chantier RECHERCHE de Reel Predictor (rédigé le 2026-10-11).
Machine de mesure : Apple M5, 16 Go de mémoire unifiée, macOS 27, Python 3.12, torch 2.6.0 (MPS),
transformers 5.17.0, tribev2 @ af58661, neuralset 0.0.2.

Conventions :
- chaque affirmation factuelle est suivie de sa source (URL, ou fichier du code installé
  `tribe_worker/.venv/lib/python3.12/site-packages/...`, abrégé `site-packages/...`) ;
- **[MESURÉ]** = mesuré sur cette machine pendant ce chantier (scripts dans le dossier brouillon) ;
- **[SPÉCULATIF]** = hypothèse ou extrapolation non validée scientifiquement ;
- **[NON VÉRIFIÉ]** = rapporté par une source secondaire, non contrôlé ici.

---

## 1. TRIBE v2

### 1.1 Identité et licences

- Modèle d'encodage cérébral multimodal de Meta FAIR : prédit la réponse fMRI (BOLD) à des stimuli
  naturalistes (vidéo, audio, texte), pour un « sujet moyen », sur le maillage cortical **fsaverage5**.
  Code `facebookresearch/tribev2`, poids `facebook/tribev2` et `facebook/tribev2-mini` sur Hugging Face,
  article d'Ascoli et al. 2026 (arXiv 2605.04326).
  — https://huggingface.co/facebook/tribev2 · https://github.com/facebookresearch/tribev2 · https://arxiv.org/html/2605.04326
- **Licence des poids et du code TRIBE v2 : CC BY-NC 4.0 (usage non commercial uniquement).** Champ
  `license: cc-by-nc-4.0` des cartes HF de `facebook/tribev2` et `facebook/tribev2-mini` (vérifié via l'API HF
  le 2026-10-11) ; README : « This project is licensed under CC-BY-NC-4.0 ».
  — https://huggingface.co/facebook/tribev2 · https://raw.githubusercontent.com/facebookresearch/tribev2/main/LICENSE
  ⇒ Reel Predictor ne doit pas être utilisé pour une activité commerciale (analyse de Reels pour un client
  payant, etc.) sans accord de Meta. Plusieurs demandes de licence commerciale sont restées ouvertes sur GitHub
  (#8, #17, #25, #45, #49, #66) [NON VÉRIFIÉ ici, issu du rapport de planification].
- Backbones (licences lues sur les cartes HF via l'API, 2026-10-11) :
  `meta-llama/Llama-3.2-1B` → licence **Llama 3.2 Community License** (`llama3.2`), dépôt **gated manuel** ;
  `facebook/vjepa2-vitl-fpc64-256` → MIT ; `facebook/w2v-bert-2.0` → MIT.
  — https://huggingface.co/meta-llama/Llama-3.2-1B · https://huggingface.co/facebook/vjepa2-vitl-fpc64-256 · https://huggingface.co/facebook/w2v-bert-2.0

### 1.2 Installation (venv isolé)

- tribev2 n'est pas publié sur PyPI ; installation depuis Git (`pip install -e .` ou
  `uv pip install "tribev2 @ git+https://github.com/facebookresearch/tribev2.git"`).
  — https://raw.githubusercontent.com/facebookresearch/tribev2/main/README.md
- Pins imposés par tribev2 : `torch>=2.5.1,<2.7`, `numpy==2.2.6`, `exca==0.5.20`, `neuralset==0.0.2`,
  `x_transformers==1.27.20`, `moviepy>=2.2.1` ; `transformers` non pinné.
  — https://raw.githubusercontent.com/facebookresearch/tribev2/main/pyproject.toml
- D'où un venv séparé `tribe_worker/.venv` (`tribe_worker/pyproject.toml`, tribev2 @ af58661), appelé en
  sous-process par le pipeline (contrat `docs/CONTRACTS.md` §1).
- **transformers 5.17.0 : compatible, aucun pin ajouté** [MESURÉ] : import OK avec torch 2.6.0,
  `AutoVideoProcessor`/V-JEPA 2, `Wav2Vec2BertModel`, LLaMA-3.2-1B et la tête TRIBE fonctionnent.
  Différence notable : transformers 5 charge par défaut les poids dans le dtype du checkpoint
  (`dtype="auto"`, `site-packages/transformers/modeling_utils.py`), donc LLaMA-3.2 en bf16 alors que
  transformers 4 chargeait en fp32 (régime de l'entraînement de TRIBE). Le worker choisit explicitement :
  bf16 sur MPS (écart mesuré négligeable, §1.7 C), fp32 sur CPU (`TRIBE_TEXT_DTYPE`).

### 1.3 API et sorties (lues dans le code installé)

- `TribeModel.from_pretrained(repo, checkpoint_name="best.ckpt", cache_folder=None, cluster=None,
  device="auto", config_update=None)` : télécharge `config.yaml` + `best.ckpt`, force
  `average_subjects=True`, applique `config_update`, construit la tête et la place sur `device`
  (« auto » = CUDA sinon CPU, **pas de détection MPS**). — `site-packages/tribev2/demo_utils.py`
- `get_events_dataframe(video_path=...)` → DataFrame d'événements (voir §1.9) ;
  `predict(events) -> (preds, segments)` : `preds` de forme `(n_segments_gardés, 20484)`, un segment = 1 TR
  = 1 s ; les segments sans aucun événement sont supprimés (`remove_empty_segments=True`).
  — `site-packages/tribev2/demo_utils.py`
- Espace : fsaverage5, 2 × 10 242 sommets (hémisphère gauche puis droit) = 20 484.
  — `site-packages/tribev2/utils_fmri.py` ; article §5.1 https://arxiv.org/html/2605.04326
- Unités : les cibles d'entraînement sont z-scorées par sommet et par session, détrendées, ré-échantillonnées
  à 1 Hz ; la prédiction est celle d'un « sujet moyen » (moyenne des têtes sujet).
  — article §5.8 https://arxiv.org/html/2605.04326 ; `config.yaml` (`neuro.cleaning.standardize: zscore_sample`,
  `detrend: true`, `frequency: 1.0`) https://huggingface.co/facebook/tribev2-mini/raw/main/config.yaml
- Fenêtres : la tête traite des segments de 100 TR (`data.duration_trs: 100`), sans recouvrement à l'inférence ;
  pour un Reel ≤ 100 s, une seule fenêtre. — `config.yaml` de tribev2-mini
- Amplitude observée [MESURÉ] (vidéo de test 30 s sans piste audio, modalité vidéo seule) : moyenne 0,02,
  écart-type global 0,16, écart-type temporel médian par sommet 0,067 — bien plus faible qu'un vrai z-score
  de session (effet probable de la moyenne sur les sujets [SPÉCULATIF]). Ne pas interpréter les valeurs
  absolues comme des z de session.

### 1.4 Modèles téléchargés et tailles (tribev2-mini, le seul retenu)

| Rôle | Dépôt HF | Fichiers téléchargés par le worker | Taille | Gated |
|---|---|---|---|---|
| Tête TRIBE | `facebook/tribev2-mini` | `config.yaml`, `best.ckpt` | 698 Mo | non |
| Vidéo | `facebook/vjepa2-vitl-fpc64-256` (ViT-L, 300 M param.) | `model.safetensors` + json | 1 304 Mo | non |
| Audio | `facebook/w2v-bert-2.0` (≈ 600 M param.) | `model.safetensors` + json | 2 322 Mo | non |
| Texte | `meta-llama/Llama-3.2-1B` | `model.safetensors` + tokenizer/json | 2 481 Mo | **oui (manuel)** |

Tailles lues via `HfApi.model_info(files_metadata=True)` le 2026-10-11 (les `original/*.pth` en double et
`conformer_shaw.pt` sont exclus du téléchargement). Total ≈ **6,8 Go** avec texte, **≈ 4,3 Go** sans texte.
Le config de tribev2-mini liste `meta-llama/Llama-3.2-1B`, `facebook/vjepa2-vitl-fpc64-256`,
`facebook/w2v-bert-2.0` (et `facebook/dinov2-large` en `image_feature`, **non utilisé** :
`features_to_use: [text, audio, video]`, jamais téléchargé). — https://huggingface.co/facebook/tribev2-mini/raw/main/config.yaml

`facebook/tribev2` complet (LLaMA-3.2-**3B** + V-JEPA2 **ViT-g**, ≈ 17 Go de téléchargements, 12,9 Go de RAM
pour LLaMA-3B en fp32) est **refusé par le worker** (code 2) sauf `TRIBE_ALLOW_FULL=1`.
— https://huggingface.co/facebook/tribev2/raw/main/config.yaml ; https://github.com/facebookresearch/tribev2/issues/28

La précision de tribev2-mini par rapport au modèle complet n'est documentée ni sur sa carte HF (README vide
hormis la licence) ni dans l'article (aucune mention d'une variante « mini ») [NON VÉRIFIÉ / non documenté].
— https://huggingface.co/facebook/tribev2-mini ; https://arxiv.org/html/2605.04326

### 1.5 Accès Hugging Face (LLaMA gated)

- Seul `meta-llama/Llama-3.2-1B` est gated (validation manuelle de la licence Llama 3.2 sur la page du
  modèle, puis `hf auth login` avec un token « read »). — https://huggingface.co/meta-llama/Llama-3.2-1B
- Chronologie pendant ce chantier : d'abord aucun token (`hf auth whoami` → « Not logged in ») ; puis token
  présent mais demande d'accès Llama « awaiting a review from the repo authors » (HTTP 403) ; puis accès
  accordé par Meta en quelques heures ⇒ LLaMA-3.2-1B téléchargé et mesuré en fin de chantier.
- Le worker vérifie le cache local AVANT tout calcul lourd ; si un dépôt manque il le télécharge, et un refus
  (GatedRepoError / 401 / 403) donne le code de sortie **3** et la ligne stderr `ERROR_HF_ACCESS <repo>`
  [MESURÉ sans token : `ERROR_HF_ACCESS meta-llama/Llama-3.2-1B`, code 3 ; le cas 403 « en attente » passe
  par la même branche (`GatedRepoError` / HTTP 403) : non rejoué en réel (accès accordé entre-temps) mais
  vérifié par simulation d'une `GatedRepoError` 403 avec le message réel « awaiting a review »].
  Le pipeline relance alors avec `--no-text` (`config/weights.yaml > brain.retry_without_text`).
- Une vidéo sans aucun mot transcrit n'a pas besoin de LLaMA (texte traité comme absent) : l'accès gated
  n'est alors pas requis.
- Une fois tous les fichiers en cache, le worker passe `HF_HUB_OFFLINE=1` : aucune requête réseau pendant
  une analyse.

### 1.6 Apple Silicon : blocages rencontrés et contournements appliqués dans le worker

| Problème (amont) | Contournement dans `tribe_worker/` | Source / preuve |
|---|---|---|
| `device: cuda` codé en dur pour chaque extracteur dans `config.yaml` | `config_update` → `cpu` à la construction, puis device réassigné après validation (le `Literal["auto","cpu","cuda","accelerate"]` de neuralset refuse `mps`, mais sans `validate_assignment`) | config.yaml ; `site-packages/neuralset/extractors/base.py` (HuggingFaceMixin) ; PR https://github.com/facebookresearch/tribev2/pull/13 |
| WhisperX lancé par `uvx whisperx` (venv torch 2.8), `compute_type="float16"` (refusé sur CPU), langue `"english"` codée en dur, mots décalés au-delà de 60 s | **WhisperX entièrement contourné** : DataFrame d'événements construit à partir de la transcription faster-whisper du pipeline (§1.9) | `site-packages/tribev2/eventstransforms.py` ; PR https://github.com/facebookresearch/tribev2/pull/58 , https://github.com/facebookresearch/tribev2/pull/29 |
| spaCy `fr_core_news_lg`/`en_core_web_lg` téléchargé à la volée (réseau, pip) par AddText/AddSentenceToWords | phrases et contextes reconstruits sans spaCy (§1.9) | `site-packages/neuralset/utils.py` (`get_spacy_model` → `spacy.cli.download`) |
| `ExtractAudioFromVideo` écrit le `.wav` à côté de la vidéo | WAV mono 16 kHz écrit dans `<cache-dir>/media/<sha1>.wav` | `site-packages/neuralset/events/transforms/audio.py` |
| Cache de features exca indexé par **chemin** de fichier (collision possible entre vidéos homonymes) | vidéo référencée via un lien symbolique adressé par contenu `<cache-dir>/media/<sha1>.mp4` (ré-analyse de la même vidéo = features réutilisées) | `site-packages/neuralset/extractors/video.py` (`item_uid`) |
| DataLoader `num_workers: 20` (nécessiterait `set_start_method("fork")` sur macOS) | `data.num_workers = 0` | config.yaml ; PR #13 |
| V-JEPA 2 sur MPS : la SDPA de torch 2.6 matérialise l'attention 8192×8192×16 têtes → « Invalid buffer size: 12.00 GB » [MESURÉ] | attention exacte **par blocs de 1024 requêtes** (fonction d'attention enregistrée dans transformers) | bench brouillon `bench_vjepa*.py` |
| V-JEPA 2 : le prédicteur JEPA (12 couches) est calculé alors que TRIBE n'utilise que l'encodeur | `skip_predictor=True` (sorties de l'encodeur identiques) | `site-packages/transformers/models/vjepa2/modeling_vjepa2.py` |
| V-JEPA 2 trop lent sur CPU (68 s par clip avec le code d'origine, ~22 s même optimisé ⇒ ~44 s de calcul par seconde de vidéo) [MESURÉ] | V-JEPA 2 sur **MPS en fp16** (3,0–4,6 s par clip dans le worker) ; attention fp16 explicite ; moyenne des tokens faite sur GPU en fp32 ; cache des images décodées | §1.7 |
| transformers 5 charge LLaMA en bf16 (TRIBE entraîné avec features fp32) ; le chargement d'origine (fp32 sur CPU puis `.to(mps)`) crée un pic de 12 Go d'empreinte [MESURÉ] | dtype explicite : **bf16 sur MPS** (prédictions corrélées à 0,9999976 avec fp32), fp32 sur CPU ; chargement mmappé puis transfert/conversion paramètre par paramètre vers MPS (`device_map` exigerait accelerate) | `site-packages/transformers/modeling_utils.py` ; §1.7 C |
| w2v-BERT 2.0 (`relative_key`) matérialise un tenseur (L, L, 64) par couche : 1,1 Go pour 41 s d'audio, 2,3 Go pour un morceau de 60 s (empreinte 8 Go à l'étape audio [MESURÉ]) | biais de position relatif calculé sur les 73 distances distinctes puis indexé (`gather`) : résultat identique (écart 0,0), empreinte de l'étape 8,0 → 6,4 Go | `site-packages/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py` |
| Repli MPS → CPU impossible après un échec : exca « gèle » l'extracteur (« Cannot proceed to update … device … frozen ») [MESURÉ] | réassignation de `device` en contournant le gel (`device` est exclu de l'uid de cache) ; repli testé (échec MPS simulé → texte sur CPU, prédictions identiques) | `site-packages/exca` ; `tribe_worker/tribe_compat.py` |
| Bug des horodatages > 60 s (mots dupliqués/décalés par morceau) | sans objet (mots en temps absolu) | PR https://github.com/facebookresearch/tribev2/pull/29 |

Équivalence numérique des contournements [MESURÉ] :
- V-JEPA 2 MPS fp32 (attention par blocs) vs CPU fp32 : écart relatif max 7,7e-7 sur la moyenne des tokens
  de la couche 16 (corrélation 0,9999999999996).
- V-JEPA 2 MPS **fp16** vs CPU fp32 : écart relatif max 6,8e-5, corrélation 0,99999998 (bf16 : 3,4e-4,
  0,9999988 ⇒ fp16 retenu).
- Prédictions TRIBE finales avant/après les optimisations (attention fp16 explicite, moyenne GPU, cache
  d'images) : corrélation 0,99999998, écart max = 1 pas de quantification float16 (4,9e-4).
- Tête TRIBE sur MPS vs CPU (mêmes features) : corrélation 0,9999999998, écart max 4,9e-4 (quantification fp16).

### 1.7 Mesures sur cette machine

Conditions : Apple M5 16 Go, `--device auto` (= MPS pour la tête et les trois extracteurs), V-JEPA 2 en
fp16. D'autres agents tournaient en parallèle pendant une partie des mesures : variabilité observée ±30 %
sur les temps. « Empreinte » = `ri_lifetime_max_phys_footprint` (inclut la mémoire Metal/MPS, absente de
`ru_maxrss`) ; c'est le chiffre à comparer aux 16 Go. Scripts : dossier brouillon `research/`.

**A. Bench V-JEPA 2 ViT-L, un clip (64 images 256×256 = 8192 tokens), encodeur seul** [MESURÉ]

| Configuration | s / clip |
|---|---|
| CPU fp32, SDPA torch + prédicteur JEPA (code d'origine) | 68 |
| CPU fp32 dans le worker (blocs, sans prédicteur, lecture incluse) | 21,7 |
| MPS fp32, SDPA d'origine | échec « Invalid buffer size: 12.00 GB » |
| MPS fp32, attention par blocs | 10,6 |
| MPS fp16, attention par blocs (SDPA) | 3,5 – 4,8 |
| MPS fp16, blocs + softmax explicite, dans le worker (720p / 1080p60, lecture incluse) | 3,0 / 4,6 |

TRIBE extrait un clip tous les 0,5 s (features vidéo à 2 Hz) ⇒ 2 clips par seconde de vidéo.

**B. Worker de bout en bout, `--no-text` (audio + vidéo)** [MESURÉ]

| Vidéo | Durée | Total | s de calcul / s de vidéo | Audio (w2v-BERT) | Vidéo (V-JEPA 2) | Tête | RSS pic | Empreinte pic |
|---|---|---|---|---|---|---|---|---|
| Synthétique FR 720×1280 30 i/s (`say` + mires) | 30 s | 196 s | **6,5** | 5,6 s | 181 s | 0,5 s | 2,7 Go | 6,6 Go |
| Reel réel (exemple 1), 1080×1920 60 i/s | 32,1 s | 307 – 312 s | **9,6** | 4,2 – 4,4 s | 293 – 298 s | 0,6 s | 4,4 – 4,9 Go | 8,7 – 9,7 Go |
| Gris + 3 salves de parole, 720×1280 | 48 s | 290 s | 6,0 | 5,7 s | 276 s | – | 2,6 Go | 7,3 Go |
| Synthétique FR, **tout CPU** (`--device cpu`) | 30 s | 1 321 s | **44** | 7,0 s | 1 304 s | – | 4,2 Go | 2,6 Go |
| Ré-analyse de la même vidéo (features en cache) | 30 s | 9,4 s | – | 0 | 0,1 s | 1,4 s | 1,9 Go | – |

Chargement de TRIBE (tête + config) : 8 s ; construction des événements : < 1 s. `/usr/bin/time -l`
confirme les empreintes (6,9 Go et 9,2 – 10,1 Go). La vidéo représente 93 – 96 % du temps. Les Reels
1080p 60 i/s coûtent ~50 % de plus que le 720p 30 i/s (décodage moviepy, redimensionnement, conversions).

Équivalence MPS/CPU de bout en bout (audio + vidéo + tête, fp16 MPS vs fp32 CPU) : corrélation 0,9999994,
écart max 1,5e-3 [MESURÉ].

**C. Avec texte (LLaMA-3.2-1B)** [MESURÉ]

Étape texte seule (features audio/vidéo déjà en cache) :

| Vidéo | Mots | Variante | Étape texte | Empreinte pic de l'étape |
|---|---|---|---|---|
| Synthétique FR 30 s | 27 | fp32 MPS, chargement d'origine | 15,5 s | 8,3 – 8,9 Go |
| Synthétique FR 30 s | 27 | fp32 CPU (repli simulé) | 10,4 s | – |
| Reel « Maroc » 32 s | 116 | fp32 MPS, chargement d'origine | 41,4 s | 8,7 Go |
| Reel « Maroc » 32 s | 116 | fp32 MPS, chargement paramètre par paramètre | 24,1 s | 8,9 Go |
| Reel « Maroc » 32 s | 116 | **bf16 MPS (défaut)** | **11,5 s** | **6,4 Go** |

bf16 vs fp32 : prédictions corrélées à 0,9999976, écart moyen 3e-4 (à comparer à 6,3e-2 entre avec et
sans texte) ⇒ bf16 retenu par défaut sur MPS. Le coût du texte croît plus vite que le nombre de mots
(chaque mot = une passe LLaMA sur tout le contexte précédent, jusqu'à 1024 mots).

Bout en bout, vidéo jamais vue (Reel réel, exemple 2, 40,9 s, 1080×1920 60 i/s,
167 mots), réglages par défaut (`--device auto`) :

| Variante | Total | s de calcul / s de vidéo | Audio | Vidéo | Texte | Tête | RSS pic | Empreinte pic |
|---|---|---|---|---|---|---|---|---|
| **Défaut final** (LLaMA bf16, w2v-BERT compact) | ≈ 416 s | **≈ 10,2** | 4,6 s | 386 s | 16,4 s | 0,5 s | 4,4 Go | **9,0 Go** (9,5 Go `time -l`) |
| Première version (LLaMA fp32 chargé sur CPU puis `.to(mps)`) | 518 s | 12,7 | 7,1 s | 431 s | 70,7 s | 0,7 s | 5,1 Go | 12,1 Go (12,7 Go `time -l`) |

(La ligne « défaut final » combine le run complet à 416 s avec le remplacement de l'étape audio par la
version compacte de w2v-BERT, mesurée séparément sur la même vidéo : 4,6 s, empreinte de l'étape audio
8,0 → 6,4 Go, prédictions identiques à 0,0 près.)

Comparaison **avec / sans texte** (mêmes features audio/vidéo, z intra-ROI non appliqué) :

| | Synthétique 30 s (27 mots, 11 s de parole) | Reel « Maroc » 32 s (116 mots, parole continue) |
|---|---|---|
| Corrélation spatiale moyenne (par seconde) | 0,988 | 0,803 |
| Corrélation temporelle médiane par sommet | 0,973 | 0,894 |
| r(avec, sans) — attention / tension | 0,99 / 0,97 | 0,97 / 0,98 |
| r(avec, sans) — mémoire / émotion | 0,92 / 0,97 | 0,92 / 0,88 |
| r(avec, sans) — action / manipulation | 0,70 / 0,96 | 0,81 / **0,69** |
| r(avec, sans) — ROI `psts` / `temporal_pole` / `auditory` | 0,96 / 0,99 / 0,93 | **0,35 / −0,03** / 0,91 |

Avec texte, la réponse à la parole de la vidéo synthétique est plus nette : `auditory` +1,22 z
(r = 0,59 avec l'indicateur de parole) et `psts` +1,28 z (r = 0,62), contre +1,01 et +0,83 z sans texte.

**D. Plausibilité** [MESURÉ] (réduction en ROI avec l'atlas Destrieux de nilearn, `config/roi_mapping.yaml`)
- Vidéo synthétique 30 s (parole FR à 5,0–10,7 s et 21,0–25,6 s, silence ailleurs, sans texte) : ROI
  `auditory` +0,10 pendant la parole vs −0,05 en silence, soit **+1,0 z** (z intra-vidéo), corrélation
  +0,48 avec l'indicateur de parole ; `psts` +0,83 z. Contrôle : la même vidéo **sans piste audio**
  donne +0,02 z (aucune réponse).
- Salves de parole de 1,8 s sur fond gris (8, 24, 40 s) : la ROI auditive culmine à t = b ou b+1
  (b = début de la salve), commence à monter ~3–4 s **avant** et retombe dès b+2 s. Cohérent avec
  l'alignement de TRIBE (ligne t ↔ BOLD(t+5 s), or le BOLD à t+5 reflète les stimuli de ≈ t−3 à t+3) :
  le pic est bien aligné sur le stimulus, mais la prédiction est **floue de ±3 s** et anticipe les
  événements (le Transformer de TRIBE voit toute la fenêtre de 100 s, futur compris).
- Effets de bord : la ROI auditive est anormalement basse sur les ~4 premières secondes et remonte sur les
  ~3 dernières secondes, en silence, sur les trois vidéos de test ⇒ début et fin de vidéo peu fiables.

**E. Téléchargements** : débit observé 0,2 – 4 Mo/s (Xet / CDN HF) ; un téléchargement Xet s'est bloqué
et a été repris en HTTP (`HF_HUB_DISABLE_XET=1` reprend le fichier `.incomplete`). Prévoir 20 min à
plusieurs heures pour les 6,8 Go selon la connexion.

### 1.8 Mode `--no-text` (sans LLaMA)

- **Mécanisme natif** : TRIBE est entraîné avec un *modality dropout* p = 0,3 : chaque modalité est
  masquée en mettant son tenseur à zéro (une modalité au moins est conservée) ; les auteurs mettent en avant
  « its ability to provide meaningful predictions in the absence of one or several modalities ».
  — article §5.3 https://arxiv.org/html/2605.04326 ; `config.yaml` de tribev2-mini
  (`brain_model_config.modality_dropout: 0.3`).
- Implémentation dans le code : si la modalité est absente du batch, `FmriEncoderModel.aggregate_features`
  la remplace par un tenseur de zéros de même taille que la sortie du projecteur — **exactement** l'état vu
  pendant l'entraînement quand le dropout la masque. — `site-packages/tribev2/model.py`
- Le worker `--no-text` n'instancie pas l'extracteur LLaMA (`data.features_to_use: [audio, video]`) et
  n'ajoute aucun événement `Word` ; c'est aussi ce que fait TRIBE pour une vidéo sans parole (aucun mot
  transcrit ⇒ extracteur texte retiré, « Removing extractor text as there are no corresponding events »,
  `site-packages/tribev2/main.py`).
- **Qualité attendue** : l'article montre que le texte domine la prédiction des cortex du langage et d'une
  grande partie du préfrontal, l'audio le cortex auditif, la vidéo les cortex occipitaux et pariétaux
  (§2.8 https://arxiv.org/html/2605.04326). Sans texte, la prédiction est donc dégradée surtout dans les ROI
  langage/préfrontales.
- **Mesure de l'écart avec/sans texte** (mêmes features audio/vidéo, §1.7 C) [MESURÉ] : sur un vrai Reel à
  parole continue (116 mots), corrélation temporelle avec/sans texte : `psts` 0,35, `temporal_pole` −0,03,
  `ifg` 0,80, `angular` 0,86, `auditory` 0,91 ; indices : attention 0,97, tension 0,98, mémoire 0,92,
  émotion 0,88, action 0,81, **manipulation 0,69**. Le niveau moyen change aussi (ex. `psts` +0,06 avec texte
  vs −0,07 sans). On ne sait pas laquelle des deux versions est « vraie » (pas de fMRI) ; la version avec
  texte est celle du régime nominal de TRIBE. Sur la vidéo synthétique (27 mots, 11 s de parole sur 30), les
  écarts sont faibles (indices ≥ 0,92 sauf action 0,70).
- Recommandation : `--no-text` est scientifiquement acceptable (régime vu à l'entraînement) et quasi
  équivalent pour **attention** et **tension** ; pour **manipulation** (et dans une moindre mesure émotion,
  action) il faut afficher une confiance réduite (`text_feature: "disabled"` dans le `.meta.json`). La
  référence de corpus (médiane/MAD par ROI) doit être calculée séparément pour chaque mode, sinon le
  décalage de niveau sera lu comme un effet de la vidéo.

### 1.9 Schéma d'événements retenu (remplace `get_events_dataframe`)

`tribe_worker/tribe_events.py` reproduit le DataFrame que produit `get_events_dataframe(video_path=...)`
(`site-packages/tribev2/demo_utils.py` → `ExtractAudioFromVideo`, `ChunkEvents`, `ExtractWordsFromAudio`,
`AddText`, `AddSentenceToWords`, `AddContextToWords(sentence_only=False, max_context_len=1024)`,
`RemoveMissing`), sans WhisperX ni spaCy :

| Type | Champs | Construction |
|---|---|---|
| `Video` | `filepath` (lien `<cache>/media/<sha1>.mp4`), `start=0`, `offset`, `duration`, `frequency`=fps, `timeline="default"`, `subject="default"` | découpé en morceaux de 30–60 s par le vrai `ChunkEvents` de neuralset |
| `Audio` | idem, `filepath` = WAV mono 16 kHz (ffmpeg) | idem ; absent si la vidéo n'a pas de piste audio (modalité mise à zéro) |
| `Word` | `text`, `start`, `duration` (≥ 0,05 s), `language`, `sentence`, `sentence_char`, `context`, `modality="heard"` | depuis `--words` (faster-whisper, temps absolus) |

Détails du texte (ce que lit réellement l'extracteur `HuggingFaceText`, `contextualized=True`,
`site-packages/neuralset/extractors/text.py`) :
- phrases = fin sur `. ! ? …` ou pause > 1,5 s ; casse identique à `AddText` (`str.capitalize()` par
  phrase : majuscule initiale, reste en minuscules — y compris les noms propres, comme TRIBE) ; un point est
  ajouté aux phrases non ponctuées, visible seulement dans le contexte des mots suivants (comme TRIBE) ;
- `context` = texte depuis le début de la vidéo jusqu'au mot inclus (fenêtre de 1024 mots) ; l'extracteur
  prend les tokens finaux du contexte correspondant au mot (`n_tokens(contexte) − n_tokens(préfixe)`) et la
  couche à 2/3 de profondeur de LLaMA ;
- les mots faster-whisper commençant par `'`, `’` ou `-` (« d » + « 'étudier », « abonne » + « -toi ») sont
  recollés au précédent dans le contexte pour que LLaMA lise un français naturel ;
- la langue (code ISO de faster-whisper) n'est qu'informative : LLaMA 3.2 est multilingue, mais TRIBE a été
  entraîné uniquement sur des stimuli anglophones ⇒ features d'un Reel français hors distribution
  [SPÉCULATIF sur l'ampleur].

### 1.10 Sémantique de `seconds` (décalage hémodynamique)

- Entraînement : l'extracteur fMRI a `offset: 5.0` et replace chaque échantillon BOLD mesuré à τ au temps
  stimulus τ − 5 s (`TimedArray(start=ta.start - self.offset)`) ; l'article : « we offset the fMRI timeseries
  by 5 seconds relative to the stimuli » (pour prédire le BOLD sur [0, T], le modèle voit les stimuli sur
  [−5, T−5]). — `config.yaml` (`neuro.offset: 5.0`) ; `site-packages/neuralset/extractors/neuro.py` ;
  article §5.8 https://arxiv.org/html/2605.04326
- À l'inférence il n'y a pas de fMRI : les segments sont en **temps stimulus**. La ligne `i` de `preds`
  correspond au segment `[t, t+1)` de la vidéo (t = `seconds[i]`) et prédit le BOLD attendu ≈ 5 s plus tard.
  **Aucun décalage supplémentaire** ne doit être appliqué (`weights.yaml > hrf_shift_s: 0`).
- `seconds[i] = floor(segment.start − début vidéo)` ; pour une vidéo de durée D, toutes les secondes
  `0..ceil(D)−1` sont présentes (l'événement `Video` couvre toute la durée, donc aucun segment n'est vide ;
  vérifié sur toutes les vidéos testées). La piste audio dépasse parfois la vidéo de quelques ms (30,016 s
  pour 30,000 s) : TRIBE produit alors une ligne de plus, que le worker supprime.
- Mesure de l'alignement (§1.7 D) [MESURÉ] : pour une salve de parole de 1,8 s commençant à b, la ROI
  auditive culmine à t = b ou b+1, monte dès b−3/b−4 et retombe à b+2. Le pic est donc bien aligné sur le
  stimulus, mais chaque ligne mélange les stimuli d'environ [t−2, t+4] : **résolution effective ±3 s**,
  avec anticipation (Transformer non causal sur la fenêtre de 100 s). Ne pas lire une valeur à la seconde
  près ; lisser/agréger sur 2–3 s pour les recommandations.

---

## 2. fsaverage5 → ROI → 6 indices

### 2.1 Atlas

- **Destrieux** (`nilearn.datasets.fetch_atlas_surf_destrieux`) : `map_left`/`map_right` de 10 242 entiers
  chacun directement sur fsaverage5, 76 libellés (74 régions + `Unknown` + `Medial_wall`), ~170 Ko, sans
  clic de licence. Choisi pour cela. — https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_atlas_surf_destrieux.html ;
  Destrieux et al. 2010, NeuroImage 53(1):1-15, https://doi.org/10.1016/j.neuroimage.2010.06.010
- Les 27 libellés de `config/roi_mapping.yaml` existent tous dans l'atlas (résolution par nom, vérifiée
  [MESURÉ] avec nilearn sur `data/assets/nilearn`). Ordre des sommets : gauche (0..10 241) puis droite,
  comme la sortie TRIBE.
- Alternative non retenue : HCP-MMP1 (Glasser, 360 aires, plus fin pour FEF/IPS/TPJ) via MNE
  (`fetch_hcp_mmp_parcellation`, licence HCP à accepter) ; tribev2 fournit `get_hcp_roi_indices`,
  `summarize_by_roi`. — https://mne.tools/1.5/generated/mne.datasets.fetch_hcp_mmp_parcellation.html ;
  `site-packages/tribev2/utils.py`

### 2.2 ROI (sommets fsaverage5 comptés avec nilearn [MESURÉ])

| ROI | Libellés Destrieux | Sommets G / D |
|---|---|---|
| `ips` | `S_intrapariet_and_P_trans`, `G_parietal_sup` | 534 / 492 |
| `fef` | `S_precentral-sup-part`, `S_front_sup` | 317 / 309 |
| `early_visual` | `S_calcarine`, `G_cuneus`, `Pole_occipital` | 404 / 440 |
| `mt` | `G_occipital_middle` | 132 / 124 |
| `precentral` | `G_precentral` | 244 / 237 |
| `premotor_ventral` | `S_precentral-inf-part`, `G_front_inf-Opercular` | 239 / 208 |
| `sma` | `G_and_S_paracentral` | 160 / 136 |
| `supramarginal` | `G_pariet_inf-Supramar` | 291 / 250 |
| `subcentral` | `G_and_S_subcentral` | 141 / 139 |
| `insula_ant` | `G_insular_short`, `S_circular_insula_ant` | 107 / 107 |
| `dacc` | `G_and_S_cingul-Mid-Ant`, `G_and_S_cingul-Ant` | 285 / 332 |
| `angular` | `G_pariet_inf-Angular` | 171 / 242 |
| `psts` | `S_temporal_sup` | 426 / 479 |
| `dmpfc` | `G_front_sup` | 494 / 494 |
| `frontopolar` | `G_and_S_transv_frontopol` | 38 / 50 |
| `rectus` | `G_rectus` | 77 / 62 |
| `temporal_pole` | `Pole_temporal` | 105 / 117 |
| `ofc` | `G_orbital`, `S_orbital-H_Shaped`, `S_suborbital` | 255 / 271 |
| `subcallosal` | `G_subcallosal` | 14 / 15 |
| `parahippocampal` | `G_oc-temp_med-Parahip` | 137 / 131 |
| `retrosplenial` | `G_cingul-Post-ventral` | 26 / 41 |
| `pcc` | `G_cingul-Post-dorsal` | 66 / 50 |
| `precuneus` | `G_precuneus`, `S_subparietal` | 307 / 376 |
| `ifg` | `G_front_inf-Triangul`, `G_front_inf-Opercular` | 187 / 156 |
| `fusiform` | `G_oc-temp_lat-fusifor` | 113 / 110 |
| `mfg` | `G_front_middle` | 269 / 262 |
| `auditory` (diagnostic) | `G_temp_sup-G_T_transv`, `S_temporal_transverse`, `G_temp_sup-Plan_tempo` | 155 / 122 |

Remarques : `subcallosal` (29 sommets), `retrosplenial` (67) et `frontopolar` (88) sont petites ; la sortie
TRIBE étant spatialement lisse, leur moyenne reste stable mais elles apportent peu d'information propre
[SPÉCULATIF]. `G_front_inf-Opercular` est partagé entre `premotor_ventral` et `ifg` (autorisé par le schéma).
Anatomie : insula antérieure = `G_insular_short` (+ sillon circulaire antérieur), postérieure =
`G_Ins_lg_and_S_cent_ins`. — https://mail.nmr.mgh.harvard.edu/archives/list/freesurfer@nmr.mgh.harvard.edu/message/4S6G4L5XZQ5LZPBXCNQVC5VWA6HOMBZ5/

### 2.3 Indices (poids de `config/roi_mapping.yaml`)

Mise en garde générale : amygdale, hippocampe, striatum/noyau accumbens et ATV — les structures les plus
directement liées à l'émotion, la mémoire et la récompense — **ne sont pas sur la surface corticale** ;
TRIBE v2 a un modèle sous-cortical séparé (`facebook/tribev2-subcortical`) aux scores d'encodage 2 à 3 fois
plus faibles, non utilisé ici. Chaque indice repose donc sur des **proxys corticaux**.
— https://arxiv.org/html/2605.04326 ; https://github.com/facebookresearch/tribev2/issues/23

| Indice | ROI (poids) | Confiance | Base | Statut |
|---|---|---|---|---|
| **attention** | ips 1.0, fef 0.8, mt 0.6, early_visual 0.5 | medium | réseau attentionnel dorsal IPS/FEF (Corbetta & Shulman 2002) ; synchronie visuo-pariétale et engagement (Bezdek et al. 2015 ; Song, Finn & Rosenberg 2021) | lien IPS/FEF ↔ attention établi ; « attention du spectateur d'un Reel » = extrapolation [SPÉCULATIF] |
| **action** (signal d'action) | premotor_ventral 1.0, precentral 0.8, sma 0.8, supramarginal 0.5, subcentral 0.3 | low | observation/préparation de l'action (Caspers et al. 2010 ; Nachev et al. 2008) | « envie d'agir / d'acheter » [SPÉCULATIF] : le signal validé en neuromarketing est le striatum ventral (Venkatraman et al. 2015), sous-cortical |
| **tension** | insula_ant 1.0, dacc 1.0 | medium | réseau de saillance (Seeley et al. 2007) ; suspense/arousal (Nummenmaa et al. 2012 ; Lehne et al. 2015) | |
| **manipulation** | angular 0.8, psts 0.8, dmpfc 0.7, frontopolar 0.6, temporal_pole 0.6, supramarginal 0.4, rectus 0.4 | low | mentalisation/ToM (Schurz et al. 2014) ; persuasion (Falk et al. 2010, 2012) | indexe un traitement socio-cognitif, **pas** le fait d'être manipulé [SPÉCULATIF] ; Destrieux ne sépare pas dmPFC médian/latéral |
| **emotion** | ofc 1.0, insula_ant 0.7, subcallosal 0.6, rectus 0.5, temporal_pole 0.5 | low | valeur affective OFC (Chikazoe et al. 2014 ; Kringelbach 2005) ; méta-analyse émotion (Lindquist et al. 2012) | intensité, pas valence [SPÉCULATIF] |
| **memory** | parahippocampal 1.0, ifg 0.8, fusiform 0.6, mfg 0.5, retrosplenial 0.4, angular 0.3, precuneus 0.2, pcc 0.2 | low | encodage mnésique (Ranganath & Ritchey 2012 ; Kim 2011) ; synchronie naturaliste et mémorisation (Hasson et al. 2008 ; Chen et al. 2017) | DMN ↑ prédit l'oubli (Kim 2011) vs synchronie ⇒ précuneus/PCC faiblement pondérés [SPÉCULATIF] |

Sources des indices :
Corbetta & Shulman 2002 https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:11994752%20AND%20SRC:MED&resultType=core&format=json ·
Bezdek et al. 2015 https://control.gatech.edu/wp-content/uploads/pubs/Bezdek-et-al-2015-Neuroscience.pdf ·
Song, Finn & Rosenberg 2021 https://pmc.ncbi.nlm.nih.gov/articles/PMC8379980 ·
Caspers et al. 2010 https://neurovault.org/collections/824/ ·
Nachev et al. 2008 https://doi.org/10.1038/nrn2478 ·
Venkatraman et al. 2015 https://stanford.edu/~knutson/nfc/venkatraman15.pdf ·
Seeley et al. 2007 https://pubmed.ncbi.nlm.nih.gov/17329432/ ·
Nummenmaa et al. 2012 https://pmc.ncbi.nlm.nih.gov/articles/PMC3386135 ·
Lehne et al. 2015 https://pmc.ncbi.nlm.nih.gov/articles/PMC4422438 ·
Schurz et al. 2014 https://www.ndcn.ox.ac.uk/publications/723944/modal ·
Falk et al. 2010 https://pmc.ncbi.nlm.nih.gov/articles/PMC3027351 ·
Falk et al. 2012 https://repository.upenn.edu/asc_papers/448/ ·
Chikazoe et al. 2014 https://pmc.ncbi.nlm.nih.gov/articles/PMC4317366 ·
Kringelbach 2005 https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:16136173%20AND%20SRC:MED&resultType=core&format=json ·
Lindquist et al. 2012 https://canlab.yale.edu/sites/default/files/lindquist_2012_meta.pdf ·
Ranganath & Ritchey 2012 https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:22992647%20AND%20SRC:MED&resultType=core&format=json ·
Kim 2011 https://doi.org/10.1016/j.neuroimage.2010.09.045 ·
Hasson et al. 2008 https://pmc.ncbi.nlm.nih.gov/articles/PMC2789242 ·
Chen et al. 2017 https://pmc.ncbi.nlm.nih.gov/articles/PMC5191958

Ce que TRIBE prédit bien : v1 — meilleures performances dans les cortex auditif et du langage, gains
multimodaux en préfrontal et à la jonction pariéto-occipito-temporale (https://arxiv.org/pdf/2507.22229) ;
v2 — plus grands gains multimodaux autour de la jonction temporo-pariéto-occipitale, la vidéo domine
occipital/pariétal, l'audio le cortex auditif, le texte le langage et une grande partie du préfrontal
(§2.8 https://arxiv.org/html/2605.04326).

---

## 3. Normalisation, décalage hémodynamique, limites

- **Niveau absolu** : les prédictions sont des z « de session » d'un sujet moyen, de faible amplitude
  (§1.3). Re-z-scorer à l'intérieur d'un Reel de 15–90 s efface le niveau moyen : pour comparer des vidéos,
  utiliser une référence de corpus (médiane/MAD par ROI sur plusieurs Reels — `data/reference.json`) ; le
  z intra-vidéo ne sert qu'à la dynamique. — article §5.8 https://arxiv.org/html/2605.04326 ; recommandation
  du rapport de planification.
- **Décalage HRF** : déjà absorbé par TRIBE (§1.10) ; ne pas re-décaler. La réponse hémodynamique
  canonique culmine vers 5–6 s avec une largeur à mi-hauteur ≈ 5 s
  (https://nipy.org/nipy/api/generated/nipy.modalities.fmri.hrf.html) et sa latence varie selon les régions
  (https://arxiv.org/pdf/1402.7015 ; https://www.biostat.jhsph.edu/~mlindqui/Papers/HRF_NI2009.pdf) : la
  résolution temporelle effective est de l'ordre de **±2–3 s**, pas 1 s. Pas de déconvolution (une HRF
  explicite a dégradé les performances dans Algonauts 2025, https://arxiv.org/html/2508.10784).
- **Début et fin de vidéo** : les 5 premières secondes n'ont pas de contexte antérieur (TRIBE a un contexte
  de 100 s) et coïncident avec un début de fenêtre ⇒ valeurs moins fiables (le pipeline les marque
  `low_context`). Sur les vidéos de test, la ROI auditive part de −0,18 à −0,30 à t = 0 et remonte sur ~4 s,
  en silence, et remonte aussi sur les ~3 dernières secondes [MESURÉ] — artefacts de bord probables (début
  de fenêtre / zéros après la fin du stimulus) [SPÉCULATIF sur la cause]. Il serait prudent d'étendre le
  marquage aux 3 dernières secondes.
- **Format Reel** : le préprocesseur V-JEPA 2 redimensionne le petit côté à 292 px puis recadre au centre
  256×256 (`video_preprocessor_config.json`, `size.shortest_edge: 292`, crop 256 [MESURÉ]) : sur une vidéo
  verticale 9:16, **le haut et le bas de l'image (souvent les sous-titres et CTA) sont coupés** pour la
  modalité vidéo. Les tokens sont ensuite moyennés spatialement (article §5.2), ce qui efface l'information
  rétinotopique.
- **Hors distribution** : entraînement sur des films/séries/podcasts anglophones (Algonauts 2025/Friends,
  Lahner 2024, Lebel 2023, Wen 2017 : `config.yaml > data.study.names`) ; Reels verticaux, montage très
  rapide, sous-titres incrustés, français ⇒ généralisation non garantie [SPÉCULATIF].
- **Sujet moyen** : une seule prédiction moyenne ; pas de variabilité inter-individuelle
  (https://github.com/facebookresearch/tribev2/issues/62).

---

## 4. Ce qui est spéculatif (à lire avant d'interpréter un score)

1. **Toute la chaîne « activité corticale → indice → viralité » est spéculative.** TRIBE prédit une
   activité BOLD moyenne ; aucun travail cité ne valide qu'une ROI corticale prédite par TRIBE prédise la
   rétention ou le partage d'un Reel. [SPÉCULATIF]
2. Les noms d'indices (« attention », « signal d'action », « tension », « manipulation », « émotion »,
   « mémoire ») sont des étiquettes interprétatives de proxys corticaux ; « manipulation » mesure au mieux un
   traitement social/mentalisant, « action » une activité prémotrice, pas une intention d'achat. [SPÉCULATIF]
3. Les poids ROI → indice sont des choix d'expert non ajustés sur données. [SPÉCULATIF]
4. Les structures sous-corticales clés (amygdale, hippocampe, striatum) sont absentes. [établi]
5. Performance de tribev2-mini vs complet, effet du français et du format vertical : non mesurés
   [NON VÉRIFIÉ] ; l'impact de `--no-text` n'est mesuré que comme écart entre deux prédictions (pas contre
   de vraies données fMRI).
6. Résolution temporelle réelle ±3 s (mesurée) ; les ~5 premières et ~3 dernières secondes sont peu
   fiables. [MESURÉ sur vidéos de test / cause SPÉCULATIVE]
7. Usage non commercial uniquement (CC BY-NC 4.0). [établi]
