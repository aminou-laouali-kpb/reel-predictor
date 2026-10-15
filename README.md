# Reel Predictor

Application **100 % locale** (macOS, Apple Silicon) pour analyser un Reel Instagram (`.mp4`)
**seconde par seconde avant de le publier**, puis comparer la prédiction avec les **vraies
statistiques Instagram** une fois le reel en ligne.

- **Avant publication** : déposez la vidéo. L'app combine un modèle cérébral (**TRIBE v2**, Meta FAIR,
  qui prédit l'activité corticale à partir de la vidéo, du son et du texte) et des signaux vidéo
  (cuts, visages, texte à l'écran, parole, audio). Vous obtenez :
  - 6 indices seconde par seconde : attention, signal d'action, tension, manipulation, émotion, mémoire ;
  - une courbe de rétention prédite et un score sur 100 ;
  - une carte cérébrale animée synchronisée avec la vidéo ;
  - des recommandations datées (« couper / ajouter un texte à 7 s »), cliquables ;
  - la transcription.
- **Après publication** : déposez jusqu'à 3 captures d'écran Instagram Insights (48 h ou 7 jours).
  Les chiffres sont lus par OCR local, **vous les vérifiez à la main**, puis l'app compare
  prédit vs réel et peut **recalibrer** le modèle de rétention (à partir de 3 reels renseignés).

> Les indices sont des **estimations exploratoires**, pas des mesures : le lien entre activité
> cérébrale prédite et comportement des spectateurs reste incertain (niveau de confiance affiché par indice).

> **Licence** : code sous [PolyForm Noncommercial 1.0.0](LICENSE) ; le modèle cérébral TRIBE v2 (Meta FAIR) est sous
> **CC BY-NC 4.0**. → **Usage personnel, éducatif ou de recherche uniquement, pas d'usage commercial** (ni vente,
> ni analyses pour des clients, ni installation payante). Détails : [NOTICE](NOTICE). *Built with Llama.*

## Ce qu'il vous faut

| | Minimum |
|---|---|
| Ordinateur | **Mac avec puce Apple** (M1, M2, M3, M4, M5…). Les Mac Intel, Windows et Linux ne sont pas pris en charge. |
| Mémoire | **16 Go** de RAM conseillés (pic mesuré ≈ 9 Go pendant l'analyse) |
| Disque | ≈ **15 Go** libres (dépendances + modèles) |
| Réseau | Uniquement à l'installation (≈ 10 Go de téléchargements). Les analyses tournent hors ligne. |
| Comptes | Un compte **Hugging Face** gratuit (pour le modèle Llama 3.2) ; une clé **API Anthropic** seulement si vous voulez le coach (payant, optionnel) |

Temps d'analyse : ≈ 10 s de calcul par seconde de vidéo (≈ 5 min pour un Reel de 30 s).

## Installation pas à pas (de A à Z)

Toutes les commandes se tapent dans l'app **Terminal** (Applications → Utilitaires → Terminal).

**1. Outils de développement d'Apple** (si ce n'est pas déjà fait ; une fenêtre s'ouvre, cliquez « Installer ») :

```bash
xcode-select --install
```

**2. Homebrew** (gestionnaire de paquets pour Mac ; suivez les instructions affichées à la fin, notamment
les deux lignes `echo … >> ~/.zprofile` / `eval …` à copier-coller) :

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**3. uv, ffmpeg et git** :

```bash
brew install uv ffmpeg git
```

**4. Accès au modèle Llama 3.2** (utilisé par TRIBE v2 pour lire le texte et la parole) :

1. créez un compte sur <https://huggingface.co/join> ;
2. ouvrez <https://huggingface.co/meta-llama/Llama-3.2-1B>, remplissez le formulaire et acceptez la licence
   (validation par Meta : de quelques minutes à quelques heures, vous recevez un e-mail) ;
3. créez un jeton **Read** sur <https://huggingface.co/settings/tokens> (gardez-le pour l'étape 6).

Sans cet accès l'app fonctionne quand même, en mode dégradé (TRIBE sans texte), signalé dans l'interface.

**5. Télécharger Reel Predictor** :

```bash
cd ~/Documents
git clone https://github.com/aminou-laouali-kpb/reel-predictor.git
cd reel-predictor
```

**6. Installer et se connecter à Hugging Face** (collez le jeton de l'étape 4 quand il est demandé ; il n'est pas affiché) :

```bash
make setup
uv run hf auth login
uv run python scripts/check_hf_access.py
```

**7. Pré-télécharger les modèles** (≈ 10 Go, une seule fois ; sinon ils se téléchargent à la première analyse) :

```bash
make models
```

**8. Lancer l'app** (le navigateur s'ouvre sur <http://127.0.0.1:8765>) :

```bash
make run
```

Pour arrêter : `Ctrl + C` dans le Terminal. Pour relancer plus tard : `cd ~/Documents/reel-predictor` puis `make run`.

**9. (Optionnel) Coach script & montage** : créez une clé sur <https://console.anthropic.com> (facturation à
l'usage, ≈ 0,25–0,50 $ par génération), puis rangez-la dans le trousseau macOS (saisie masquée) :

```bash
make api-key
```

Seul du **texte** (transcription, textes lus à l'écran, indices) est envoyé à l'API Claude, jamais la vidéo.

### Mettre à jour / désinstaller

```bash
cd ~/Documents/reel-predictor && git pull && make setup     # mise à jour
rm -rf ~/Documents/reel-predictor ~/.cache/huggingface        # désinstallation (supprime aussi vos analyses)
```

### En cas de problème

| Symptôme | Solution |
|---|---|
| `command not found: brew` | Refaites les deux lignes affichées à la fin de l'installation de Homebrew, puis rouvrez le Terminal. |
| « TRIBE v2 (sans texte) » dans l'app | L'accès Llama 3.2 n'est pas encore validé ou le jeton manque : étapes 4 et 6. |
| Analyse en « mode heuristique » | Mémoire insuffisante ou modèle absent : fermez les autres apps, relancez `make models`. |
| Le cerveau 3D n'apparaît pas | WebGL désactivé dans le navigateur : l'app bascule sur la carte « À plat ». Essayez Chrome ou Safari. |
| Port déjà utilisé | `REEL_PREDICTOR_PORT=8766 make run` |

## Commandes

| Commande | Effet |
|---|---|
| `make run` | (défaut) lance l'app sur `http://127.0.0.1:8765` et ouvre le navigateur (installe d'abord si besoin) |
| `make setup` | installe les dépendances (`uv sync`, worker TRIBE), télécharge uPlot, génère les maillages cérébraux, télécharge les modèles légers (visage, atlas, transcription), vérifie l'accès Hugging Face |
| `make models` | pré-télécharge TRIBE v2 (`facebook/tribev2-mini`) et ses backbones |
| `make api-key` | range la clé API Anthropic dans le trousseau macOS (coach, optionnel) |
| `make test` | lance les tests (`pytest`) |
| `make demo` | génère une vidéo synthétique et l'analyse en ligne de commande |
| `make clean` | supprime résultats, envois et caches (**jamais** `data/samples/`) |
| `make help` | liste des commandes |

Variables d'environnement utiles :

- `REEL_PREDICTOR_FAKE=1 make run` : analyses **factices** (instantanées) pour essayer l'interface sans le pipeline ;
- `REEL_PREDICTOR_NO_BROWSER=1` : ne pas ouvrir le navigateur ;
- `REEL_PREDICTOR_PORT=8766` : autre port (toujours sur `127.0.0.1` uniquement).

Le réseau n'est utilisé qu'au premier téléchargement des modèles (Hugging Face, MediaPipe, nilearn, uPlot) et,
si vous l'activez, par le coach (API Claude). Le serveur n'écoute que sur `127.0.0.1`.

## Où sont les données

| Emplacement | Contenu |
|---|---|
| `data/results/<job_id>/` | une analyse : `video.mp4`, `summary.json`, `timeline.jsonl`, `brain.u8` + `brain_meta.json` (carte cérébrale), `app_job.json` (nom d'origine) |
| `data/reel_predictor.sqlite` | reels analysés et statistiques Instagram validées |
| `data/calibration.json` | paramètres de rétention recalibrés (créé par « Recalibrer la rétention ») |
| `data/samples/` | vos vidéos / captures d'exemple (jamais effacé par `make clean`) |

Le dossier `data/` n'est jamais versionné (`.gitignore`) : vos vidéos et analyses restent sur votre Mac.
| `data/assets/`, `data/cache/` | modèles légers, caches |
| `~/.cache/huggingface/` | modèles Hugging Face (TRIBE v2, Llama 3.2, faster-whisper…) |
| `app/static/vendor/`, `app/static/brain_mesh.json`, `app/static/brain3d.json` | uPlot, maillage à plat et cerveau 3D (générés par `make setup`) |

`job_id` = `AAAAMMJJ-HHMMSS-<8 premiers caractères du sha256 de la vidéo>`. Les résultats restent
consultables après un redémarrage (liste « Analyses précédentes »).

## Architecture (résumé)

- `app/` : serveur FastAPI (`server.py`, API documentée dans `docs/CONTRACTS.md` §5 et sur `/api/docs`),
  file d'attente à une analyse à la fois (`jobs.py`), base SQLite (`db.py`), OCR des captures Insights
  (`insights_ocr.py`, RapidOCR), interface HTML/JS sans dépendance réseau (`static/`, graphiques uPlot,
  carte cérébrale en Canvas 2D).
- `reel_predictor/` : pipeline d'analyse (signaux vidéo, indices, rétention, règles, calibration).
- `tribe_worker/` : worker isolé (venv séparé) qui exécute TRIBE v2.
- `config/` : poids, règles et correspondance ROI → indices (YAML commentés).

## Licences

- **Code de ce dépôt** : [PolyForm Noncommercial 1.0.0](LICENSE) — utilisation, modification et partage libres
  pour tout usage **non commercial** ; tout usage commercial nécessite l'accord de l'auteur.
- **TRIBE v2** (Meta FAIR) : **CC BY-NC 4.0** → **usage non commercial uniquement**. Cette restriction
  s'applique à toute utilisation des prédictions produites par l'app.
- **Llama 3.2** (backbone texte de TRIBE) : **Llama 3.2 Community License** (acceptation requise sur Hugging Face). *Built with Llama.*
- Autres composants (V-JEPA 2, w2v-BERT, faster-whisper, MediaPipe, RapidOCR, PySceneDetect, nilearn, uPlot…) :
  licences permissives, listées dans [NOTICE](NOTICE). Aucun poids de modèle n'est inclus dans ce dépôt.
- Projet indépendant, ni affilié à Meta ni approuvé par Meta.

## Ce qui marche / ce qui ne marche pas

État au 11/10/2026, mesuré sur Apple M5 16 Go. Détails et sources : `docs/RESEARCH.md`.

**Marche (vérifié)**

- `make run` installe, démarre le serveur et ouvre l'app ; 153 tests verts (`make test`).
- Analyse complète d'un vrai Reel (32 s, 1080×1920, 60 i/s) via l'app avec TRIBE v2-mini + Llama 3.2-1B
  sur MPS : **≈ 5 min 20 s au total** (signaux vidéo 12 s, modèle cérébral 308 s), pic mémoire ≈ 8,9 Go.
  Compter **≈ 10 s de calcul par seconde de vidéo** (V-JEPA 2 ≈ 93 % du temps). Ré-analyse d'une même vidéo : ~15 s (cache).
- Sans le modèle cérébral (mode heuristique) : 12 à 25 s par Reel.
- Cuts, transcription FR horodatée (faster-whisper), visages (MediaPipe), texte à l'écran (RapidOCR),
  6 indices, score 0-100, rétention prédite, recommandations horodatées cliquables.
- Lecteur synchronisé avec le **cerveau 3D interactif** (WebGL, surface piale fsaverage5 : glisser pour tourner, molette
  pour zoomer, survol = nom de la région et indices qu'elle alimente, boutons « zones qui alimentent » un indice ; animation
  continue entre deux secondes) ou la carte à plat (4 vues), et les courbes. Le cerveau s'anime aussi en accueil et pendant l'analyse.
- Replis : sans accès Llama → TRIBE sans texte (`--no-text`) ; TRIBE en échec / mémoire / délai → mode
  heuristique signalé dans l'UI avec la raison.
- Onglet « Après publication » : OCR FR/EN des captures Insights (10/10 champs sur captures synthétiques
  claires et sombres), validation manuelle, SQLite, courbe prédit vs réel, recalibration de la rétention (≥ 3 reels).

**Coach script & montage** (API Claude, payant ; seul du texte quitte le Mac)

- `make api-key` range la clé Anthropic dans le trousseau macOS (jamais en clair).
- Claude Opus 5.5 produit un diagnostic, un plan de montage horodaté de la vidéo actuelle et 3 variantes de script ;
  une 2ᵉ passe corrige la meilleure variante en priorité sur les indices qui **baissent** vs l'original
  (`config/coach.yaml > rounds`). Coût réel mesuré : 0,26 $ pour 1 passe (≈ 0,40–0,50 $ estimé pour 2 passes).
- Chaque variante est simulée localement (voix de synthèse macOS + TRIBE texte + audio, sans image : ≈ 0,5 s de
  calcul par seconde, ~1 min 20 pour 4 scripts) et comparée à l'original **re-voicé par la même voix**.
- Cibles **relatives à vos Reels** (`targets.mode: relative`) : un moment clé est atteint quand l'indice égale votre
  90ᵉ percentile ; converties à l'échelle de la simulation par correspondance de quantiles avec l'original.
  Avec seulement 2 Reels analysés, ces cibles sont encore instables.

**Ne marche pas / non vérifié**

- **Aucune validation prédictive** : ni les indices, ni le score, ni la rétention n'ont été comparés à de
  vraies performances. Les pondérations (`config/weights.yaml`) sont des valeurs de départ.
- Lien ROI → indice **spéculatif** pour action, manipulation, émotion, mémoire (structures sous-corticales absentes
  de la surface corticale) ; TRIBE prédit un « sujet moyen » entraîné sur films/podcasts, pas sur des Reels.
- Le score récompense en partie le **nombre de coupes et de nouveaux textes** (boosters heuristiques, y compris en
  simulation où TRIBE ne voit pas l'image) : des écarts < ~10 points entre deux scripts sont du bruit ; vérifier sur
  plusieurs voix (`LAB_VOICE`/`LAB_RATE`) et trancher en conditions réelles (Instagram Trial Reels).
- V-JEPA 2 recadre l'image au centre en carré : le haut et le bas d'un Reel 9:16 ne sont pas vus par le modèle cérébral.
- Prédiction cérébrale floue d'environ ±3 s ; 5 premières et 3 dernières secondes peu fiables (signalées).
- OCR Insights jamais testé sur de vraies captures Instagram ; « Interactions » lu mais non stocké.
- OCR du texte à l'écran : mots parfois collés (« TUPEUXFAIRE ») ; Whisper déforme certains noms propres.
- Modèle complet `facebook/tribev2` (Llama 3B, ViT-g) non testé : risque de dépassement mémoire sur 16 Go.
