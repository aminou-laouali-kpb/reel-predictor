# Reel Predictor — commandes usuelles (GNU make 3.81 compatible).
# Le chemin du projet peut contenir des espaces : toutes les commandes utilisent des
# chemins relatifs à ce Makefile, et les chemins absolus sont entre guillemets.
#
#   make            = make run
#   make help       liste des commandes

SHELL := /bin/bash
UV ?= uv
STAMP := .venv/.ready
TRIBE_PY := tribe_worker/.venv/bin/python
TRIBE_MODEL ?= facebook/tribev2-mini

# Première cible = cible par défaut
run: $(STAMP)
	$(UV) run python -m app

$(STAMP): pyproject.toml uv.lock tribe_worker/pyproject.toml
	@$(MAKE) --no-print-directory setup

setup:
	$(UV) sync
	$(UV) sync --project tribe_worker
	$(UV) run python scripts/fetch_assets.py
	$(UV) run python scripts/check_hf_access.py
	@mkdir -p .venv && touch "$(STAMP)"
	@echo "Installation terminée. Lancez : make run"

models: $(STAMP)
	@test -x "$(TRIBE_PY)" || { echo "Worker TRIBE non installé : lancez make setup"; exit 1; }
	"$(TRIBE_PY)" tribe_worker/run_tribe.py --download-only --model "$(TRIBE_MODEL)"

test: $(STAMP)
	$(UV) run pytest -q

demo: $(STAMP)
	$(UV) run python scripts/make_samples.py
	$(UV) run python -m reel_predictor.cli data/samples/demo_synthetic.mp4

# Supprime résultats, envois et caches. Ne touche JAMAIS à data/samples/ (vos vidéos et captures),
# ni à data/assets/ (modèles), ni à la base data/reel_predictor.sqlite (statistiques validées).
clean:
	@echo "Suppression des résultats, envois et caches (data/samples conservé)…"
	@find data/results data/uploads data/cache -mindepth 1 -maxdepth 1 -exec rm -rf {} + 2>/dev/null || true
	@find . -path ./.venv -prune -o -path ./tribe_worker/.venv -prune -o -type d -name __pycache__ -print \
		| while IFS= read -r d; do rm -rf "$$d"; done
	@rm -rf .pytest_cache
	@echo "OK."

# Clé API Anthropic (coach « script & montage ») : saisie MASQUÉE, rangée dans le trousseau macOS
# (service reel-predictor-anthropic). C'est vous qui tapez la clé ; elle n'est jamais affichée ni écrite sur disque.
api-key:
	@command -v security >/dev/null 2>&1 || { echo "Commande « security » introuvable : macOS requis (sinon : export ANTHROPIC_API_KEY=...)."; exit 1; }
	@echo "Collez votre clé API Anthropic (sk-ant-...) quand le trousseau la demande (saisie masquée, deux fois)."
	@# -w en DERNIER argument : `security` demande la clé lui-même, elle n'apparaît jamais dans la liste des processus.
	@if security add-generic-password -U -s reel-predictor-anthropic -a "$$USER" -l "Reel Predictor — clé API Anthropic" -w; then \
		echo "Clé enregistrée dans le trousseau (service reel-predictor-anthropic). Rechargez la page de l'app."; \
	else echo "Échec de l'enregistrement dans le trousseau."; exit 1; fi

help:
	@echo "Reel Predictor — commandes :"
	@echo "  make run     (défaut) lance l'app sur http://127.0.0.1:8765 (installe d'abord si besoin)"
	@echo "  make setup   installe les dépendances, uPlot, le maillage cérébral, les modèles légers ; vérifie l'accès Hugging Face"
	@echo "  make models  pré-télécharge TRIBE v2 ($(TRIBE_MODEL)) et ses backbones"
	@echo "  make test    lance les tests (pytest)"
	@echo "  make demo    génère une vidéo synthétique et l'analyse en ligne de commande"
	@echo "  make clean   supprime résultats / envois / caches (jamais data/samples/)"
	@echo "  make api-key enregistre la clé API Anthropic (coach script & montage) dans le trousseau macOS"
	@echo "Variables utiles : REEL_PREDICTOR_FAKE=1 make run (analyses factices), REEL_PREDICTOR_NO_BROWSER=1"

.PHONY: run setup models test demo clean help api-key
