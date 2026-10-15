"""Vérifie l'accès Hugging Face nécessaire au backbone texte de TRIBE v2 (Llama 3.2, modèle « gated »).

N'affiche JAMAIS le token. Ne bloque jamais : code de sortie toujours 0.
Usage : uv run python scripts/check_hf_access.py
"""
from __future__ import annotations

import os
import sys
import threading

REPO = "meta-llama/Llama-3.2-1B"
TIMEOUT_S = 10.0

INSTRUCTIONS = f"""
Pour activer le backbone texte de TRIBE v2 (sinon l'analyse tourne sans le texte, ou en mode dégradé) :
  1. Créez un compte sur https://huggingface.co puis acceptez la licence du modèle :
     https://huggingface.co/{REPO}   (bouton « Agree and access repository » ; validation parfois en quelques minutes)
  2. Créez un token de type « read » : https://huggingface.co/settings/tokens
  3. Enregistrez-le sur cette machine :   uv run hf auth login
  4. Vérifiez :                            uv run python scripts/check_hf_access.py
"""


def _token_present() -> bool:
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:
        try:
            from huggingface_hub import HfFolder

            return bool(HfFolder.get_token())
        except Exception:
            return False


def _check_access() -> tuple[bool | None, str]:
    """(accès, explication). None = indéterminé (réseau, délai)."""
    result: dict[str, tuple[bool | None, str]] = {}

    def run() -> None:
        try:
            from huggingface_hub import auth_check
            from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
        except ImportError as exc:
            result["r"] = (None, f"huggingface_hub trop ancien pour auth_check ({exc})")
            return
        try:
            auth_check(REPO)
            result["r"] = (True, "accès accordé")
        except GatedRepoError:
            result["r"] = (False, "licence non acceptée (ou demande en attente) pour ce compte")
        except RepositoryNotFoundError:
            result["r"] = (False, "dépôt introuvable avec ce token (token invalide ou licence non acceptée)")
        except Exception as exc:  # réseau, 401, etc. — on n'affiche que le type et un message court
            msg = str(exc).splitlines()[0][:160] if str(exc) else exc.__class__.__name__
            if "401" in msg:
                result["r"] = (False, "token refusé (401) : recréez un token « read »")
            else:
                result["r"] = (None, f"vérification impossible ({exc.__class__.__name__}: {msg})")

    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(int(TIMEOUT_S)))
    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(TIMEOUT_S)
    if "r" not in result:
        return None, f"pas de réponse de huggingface.co en {TIMEOUT_S:.0f} s (hors ligne ?)"
    return result["r"]


def main() -> int:
    print(f"Accès Hugging Face pour {REPO} (backbone texte de TRIBE v2)")
    if not _token_present():
        print("✗ Aucun token Hugging Face trouvé sur cette machine.")
        print(INSTRUCTIONS)
        return 0
    print("✓ Token Hugging Face présent (non affiché).")
    ok, why = _check_access()
    if ok is True:
        print(f"✓ Accès à {REPO} : {why}.")
    elif ok is False:
        print(f"✗ Accès à {REPO} refusé : {why}.")
        print(INSTRUCTIONS)
    else:
        print(f"? Accès à {REPO} non vérifié : {why}.")
        print("  (Non bloquant : l'accès sera retesté au premier lancement de TRIBE.)")
    return 0


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # ne bloque jamais `make setup`
        print(f"Vérification Hugging Face ignorée : {exc.__class__.__name__}")
    sys.exit(0)
