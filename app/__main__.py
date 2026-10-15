"""`uv run python -m app` : lance le serveur local et ouvre le navigateur quand il est prêt.

Variables d'environnement :
  REEL_PREDICTOR_NO_BROWSER=1  ne pas ouvrir le navigateur
  REEL_PREDICTOR_FAKE=1        analyses factices (développement de l'UI, sans pipeline)
  REEL_PREDICTOR_PORT=8765     port d'écoute (127.0.0.1 uniquement)
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser

from .server import HOST, PORT


def _port_busy(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def _is_our_server(url: str) -> bool:
    try:
        with urllib.request.urlopen(url + "/api/health", timeout=2) as r:
            return bool(json.load(r).get("ok"))
    except Exception:
        return False


def _open_when_ready(server, url: str) -> None:
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if getattr(server, "started", False):
            webbrowser.open(url)
            return
        time.sleep(0.2)


def main() -> int:
    import uvicorn

    from .server import create_app

    port = int(os.environ.get("REEL_PREDICTOR_PORT", PORT))
    url = f"http://{HOST}:{port}"
    no_browser = os.environ.get("REEL_PREDICTOR_NO_BROWSER", "") == "1"

    if _port_busy(HOST, port):
        if _is_our_server(url):
            print(f"Reel Predictor tourne déjà sur {url}")
            if not no_browser:
                webbrowser.open(url)
            return 0
        print(f"Le port {port} est déjà utilisé par une autre application. "
              f"Libérez-le ou lancez avec REEL_PREDICTOR_PORT=<autre port>.", file=sys.stderr)
        return 1

    if os.environ.get("REEL_PREDICTOR_FAKE", "") == "1":
        print("Mode FACTICE actif (REEL_PREDICTOR_FAKE=1) : les analyses sont simulées.")
    config = uvicorn.Config(create_app(), host=HOST, port=port,
                            log_level=os.environ.get("REEL_PREDICTOR_LOG", "info"))
    server = uvicorn.Server(config)
    if not no_browser:
        threading.Thread(target=_open_when_ready, args=(server, url), daemon=True).start()
    print(f"Reel Predictor : {url}  (Ctrl+C pour arrêter)")
    server.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
