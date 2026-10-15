"""Analyse d'un Reel en ligne de commande.

uv run python -m reel_predictor.cli video.mp4 [--out DIR] [--brain tribe|heuristic|mock]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from . import paths


def _bar(pct: float, width: int = 28) -> str:
    n = int(round(pct * width))
    return "█" * n + "·" * (width - n)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reel_predictor", description="Analyse un Reel (.mp4) seconde par seconde.")
    ap.add_argument("video", type=Path)
    ap.add_argument("--out", type=Path, default=None, help="dossier de sortie (défaut : data/results/cli_<nom>_<date>)")
    ap.add_argument("--brain", choices=["tribe", "heuristic", "mock"], default=None,
                    help="fournisseur cérébral (défaut : config/weights.yaml > brain.provider)")
    ap.add_argument("--config-dir", type=Path, default=paths.CONFIG_DIR)
    ap.add_argument("--quiet", action="store_true", help="pas de barre de progression")
    a = ap.parse_args(argv)

    from .config import load_config
    from .media import VideoUnreadableError
    from .pipeline import analyze

    cfg = load_config(a.config_dir)
    if a.brain:
        cfg = cfg.with_brain_provider(a.brain)
    out = a.out or paths.RESULTS_DIR / f"cli_{a.video.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    video = out / "video.mp4"
    if a.video.resolve() != video.resolve():
        shutil.copyfile(a.video, video)

    t0 = time.time()
    last = {"stage": None}

    def progress(stage: str, pct: float, msg: str) -> None:
        if a.quiet:
            return
        end = "\n" if stage != last["stage"] and last["stage"] is not None else ""
        last["stage"] = stage
        sys.stderr.write(f"{end}\r[{_bar(pct)}] {pct * 100:5.1f}%  {stage:<8} {msg[:60]:<60}")
        sys.stderr.flush()

    try:
        res = analyze(video, out, cfg, progress=progress)
    except VideoUnreadableError as e:
        sys.stderr.write(f"\nVidéo illisible : {e}\n")
        return 2
    sys.stderr.write("\n")
    s = res.summary
    b = s["brain"]
    print(f"\n=== {a.video.name} — {s['video']['duration_s']} s — {s['video']['width']}x{s['video']['height']} ===")
    print(f"Score : {s['score']}/100   (base {s['score_breakdown']['base']}, accroche +{s['score_breakdown']['hook_bonus']},"
          f" creux {s['score_breakdown']['dip_penalty']})")
    r = s["retention"]
    print(f"Rétention prédite : visionnage moyen {r['avg_watch_s']} s ({r['avg_watch_ratio'] * 100:.0f} %),"
          f" complétion {r['completion'] * 100:.0f} %")
    print("Moyennes des indices : " + ", ".join(f"{k} {v:.2f}" for k, v in s["score_breakdown"]["index_means"].items()))
    print(f"Cerveau : {b['provider']}" + (f" ({b['model']})" if b["model"] else "")
          + (f" — dégradé : {b['reason']}" if b["degraded"] else ""))
    print(f"Cuts ({len(s['cuts'])}) : {', '.join(f'{c:.2f}' for c in s['cuts']) or 'aucun'}")
    tr = s["transcript"]
    print(f"Transcription [{tr['language']}] : {tr['text'][:300]}")
    texts = [(row["t"], row["text"]) for row in res.timeline if row["text_new"]]
    if texts:
        print("Texte à l'écran : " + " | ".join(f"{t}s « {x[:50]} »" for t, x in texts[:8]))
    print(f"\nRecommandations ({len(s['recommendations'])}) :")
    for rec in s["recommendations"]:
        print(f"  [{rec['severity']:<6}] {rec['t']:>5.1f}–{rec['t_end']:<5.1f} s  {rec['message']}")
    if s.get("warnings"):
        print("\nAvertissements :")
        for w in s["warnings"]:
            print(f"  - {w}")
    print("\nTemps par étape (s) : " + ", ".join(f"{k} {v:.2f}" for k, v in s["timings"].items()))
    print(f"Résultats : {out}  ({time.time() - t0:.1f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
