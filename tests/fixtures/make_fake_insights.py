"""Capture d'écran Instagram Insights SYNTHÉTIQUE (PIL) pour tester l'OCR — aucune donnée réelle.

Imite la mise en page FR d'un écran « Statistiques du reel » : libellés à gauche et valeurs à droite,
plus quelques blocs « libellé puis grand nombre en dessous ».

Usage CLI : uv run python tests/fixtures/make_fake_insights.py sortie.png [--dark]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
]
BOLD_CANDIDATES = ["/System/Library/Fonts/Supplemental/Arial Bold.ttf"] + FONT_CANDIDATES

# Valeurs attendues après lecture (unités de l'app : secondes, %, nombres)
EXPECTED = {
    "views": 12300,
    "reach": 8734,
    "watch_time_s": 4800,
    "likes": 1234,
    "comments": 56,
    "shares": 78,
    "saves": 90,
    "follows": 12,
    "skip_rate": 45.2,
    "avg_watch_s": 4,
}


def find_font(candidates: list[str]) -> str | None:
    for c in candidates:
        if Path(c).is_file():
            return c
    return None


def make_insights_png(path: Path | str, dark: bool = False) -> Path:
    regular = find_font(FONT_CANDIDATES)
    bold = find_font(BOLD_CANDIDATES)
    if regular is None:
        raise FileNotFoundError("Aucune police système trouvée (Arial/Helvetica)")
    W, H = 1080, 1900
    bg, fg, muted = ((0, 0, 0), (245, 245, 245), (160, 160, 160)) if dark else ((255, 255, 255), (20, 20, 20), (115, 115, 115))
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(bold or regular, 52)
    f = ImageFont.truetype(regular, 40)
    f_big = ImageFont.truetype(bold or regular, 66)
    f_small = ImageFont.truetype(regular, 32)

    y = 60
    d.text((60, y), "9:41", font=f_small, fill=fg)  # barre d'état, doit être ignorée
    y += 90
    d.text((60, y), "Statistiques du reel", font=f_title, fill=fg)
    y += 80
    d.text((60, y), "11 oct. · 7 derniers jours", font=f_small, fill=muted)
    y += 100
    d.text((60, y), "Aperçu", font=f_title, fill=fg)
    y += 95

    def row(label: str, value: str) -> None:
        nonlocal y
        d.text((60, y), label, font=f, fill=fg)
        tw = d.textlength(value, font=f)
        d.text((W - 60 - tw, y), value, font=f, fill=fg)
        y += 100

    row("Vues", "12,3 k")
    row("Comptes touchés", "8 734")
    row("Durée de visionnage", "1 h 20 min")
    row("Taux de skip", "45,2 %")
    y += 40
    d.text((60, y), "Interactions", font=f_title, fill=fg)
    y += 95
    row("J'aime", "1 234")
    row("Commentaires", "56")
    row("Partages", "78")
    row("Enregistrements", "90")
    row("Abonnements", "12")
    y += 30
    d.text((60, y), "Durée moyenne de visionnage", font=f, fill=fg)
    y += 60
    d.text((60, y), "4 s", font=f_big, fill=fg)
    img.save(path)
    return Path(path)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("out", type=Path)
    p.add_argument("--dark", action="store_true")
    a = p.parse_args(argv)
    print(make_insights_png(a.out, dark=a.dark))
    return 0


if __name__ == "__main__":
    sys.exit(main())
