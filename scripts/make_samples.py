"""Génère une vidéo synthétique de démonstration : data/samples/demo_synthetic.mp4.

~20 s, verticale 720x1280, 30 i/s, voix française (commande macOS `say -v Thomas`),
4 plans séparés par des cuts nets (3 s, 6 s, 15 s), du texte à l'écran sur certaines secondes
(dessiné avec Pillow : le ffmpeg local n'a pas `drawtext`) et un passage silencieux (~9,5–14 s).

Scénario (sert de vérité terrain aux tests) :
  plans  : A 0–3 s (rouge, mouvement) · B 3–6 s (bleu) · C 6–15 s (gris, quasi statique) · D 15–20 s (vert)
  texte  : 0–3 s « 3 ASTUCES / POUR PERCER » · 4–6 s « ASTUCE 1 : COUPEZ »
           7–9 s paragraphe très long (surcharge) · 16–20 s « ABONNE-TOI / LIEN EN BIO » (CTA)
  voix   : 0,3 s phrase 1 · 7,5 s phrase 2 · silence ~9,5–14 s · 14,2 s phrase 3 (CTA parlé)

Usage : uv run python scripts/make_samples.py [--force]
Ne touche à aucun autre fichier de data/samples/.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "samples" / "demo_synthetic.mp4"

W, H, FPS, DURATION = 720, 1280, 30, 20.0
SR = 44100
FONT_BOLD = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

# (début, fin, couleur RGB de fond, amplitude du mouvement)
SHOTS = [
    (0.0, 3.0, (200, 40, 40), 1.0),
    (3.0, 6.0, (30, 70, 200), 0.6),
    (6.0, 15.0, (110, 110, 110), 0.03),
    (15.0, 20.0, (40, 170, 70), 0.6),
]
LONG_TEXT = ("Voici un très long texte que personne n'aura le temps de lire en entier "
             "car il contient beaucoup trop de mots pour un affichage aussi court "
             "sur un petit écran de téléphone pendant deux secondes")
# (début, fin, lignes, taille de police)
TEXTS = [
    (0.0, 3.0, ["3 ASTUCES", "POUR PERCER"], 84),
    (4.0, 6.0, ["ASTUCE 1 : COUPEZ"], 64),
    (7.0, 9.0, None, 34),  # paragraphe long, découpé automatiquement
    (16.0, 20.0, ["ABONNE-TOI", "LIEN EN BIO"], 84),
]
SPEECH = [
    (0.3, "Voici trois astuces pour percer sur Instagram. Première astuce : coupez vos vidéos "
          "toutes les trois secondes."),
    (7.5, "Deuxième astuce : gardez le rythme."),
    (14.2, "Troisième astuce : abonne-toi et commente pour la suite. Le lien est en bio."),
]


def _wrap(text: str, font: ImageFont.FreeTypeFont, max_w: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if font.getlength(cand) <= max_w:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _text_layers() -> list[tuple[float, float, Image.Image]]:
    layers = []
    for start, end, lines, size in TEXTS:
        font = ImageFont.truetype(FONT_BOLD, size)
        if lines is None:
            lines = _wrap(LONG_TEXT, font, W - 80)
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        line_h = int(size * 1.25)
        y = H // 2 - len(lines) * line_h // 2 - 150
        for line in lines:
            w = d.textlength(line, font=font)
            d.text(((W - w) / 2, y), line, font=font, fill=(255, 255, 255, 255),
                   stroke_width=max(2, size // 14), stroke_fill=(0, 0, 0, 255))
            y += line_h
        layers.append((start, end, img))
    return layers


def _frame(t: float, layers) -> bytes:
    shot = next(s for s in SHOTS if s[0] <= t < s[1]) if t < SHOTS[-1][1] else SHOTS[-1]
    start, _, color, amp = shot
    # Fond : dégradé vertical de la couleur du plan
    grad = np.linspace(0.75, 1.15, H, dtype=np.float32)[:, None, None]
    base = np.clip(np.array(color, np.float32)[None, None, :] * grad, 0, 255)
    frame = np.broadcast_to(base, (H, W, 3)).astype(np.uint8).copy()
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    # Formes en mouvement (amplitude propre au plan)
    local = t - start
    cx = W / 2 + amp * 220 * np.sin(local * 2.2)
    cy = H * 0.72 + amp * 120 * np.cos(local * 1.7)
    r = 90
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(250, 220, 60))
    bx = 60 + (amp * 400 * ((local * 0.35) % 1.0))
    d.rectangle((bx, 150, bx + 160, 260), fill=(20, 20, 20))
    for s, e, layer in layers:
        if s <= t < e:
            img.paste(layer, (0, 0), layer)
    return img.tobytes()


def _speech_track(tmp: Path) -> Path:
    if shutil.which("say") is None:
        raise SystemExit("La commande macOS `say` est introuvable.")
    total = np.zeros(int(SR * DURATION), np.float32)
    for i, (start, text) in enumerate(SPEECH):
        aiff = tmp / f"seg{i}.aiff"
        subprocess.run(["say", "-v", "Thomas", "-o", str(aiff), text], check=True)
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(aiff), "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
            check=True, capture_output=True).stdout
        seg = np.frombuffer(raw, np.float32)
        a = int(start * SR)
        b = min(len(total), a + len(seg))
        total[a:b] += seg[: b - a]
    peak = float(np.max(np.abs(total))) or 1.0
    total = total / peak * 0.8
    wav = tmp / "voice.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((total * 32767).astype("<i2").tobytes())
    return wav


def make(out: Path = OUT, force: bool = False) -> Path:
    if out.exists() and not force:
        print(f"Déjà présent : {out}")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    layers = _text_layers()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        wav = _speech_track(tmp)
        tmp_out = tmp / "out.mp4"
        cmd = ["ffmpeg", "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
               "-i", str(wav),
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
               "-g", "30", "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart",
               str(tmp_out)]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        assert proc.stdin is not None
        n = int(DURATION * FPS)
        for i in range(n):
            proc.stdin.write(_frame(i / FPS, layers))
        proc.stdin.close()
        if proc.wait() != 0:
            raise SystemExit("ffmpeg a échoué")
        shutil.move(str(tmp_out), out)
    print(f"Écrit : {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="régénérer même si le fichier existe")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    make(a.out, a.force)
    sys.exit(0)
