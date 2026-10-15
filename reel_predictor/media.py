"""Accès média via ffmpeg/ffprobe : métadonnées, piste audio WAV, images échantillonnées."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np


class VideoUnreadableError(RuntimeError):
    """La vidéo ne peut pas être lue (fichier absent, corrompu, sans piste vidéo...)."""


@dataclass
class VideoInfo:
    filename: str
    duration_s: float
    fps: float
    width: int
    height: int
    has_audio: bool
    rotation: int = 0

    @property
    def n_seconds(self) -> int:
        """Nombre de secondes de la timeline : t = 0..ceil(durée)-1.
        Tolérance de 50 ms pour ne pas créer une dernière seconde quasi vide (arrondis de conteneur)."""
        return max(1, int(math.ceil(self.duration_s - 0.05)))

    def as_summary(self) -> dict:
        d = asdict(self)
        d.pop("rotation", None)
        d["duration_s"] = round(self.duration_s, 3)
        d["fps"] = round(self.fps, 3)
        return d


def _bin(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        for cand in (f"/opt/homebrew/bin/{name}", f"/usr/local/bin/{name}"):
            if Path(cand).exists():
                return cand
        raise VideoUnreadableError(f"{name} introuvable : installez ffmpeg (brew install ffmpeg)")
    return path


def _rate(s: str | None) -> float:
    if not s or s in ("0/0", "N/A"):
        return 0.0
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            return float(a) / float(b) if float(b) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def probe(path: Path) -> VideoInfo:
    path = Path(path)
    if not path.exists():
        raise VideoUnreadableError(f"Fichier introuvable : {path}")
    cmd = [_bin("ffprobe"), "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoUnreadableError(f"Vidéo illisible : {proc.stderr.strip()[:300]}")
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise VideoUnreadableError("Vidéo illisible (ffprobe)") from e
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"
                  and not s.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise VideoUnreadableError("Aucune piste vidéo dans le fichier")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    duration = 0.0
    for cand in (data.get("format", {}).get("duration"), video.get("duration")):
        try:
            duration = float(cand)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue
    if not duration or duration <= 0 or not math.isfinite(duration):
        raise VideoUnreadableError("Durée de la vidéo inconnue ou nulle")
    fps = _rate(video.get("avg_frame_rate")) or _rate(video.get("r_frame_rate")) or 30.0
    w, h = int(video.get("width") or 0), int(video.get("height") or 0)
    if w <= 0 or h <= 0:
        raise VideoUnreadableError("Dimensions de la vidéo invalides")
    rotation = 0
    try:
        rotation = int(float(video.get("tags", {}).get("rotate", 0)))
    except (TypeError, ValueError):
        rotation = 0
    for sd in video.get("side_data_list", []) or []:
        if "rotation" in sd:
            try:
                rotation = int(float(sd["rotation"]))
            except (TypeError, ValueError):
                pass
    if abs(rotation) % 180 == 90:  # ffmpeg applique la rotation au décodage : dimensions affichées
        w, h = h, w
    return VideoInfo(filename=path.name, duration_s=duration, fps=fps, width=w, height=h,
                     has_audio=has_audio, rotation=rotation)


def extract_audio(path: Path, wav_path: Path, sample_rate: int = 16000) -> Path:
    """Extrait la piste audio en WAV mono PCM 16 bits."""
    cmd = [_bin("ffmpeg"), "-v", "error", "-y", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate),
           "-c:a", "pcm_s16le", "-f", "wav", str(wav_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 or not Path(wav_path).exists():
        raise RuntimeError(f"Extraction audio impossible : {proc.stderr.strip()[:300]}")
    return Path(wav_path)


def read_wav(wav_path: Path) -> tuple[np.ndarray, int]:
    """Lit un WAV PCM 16 bits (mono ou multi-canal) -> (float32 mono [-1, 1], sample_rate)."""
    import wave

    with wave.open(str(wav_path), "rb") as w:
        sr = w.getframerate()
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise RuntimeError(f"WAV non supporté (largeur {width} octets)")
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1)
    return x, sr


@dataclass
class FrameSet:
    """Images échantillonnées à `fps` : fichier i <-> instant i / fps."""
    directory: Path
    fps: float
    files: list[Path]

    def time_of(self, i: int) -> float:
        return i / self.fps

    def index_at(self, t: float) -> int:
        if not self.files:
            return -1
        return int(min(len(self.files) - 1, max(0, round(t * self.fps))))

    def __len__(self) -> int:
        return len(self.files)


def extract_frames(path: Path, out_dir: Path, fps: float, max_width: int = 720) -> FrameSet:
    """Extrait des JPEG à `fps` images/s, réduites à `max_width` px de large au plus."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vf = f"fps={fps:g},scale='min({max_width},iw)':-2"
    cmd = [_bin("ffmpeg"), "-v", "error", "-y", "-i", str(path), "-an", "-vf", vf, "-q:v", "3",
           str(out_dir / "f_%06d.jpg")]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    files = sorted(out_dir.glob("f_*.jpg"))
    if proc.returncode != 0 and not files:
        raise RuntimeError(f"Extraction des images impossible : {proc.stderr.strip()[:300]}")
    return FrameSet(directory=out_dir, fps=fps, files=files)


def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))
