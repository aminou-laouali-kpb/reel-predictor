"""Transcription mot à mot (faster-whisper, CPU int8, VAD)."""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .config import SpeechConfig

ProgressFn = Callable[[float, str], None]

_MODEL_CACHE: dict[tuple, object] = {}
_LOCK = threading.Lock()


@dataclass
class Word:
    w: str
    start: float
    end: float
    prob: float = 1.0


@dataclass
class Transcript:
    language: str | None
    language_probability: float
    text: str
    words: list[Word] = field(default_factory=list)

    def words_json(self, duration_s: float) -> dict:
        """Format `--words` du worker TRIBE (contrat §1)."""
        return {"language": self.language or "fr", "duration_s": round(float(duration_s), 3),
                "words": [{"w": w.w, "start": round(w.start, 3), "end": round(w.end, 3)} for w in self.words]}


def get_model(cfg: SpeechConfig):
    from faster_whisper import WhisperModel

    key = (cfg.model, cfg.device, cfg.compute_type, cfg.cpu_threads)
    with _LOCK:
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = WhisperModel(cfg.model, device=cfg.device, compute_type=cfg.compute_type,
                                             cpu_threads=cfg.cpu_threads)
        return _MODEL_CACHE[key]


def transcribe(audio: np.ndarray, sample_rate: int, cfg: SpeechConfig,
               progress: ProgressFn | None = None) -> Transcript:
    """`audio` : float32 mono. Renvoie la transcription avec horodatage par mot."""
    if audio.size == 0:
        return Transcript(language=cfg.language, language_probability=0.0, text="")
    if sample_rate != 16000:
        n = int(round(audio.size * 16000 / sample_rate))
        audio = np.interp(np.linspace(0, audio.size - 1, n), np.arange(audio.size), audio).astype(np.float32)
    duration = audio.size / 16000.0
    if progress:
        progress(0.02, "Chargement du modèle de transcription")
    model = get_model(cfg)
    segments, info = model.transcribe(
        audio.astype(np.float32), language=cfg.language, beam_size=cfg.beam_size,
        word_timestamps=True, vad_filter=cfg.vad_filter, condition_on_previous_text=False,
    )
    words: list[Word] = []
    texts: list[str] = []
    for seg in segments:
        texts.append(seg.text.strip())
        for w in seg.words or []:
            token = w.word.strip()
            if not token:
                continue
            words.append(Word(w=token, start=float(w.start), end=float(max(w.end, w.start)),
                              prob=float(getattr(w, "probability", 1.0))))
        if progress and duration > 0:
            progress(min(0.99, 0.05 + 0.95 * seg.end / duration), "Transcription en cours")
    words.sort(key=lambda w: w.start)
    return Transcript(language=info.language, language_probability=float(info.language_probability or 0.0),
                      text=" ".join(t for t in texts if t), words=words)
