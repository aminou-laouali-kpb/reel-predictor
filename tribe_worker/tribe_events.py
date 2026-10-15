"""Construction du DataFrame d'événements attendu par TRIBE v2, SANS WhisperX ni spaCy.

Pourquoi : `TribeModel.get_events_dataframe(video_path=...)` (tribev2/demo_utils.py) enchaîne
  ExtractAudioFromVideo -> ChunkEvents(Audio/Video, 30-60 s) -> ExtractWordsFromAudio (`uvx whisperx`,
  langue "english" et compute_type "float16" codés en dur, décalage des mots au-delà de 60 s)
  -> AddText / AddSentenceToWords / AddContextToWords (spaCy `*_core_*_lg`, téléchargé à la volée).
Sur Mac (CPU, pas de CUDA) WhisperX plante (float16), la langue est fausse pour un Reel français et
spaCy déclenche un téléchargement réseau. On reproduit donc le même schéma d'événements à partir de
la transcription faster-whisper déjà calculée par le pipeline (`--words`).

Schéma produit (une seule timeline "default", sujet "default") :
  * `Video` : filepath (lien symbolique adressé par contenu), start/offset/duration, frequency = fps,
    découpé en morceaux de 30-60 s par `ChunkEvents` (identique à TRIBE).
  * `Audio` : WAV mono 16 kHz extrait par ffmpeg (le wav2vec-BERT de TRIBE ré-échantillonne de toute
    façon à 16 kHz et moyenne les canaux), même découpage 30-60 s.
  * `Word`  : text, start, duration, language, sentence, sentence_char, context. `context` = texte
    ponctué depuis le début de la vidéo jusqu'au mot inclus (fenêtre de 1024 mots), avec la même
    casse que `AddText` (`str.capitalize()` par phrase) : c'est exactement ce que lit l'extracteur
    texte (`HuggingFaceText`, contextualized=True), qui prend les derniers tokens du contexte.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

END_PUNCT = (".", "!", "?", "…")
PAUSE_SENTENCE_S = 1.5        # silence au-delà duquel on force une fin de phrase (pas de ponctuation)
MAX_CONTEXT_WORDS = 1024      # = AddContextToWords(max_context_len=1024) dans TRIBE
MIN_WORD_DUR = 0.05
GLUE_PREFIX = ("'", "’", "-")  # mots collés au précédent (élision, trait d'union)
CHUNK_MIN_S, CHUNK_MAX_S = 30.0, 60.0   # = ChunkEvents(min_duration=30, max_duration=60) dans TRIBE

# Codes ISO 639-1 -> nom utilisé par neuralset (champ `language` des Word ; informatif seulement).
ISO_TO_NAME = {"en": "english", "fr": "french", "es": "spanish", "nl": "dutch", "zh": "chinese",
               "de": "german", "it": "italian", "pt": "portuguese"}


class InputError(ValueError):
    """Entrée invalide (code de sortie 2)."""


# --------------------------------------------------------------------------------------------
# Lecture / validation des entrées
# --------------------------------------------------------------------------------------------
def load_words(path: Path) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as e:
        raise InputError(f"fichier --words introuvable : {path}") from e
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise InputError(f"fichier --words illisible (JSON invalide) : {e}") from e
    if not isinstance(data, dict) or not isinstance(data.get("words", []), list):
        raise InputError("--words doit être un objet JSON avec une liste 'words'")
    words = []
    for i, w in enumerate(data.get("words", [])):
        try:
            text = str(w["w"]).strip()
            start, end = float(w["start"]), float(w["end"])
        except (KeyError, TypeError, ValueError) as e:
            raise InputError(f"mot n°{i} invalide dans --words : {w!r}") from e
        if text and start == start and end == end:  # ignore NaN / vides
            words.append({"w": text, "start": start, "end": end})
    lang = str(data.get("language") or "").strip().lower()
    return {"language": lang, "duration_s": data.get("duration_s"), "words": words}


def ffprobe(video: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-show_entries",
           "format=duration:stream=codec_type,avg_frame_rate,width,height", "-of", "json", str(video)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60).stdout
        info = json.loads(out)
    except FileNotFoundError as e:
        raise RuntimeError("ffprobe introuvable (installer ffmpeg)") from e
    except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
        raise InputError(f"vidéo illisible par ffprobe : {video}") from e
    streams = info.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    try:
        duration = float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        duration = 0.0
    if not has_video or duration <= 0:
        raise InputError(f"pas de piste vidéo exploitable dans {video}")
    return {"duration": duration, "has_audio": has_audio}


def ffprobe_audio(path: Path) -> dict:
    """Entrée audio seule (mode `--no-video`) : durée de la première piste audio."""
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type",
           "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60).stdout
        info = json.loads(out)
    except FileNotFoundError as e:
        raise RuntimeError("ffprobe introuvable (installer ffmpeg)") from e
    except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
        raise InputError(f"audio illisible par ffprobe : {path}") from e
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    try:
        duration = float(info["format"]["duration"])
    except (KeyError, TypeError, ValueError):
        duration = 0.0
    if not has_audio or duration <= 0:
        raise InputError(f"pas de piste audio exploitable dans {path}")
    return {"duration": duration, "has_audio": True}


def prepare_audio(audio: Path, media_dir: Path) -> tuple[Path, str]:
    """Mode `--no-video` : `<media_dir>/<sha>.wav` (mono 16 kHz) adressé par contenu (cache de features)."""
    media_dir.mkdir(parents=True, exist_ok=True)
    digest = file_digest(audio)
    wav = media_dir / f"{digest}.audio.wav"
    if not wav.exists():
        part = media_dir / f".{digest}.{os.getpid()}.wav"
        cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", str(audio),
               "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(part)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode != 0 or not part.exists() or part.stat().st_size < 1000:
            part.unlink(missing_ok=True)
            raise InputError(f"conversion audio impossible : {audio} ({r.stderr.strip()[:200]})")
        os.replace(part, wav)
    return wav, digest


# --------------------------------------------------------------------------------------------
# Médias adressés par contenu (le cache de features exca est indexé par chemin de fichier)
# --------------------------------------------------------------------------------------------
def file_digest(path: Path, n_hex: int = 20) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:n_hex]


def prepare_media(video: Path, media_dir: Path, has_audio: bool) -> tuple[Path, Path | None, str]:
    """Crée `<media_dir>/<sha>.mp4` (lien symbolique vers la vidéo) et `<sha>.wav` (mono 16 kHz).

    Le cache de features de TRIBE/neuralset est indexé par le chemin du fichier : passer par un chemin
    dérivé du contenu évite toute collision entre deux vidéos différentes ayant le même nom et permet de
    réutiliser les features si la même vidéo est ré-analysée. Rien n'est écrit à côté de la vidéo.
    """
    media_dir.mkdir(parents=True, exist_ok=True)
    digest = file_digest(video)
    suffix = video.suffix.lower() if video.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".avi"} else ".mp4"
    link = media_dir / f"{digest}{suffix}"
    tmp = media_dir / f".{digest}.{os.getpid()}.lnk"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    os.symlink(video.resolve(), tmp)
    os.replace(tmp, link)  # remplace atomiquement un lien éventuellement cassé
    wav = None
    if has_audio:
        wav = media_dir / f"{digest}.wav"
        if not wav.exists():
            part = media_dir / f".{digest}.{os.getpid()}.wav"
            cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", str(video),
                   "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(part)]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if r.returncode != 0 or not part.exists() or part.stat().st_size < 1000:
                part.unlink(missing_ok=True)
                wav = None  # piste audio vide / illisible : TRIBE tourne sans audio
            else:
                os.replace(part, wav)
    return link, wav, digest


# --------------------------------------------------------------------------------------------
# Mots -> événements Word avec contexte (équivalent AddText + AddSentenceToWords + AddContextToWords)
# --------------------------------------------------------------------------------------------
def _ends_sentence(text: str) -> bool:
    t = text.rstrip("\"'»”)]")
    return t.endswith(END_PUNCT)


def split_sentences(words: list[dict]) -> list[list[dict]]:
    sentences: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur and w["start"] - cur[-1]["end"] > PAUSE_SENTENCE_S:
            sentences.append(cur)
            cur = []
        cur.append(w)
        if _ends_sentence(w["w"]):
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    return sentences


def build_word_rows(words: list[dict], language: str, duration: float) -> list[dict]:
    """Événements `Word` avec `context` (texte ponctué depuis le début jusqu'au mot inclus)."""
    lang_name = ISO_TO_NAME.get(language, language or "english")
    clean = []
    for w in sorted(words, key=lambda x: x["start"]):
        start = min(max(0.0, w["start"]), max(0.0, duration - MIN_WORD_DUR))
        end = min(max(w["end"], start + MIN_WORD_DUR), duration) if duration > 0 else w["end"]
        clean.append({"w": w["w"], "start": start, "end": max(end, start + MIN_WORD_DUR)})
    rows: list[dict] = []
    history: list[str] = []  # mots (casse finale) déjà vus, pour la fenêtre de contexte

    def join(ws: list[str]) -> str:
        # faster-whisper découpe « d'étudier » en « d » + « 'étudier » et « abonne-toi » en « abonne »
        # + « -toi » : on recolle ces mots au précédent pour que LLaMA lise un texte naturel.
        out = ""
        for k, x in enumerate(ws):
            out += x if (k == 0 or x[:1] in GLUE_PREFIX) else " " + x
        return out

    for sent in split_sentences(clean):
        # Équivalent de `" ".join(mots).capitalize()` (AddText) : 1er caractère en majuscule, le reste
        # en minuscules ; fait mot à mot pour garder l'alignement mot <-> fin du contexte.
        pieces = [w["w"].lower() for w in sent]
        pieces[0] = pieces[0][:1].upper() + pieces[0][1:]
        ends = _ends_sentence(pieces[-1])
        sentence_text = join(pieces) + ("" if ends else ".")
        char = 0
        for w, piece in zip(sent, pieces):
            history.append(piece)
            # Le contexte se termine exactement par le mot : l'extracteur texte en déduit les tokens
            # du mot (n_tokens(contexte) - n_tokens(préfixe)).
            context = join(history[-MAX_CONTEXT_WORDS:])
            rows.append({
                "type": "Word", "text": piece, "start": round(w["start"], 4),
                "duration": round(w["end"] - w["start"], 4), "timeline": "default", "subject": "default",
                "language": lang_name, "sentence": sentence_text, "sentence_char": char,
                "context": context, "modality": "heard",
            })
            char += len(piece) + (0 if piece[:1] in GLUE_PREFIX else 1)
        if not ends:  # comme TRIBE : le point ajouté n'apparaît que dans le contexte des mots suivants
            history[-1] = history[-1] + "."
    return rows


# --------------------------------------------------------------------------------------------
# DataFrame complet
# --------------------------------------------------------------------------------------------
def build_events(video_link: Path | None, wav: Path | None, words: list[dict], language: str,
                 duration: float, use_text: bool):
    """Renvoie (events DataFrame standardisé, n_mots). `video_link=None` : mode sans vidéo (audio + mots)."""
    import pandas as pd
    from neuralset.events.transforms import ChunkEvents
    from neuralset.events.utils import standardize_events

    base = {"start": 0.0, "timeline": "default", "subject": "default"}
    rows = []
    if video_link is not None:
        rows.append({"type": "Video", "filepath": str(video_link), **base})
    if wav is not None:
        rows.append({"type": "Audio", "filepath": str(wav), **base})
    if not rows:
        raise InputError("ni vidéo ni audio : aucune modalité à prédire")
    events = standardize_events(pd.DataFrame(rows))
    events = ChunkEvents(event_type_to_chunk="Audio", max_duration=CHUNK_MAX_S, min_duration=CHUNK_MIN_S)(events)
    if video_link is not None:
        events = ChunkEvents(event_type_to_chunk="Video", max_duration=CHUNK_MAX_S, min_duration=CHUNK_MIN_S)(events)
    n_words = 0
    if use_text and words:
        main = "Video" if video_link is not None else "Audio"
        video_dur = float(events.loc[events.type == main, "stop"].max())
        word_rows = build_word_rows(words, language, min(duration, video_dur))
        n_words = len(word_rows)
        if word_rows:
            events = pd.concat([events, pd.DataFrame(word_rows)], ignore_index=True)
    return standardize_events(events), n_words
