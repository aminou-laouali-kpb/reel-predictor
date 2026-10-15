"""Simulation locale de variantes de script avant tournage (contrat docs/CONTRACTS.md §7).

    baseline_variant(job_dir) -> Variant      # script ORIGINAL (transcription + OCR + cuts) en segments
    simulate(variants, job_dir, config, coach_cfg, progress=None) -> dict

Pour chaque variante :
1. voix off segment par segment avec la voix de synthèse macOS (`say -v <voix> -r <débit>`), convertie en
   WAV 16 kHz mono puis concaténée : les durées (segments, vidéo) viennent de l'audio réel ;
2. mots horodatés DIRECTEMENT depuis les segments (texte exact du script, réparti dans la zone parlée de
   chaque segment au prorata de la longueur prononcée). Pas de faster-whisper ici : le texte est connu (pas
   d'erreur de transcription « Erasmus-Mindu », chiffres conservés tels qu'écrits), les frontières de
   segment sont exactes à l'échantillon près, l'erreur intra-segment (≈ 0,2 s) est très inférieure à la
   résolution effective de TRIBE (± 2–3 s, docs/RESEARCH.md §1.10), et on économise ≈ 15 s par variante ;
3. TRIBE v2 sans image (`run_tribe.py --no-video --audio`) -> ROI (atlas, comme le pipeline) ;
4. signaux non cérébraux reconstruits depuis les segments (cuts = `cut_at_start` ou changement de type de
   plan, texte à l'écran et nouveauté, visage et mouvement selon `shot`, audio réel de la voix de synthèse :
   loudness/silences), puis `features.build_features`, `indices.compute_indices`, `retention.simulate`,
   `score.compute_score` du pipeline (aucune logique dupliquée) ;
5. rôles -> moments clés (`targets.moments_from_roles`), évaluation (`targets.evaluate`), classement
   (`targets.rank_key`).

Normalisation (choix documenté) : les prédictions sans image ne sont PAS comparables aux Reels réels (les ROI
visuelles ne voient ni cuts ni texte) et ne sont JAMAIS ajoutées à `data/reference.json`. Toutes les variantes
d'un même Reel sont exprimées en z robuste relatif à l'ORIGINAL re-voicé : roi = (valeur − médiane_original) /
(1,4826 · MAD_original), par ROI (MAD plancher = moitié de la MAD médiane des ROI, pour ne pas amplifier une ROI
quasi plate). L'original (même voix, même débit) sert de référence commune : comparaison équitable, résultat
indépendant des autres variantes proposées, et une variante simulée seule (seconde passe du coach) reste
comparable grâce à la référence enregistrée dans `<job>/coach/sim_reference.json` (recalculée en simulant
l'original si elle manque ou ne correspond plus : voix, débit, mode texte ou script original différents).

Jamais d'exception pour un échec TRIBE : repli heuristique pour TOUTES les variantes (`mode="heuristic"`,
`reason` en français) afin de ne jamais comparer une variante « avec cerveau » à une variante « sans ».
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .. import paths
from ..config import INDEX_NAMES, Config, load_config
from .schema import Segment, Variant
from .targets import evaluate, moments_from_roles, rank_key

log = logging.getLogger("reel_predictor.coach.simulate")

ProgressFn = Callable[[float, str], None]

BASELINE_ANGLE = "original"
SR = 16000
REFERENCE_FILE = "sim_reference.json"     # dans <job>/coach/

# Plans -> (nb de visages, aire du visage / image, mouvement moyen image à image 0..255 à 4 i/s).
# Ordres de grandeur d'un Reel vertical (face caméra ≈ 5–8 % de l'image) ; approximations communes à toutes les
# variantes, original compris : elles ne biaisent pas la comparaison.
SHOT_PARAMS: dict[str, tuple[int, float, float]] = {
    "face_cam": (1, 0.06, 3.0),
    "close_up": (1, 0.15, 3.5),
    "screen": (0, 0.0, 2.0),
    "broll": (0, 0.0, 8.0),
    "text_card": (0, 0.0, 1.0),
}
CUT_MOTION = 40.0       # différence d'image de l'image qui suit un cut (moyennée sur la seconde : ≈ +10)
ZOOM_MOTION = 4.0       # consigne « zoom » dans `edit` : mouvement ajouté sur la 1re seconde du segment
FRAME_FPS = 4.0

# Découpage de l'original en phrases
_END = re.compile(r"[.!?…]+[»\"')\]]*$")
PAUSE_SPLIT_S = 1.0
MAX_SEG_S = 9.0
MAX_SEG_WORDS = 30
MIN_SEG_WORDS = 3
MAX_CTA_SEGMENTS = 2
GLUE_PREFIX = ("'", "’", "-")
# Nombres saillants (rôle « fact ») : montants, pourcentages, grands nombres
_MONEY = re.compile(r"\d[\d\s.,]*\s*(?:€|k€|euros?|eur\b|\$|dollars?|fcfa|francs?|%|pour\s?cents?)", re.I)
_BIG = re.compile(r"\b\d{3,}(?:[\s.,]\d{3})*\b")
_NUM_WORDS = re.compile(r"\b(?:mille|million|millions|milliard|milliards|pourcent)\b", re.I)


# ============================================================================= original -> Variant
def _read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _join_words(ws: list[str]) -> str:
    out = ""
    for k, w in enumerate(ws):
        out += w if (k == 0 or w[:1] in GLUE_PREFIX) else " " + w
    return out.strip()


def _load_words(job_dir: Path, rows: list[dict]) -> list[dict]:
    p = job_dir / "words.json"
    if p.exists():
        try:
            words = _read_json(p).get("words") or []
            return [{"w": str(w["w"]), "start": float(w["start"]), "end": float(w["end"])} for w in words
                    if str(w.get("w", "")).strip()]
        except (OSError, ValueError, KeyError, TypeError):
            pass
    out = []                                   # repli : mots de la timeline répartis dans leur seconde
    for r in rows:
        ws = r.get("words") or []
        for k, w in enumerate(ws):
            s = r["t"] + k / max(1, len(ws))
            out.append({"w": w, "start": s, "end": s + 1 / max(1, len(ws))})
    return out


def _split_sentences(words: list[dict]) -> list[list[dict]]:
    sents: list[list[dict]] = []
    cur: list[dict] = []
    for w in words:
        if cur and (w["start"] - cur[-1]["end"] > PAUSE_SPLIT_S or len(cur) >= MAX_SEG_WORDS
                    or w["end"] - cur[0]["start"] > MAX_SEG_S):
            sents.append(cur)
            cur = []
        cur.append(w)
        if _END.search(w["w"]):
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    merged: list[list[dict]] = []
    for s in sents:                            # fragments très courts (« Et », « Donc ») : avec la phrase d'avant
        n = sum(1 for w in s if w["w"][:1] not in GLUE_PREFIX)
        if merged and n < MIN_SEG_WORDS:
            merged[-1] = merged[-1] + s
        else:
            merged.append(s)
    return merged


def _dedupe_texts(texts: list[str]) -> str:
    from ..ocr import similarity
    from ..textutils import squash

    kept: list[str] = []
    for txt in texts:
        txt = txt.strip()
        if not txt:
            continue
        sq = squash(txt)
        if not sq:
            continue
        done = False
        for i, k in enumerate(kept):
            sk = squash(k)
            if sq in sk:
                done = True
            elif sk in sq or similarity(txt, k) >= 0.8:
                kept[i] = txt if len(sq) >= len(sk) else k
                done = True
            if done:
                break
        if not done:
            kept.append(txt)
    return " / ".join(kept)


def _salient_number(text: str) -> bool:
    return bool(_MONEY.search(text) or _BIG.search(text) or _NUM_WORDS.search(text))


def baseline_variant(job_dir: Path) -> Variant:
    """Script ORIGINAL du Reel analysé, sous forme de `Variant` (segments = phrases de la transcription).

    on_screen_text = textes OCR du segment dédoublonnés ; cut_at_start = un cut tombe pendant le segment (ramené
    à son début ; jamais pour le 1er : le début de la vidéo n'est pas un cut) ; shot = face_cam si un visage est
    vu sur au moins la moitié des secondes, sinon screen ; rôles : 1er segment = hook, 1er autre segment avec un
    nombre saillant (montant, %, grand nombre) = fact, dernier(s) segment(s) avec un appel à l'action = cta.
    """
    from ..textutils import find_in_text, find_in_tokens, keyword_tokens, tokens
    from ..timeline import read_timeline

    job_dir = Path(job_dir)
    summary = _read_json(job_dir / "summary.json")
    rows = read_timeline(job_dir / "timeline.jsonl")
    duration = float(summary.get("video", {}).get("duration_s") or len(rows))
    cuts = [float(c) for c in summary.get("cuts", [])]
    words = sorted(_load_words(job_dir, rows), key=lambda w: w["start"])
    sents = _split_sentences(words)
    if not sents:                              # pas de parole : un segment par plan (texte seul)
        bounds = [0.0] + [c for c in cuts if 0.5 < c < duration - 0.5] + [duration]
        spans = [(bounds[i], bounds[i + 1], "") for i in range(len(bounds) - 1)]
    else:
        starts = [0.0] + [s[0]["start"] for s in sents[1:]]
        ends = starts[1:] + [duration]
        spans = [(a, b, _join_words([w["w"] for w in s])) for a, b, s in zip(starts, ends, sents)]

    try:
        cta_kw = keyword_tokens(load_config().weights.get("cta_keywords", []))
    except Exception:  # noqa: BLE001 — config illisible : pas de détection de CTA
        cta_kw = []
    segs: list[dict] = []
    for i, (a, b, vo) in enumerate(spans):
        secs = [r for r in rows if a <= r["t"] + 0.5 < b] or rows[max(0, min(int(a), len(rows) - 1)):][:1]
        screen = _dedupe_texts([r.get("text", "") for r in secs])
        faces = [int(r.get("faces", 0)) > 0 for r in secs]
        shot = "face_cam" if faces and sum(faces) >= 0.5 * len(faces) else "screen"
        cut = i > 0 and any(a - 0.5 <= c < b - 0.25 for c in cuts)
        has_cta = bool(find_in_tokens(tokens(vo), cta_kw) or find_in_text(screen, cta_kw)) if cta_kw else False
        segs.append({"a": a, "b": b, "vo": vo, "screen": screen, "shot": shot, "cut": cut, "cta": has_cta,
                     "num_vo": _salient_number(vo), "num": _salient_number(vo + " " + screen)})
    roles = ["body"] * len(segs)
    if segs:
        roles[0] = "hook"
        # le nombre dit prime (l'OCR d'un segment déborde souvent sur le voisin) ; sinon nombre affiché
        fact = next((i for i in range(1, len(segs)) if segs[i]["num_vo"]), None)
        if fact is None:
            fact = next((i for i in range(1, len(segs)) if segs[i]["num"]), None)
        if fact is not None:
            roles[fact] = "fact"
        # CTA : le dernier segment qui en contient un, + le précédent s'il en contient aussi (≤ MAX_CTA_SEGMENTS,
        # 2nde moitié de la vidéo) : une fenêtre CTA comparable à celle d'une variante
        last = next((i for i in range(len(segs) - 1, 0, -1) if segs[i]["cta"] and roles[i] == "body"), None)
        if last is not None:
            roles[last] = "cta"
            j = last - 1
            while (j >= 1 and last - j < MAX_CTA_SEGMENTS and segs[j]["cta"] and roles[j] == "body"
                   and segs[j]["a"] >= duration / 2):
                roles[j] = "cta"
                j -= 1
    return Variant(
        angle=BASELINE_ANGLE,
        hook_summary="Script actuel du Reel (transcription Whisper + texte à l'écran détecté)",
        segments=[Segment(role=r, voice_over=s["vo"], on_screen_text=s["screen"], shot=s["shot"],
                          cut_at_start=s["cut"], edit="") for r, s in zip(roles, segs)],
        rationale="Référence : le script original re-voicé avec la même voix de synthèse que les variantes",
    )


# ============================================================================= voix de synthèse
@dataclass
class _Rendered:
    """Variante rendue : audio, segments horodatés, mots."""
    variant: Variant
    samples: np.ndarray
    spans: list[tuple[float, float]]          # (début, fin) de chaque segment (s)
    words: list[dict]
    duration: float
    wav: Path | None = None
    words_path: Path | None = None
    tts_ok: bool = True
    raw_roi: np.ndarray | None = None          # (T, R)
    meta: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return max(1, int(math.ceil(self.duration - 0.05)))   # = media.VideoInfo.n_seconds


def _clean_tts_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()


def _say(text: str, voice: str, rate: int, out_aiff: Path) -> None:
    cmd = ["say", "-v", voice, "-r", str(int(rate)), "-o", str(out_aiff), "--", text]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0 or not out_aiff.exists() or out_aiff.stat().st_size < 100:
        raise RuntimeError(f"say a échoué ({r.returncode}) : {r.stderr.strip()[:200]}")


def _to_samples(src: Path, dst_wav: Path) -> np.ndarray:
    from ..media import read_wav

    cmd = ["ffmpeg", "-nostdin", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), "-ac", "1",
           "-ar", str(SR), "-c:a", "pcm_s16le", str(dst_wav)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"conversion audio impossible : {r.stderr.strip()[:200]}")
    x, sr = read_wav(dst_wav)
    if sr != SR:
        raise RuntimeError(f"fréquence inattendue {sr}")
    return x


def _write_wav(path: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def _spoken_weight(word: str) -> float:
    """Longueur prononcée approximative (un chiffre se dit en plusieurs syllabes : « 1400 »)."""
    letters = sum(c.isalpha() for c in word)
    digits = sum(c.isdigit() for c in word)
    return max(1.0, letters + 4.0 * digits) + 1.0


def _voiced_zone(x: np.ndarray) -> tuple[float, float]:
    """(début, fin) en secondes de la zone parlée d'un segment (énergie par fenêtres de 20 ms)."""
    if x.size == 0:
        return 0.0, 0.0
    win = int(0.02 * SR)
    n = max(1, x.size // win)
    rms = np.sqrt(np.mean(x[: n * win].reshape(n, win) ** 2, axis=1)) if x.size >= win else np.array([0.0])
    thr = max(1e-3, 0.05 * float(rms.max()))
    on = np.flatnonzero(rms > thr)
    if on.size == 0:
        return 0.0, x.size / SR
    return on[0] * win / SR, min(x.size, (on[-1] + 1) * win) / SR


def _segment_words(text: str, x: np.ndarray, offset: float) -> list[dict]:
    raw = text.split()
    ws: list[str] = []
    for w in raw:                               # ponctuation isolée (« ? », « — ») collée au mot précédent
        if ws and not any(c.isalnum() for c in w):
            ws[-1] += w
        elif any(c.isalnum() for c in w):
            ws.append(w)
    if not ws:
        return []
    a, b = _voiced_zone(x)
    if b - a < 0.05 * len(ws):
        a, b = 0.0, max(x.size / SR, 0.05 * len(ws))
    wts = np.array([_spoken_weight(w) for w in ws])
    edges = a + (b - a) * np.concatenate([[0.0], np.cumsum(wts) / wts.sum()])
    return [{"w": w, "start": round(offset + float(edges[k]), 3),
             "end": round(offset + float(edges[k + 1]) - 0.02, 3)} for k, w in enumerate(ws)]


def _render_all(variants: list[Variant], work: Path, voice: str, rate: int, reading_wps: float,
                progress: Callable[[float], None]) -> list[_Rendered]:
    """Synthèse de tous les segments (en parallèle), puis assemblage par variante."""
    jobs = []
    for vi, v in enumerate(variants):
        for si, s in enumerate(v.segments):
            jobs.append((vi, si, _clean_tts_text(s.voice_over)))
    audio: dict[tuple[int, int], np.ndarray] = {}
    done = [0]

    def synth(job):
        vi, si, text = job
        if not text:
            return job, None
        aiff = work / f"v{vi}_s{si}.aiff"
        _say(text, voice, rate, aiff)
        x = _to_samples(aiff, work / f"v{vi}_s{si}.wav")
        return job, x

    with ThreadPoolExecutor(max_workers=6, thread_name_prefix="tts") as ex:
        for job, x in ex.map(synth, jobs):
            audio[(job[0], job[1])] = x
            done[0] += 1
            progress(done[0] / max(1, len(jobs)))
    out = []
    for vi, v in enumerate(variants):
        chunks, spans, words, t = [], [], [], 0.0
        for si, s in enumerate(v.segments):
            x = audio.get((vi, si))
            if x is None:                       # segment sans voix off : silence le temps de lire le texte
                n_read = len(s.on_screen_text.split())
                x = np.zeros(int(SR * max(1.5, n_read / max(0.5, reading_wps))), np.float32)
            words.extend(_segment_words(_clean_tts_text(s.voice_over), x, t))
            chunks.append(x)
            spans.append((t, t + x.size / SR))
            t += x.size / SR
        samples = np.concatenate(chunks) if chunks else np.zeros(SR, np.float32)
        out.append(_Rendered(variant=v, samples=samples.astype(np.float32), spans=spans, words=words,
                             duration=max(samples.size / SR, 1.0)))
    return out


def _render_estimated(variants: list[Variant], rate: int, reading_wps: float) -> list[_Rendered]:
    """Repli sans voix de synthèse : durées estimées au débit configuré, audio muet (mode heuristique)."""
    out = []
    for v in variants:
        spans, words, t = [], [], 0.0
        for s in v.segments:
            n = len(_clean_tts_text(s.voice_over).split())
            d = n * 60.0 / max(60, rate) + 0.25 if n else max(1.5, len(s.on_screen_text.split()) / reading_wps)
            words.extend(_segment_words(_clean_tts_text(s.voice_over), np.ones(int(d * SR), np.float32) * 0.1, t))
            spans.append((t, t + d))
            t += d
        out.append(_Rendered(variant=v, samples=np.zeros(int(max(t, 1.0) * SR), np.float32), spans=spans,
                             words=words, duration=max(t, 1.0), tts_ok=False))
    return out


# ============================================================================= signaux -> indices
def _signals(r: _Rendered, cfg: Config):
    from ..audio import analyze as analyze_audio
    from ..faces import FrameStats
    from ..features import Signals
    from ..ocr import OcrSecond, count_words
    from ..speech import Transcript, Word

    T = r.T
    segs = r.variant.segments
    cuts: list[float] = []
    for i, s in enumerate(segs):
        if i > 0 and (s.cut_at_start or s.shot != segs[i - 1].shot):
            cuts.append(round(r.spans[i][0], 3))

    def seg_at(x: float) -> int:
        for i, (a, b) in enumerate(r.spans):
            if a <= x < b:
                return i
        return len(segs) - 1

    times = np.arange(int(T * FRAME_FPS), dtype=np.float32) / FRAME_FPS
    f_faces = np.zeros(times.size, np.int32)
    f_area = np.zeros(times.size, np.float32)
    f_motion = np.zeros(times.size, np.float32)
    for k, x in enumerate(times):
        i = seg_at(float(x) + 1e-3)
        n, area, mot = SHOT_PARAMS.get(segs[i].shot, SHOT_PARAMS["face_cam"])
        f_faces[k], f_area[k], f_motion[k] = n, area, mot
        if "zoom" in (segs[i].edit or "").lower() and x < r.spans[i][0] + 1.0:
            f_motion[k] += ZOOM_MOTION
    for c in cuts:                               # la 1re image après le cut diffère fortement de la précédente
        k = int(np.searchsorted(times, c, side="left"))
        if k < times.size:
            f_motion[k] += CUT_MOTION
    frames = FrameStats(times=times, faces=f_faces, face_area=f_area, motion=f_motion, face_ok=True)

    ocr, prev = [], ""
    for t in range(T):
        txt = _clean_tts_text(segs[seg_at(t + 0.5)].on_screen_text) if segs else ""
        ocr.append(OcrSecond(text=txt, words=count_words(txt), new=bool(txt) and txt != prev, score=1.0 if txt else 0.0))
        prev = txt
    ac = cfg.section("audio")
    audio = analyze_audio(r.samples, SR, T, float(ac["window_s"]), float(ac["silence_dbfs"]), float(ac["silence_rel"]))
    ws = [Word(w["w"], float(w["start"]), float(w["end"])) for w in r.words]
    tr = Transcript(language="fr", language_probability=1.0, text=" ".join(s.voice_over for s in segs), words=ws)
    return Signals(duration_s=r.duration, n_seconds=T, cuts=cuts, transcript=tr, frames=frames, ocr=ocr,
                   audio=audio), (ws[0].start if ws else None)


def _score_variant(r: _Rendered, cfg: Config, roi_norm: np.ndarray | None, roi_names: list[str] | None,
                   tcfg: dict) -> dict:
    from ..features import build_features
    from ..indices import compute_indices
    from ..retention import simulate as simulate_retention
    from ..score import compute_score

    sig, first_word = _signals(r, cfg)
    df = build_features(sig, cfg)
    idx = compute_indices(df, cfg, roi_norm, roi_names if roi_norm is not None else None)
    rp = cfg.retention
    ret = simulate_retention(idx.values["attention"], df["f_cut_recent"].to_numpy(float), r.duration,
                             hook_seconds=rp.hook_seconds, hook_drop=rp.hook_drop, base_drop=rp.base_drop,
                             k_attention=rp.k_attention, cut_bonus=rp.cut_bonus, allow_increase=rp.allow_increase)
    sc = compute_score(df, idx.values, ret, first_word, cfg)
    rows = [{"t": t, "indices": {k: round(float(np.clip(idx.values[k][t], 0, 1)), 4) for k in INDEX_NAMES},
             "retention": round(float(np.clip(ret.per_second[t], 0, 1)), 4)} for t in range(r.T)]
    spans = [(s.role, a, b) for s, (a, b) in zip(r.variant.segments, r.spans)]
    moments = moments_from_roles(spans, r.duration, tcfg)
    ev = evaluate(rows, moments, tcfg)
    return {
        "duration_s": round(r.duration, 2),
        "score": int(sc.score),
        "segments": [{"role": role, "t": round(a, 2), "t_end": round(b, 2)} for role, a, b in spans],
        "timeline": rows,
        "evaluation": ev,
        # informatif (hors contrat)
        "angle": r.variant.angle,
        "retention_summary": {"avg_watch_ratio": round(float(ret.avg_watch_ratio), 4), "avg_watch_s": round(float(ret.avg_watch_s), 2),
                              "completion": round(float(ret.completion), 4)},
        "cuts": [round(c, 2) for c in sig.cuts],
    }


# ============================================================================= cerveau (TRIBE sans image)
class _BrainFailure(Exception):
    def __init__(self, reason: str, detail: str = "", hf_text: bool = False):
        super().__init__(reason)
        self.reason, self.detail, self.hf_text = reason, detail, hf_text


def _run_brain(wav: Path, words_path: Path, out_path: Path, cfg: Config, no_text: bool, timeout_s: float,
               progress: ProgressFn | None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Un appel du worker `--no-video`. Lève `_BrainFailure` (jamais d'autre exception attendue)."""
    from ..brain.tribe import EXIT_HF, EXIT_INPUT, EXIT_OK, EXIT_OOM, _HF, _last_error, _run_worker, load_npz

    cmd = [str(paths.TRIBE_WORKER_PYTHON), str(paths.TRIBE_WORKER_SCRIPT), "--no-video", "--audio", str(wav),
           "--words", str(words_path), "--out", str(out_path), "--model", cfg.brain.model,
           "--cache-dir", str(paths.CACHE_DIR / "tribe"), "--device", cfg.brain.device]
    if no_text:
        cmd.append("--no-text")
    run = _run_worker(cmd, timeout_s, progress, "Modèle cérébral (sans image)")
    if run.timed_out:
        raise _BrainFailure(f"Délai dépassé pour TRIBE ({int(timeout_s)} s)", _last_error(run))
    code = run.returncode
    if code == EXIT_OK and out_path.exists():
        return load_npz(out_path)
    if code == EXIT_HF:
        m = next((_HF.search(l) for l in run.stderr_tail + run.stdout_tail if _HF.search(l)), None)
        repo = m.group(1) if m else "un modèle gated"
        raise _BrainFailure(f"Accès Hugging Face à {repo} manquant", _last_error(run), hf_text="llama" in repo.lower())
    if code == EXIT_OOM:
        raise _BrainFailure("Mémoire insuffisante pour TRIBE v2", _last_error(run))
    if code == EXIT_INPUT:
        raise _BrainFailure("Entrée refusée par le worker TRIBE", _last_error(run))
    if code is not None and code < 0:
        raise _BrainFailure(f"Worker TRIBE interrompu (signal {-code})", _last_error(run))
    raise _BrainFailure(f"Erreur du worker TRIBE (code {code})", _last_error(run))


def _robust_reference(raw: np.ndarray, mad_floor_frac: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Médiane / MAD par ROI d'une série (T, R) ; MAD plancher = `mad_floor_frac` × MAD médiane des ROI."""
    ok = np.isfinite(raw).all(axis=1)
    x = raw[ok] if ok.any() else np.zeros((1, raw.shape[1]))
    med = np.median(x, axis=0)
    mad = np.median(np.abs(x - med), axis=0)
    pos = mad[mad > 1e-9]
    floor = mad_floor_frac * float(np.median(pos)) if pos.size else 1.0
    # arrondi = valeurs enregistrées dans sim_reference.json : un appel ultérieur obtient exactement les mêmes z
    return np.round(med, 6), np.round(np.maximum(mad, max(floor, 1e-6)), 6)


def _baseline_digest(v: Variant, voice: str, rate: int) -> str:
    h = hashlib.sha1(f"{voice}|{rate}|".encode())
    for s in v.segments:
        h.update(json.dumps([s.voice_over, s.on_screen_text], ensure_ascii=False).encode())
    return h.hexdigest()[:16]


def _load_reference(job_dir: Path, digest: str, mode: str, roi_names: list[str]):
    p = Path(job_dir) / "coach" / REFERENCE_FILE
    try:
        d = _read_json(p)
        if d.get("digest") == digest and d.get("mode") == mode and d.get("rois") == roi_names:
            return np.asarray(d["median"], float), np.asarray(d["mad"], float)
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _save_reference(job_dir: Path, digest: str, mode: str, roi_names: list[str], med, mad, meta: dict) -> None:
    p = Path(job_dir) / "coach" / REFERENCE_FILE
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        from ..timeline import write_json

        write_json(p, {"version": 1, "digest": digest, "mode": mode, "rois": roi_names,
                       "median": [round(float(x), 6) for x in med], "mad": [round(float(x), 6) for x in mad],
                       "method": "z robuste par ROI relatif à l'original re-voicé (TRIBE --no-video)", **meta})
    except OSError as e:
        log.warning("Référence de simulation non enregistrée : %s", e)


# ============================================================================= API
def simulate(variants: list[Variant], job_dir: Path, config: Config, coach_cfg: dict,
             progress: ProgressFn | None = None) -> dict:
    """Simule chaque variante (voix de synthèse + TRIBE sans image + signaux reconstruits). Contrat §7."""
    t_all = time.time()
    job_dir = Path(job_dir)
    variants = [v if isinstance(v, Variant) else Variant.model_validate(v) for v in variants]
    if not variants:
        raise ValueError("aucune variante à simuler")
    sim_cfg = coach_cfg.get("simulation", {}) or {}
    tcfg = coach_cfg["targets"]
    voice = str(sim_cfg.get("tts_voice", "Thomas"))
    rate = int(sim_cfg.get("tts_rate_wpm", 190))
    reading_wps = float(coach_cfg.get("reading_wps", 3.5))
    timeout_s = float(sim_cfg.get("timeout_s", 900))
    warnings: list[str] = []
    timings: dict[str, float] = {}
    last = [0.0]

    def report(p: float, msg: str) -> None:
        if progress is None:
            return
        p = max(last[0], min(1.0, float(p)))
        last[0] = p
        try:
            progress(round(p, 4), msg)
        except Exception:  # noqa: BLE001
            pass

    is_orig = variants[0].angle == BASELINE_ANGLE
    names = (["original"] if is_orig else []) + [f"variante {k}" for k in range(1, len(variants) + 1 - int(is_orig))]

    mode, reason = "tribe_no_video", None
    if str(sim_cfg.get("brain_mode", "no_video")) != "no_video":
        mode, reason = "heuristic", "Simulation sans modèle cérébral (coach.simulation.brain_mode = heuristic)"
    elif config.brain.provider == "heuristic":
        mode, reason = "heuristic", "Modèle cérébral désactivé dans la configuration (brain.provider = heuristic)"
    else:
        from ..brain.tribe import worker_installed

        if not worker_installed():
            mode, reason = "heuristic", "Worker TRIBE non installé : simulation sans modèle cérébral"

    # Variante « original » cachée : nécessaire à la référence de normalisation si elle n'est pas fournie
    roi_names: list[str] | None = None
    ref = None
    hidden_base: Variant | None = None
    no_text = bool(config.brain.no_text)
    if mode == "tribe_no_video" and not is_orig:
        try:
            from ..brain.atlas import roi_matrix

            roi_names, _ = roi_matrix(config.roi_mapping)
            hidden_base = baseline_variant(job_dir)
            digest = _baseline_digest(hidden_base, voice, rate)
            ref = _load_reference(job_dir, digest, "no_video:" + ("disabled" if no_text else "llama"), roi_names)
            if ref is not None:
                hidden_base = None
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Original indisponible pour la normalisation ({type(e).__name__}) : z intra-simulation")
            hidden_base = None

    pool = variants + ([hidden_base] if hidden_base is not None else [])
    work_root = paths.CACHE_DIR / "coach_sim"
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{job_dir.name}_", dir=work_root))
    try:
        # ------------------------------------------------------------ 1. voix de synthèse
        t0 = time.time()
        report(0.01, f"Voix de synthèse ({voice}, {rate} mots/min)")
        try:
            rendered = _render_all(pool, work, voice, rate, reading_wps,
                                   lambda f: report(0.01 + 0.13 * f, "Voix de synthèse des segments"))
        except Exception as e:  # noqa: BLE001
            log.warning("TTS impossible : %s", e)
            rendered = _render_estimated(pool, rate, reading_wps)
            if mode != "heuristic":
                mode, reason = "heuristic", f"Voix de synthèse indisponible ({type(e).__name__}) : durées estimées, sans cerveau"
        for i, r in enumerate(rendered):
            if r.tts_ok:
                r.wav = work / f"variant{i}.wav"
                _write_wav(r.wav, r.samples)
            r.words_path = work / f"variant{i}.words.json"
            r.words_path.write_text(json.dumps({"language": "fr", "duration_s": round(r.duration, 3),
                                                "words": r.words}, ensure_ascii=False), encoding="utf-8")
        timings["tts"] = round(time.time() - t0, 2)

        # ------------------------------------------------------------ 2. TRIBE sans image
        t0 = time.time()
        brain_meta: list[dict] = []
        if mode == "tribe_no_video":
            from ..brain.atlas import roi_matrix
            from ..indices import align_predictions, roi_values

            roi_names, W = roi_matrix(config.roi_mapping)
            hrf = float(config.section("normalization").get("hrf_shift_s", 0))
            deadline = time.time() + timeout_s
            n = len(rendered)
            try:
                i = 0
                while i < n:
                    r = rendered[i]
                    a, b = 0.15 + 0.77 * i / n, 0.15 + 0.77 * (i + 1) / n
                    label = names[i] if i < len(names) else "original (référence)"
                    report(a, f"Modèle cérébral : {label} ({i + 1}/{n})")
                    remaining = deadline - time.time()
                    if remaining < 5:
                        raise _BrainFailure(f"Délai de simulation dépassé ({int(timeout_s)} s)")
                    try:
                        preds, secs, meta = _run_brain(
                            r.wav, r.words_path, work / f"variant{i}.npz", config, no_text, remaining,
                            lambda f, m, a=a, b=b, label=label: report(a + (b - a) * f, f"{label} : {m}"))
                    except _BrainFailure as e:
                        if e.hf_text and not no_text and config.brain.retry_without_text and i == 0:
                            no_text = True
                            warnings.append(f"{e.reason} : simulation sans la composante texte")
                            continue
                        raise
                    aligned = align_predictions(preds, secs, r.T, hrf)
                    r.raw_roi = roi_values(aligned, W)
                    r.meta = meta
                    brain_meta.append({k: meta.get(k) for k in ("runtime_s", "peak_footprint_mb", "text_feature",
                                                                "video_feature", "n_rows")})
                    i += 1
            except _BrainFailure as e:
                mode = "heuristic"
                reason = f"{e.reason} : simulation sans modèle cérébral (indices issus des signaux seuls)"
                if e.detail:
                    warnings.append(f"Détail TRIBE : {e.detail}")
                log.warning("TRIBE sans image en échec : %s (%s)", e.reason, e.detail)
            except Exception as e:  # noqa: BLE001 — jamais d'exception pour un échec du cerveau
                mode = "heuristic"
                reason = f"Erreur du modèle cérébral ({type(e).__name__}) : simulation sans modèle cérébral"
                log.exception("TRIBE sans image : erreur inattendue")
        timings["brain"] = round(time.time() - t0, 2)

        # ------------------------------------------------------------ 3. normalisation commune
        t0 = time.time()
        report(0.93, "Indices, rétention et score des variantes")
        roi_norm: list[np.ndarray | None] = [None] * len(rendered)
        norm_info: dict = {"method": None}
        if mode == "tribe_no_video":
            from ..indices import normalize_corpus

            text_mode = "disabled" if no_text else "llama"
            mode_key = f"no_video:{text_mode}"
            mad_scale = float(config.section("normalization")["mad_scale"])
            if is_orig:
                med, mad = _robust_reference(rendered[0].raw_roi)
                digest = _baseline_digest(variants[0], voice, rate)
                _save_reference(job_dir, digest, mode_key, roi_names, med, mad,
                                {"voice": voice, "rate_wpm": rate, "duration_s": round(rendered[0].duration, 2)})
                source = "original de cette simulation"
            elif ref is not None:
                med, mad = ref
                source = "original enregistré (coach/sim_reference.json)"
            elif hidden_base is not None:
                med, mad = _robust_reference(rendered[-1].raw_roi)
                _save_reference(job_dir, _baseline_digest(hidden_base, voice, rate), mode_key, roi_names, med, mad,
                                {"voice": voice, "rate_wpm": rate, "duration_s": round(rendered[-1].duration, 2)})
                source = "original re-simulé pour la référence"
            else:
                med, mad = _robust_reference(np.concatenate([r.raw_roi for r in rendered], axis=0))
                source = "variantes de cette simulation (original indisponible)"
            for i, r in enumerate(rendered):
                roi_norm[i] = normalize_corpus(r.raw_roi, med, mad, mad_scale)
            norm_info = {"method": "z robuste par ROI (médiane/MAD) relatif à l'original re-voicé, mode sans image "
                                   "séparé, jamais ajouté à la référence de corpus",
                         "source": source, "text_feature": text_mode}

        results = []
        for i, v in enumerate(variants):
            res = {"name": names[i], **_score_variant(rendered[i], config, roi_norm[i], roi_names, tcfg)}
            results.append(res)
        timings["indices"] = round(time.time() - t0, 2)

        cand = list(range(1, len(results))) if (is_orig and len(results) > 1) else list(range(len(results)))
        best = max(cand, key=lambda i: rank_key(results[i]["evaluation"], results[i]["score"])) if cand else None
        timings["total"] = round(time.time() - t_all, 2)
        report(1.0, "Simulation terminée")
        out = {"mode": mode, "reason": reason, "results": results, "best": best, "timings": timings,
               # informatif (hors contrat)
               "normalization": norm_info, "voice": {"name": voice, "rate_wpm": rate},
               "brain_runs": brain_meta, "warnings": warnings}
        return out
    finally:
        if os.environ.get("REEL_SIM_KEEP") != "1":          # REEL_SIM_KEEP=1 : garder les fichiers (débogage)
            shutil.rmtree(work, ignore_errors=True)
