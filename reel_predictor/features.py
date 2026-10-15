"""Fusion des signaux en un DataFrame à 1 ligne par seconde.

Colonnes « timeline » (contrat §3) : t, cut, since_cut, faces, face_area, words, speech_rate, text,
text_words, text_new, audio_rms, silence. Colonnes `f_<nom>` : features normalisées 0..1
utilisées par les boosters de `weights.yaml > indices`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .audio import AudioStats
from .config import FEATURE_NAMES, Config
from .faces import FrameStats
from .ocr import OcrSecond
from .speech import Transcript
from .textutils import find_in_text, find_in_tokens, keyword_tokens, tokens


@dataclass
class Signals:
    duration_s: float
    n_seconds: int
    cuts: list[float] = field(default_factory=list)
    transcript: Transcript | None = None
    frames: FrameStats | None = None
    ocr: list[OcrSecond] | None = None
    audio: AudioStats | None = None


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def since_cut_series(cuts: list[float], T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cut dans [t,t+1), secondes depuis le dernier cut ou le début à la fin de t, un vrai cut a eu lieu)."""
    cut = np.zeros(T, bool)
    since = np.zeros(T, np.float32)
    had = np.zeros(T, bool)
    for t in range(T):
        cut[t] = any(t <= c < t + 1 for c in cuts)
        prev = [c for c in cuts if c < t + 1]
        if prev:
            since[t] = (t + 1) - max(prev)
            had[t] = True
        else:
            since[t] = t + 1
    return cut, since, had


def cut_recent_from(since_cut: np.ndarray, had_cut: np.ndarray, tau: float) -> np.ndarray:
    """exp(-(temps depuis le dernier vrai cut, mesuré au milieu de la seconde) / tau) ; 0 sans cut."""
    v = np.exp(-np.maximum(0.0, np.asarray(since_cut, float) - 0.5) / max(1e-6, tau))
    return np.where(np.asarray(had_cut, bool), v, 0.0)


def _per_second_frames(fs: FrameStats | None, T: int):
    faces = np.zeros(T, np.int32)
    area = np.zeros(T, np.float32)
    face_frac = np.zeros(T, np.float32)
    motion = np.zeros(T, np.float32)
    if fs is None or fs.times.size == 0:
        return faces, area, face_frac, motion
    sec = np.floor(fs.times + 1e-6).astype(int)
    for t in range(T):
        m = sec == t
        if not m.any():
            # pas d'image dans la seconde (fin de vidéo) : on reprend la dernière connue
            j = int(np.clip(np.searchsorted(fs.times, t, side="right") - 1, 0, fs.times.size - 1))
            m = np.zeros_like(sec, bool)
            m[j] = True
        faces[t] = int(fs.faces[m].max())
        area[t] = float(fs.face_area[m].max())
        face_frac[t] = float((fs.faces[m] > 0).mean())
        mv = fs.motion[m]
        motion[t] = float(mv.mean()) if mv.size else 0.0
    return faces, area, face_frac, motion


def build_features(sig: Signals, cfg: Config) -> pd.DataFrame:
    T = sig.n_seconds
    fc = cfg.section("features")
    t_idx = np.arange(T)
    df = pd.DataFrame({"t": t_idx})

    # --- plans
    cut, since, had = since_cut_series(sig.cuts, T)
    df["cut"] = cut
    df["since_cut"] = since
    df["had_cut"] = had
    df["f_cut"] = cut.astype(float)
    df["f_cut_recent"] = cut_recent_from(since, had, float(fc["cut_tau_s"]))
    a0, a1 = float(fc["since_cut_long"]["start_s"]), float(fc["since_cut_long"]["full_s"])
    df["f_since_cut_long"] = _clip01((since - a0) / (a1 - a0))

    # --- visages + mouvement
    faces, area, face_frac, motion = _per_second_frames(sig.frames, T)
    df["faces"] = faces
    df["face_area"] = area
    df["motion_raw"] = motion
    df["f_motion"] = _clip01(motion / float(fc["motion_ref"]))
    df["f_face_present"] = face_frac
    df["f_face_close"] = _clip01(area / float(fc["face_close_area"]))

    # --- parole
    words = sig.transcript.words if sig.transcript else []
    per_sec_words: list[list[str]] = [[] for _ in range(T)]
    coverage = np.zeros(T, np.float32)
    starts = np.array([w.start for w in words], float)
    for w in words:
        t0 = int(math.floor(w.start))
        if 0 <= t0 < T:
            per_sec_words[t0].append(w.w)
        for t in range(max(0, t0), min(T, int(math.ceil(w.end)) + 1)):
            coverage[t] += max(0.0, min(w.end, t + 1) - max(w.start, t))
    coverage = np.minimum(coverage, 1.0)
    df["words"] = per_sec_words
    df["n_words"] = [len(x) for x in per_sec_words]
    win = max(1, int(fc["speech_rate_window_s"]))
    rate = np.zeros(T, np.float32)
    for t in range(T):
        lo = max(0, t - (win - 1) // 2)
        hi = min(T, lo + win)
        lo = max(0, hi - win)
        n = int(((starts >= lo) & (starts < hi)).sum()) if starts.size else 0
        rate[t] = n / float(hi - lo)
    df["speech_rate"] = rate
    df["speech_cov"] = coverage
    df["f_speech"] = _clip01(coverage / 0.5)
    df["f_speech_rate"] = _clip01(rate / float(fc["speech_rate_ref"]))
    onset = np.zeros(T, np.float32)
    for t in range(T):
        if coverage[t] >= 0.2 and (t == 0 or coverage[t - 1] < 0.1):
            onset[t] = 1.0
    df["f_speech_onset"] = onset

    # --- texte à l'écran
    ocr = sig.ocr or []
    texts = [o.text for o in ocr] + [""] * (T - len(ocr))
    df["text"] = texts[:T]
    df["text_words"] = [o.words for o in ocr][:T] + [0] * max(0, T - len(ocr))
    df["text_new"] = [o.new for o in ocr][:T] + [False] * max(0, T - len(ocr))
    df["f_text_present"] = (df["text_words"] > 0).astype(float)
    df["f_text_new"] = df["text_new"].astype(float)
    df["f_text_density"] = _clip01(df["text_words"].to_numpy(float) / float(fc["text_density_ref"]))

    # --- audio
    au = sig.audio
    if au is not None:
        rms = au.rms[:T]
        db = np.maximum(au.dbfs[:T], -60.0)
        dstd = au.db_std[:T]
        sil = au.silence[:T]
    else:
        rms = np.zeros(T, np.float32)
        db = np.full(T, -60.0, np.float32)
        dstd = np.zeros(T, np.float32)
        sil = np.ones(T, bool)
    df["audio_rms"] = rms
    df["audio_db"] = db
    df["silence"] = sil.astype(bool)
    df["f_loudness"] = rms
    df["f_silence"] = sil.astype(float)
    lw = max(1, int(fc["loud_var_window_s"]))
    lv = np.zeros(T, np.float32)
    for t in range(T):
        lo = max(0, t - (lw - 1) // 2)
        hi = min(T, lo + lw)
        seg = db[lo:hi]
        lv[t] = math.sqrt(float(np.var(seg)) + float(np.mean(dstd[lo:hi] ** 2)))
    df["f_loud_var"] = _clip01(lv / float(fc["loud_var_ref_db"]))
    jump = np.zeros(T, np.float32)
    jump[1:] = np.abs(np.diff(db))
    df["f_loud_jump"] = _clip01(jump / float(fc["loud_jump_ref_db"]))

    # --- mots-clés : CTA, persuasion, nombres, répétitions, double codage
    cta_kw = keyword_tokens(cfg.weights.get("cta_keywords", []))
    pers_kw = keyword_tokens(cfg.weights.get("persuasion_keywords", []))
    num_words = {t for k in cfg.weights.get("number_words", []) for t in tokens(k)}
    stop = {t for k in cfg.weights.get("stopwords", []) for t in tokens(k)}
    min_len = int(fc["keyword_min_len"])

    word_toks: list[tuple[float, str]] = []      # (instant, token) de la parole
    for w in words:
        for tk in tokens(w.w):
            word_toks.append((w.start, tk))
    tok_list = [tk for _, tk in word_toks]
    cta_terms: list[set[str]] = [set() for _ in range(T)]
    pers = np.zeros(T, np.float32)
    for idx, label in find_in_tokens(tok_list, cta_kw):
        t = int(math.floor(word_toks[idx][0]))
        if 0 <= t < T:
            cta_terms[t].add(label)
    for idx, _label in find_in_tokens(tok_list, pers_kw):
        t = int(math.floor(word_toks[idx][0]))
        if 0 <= t < T:
            pers[t] = 1.0
    number = np.zeros(T, np.float32)
    for t in range(T):
        for label in find_in_text(texts[t], cta_kw):
            cta_terms[t].add(label)
        if find_in_text(texts[t], pers_kw):
            pers[t] = 1.0
        sec_toks = [tk for (s, tk) in word_toks if t <= s < t + 1] + (tokens(texts[t]) if texts[t] else [])
        if any(tk.isdigit() or tk in num_words for tk in sec_toks):
            number[t] = 1.0
    df["cta"] = [bool(x) for x in cta_terms]
    df["cta_terms"] = [sorted(x) for x in cta_terms]
    df["f_cta"] = df["cta"].astype(float)
    df["f_persuasion"] = pers
    df["f_number"] = number

    def content(tk: str) -> bool:
        return len(tk) >= min_len and tk.isalpha() and tk not in stop

    # occurrences (parole : chaque mot ; écran : une fois par nouveau texte)
    events: list[tuple[float, str]] = [(s, tk) for s, tk in word_toks if content(tk)]
    for t in range(T):
        if df.at[t, "text_new"] and texts[t]:
            events.extend((t + 0.5, tk) for tk in set(tokens(texts[t])) if content(tk))
    events.sort()
    seen: dict[str, int] = {}
    repeat = np.zeros(T, np.float32)
    for s, tk in events:
        if seen.get(tk, 0) >= 1:
            t = int(math.floor(s))
            if 0 <= t < T:
                repeat[t] += 1
        seen[tk] = seen.get(tk, 0) + 1
    df["f_keyword_repeat"] = _clip01(repeat / float(fc["keyword_repeat_ref"]))
    ow = float(fc["overlap_window_s"])
    overlap = np.zeros(T, np.float32)
    for t in range(T):
        if not texts[t]:
            continue
        screen = {tk for tk in tokens(texts[t]) if content(tk)}
        if not screen:
            continue
        spoken = {tk for s, tk in word_toks if t - ow <= s < t + 1 + ow}
        overlap[t] = len(screen & spoken) / len(screen)
    df["f_text_speech_overlap"] = overlap

    hook_w = float(cfg.section("score")["hook"]["window_s"])
    df["f_hook_window"] = (t_idx < hook_w).astype(float)

    missing = [f for f in FEATURE_NAMES if f"f_{f}" not in df.columns]
    assert not missing, missing
    return df


def feature_matrix(df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {f: df[f"f_{f}"].to_numpy(float) for f in FEATURE_NAMES}
