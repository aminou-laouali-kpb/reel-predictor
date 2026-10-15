"""Pipeline complet : mp4 -> timeline 1 ligne/seconde + résumé + carte cérébrale (contrat §2-§3).

Ordre : étapes légères (probe, scenes, speech, faces, ocr, audio) puis brain (long), indices, rules, write.
`analyze` ne lève une exception (`VideoUnreadableError`) que si la vidéo est illisible : toute autre
défaillance d'étape est absorbée (signal vide + avertissement), un échec TRIBE bascule en heuristique.
"""
from __future__ import annotations

import logging
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import paths
from .audio import analyze as analyze_audio
from .brain.heuristic import HeuristicProvider
from .brain.provider import BrainPrediction, BrainProvider, BrainRequest, BrainUnavailable
from .config import INDEX_NAMES, Config, load_config
from .features import Signals, build_features
from .media import VideoInfo, VideoUnreadableError, extract_audio, extract_frames, probe, read_wav
from .stages import STAGE_LABELS, global_progress

log = logging.getLogger("reel_predictor")

ProgressCallback = Callable[[str, float, str], None]

__all__ = ["AnalysisResult", "analyze", "VideoUnreadableError", "make_provider"]


@dataclass
class AnalysisResult:
    out_dir: Path
    summary: dict
    timeline: list[dict]


class _Reporter:
    """Progression globale = Σ (avancement de chaque étape × son poids), via `stages.global_progress`.

    Les étapes légères tournent en parallèle : chaque étape garde son propre avancement (monotone) et
    la progression globale est la somme de leurs contributions, donc elle ne recule jamais.
    """

    def __init__(self, cb: ProgressCallback | None):
        self.cb = cb
        self._frac: dict[str, float] = {}
        self._last_g = -1.0
        self._lock = threading.Lock()

    def __call__(self, stage: str, pct: float, message: str | None = None) -> None:
        if self.cb is None:
            return
        with self._lock:
            pct = max(0.0, min(1.0, float(pct)))
            self._frac[stage] = max(self._frac.get(stage, 0.0), pct)
            g = sum(global_progress(s, f) - global_progress(s, 0.0) for s, f in self._frac.items())
            g = round(min(1.0, max(self._last_g, g)), 4)
            if g == self._last_g and message is None:
                return
            self._last_g = g
            try:
                self.cb(stage, g, message or STAGE_LABELS.get(stage, stage))
            except Exception:  # noqa: BLE001 — un callback défaillant ne doit pas casser l'analyse
                log.exception("callback de progression en erreur")

    def sub(self, stage: str) -> Callable[[float, str], None]:
        return lambda pct, msg=None: self(stage, pct, msg)


def make_provider(cfg: Config) -> BrainProvider:
    """Fournisseur selon `brain.provider` (tribe | heuristic | mock)."""
    p = cfg.brain.provider
    if p == "tribe":
        from .brain.tribe import TribeProvider

        return TribeProvider(cfg.brain)
    if p == "mock":
        from .brain.mock import MockProvider

        return MockProvider(cfg.roi_mapping)
    return HeuristicProvider()


def _first_word(signals: Signals) -> float | None:
    if signals.transcript and signals.transcript.words:
        return float(signals.transcript.words[0].start)
    return None


def analyze(video_path: Path, out_dir: Path, config: Config | None = None,
            progress: ProgressCallback | None = None, brain: BrainProvider | None = None) -> AnalysisResult:
    t_all = time.time()
    video_path, out_dir = Path(video_path), Path(out_dir)
    cfg = config or load_config()
    out_dir.mkdir(parents=True, exist_ok=True)
    paths.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rep = _Reporter(progress)
    timings: dict[str, float] = {}
    warnings: list[str] = []

    def warn(msg: str) -> None:
        log.warning(msg)
        warnings.append(msg)

    # ------------------------------------------------------------------ probe
    rep("probe", 0.0)
    t0 = time.time()
    info: VideoInfo = probe(video_path)              # VideoUnreadableError si illisible
    T = info.n_seconds
    timings["probe"] = round(time.time() - t0, 3)
    rep("probe", 1.0, f"Vidéo de {info.duration_s:.1f} s ({info.width}x{info.height})")
    sig = Signals(duration_s=info.duration_s, n_seconds=T)
    tmp = Path(tempfile.mkdtemp(prefix=f"job_{out_dir.name}_", dir=paths.CACHE_DIR))
    try:
        # ---------------------------------------------- étapes légères (en parallèle)
        # scenes | speech | images -> visages -> OCR : trois fils (ffmpeg, CTranslate2, onnxruntime et
        # OpenCV relâchent le GIL). Les résultats sont relus dans l'ordre des étapes.
        smp = cfg.sampling

        def run_scenes() -> dict:
            t0 = time.time()
            rep("scenes", 0.0)
            sc = cfg.section("scenes")
            out: dict = {"cuts": [], "warnings": []}
            try:
                from .scenes import detect_cuts

                cuts, _method, w = detect_cuts(video_path, info.duration_s, info.fps, sc["detector"],
                                               float(sc["threshold"]), float(sc["min_scene_len_s"]),
                                               float(sc["ffmpeg_threshold"]), progress=rep.sub("scenes"))
                out["cuts"] = cuts
                if w:
                    out["warnings"].append(w)
            except Exception as e:  # noqa: BLE001
                out["warnings"].append(f"Détection des plans impossible : {type(e).__name__}: {e}")
            out["time"] = time.time() - t0
            rep("scenes", 1.0, f"{len(out['cuts'])} changement(s) de plan")
            return out

        def run_speech() -> dict:
            from .speech import Transcript

            t0 = time.time()
            rep("speech", 0.0)
            out: dict = {"samples": None, "sr": 16000, "warnings": [],
                         "transcript": Transcript(language=cfg.speech.language, language_probability=0.0, text="")}
            if info.has_audio:
                try:
                    wav = extract_audio(video_path, tmp / "audio.wav", 16000)
                    out["samples"], out["sr"] = read_wav(wav)
                except Exception as e:  # noqa: BLE001
                    out["warnings"].append(f"Piste audio illisible : {e}")
            if out["samples"] is not None and out["samples"].size:
                try:
                    from .speech import transcribe

                    out["transcript"] = transcribe(out["samples"], out["sr"], cfg.speech, progress=rep.sub("speech"))
                except Exception as e:  # noqa: BLE001
                    out["warnings"].append(f"Transcription impossible : {type(e).__name__}: {e}")
            out["time"] = time.time() - t0
            rep("speech", 1.0, f"{len(out['transcript'].words)} mot(s) transcrit(s)")
            return out

        def run_visual() -> dict:
            out: dict = {"frames": None, "ocr": None, "warnings": [], "error": None}
            t0 = time.time()
            rep("faces", 0.0, "Extraction des images")
            try:
                frames = extract_frames(video_path, tmp / "frames", smp.frames_fps, smp.frame_width)
            except Exception as e:  # noqa: BLE001
                out["error"] = f"Impossible de décoder les images de la vidéo : {e}"
                return out
            if len(frames) == 0:
                out["error"] = "Impossible de décoder les images de la vidéo"
                return out
            out["time_frames"] = time.time() - t0
            t1 = time.time()
            try:
                from .faces import analyze_frames

                fstats, w = analyze_frames(frames, float(cfg.section("faces")["min_confidence"]),
                                           progress=lambda p, m=None: rep("faces", 0.1 + 0.9 * p, m))
                out["frames"] = fstats
                if w:
                    out["warnings"].append(w)
            except Exception as e:  # noqa: BLE001
                out["warnings"].append(f"Analyse des visages impossible : {type(e).__name__}: {e}")
            out["time_faces"] = time.time() - t1
            rep("faces", 1.0)
            t2 = time.time()
            rep("ocr", 0.0)
            oc = cfg.section("ocr")
            try:
                from .ocr import read_text

                out["ocr"] = read_text(frames, T, smp.ocr_fps, float(oc["min_score"]), int(oc["min_chars"]),
                                       float(oc["same_text_similarity"]), int(oc["det_limit_side_len"]),
                                       str(oc["det_limit_type"]), bool(oc["use_angle_cls"]),
                                       float(oc["reuse_if_diff_below"]), progress=rep.sub("ocr"))
            except Exception as e:  # noqa: BLE001
                out["warnings"].append(f"OCR impossible : {type(e).__name__}: {e}")
            out["time_ocr"] = time.time() - t2
            rep("ocr", 1.0)
            return out

        with ThreadPoolExecutor(max_workers=3, thread_name_prefix="reel") as ex:
            f_scenes = ex.submit(run_scenes)
            f_visual = ex.submit(run_visual)
            f_speech = ex.submit(run_speech)
            r_scenes = f_scenes.result()
            r_speech = f_speech.result()
            r_visual = f_visual.result()
        if r_visual["error"]:
            raise VideoUnreadableError(r_visual["error"])
        for r in (r_scenes, r_speech, r_visual):
            for w in r["warnings"]:
                warn(w)
        sig.cuts = r_scenes["cuts"]
        transcript = r_speech["transcript"]
        samples, sr = r_speech["samples"], r_speech["sr"]
        sig.transcript = transcript
        sig.frames = r_visual["frames"]
        sig.ocr = r_visual["ocr"]
        timings["scenes"] = round(r_scenes["time"], 3)
        timings["speech"] = round(r_speech["time"], 3)
        timings["frames"] = round(r_visual.get("time_frames", 0.0), 3)
        timings["faces"] = round(r_visual.get("time_faces", 0.0), 3)
        timings["ocr"] = round(r_visual.get("time_ocr", 0.0), 3)
        words_path = out_dir / "words.json"
        from .timeline import write_json

        write_json(words_path, transcript.words_json(info.duration_s))
        timings["light_stages_wall"] = round(time.time() - t_all, 3)

        # -------------------------------------------------------------- audio
        rep("audio", 0.0)
        t0 = time.time()
        ac = cfg.section("audio")
        sig.audio = analyze_audio(samples, sr, T, float(ac["window_s"]), float(ac["silence_dbfs"]),
                                  float(ac["silence_rel"]))
        df = build_features(sig, cfg)
        timings["audio"] = round(time.time() - t0, 3)
        rep("audio", 1.0)

        # -------------------------------------------------------------- brain
        rep("brain", 0.0)
        t0 = time.time()
        provider = brain if brain is not None else make_provider(cfg)
        req = BrainRequest(video_path=video_path, words_path=words_path, duration_s=info.duration_s,
                           n_seconds=T, work_dir=out_dir, features=df, progress=rep.sub("brain"))
        pred: BrainPrediction
        try:
            pred = provider.predict(req)
        except BrainUnavailable as e:
            warn(f"Modèle cérébral indisponible : {e.reason}" + (f" ({e.detail})" if e.detail else ""))
            pred = HeuristicProvider(e.reason, e.detail).predict(req)
        except Exception as e:  # noqa: BLE001
            warn(f"Erreur du modèle cérébral : {type(e).__name__}: {e}")
            log.debug(traceback.format_exc())
            pred = HeuristicProvider(f"Erreur inattendue du modèle cérébral ({type(e).__name__})", str(e)).predict(req)
        warnings.extend(pred.warnings)
        roi_names = list(cfg.roi_mapping["rois"])
        roi_corpus = roi_video = aligned = None
        ref_info = {"n_reels": 0, "used": False}
        if pred.available:
            try:
                from .brain.atlas import roi_matrix
                from .brain.reference import Reference
                from .indices import align_predictions, normalize_corpus, roi_values, zscore_video

                norm = cfg.section("normalization")
                roi_names, W = roi_matrix(cfg.roi_mapping)
                aligned = align_predictions(pred.preds, pred.seconds, T, float(norm["hrf_shift_s"]))
                raw = roi_values(aligned, W)
                ref = Reference.load()
                text_mode = pred.meta.get("text_feature") or "llama"
                med, mad, used = ref.stats(roi_names, int(norm["reference_min_reels"]), mode=text_mode)
                ref_info = {"n_reels": ref.count(text_mode), "used": bool(used), "text_feature": text_mode}
                roi_corpus = normalize_corpus(raw, med, mad, float(norm["mad_scale"]))
                roi_video = zscore_video(raw)
                if pred.provider == "tribe":
                    try:
                        ref.add(out_dir.name, roi_names, raw, int(norm["reference_max_seconds_per_reel"]), mode=text_mode)
                        ref.save()
                    except Exception as e:  # noqa: BLE001
                        warn(f"Référence de corpus non mise à jour : {e}")
            except Exception as e:  # noqa: BLE001
                warn(f"Réduction en ROI impossible : {type(e).__name__}: {e}")
                pred = HeuristicProvider("Atlas cortical indisponible", str(e)).predict(req)
                roi_names = list(cfg.roi_mapping["rois"])
                roi_corpus = roi_video = aligned = None
        timings["brain"] = round(time.time() - t0, 3)
        rep("brain", 1.0, "Modèle cérébral terminé" if pred.available else "Mode heuristique (sans cerveau)")

        # -------------------------------------------------------------- indices + rétention + score
        rep("indices", 0.0)
        t0 = time.time()
        from .indices import compute_indices
        from .retention import simulate
        from .score import compute_score

        src = None
        if roi_corpus is not None:
            src = roi_corpus if cfg.section("normalization")["brain_roi_source"] == "corpus" else roi_video
        idx = compute_indices(df, cfg, src, roi_names if src is not None else None)
        for k in INDEX_NAMES:
            df[f"idx_{k}"] = idx.values[k]
        rp = cfg.retention
        ret = simulate(idx.values["attention"], df["f_cut_recent"].to_numpy(float), info.duration_s,
                       hook_seconds=rp.hook_seconds, hook_drop=rp.hook_drop, base_drop=rp.base_drop,
                       k_attention=rp.k_attention, cut_bonus=rp.cut_bonus, allow_increase=rp.allow_increase)
        first_word = _first_word(sig)
        sc_res = compute_score(df, idx.values, ret, first_word, cfg)
        timings["indices"] = round(time.time() - t0, 3)
        rep("indices", 1.0)

        # -------------------------------------------------------------- rules
        rep("rules", 0.0)
        t0 = time.time()
        from .rules import RuleContext, evaluate

        ctx = RuleContext(df=df, duration_s=info.duration_s, first_word_s=first_word,
                          has_speech=bool(transcript.words), face_ok=bool(sig.frames and sig.frames.face_ok),
                          retention=ret.per_second)
        recs, rule_warns = evaluate(ctx, cfg.rules)
        warnings.extend(rule_warns)
        timings["rules"] = round(time.time() - t0, 3)
        rep("rules", 1.0, f"{len(recs)} recommandation(s)")

        # -------------------------------------------------------------- write
        rep("write", 0.0)
        t0 = time.time()
        from .timeline import build_rows, write_brain, write_json, write_timeline

        hook_w = float(cfg.section("score")["hook"]["window_s"])
        flags: list[list[str]] = []
        for t in range(T):
            f: list[str] = []
            has_brain_t = pred.available and src is not None and np.isfinite(src[t]).any()
            if not has_brain_t:
                f.append("no_brain")
            elif t < cfg.brain.low_context_s:
                f.append("low_context")
            elif t >= T - cfg.brain.low_context_end_s:
                f.append("low_context_end")
            if t < hook_w:
                f.append("hook")
            if bool(df.at[t, "cta"]):
                f.append("cta")
            flags.append(f)
        rows = build_rows(df, idx.values, ret.per_second, roi_names, roi_corpus, roi_video, flags)
        write_timeline(out_dir / "timeline.jsonl", rows)
        brain_meta = None
        if pred.available and aligned is not None:
            brain_meta = write_brain(out_dir, aligned)
        else:
            for name in ("brain.u8", "brain_meta.json"):
                (out_dir / name).unlink(missing_ok=True)
        conf_map = cfg.roi_mapping.get("confidence", {}) or {}
        confidence = {k: (conf_map.get(k, "low") if pred.available else "low") for k in INDEX_NAMES}
        timings["write"] = round(time.time() - t0, 3)
        timings["total"] = round(time.time() - t_all, 3)
        summary = {
            "version": 1,
            "job_id": out_dir.name,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "video": info.as_summary(),
            "score": int(sc_res.score),
            "score_breakdown": sc_res.breakdown,
            "retention": {
                "curve": [round(float(x), 4) for x in ret.curve],
                "avg_watch_ratio": round(ret.avg_watch_ratio, 4),
                "avg_watch_s": round(ret.avg_watch_s, 2),
                "completion": round(ret.completion, 4),
                "calibrated": cfg.calibration is not None,
            },
            "recommendations": recs,
            "cuts": [round(float(c), 3) for c in sig.cuts],
            "transcript": {"language": transcript.language, "text": transcript.text},
            "brain": {
                "available": bool(pred.available),
                "provider": pred.provider,
                "model": pred.model,
                "degraded": bool(pred.degraded),
                "reason": pred.reason,
                "rois": roi_names,
                "confidence": confidence,
                "text_feature": pred.meta.get("text_feature"),
                "reference": ref_info,
                "runtime_s": timings["brain"],
                "brain_meta": brain_meta,
            },
            "timings": timings,
            "config": cfg.effective_dict(),
            "warnings": warnings,
        }
        write_json(out_dir / "summary.json", summary)
        rep("write", 1.0, "Analyse terminée")
        return AnalysisResult(out_dir=out_dir, summary=summary, timeline=rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
