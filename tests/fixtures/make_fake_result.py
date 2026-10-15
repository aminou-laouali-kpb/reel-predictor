"""Génère un dossier de résultat FACTICE conforme au contrat §3 (docs/CONTRACTS.md).

Sert à développer et tester l'app sans le pipeline réel :
    summary.json, timeline.jsonl, brain.u8 + brain_meta.json (activité lisse sur fsaverage5),
    video.mp4 (généré par ffmpeg si absent du dossier).

Usage CLI :
    uv run python tests/fixtures/make_fake_result.py data/results/20260101-120000-deadbeef [--duration 30] [--no-brain]

Aucune donnée réelle : toutes les valeurs sont synthétiques (sinusoïdes lissées, déterministes).
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

N_VERTICES = 20484
HEMI_SPLIT = 10242
INDEX_KEYS = ["attention", "action", "tension", "manipulation", "emotion", "memory"]
ROIS = ["ips", "fef", "early_visual", "mt", "precentral", "insula_ant", "dacc", "ofc",
        "angular", "psts", "parahippocampal", "ifg", "auditory"]

FAKE_WORDS = ("Bonjour à tous aujourd'hui je vous montre une astuce simple pour gagner du temps "
              "chaque matin regardez bien ce qui se passe ici ensuite on passe à la deuxième étape "
              "et franchement le résultat est bluffant abonnez-vous pour la suite").split()


# --------------------------------------------------------------------------- vidéo
def _ffprobe_duration(video: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video)],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip()
        d = float(out)
        return d if d > 0 else None
    except Exception:
        return None


def _ffprobe_size(video: Path) -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
             "-of", "csv=p=0:s=x", str(video)],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout.strip().splitlines()[0]
        w, h = (int(x) for x in out.split("x")[:2])
        return w, h
    except Exception:
        return None


def make_fake_video(path: Path, duration_s: float = 30.0, width: int = 360, height: int = 640) -> Path:
    """Vidéo de test H.264 + AAC (mire animée + bip), lisible par les navigateurs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    d = f"{duration_s:.3f}"
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc2=size={width}x{height}:rate=30:duration={d}",
        "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=44100:duration={d}",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p", "-g", "30",
        "-c:a", "aac", "-b:a", "64k", "-shortest", "-movflags", "+faststart", str(path),
    ]
    try:
        subprocess.run(cmd, check=True, timeout=120)
    except Exception:
        # Repli : encodeur matériel macOS
        cmd[cmd.index("libx264")] = "h264_videotoolbox"
        i = cmd.index("-preset")
        del cmd[i:i + 2]
        subprocess.run(cmd, check=True, timeout=120)
    return path


# --------------------------------------------------------------------------- cerveau
def _vertex_positions() -> np.ndarray:
    """Positions 3D (sphère fsaverage5) pour une activité spatialement lisse ; repli sans nilearn."""
    try:
        from nilearn import datasets, surface

        fs = datasets.fetch_surf_fsaverage("fsaverage5")
        parts = []
        for hemi in ("left", "right"):
            mesh = surface.load_surf_mesh(fs[f"sphere_{hemi}"])
            c = np.asarray(mesh.coordinates if hasattr(mesh, "coordinates") else mesh[0], dtype=np.float64)
            c = c / np.linalg.norm(c, axis=1, keepdims=True)
            parts.append(c)
        pos = np.concatenate(parts, axis=0)
        if pos.shape == (N_VERTICES, 3):
            return pos
    except Exception:
        pass
    # Repli : spirale de Fibonacci par hémisphère (lisse en index)
    n = HEMI_SPLIT
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5 ** 0.5) * i
    sph = np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)
    return np.concatenate([sph, sph], axis=0)


def make_fake_brain(T: int, seed: int = 0) -> tuple[np.ndarray, dict]:
    """Activité (T, 20484) lisse dans l'espace et le temps -> (uint8, meta) selon le contrat."""
    rng = np.random.default_rng(seed)
    pos = _vertex_positions()
    hemi = np.zeros(N_VERTICES)
    hemi[HEMI_SPLIT:] = 1.0
    n_blobs = 6
    centers = rng.normal(size=(n_blobs, 3))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    widths = rng.uniform(0.35, 0.7, size=n_blobs)
    signs = np.array([1, -1, 1, 1, -1, 1], dtype=float)
    freqs = rng.uniform(0.05, 0.25, size=n_blobs)
    phases = rng.uniform(0, 2 * math.pi, size=n_blobs)
    lat = rng.uniform(-0.5, 0.5, size=n_blobs)  # latéralisation
    spatial = np.stack([
        np.exp(-np.sum((pos - centers[k]) ** 2, axis=1) / (2 * widths[k] ** 2)) * (1 + lat[k] * (2 * hemi - 1))
        for k in range(n_blobs)
    ])  # (K, V)
    t = np.arange(T)[:, None]
    amp = signs * np.sin(2 * math.pi * freqs * t + phases)  # (T, K)
    values = amp @ spatial  # (T, V)
    values += 0.15 * np.sin(0.3 * t)  # dérive lente globale
    vmax = float(np.percentile(np.abs(values), 98)) or 1.0
    vmin = -vmax
    u8 = np.clip(np.round((values - vmin) / (vmax - vmin) * 255.0), 0, 255).astype(np.uint8)
    meta = {"T": int(T), "n_vertices": N_VERTICES, "hemi_split": HEMI_SPLIT, "vmin": vmin, "vmax": vmax,
            "space": "fsaverage5", "units": "z (TRIBE, niveau session)"}
    return np.ascontiguousarray(u8), meta


# --------------------------------------------------------------------------- timeline / summary
def _smooth_series(T: int, seed: int, base: float, amp: float) -> list[float]:
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    f1, f2 = rng.uniform(0.04, 0.12), rng.uniform(0.15, 0.35)
    p1, p2 = rng.uniform(0, 2 * math.pi, size=2)
    s = base + amp * (0.7 * np.sin(2 * math.pi * f1 * t + p1) + 0.3 * np.sin(2 * math.pi * f2 * t + p2))
    return [float(round(min(1.0, max(0.0, v)), 4)) for v in s]


def make_fake_timeline(duration_s: float, brain: bool = True, seed: int = 0) -> tuple[list[dict], list[float], list[str]]:
    T = max(1, math.ceil(duration_s))
    series = {k: _smooth_series(T, seed + i, 0.5, 0.3) for i, k in enumerate(INDEX_KEYS)}
    # creux d'attention marqué autour de 40 % de la vidéo (pour déclencher des recommandations)
    dip_c = int(T * 0.4)
    for dt in range(-2, 3):
        if 0 <= dip_c + dt < T:
            series["attention"][dip_c + dt] = round(max(0.05, series["attention"][dip_c + dt] - 0.3 + 0.08 * abs(dt)), 4)
    cuts = sorted({round(x, 2) for x in (2.4, 5.1, 9.8, 14.2, 21.5, 26.7) if x < duration_s})
    words_per_sec: list[list[str]] = [[] for _ in range(T)]
    wi = 0
    for s in range(T):
        n = 0 if (dip_c <= s <= dip_c + 1) else (2 if s % 3 else 3)
        for _ in range(n):
            if wi < len(FAKE_WORDS):
                words_per_sec[s].append(FAKE_WORDS[wi])
                wi += 1
    texts = {0: "ASTUCE DU MATIN", 1: "ASTUCE DU MATIN", 2: "ASTUCE DU MATIN", 8: "Étape 1", 9: "Étape 1",
             17: "Étape 2", 18: "Étape 2", T - 2: "Abonne-toi !", T - 1: "Abonne-toi !"}
    retention, r = [], 1.0
    timeline = []
    last_cut = 0.0
    prev_text = ""
    for s in range(T):
        att = series["attention"][s]
        drop = 0.035 + 0.05 * (1 - att) + (0.06 if s < 3 else 0.0)
        r = max(0.02, r * (1 - drop))
        retention.append(round(r, 4))
        cut_here = any(s <= c < s + 1 for c in cuts)
        cuts_before = [c for c in cuts if c < s + 1]
        if cuts_before:
            last_cut = cuts_before[-1]
        rate_window = [len(words_per_sec[k]) for k in range(max(0, s - 2), s + 1)]
        text = texts.get(s, "")
        flags = []
        if s < 5:
            flags.append("low_context")
        if s < 3:
            flags.append("hook")
        if s >= T - 2:
            flags.append("cta")
        if not brain:
            flags.append("no_brain")
        roi = {name: (round(float(math.sin(0.3 * s + i)), 3) if brain else None) for i, name in enumerate(ROIS)}
        roi_z = {name: (round(float(math.cos(0.25 * s + i)), 3) if brain else None) for i, name in enumerate(ROIS)}
        timeline.append({
            "t": s,
            "roi": roi,
            "roi_video_z": roi_z,
            "indices": {k: series[k][s] for k in INDEX_KEYS},
            "cut": cut_here,
            "since_cut": round((s + 1) - last_cut, 2),
            "faces": 1 if (s % 7) < 5 else 0,
            "face_area": round(0.12 + 0.05 * math.sin(s / 4), 3) if (s % 7) < 5 else 0.0,
            "words": words_per_sec[s],
            "speech_rate": round(sum(rate_window) / len(rate_window), 2),
            "text": text,
            "text_words": len(text.split()),
            "text_new": bool(text) and text != prev_text,
            "audio_rms": round(0.0 if (dip_c <= s <= dip_c + 1) else 0.4 + 0.3 * abs(math.sin(s / 3)), 3),
            "silence": dip_c <= s <= dip_c + 1,
            "retention": round(r, 4),
            "flags": flags,
        })
        prev_text = text
    return timeline, cuts, [w for ws in words_per_sec for w in ws]


def make_fake_summary(job_id: str, filename: str, duration_s: float, timeline: list[dict], cuts: list[float],
                      words: list[str], brain: bool, width: int, height: int) -> dict:
    T = len(timeline)
    means = {k: round(float(np.mean([row["indices"][k] for row in timeline])), 4) for k in INDEX_KEYS}
    weights = {"attention": 0.3, "action": 0.1, "tension": 0.15, "manipulation": 0.1, "emotion": 0.2, "memory": 0.15}
    base = sum(means[k] * weights[k] for k in INDEX_KEYS) * 100
    hook_bonus, dip_penalty = 4.0, -6.0
    score = int(max(0, min(100, round(base + hook_bonus + dip_penalty + 10))))
    curve = [1.0] + [row["retention"] for row in timeline]
    avg_ratio = round(float(np.mean(curve[1:])), 4)
    dip_c = int(T * 0.4)
    recs = [
        {"t": 0.0, "t_end": 3.0, "rule": "weak_hook", "severity": "medium",
         "message": "Accroche : ajouter un texte à l'écran dès 0 s pour annoncer le bénéfice."},
        {"t": float(max(0, dip_c - 1)), "t_end": float(min(T, dip_c + 3)), "rule": "attention_drop_no_cut",
         "severity": "high",
         "message": f"Creux d'attention sans cut : couper ou ajouter un plan/texte à {max(0, dip_c - 1)} s."},
        {"t": float(dip_c), "t_end": float(min(T, dip_c + 2)), "rule": "silence", "severity": "medium",
         "message": f"Silence de 2 s à {dip_c} s : raccourcir ou ajouter de la musique."},
        {"t": float(max(0, T - 3)), "t_end": float(T), "rule": "late_cta", "severity": "low",
         "message": f"Appel à l'action tardif ({max(0, T - 3)} s) : l'annoncer plus tôt."},
    ]
    conf = {"attention": "medium", "action": "low", "tension": "medium", "manipulation": "low",
            "emotion": "low", "memory": "low"}
    return {
        "version": 1,
        "job_id": job_id,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "video": {"filename": filename, "duration_s": round(duration_s, 3), "fps": 30.0, "width": width,
                  "height": height, "has_audio": True},
        "score": score,
        "score_breakdown": {"index_means": means, "hook_bonus": hook_bonus, "dip_penalty": dip_penalty,
                            "weights": weights},
        "retention": {"curve": curve, "avg_watch_ratio": avg_ratio, "avg_watch_s": round(avg_ratio * duration_s, 2),
                      "completion": curve[-1]},
        "recommendations": recs,
        "cuts": cuts,
        "transcript": {"language": "fr", "text": " ".join(words)},
        "brain": {
            "available": brain,
            "provider": "tribe" if brain else "heuristic",
            "model": "facebook/tribev2-mini" if brain else None,
            "degraded": not brain,
            "reason": None if brain else "Résultat factice : TRIBE désactivé (REEL_PREDICTOR_FAKE).",
            "rois": ROIS if brain else [],
            "confidence": conf,
        },
        "timings": {"probe": 0.1, "scenes": 0.5, "speech": 2.0, "faces": 1.0, "ocr": 1.5, "audio": 0.1,
                    "brain": 30.0 if brain else 0.0, "indices": 0.1, "rules": 0.05, "write": 0.05},
        "config": {"weights": {}, "roi_mapping": {}, "rules": {}},
        "fake": True,
    }


def make_fake_result(out_dir: Path | str, duration_s: float | None = None, brain: bool = True,
                     job_id: str | None = None, filename: str | None = None, seed: int = 0) -> Path:
    """Remplit `out_dir` avec un résultat factice complet. Réutilise `out_dir/video.mp4` s'il existe."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    video = out_dir / "video.mp4"
    width, height = 360, 640
    probed = _ffprobe_duration(video) if video.exists() else None
    if probed is None:
        duration_s = float(duration_s or 30.0)
        make_fake_video(video, duration_s, width, height)
    else:
        duration_s = probed
        width, height = _ffprobe_size(video) or (width, height)
    timeline, cuts, words = make_fake_timeline(duration_s, brain=brain, seed=seed)
    summary = make_fake_summary(job_id or out_dir.name, filename or "video.mp4", duration_s, timeline, cuts, words,
                                brain, width, height)
    with open(out_dir / "timeline.jsonl", "w", encoding="utf-8") as fh:
        for row in timeline:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if brain:
        u8, meta = make_fake_brain(len(timeline), seed=seed)
        (out_dir / "brain.u8").write_bytes(u8.tobytes(order="C"))
        (out_dir / "brain_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    else:
        for name in ("brain.u8", "brain_meta.json"):
            (out_dir / name).unlink(missing_ok=True)
    # summary.json en dernier : sa présence signifie « résultat complet »
    tmp = out_dir / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(out_dir / "summary.json")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Génère un résultat d'analyse factice (contrat §3).")
    p.add_argument("out_dir", type=Path)
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--no-brain", action="store_true", help="simule le mode dégradé (pas de carte cérébrale)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--video", type=Path, help="vidéo à copier en video.mp4 (sinon mire générée)")
    a = p.parse_args(argv)
    if a.video:
        a.out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(a.video, a.out_dir / "video.mp4")
    out = make_fake_result(a.out_dir, a.duration, brain=not a.no_brain, seed=a.seed)
    print(f"Résultat factice écrit dans {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
