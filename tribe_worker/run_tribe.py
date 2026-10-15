#!/usr/bin/env python
"""Worker TRIBE v2 (venv isolé `tribe_worker/.venv`) — contrat : docs/CONTRACTS.md §1.

    tribe_worker/.venv/bin/python tribe_worker/run_tribe.py \
        --video PATH.mp4 --words PATH.json --out PATH.npz \
        --model facebook/tribev2-mini --cache-dir data/cache/tribe [--device auto|cpu|mps] [--no-text]
    tribe_worker/.venv/bin/python tribe_worker/run_tribe.py --no-video --audio PATH.wav --words PATH.json \
        --out PATH.npz [--no-text]          # sans image : audio + texte (simulation de scripts, coach)
    tribe_worker/.venv/bin/python tribe_worker/run_tribe.py --download-only --model REPO [--no-text] [--no-video]

stdout : lignes `PROGRESS <0..1> <message>` ; logs libres sur stderr.
Codes : 0 ok · 1 autre erreur · 2 entrée invalide · 3 accès HF refusé (stderr `ERROR_HF_ACCESS <repo>`)
        · 4 mémoire insuffisante.
Sortie : PATH.npz {preds float16 (N, 20484), seconds int32 (N,)} + PATH.npz.meta.json.
Détails, mesures et contournements : tribe_worker/README.md et docs/RESEARCH.md.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

# Avant tout import de torch / tokenizers.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # op non implémentée sur MPS -> CPU
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import tribe_compat as tc  # noqa: E402  (léger : pas de torch à l'import)
from tribe_events import (InputError, build_events, ffprobe, ffprobe_audio, load_words,  # noqa: E402
                          prepare_audio, prepare_media)

EXIT_OK, EXIT_ERROR, EXIT_INPUT, EXIT_HF, EXIT_OOM = 0, 1, 2, 3, 4
DEFAULT_MODEL = "facebook/tribev2-mini"

# Device effectif de chaque backbone quand --device vaut mps (ou auto sur Apple Silicon).
# Mesuré sur M5 16 Go (README) : tête, w2v-BERT, V-JEPA 2 (fp16) et LLaMA-3.2-1B (fp32) fonctionnent sur MPS,
# résultats équivalents au CPU. En cas d'erreur MPS, audio/texte/tête sont relancés sur CPU.
MPS_POLICY = {"head": "mps", "audio": "mps", "video": "mps", "text": "mps"}
# dtype du LLM texte sur MPS. TRIBE a été entraîné avec des features LLaMA fp32 (transformers 4) ; mesuré sur
# un Reel de 32 s (116 mots) : bf16 vs fp32 -> prédictions corrélées à 0,9999976 (écart moyen 3e-4, contre
# 6e-2 entre avec/sans texte), étape texte 11,5 s au lieu de 24 s, empreinte 6,4 Go au lieu de 8,9 Go.
# => bf16 par défaut sur MPS ; fp32 sur CPU (bf16 sur CPU non mesuré). Surcharge : TRIBE_TEXT_DTYPE=float32.
TEXT_DTYPE = os.environ.get("TRIBE_TEXT_DTYPE", "bfloat16")
# dtype de V-JEPA 2 sur MPS : fp16 = x3 plus rapide que fp32, features quasi identiques (corr. 0.99999998,
# repli fp32 automatique si débordement). Sur CPU : toujours fp32.
VIDEO_DTYPE_MPS = os.environ.get("TRIBE_VIDEO_DTYPE", "float16")
# Poids relatifs (durée mesurée) pour répartir la barre de progression entre les étapes.
# Mesures pour un Reel de 30 s : audio ~5 s, vidéo 180-300 s, texte 15-40 s.
STAGE_WEIGHTS = {"audio": 0.3, "video": 10.0, "text": 1.5}
FEAT_START, FEAT_END = 0.10, 0.93

_t_start = time.time()


def emit(p: float, msg: str) -> None:
    print(f"PROGRESS {min(max(p, 0.0), 1.0):.3f} {msg}", flush=True)


def log(msg: str) -> None:
    print(f"[run_tribe] {msg}", file=sys.stderr, flush=True)


def peak_rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(r / 2**20 if sys.platform == "darwin" else r / 2**10, 1)  # octets sur macOS, Kio sur Linux


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prédiction TRIBE v2 (fsaverage5) pour une vidéo.")
    p.add_argument("--video", type=Path)
    p.add_argument("--words", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--cache-dir", type=Path, default=ROOT / "data" / "cache" / "tribe")
    p.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    p.add_argument("--no-text", action="store_true", help="sans backbone texte (LLaMA, gated)")
    p.add_argument("--no-video", action="store_true",
                   help="sans image (V-JEPA 2 non chargé) : audio + texte ; entrée --audio (ou un fichier audio via --video)")
    p.add_argument("--audio", type=Path, help="fichier audio (wav/aiff/m4a...) pour --no-video")
    p.add_argument("--download-only", action="store_true")
    args = p.parse_args(argv)
    if not args.download_only:
        if args.no_video:
            if args.audio is None:
                args.audio = args.video  # un .wav passé par --video est accepté en mode --no-video
            need = ("audio", "words", "out")
        else:
            if args.audio is not None:
                p.error("--audio n'est utilisable qu'avec --no-video")
            need = ("video", "words", "out")
        missing = [n for n in need if getattr(args, n) is None]
        if missing:
            p.error("arguments requis : " + ", ".join("--" + m for m in missing))  # exit 2
    return args


def check_model_allowed(model: str) -> None:
    if model.rstrip("/") == tc.FULL_MODEL and os.environ.get("TRIBE_ALLOW_FULL") != "1":
        raise InputError("facebook/tribev2 (LLaMA-3.2-3B + V-JEPA2 ViT-g) dépasse 16 Go de RAM : utiliser "
                         "facebook/tribev2-mini (ou TRIBE_ALLOW_FULL=1 à vos risques)")


def go_offline() -> None:
    """Tous les fichiers sont en cache : on interdit tout appel réseau pendant l'analyse."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants as hc
        hc.HF_HUB_OFFLINE = True
    except Exception:  # noqa: BLE001
        pass


def ensure_models(model: str, use_text: bool, use_audio: bool, p0: float, p1: float,
                  use_video: bool = True) -> None:
    _, missing = tc.required_repos(model, use_text, use_audio, use_video)
    if missing:
        log(f"modèles absents du cache HF : {missing} -> téléchargement")
        tc.download_repos(model, use_text, lambda f, m: emit(p0 + (p1 - p0) * f, m), use_audio, use_video)
        _, missing = tc.required_repos(model, use_text, use_audio, use_video)
        if missing:
            raise RuntimeError(f"téléchargement incomplet : {missing}")
    go_offline()


def versions() -> dict:
    out = {}
    try:
        import torch
        import transformers
        out = {"torch": torch.__version__, "transformers": transformers.__version__}
    except Exception:  # noqa: BLE001
        pass
    try:
        from importlib.metadata import distribution
        d = distribution("tribev2")
        commit = ""
        try:
            commit = json.loads(d.read_text("direct_url.json") or "{}").get("vcs_info", {}).get("commit_id", "")[:7]
        except Exception:  # noqa: BLE001
            pass
        out["tribev2"] = d.version + (f"+{commit}" if commit else "")
    except Exception:  # noqa: BLE001
        out["tribev2"] = "inconnu"
    return out


# --------------------------------------------------------------------------------------------
def run_download_only(args) -> int:
    check_model_allowed(args.model)
    use_text = not args.no_text
    emit(0.0, f"Vérification des modèles ({args.model})")
    tc.download_repos(args.model, use_text, emit, use_video=not args.no_video)
    log("téléchargement terminé")
    return EXIT_OK


def resolve_devices(arg: str) -> dict[str, str]:
    import torch

    mps_ok = torch.backends.mps.is_available()
    if arg == "cpu" or (arg == "auto" and not mps_ok):
        return {k: "cpu" for k in MPS_POLICY}
    if arg == "mps" and not mps_ok:
        raise InputError("--device mps demandé mais MPS indisponible (torch.backends.mps.is_available() = False)")
    return dict(MPS_POLICY)


def run_predict(args) -> int:
    stage_times: dict[str, float] = {}
    warnings: list[str] = []
    t = time.time()

    # 1) Entrées (code 2 si invalides) -------------------------------------------------------
    check_model_allowed(args.model)
    use_video = not args.no_video
    media = (args.video if use_video else args.audio).expanduser()
    if not media.is_file():
        raise InputError(f"{'vidéo' if use_video else 'audio'} introuvable : {media}")
    out = args.out.expanduser()
    if out.suffix != ".npz":
        raise InputError(f"--out doit se terminer par .npz : {out}")
    words = load_words(args.words.expanduser())
    info = ffprobe(media) if use_video else ffprobe_audio(media)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta_path = Path(str(out) + ".meta.json")
    for stale in (out, meta_path):
        stale.unlink(missing_ok=True)
    use_text = not args.no_text and len(words["words"]) > 0
    if not args.no_text and not use_text:
        warnings.append("aucun mot transcrit : modalité texte absente (traitée comme le dropout de modalité de TRIBE)")
    if not info["has_audio"]:
        warnings.append("vidéo sans piste audio : modalité audio absente")
    emit(0.01, "Vérification des modèles")

    # 2) Modèles en cache (téléchargement au premier lancement, puis hors ligne) --------------
    ensure_models(args.model, use_text, info["has_audio"], 0.01, 0.05, use_video)
    stage_times["check_models"] = round(time.time() - t, 2)

    # 3) Chargement TRIBE ----------------------------------------------------------------------
    t = time.time()
    emit(0.05, "Chargement de TRIBE v2")
    import numpy as np

    devices = resolve_devices(args.device)
    tc.set_text_dtype({"mps": TEXT_DTYPE, "cpu": os.environ.get("TRIBE_TEXT_DTYPE", "float32")})
    if use_video:
        tc.patch_vjepa2({"mps": VIDEO_DTYPE_MPS, "cpu": "float32"})
    tc.patch_w2vbert_relative_key()
    tc.patch_sdpa_gqa_mps()
    from tribev2 import TribeModel

    cache_dir = args.cache_dir.expanduser().resolve()
    features_dir = cache_dir / "features"
    features = ["audio"] + (["video"] if use_video else []) + (["text"] if use_text else [])
    config_update = {
        # Le config.yaml du checkpoint code `device: cuda` en dur pour chaque extracteur.
        "data.text_feature.device": "cpu",
        "data.audio_feature.device": "cpu",
        "data.video_feature.image.device": "cpu",
        "data.image_feature.image.device": "cpu",
        # Pas de workers DataLoader (20 dans le config) : inutile à l'inférence, et évite le besoin
        # de multiprocessing.set_start_method("fork") sur macOS.
        "data.num_workers": 0,
        # Sans texte (ou sans vidéo) : on n'instancie pas l'extracteur correspondant ; la tête reçoit des
        # zéros pour cette modalité (FmriEncoderModel.aggregate_features), exactement comme le dropout de
        # modalité (p=0.3) vu à l'entraînement.
        "data.features_to_use": [m for m in ("text", "audio", "video") if m in features],
    }
    xp = TribeModel.from_pretrained(args.model, cache_folder=str(features_dir), device=devices["head"],
                                    config_update=config_update)
    # Raccourci exact (moyenne des tokens V-JEPA sur GPU) seulement si la config agrège bien par moyenne.
    if use_video:
        tc.VJEPA_STATE["token_mean"] = getattr(xp.data.video_feature.image, "token_aggregation", None) == "mean"
    stage_times["load_model"] = round(time.time() - t, 2)

    # 4) Événements ----------------------------------------------------------------------------
    t = time.time()
    emit(0.08, "Préparation des événements (vidéo, audio, mots)" if use_video else "Préparation des événements (audio, mots)")
    if use_video:
        link, wav, digest = prepare_media(media, cache_dir / "media", info["has_audio"])
    else:
        link = None
        wav, digest = prepare_audio(media, cache_dir / "media")
    events, n_words = build_events(link, wav, words["words"], words["language"], info["duration"], use_text)
    if wav is None and info["has_audio"]:
        warnings.append("extraction audio impossible : modalité audio absente")
    present = set(events.type.unique())
    mods = [m for m in ("audio", "video", "text") if m in features and
            {"audio": "Audio", "video": "Video", "text": "Word"}[m] in present]
    stage_times["events"] = round(time.time() - t, 2)
    log(f"événements : {events.type.value_counts().to_dict()} ; modalités : {mods} ; devices : {devices}")

    # 5) Features par modalité (cache exca sur disque, indexé par contenu) -------------------
    total_w = sum(STAGE_WEIGHTS[m] for m in mods) or 1.0
    ranges, acc = {}, FEAT_START
    for m in mods:
        span = (FEAT_END - FEAT_START) * STAGE_WEIGHTS[m] / total_w
        ranges[m] = (acc, acc + span)
        acc += span
    labels = {"audio": "audio (w2v-BERT 2.0)", "video": "vidéo (V-JEPA 2)", "text": "texte (LLaMA 3.2)"}
    current = {"m": None}

    def sub_progress(frac: float) -> None:
        m = current["m"]
        if m:
            a, b = ranges[m]
            emit(a + (b - a) * frac, f"Features {labels[m]}")

    tc.install_progress_hooks({"Encoding video": sub_progress, "Computing word embeddings": sub_progress})
    used_devices = {"head": devices["head"]}
    for m in mods:
        t = time.time()
        current["m"] = m
        emit(ranges[m][0], f"Features {labels[m]}")
        extractor = getattr(xp.data, f"{m}_feature")
        dev = devices[m]
        tc.set_extractor_device(extractor, dev)
        try:
            extractor.prepare(events)
        except Exception as e:  # noqa: BLE001
            if dev == "cpu" or tc.access_repo_from_exception(e):
                raise
            if m == "video" and tc.is_oom(e):
                # V-JEPA 2 sur CPU ≈ 22 s par clip (x5 à x7 vs MPS, ~44 s de calcul par seconde de vidéo) :
                # un repli dépasserait le délai du pipeline -> code 4.
                raise
            # Erreur MPS (op manquante, buffer trop grand, OOM GPU...) : on retente sur CPU.
            log(f"échec de l'extracteur {m} sur {dev} ({type(e).__name__}: {str(e)[:300]}) -> repli CPU")
            warnings.append(f"extracteur {m} : repli MPS -> CPU ({type(e).__name__})")
            tc.free_extractor(extractor)
            dev = "cpu"
            tc.set_extractor_device(extractor, dev)
            extractor.prepare(events)
        tc.free_extractor(extractor)
        used_devices[m] = dev
        stage_times[m] = round(time.time() - t, 2)
        log(f"features {m} : {stage_times[m]} s sur {dev} ; RSS pic {peak_rss_mb()} Mo ; "
            f"empreinte pic {tc.peak_footprint_mb()} Mo")
    current["m"] = None

    # 6) Prédiction ----------------------------------------------------------------------------
    t = time.time()
    emit(FEAT_END, "Prédiction de l'activité corticale")
    try:
        preds, segments = xp.predict(events, verbose=False)
    except Exception as e:  # noqa: BLE001
        if devices["head"] == "cpu" or tc.is_oom(e):
            raise
        log(f"échec de la tête sur {devices['head']} ({type(e).__name__}) -> repli CPU")
        warnings.append(f"tête TRIBE : repli {devices['head']} -> CPU ({type(e).__name__})")
        xp._model.to("cpu")
        used_devices["head"] = "cpu"
        preds, segments = xp.predict(events, verbose=False)
    stage_times["predict"] = round(time.time() - t, 2)

    # 7) Écriture ------------------------------------------------------------------------------
    t = time.time()
    emit(0.98, "Écriture des résultats")
    preds = np.asarray(preds, dtype=np.float32)
    # Chaque ligne = segment de 1 TR (1 s) en temps STIMULUS : TRIBE a appris à prédire le BOLD
    # mesuré 5 s plus tard (FmriExtractor offset=5 s) ; aucun décalage supplémentaire à appliquer.
    t0 = float(events.start.min())
    secs = np.array([int(np.floor(s.start - t0 + 1e-3)) for s in segments], dtype=np.int64)
    order = np.argsort(secs, kind="stable")
    secs, preds = secs[order], preds[order]
    # La piste audio dépasse parfois la vidéo de quelques ms (ex. 30,016 s pour 30,000 s) : TRIBE produit
    # alors une ligne pour la seconde 30. On ne garde que t = 0..ceil(durée)-1 (timeline du pipeline).
    n_sec = int(np.ceil(info["duration"] - 1e-3))
    keep = (secs >= 0) & (secs < n_sec)
    if not keep.all():
        log(f"lignes hors de [0, {n_sec}) ignorées : secondes {secs[~keep].tolist()}")
    secs, preds = secs[keep], preds[keep]
    uniq, inv = np.unique(secs, return_inverse=True)
    if len(uniq) != len(secs):  # ne devrait pas arriver (segments de 100 s sans recouvrement)
        warnings.append("secondes dupliquées moyennées")
        agg = np.zeros((len(uniq), preds.shape[1]), dtype=np.float64)
        np.add.at(agg, inv, preds)
        preds = (agg / np.bincount(inv)[:, None]).astype(np.float32)
        secs = uniq
    if preds.ndim != 2 or preds.shape[1] != 20484:
        raise RuntimeError(f"forme de prédiction inattendue : {preds.shape}")
    if not np.isfinite(preds).all():
        raise RuntimeError("prédictions non finies (NaN/inf)")
    tmp = out.with_name(out.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, preds=preds.astype(np.float16), seconds=secs.astype(np.int32))
    os.replace(tmp, out)
    stage_times["save"] = round(time.time() - t, 2)
    runtime = round(time.time() - _t_start, 2)
    dev_summary = used_devices["head"]
    if len(set(used_devices.values())) > 1:
        dev_summary = ",".join(f"{k}:{v}" for k, v in used_devices.items())
    meta = {
        "model": args.model,
        "device": dev_summary,
        "n_rows": int(len(secs)),
        "runtime_s": runtime,
        "stage_times": stage_times,
        "peak_rss_mb": peak_rss_mb(),
        "text_feature": "llama" if "text" in mods else "disabled",
        "video_feature": "vjepa2" if "video" in mods else "disabled",
        "versions": versions(),
        "warnings": warnings,
        # Champs additionnels (hors contrat, informatifs)
        # empreinte mémoire pic (inclut les buffers Metal/MPS, absents de ru_maxrss)
        "peak_footprint_mb": tc.peak_footprint_mb(),
        "devices": used_devices,
        "video_duration_s": round(info["duration"], 3),   # durée du média d'entrée (audio en --no-video)
        "n_words": int(n_words),
        "text_dtype": tc.TEXT_STATE.get(used_devices.get("text", ""), "float32") if "text" in mods else None,
        "video_dtype": (VIDEO_DTYPE_MPS if used_devices.get("video") == "mps" else "float32") if "video" in mods else None,
        "media_digest": digest,
    }
    tmp_meta = Path(str(meta_path) + f".tmp{os.getpid()}")
    tmp_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    os.replace(tmp_meta, meta_path)
    emit(1.0, "Prédiction TRIBE terminée")
    log(f"OK : {meta['n_rows']} lignes, {runtime} s, RSS pic {meta['peak_rss_mb']} Mo, "
        f"empreinte pic {meta['peak_footprint_mb']} Mo")
    return EXIT_OK


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run_download_only(args) if args.download_only else run_predict(args)
    except InputError as e:
        print(f"ERREUR entrée invalide : {e}", file=sys.stderr, flush=True)
        return EXIT_INPUT
    except KeyboardInterrupt:
        print("ERREUR interrompu", file=sys.stderr, flush=True)
        return EXIT_ERROR
    except BaseException as e:  # noqa: BLE001
        repo = tc.access_repo_from_exception(e)
        if repo:
            print(f"ERROR_HF_ACCESS {repo}", file=sys.stderr, flush=True)
            print(f"ERREUR accès Hugging Face refusé pour {repo} : accepter la licence sur "
                  f"https://huggingface.co/{repo} puis `hf auth login` (ou relancer avec --no-text).",
                  file=sys.stderr, flush=True)
            return EXIT_HF
        if tc.is_oom(e):
            print(f"ERREUR mémoire insuffisante : {type(e).__name__}: {str(e)[:500]}", file=sys.stderr, flush=True)
            return EXIT_OOM
        traceback.print_exc(file=sys.stderr)
        print(f"ERREUR {type(e).__name__}: {str(e)[:500]}", file=sys.stderr, flush=True)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
