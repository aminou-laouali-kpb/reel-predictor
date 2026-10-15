"""Chargement et validation de la configuration YAML (`config/`).

`load_config()` lit `weights.yaml`, `roi_mapping.yaml`, `rules.yaml`, valide leur structure et
applique la surcharge de calibration (`data/calibration.json` > `weights.retention`).
`Config.raw` contient les YAML bruts (contrat §2) ; `Config.weights` la version effective
(calibration appliquée).
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from . import paths

INDEX_NAMES: tuple[str, ...] = ("attention", "action", "tension", "manipulation", "emotion", "memory")
FEATURE_NAMES: tuple[str, ...] = (
    "cut", "cut_recent", "since_cut_long", "motion", "face_present", "face_close", "speech",
    "speech_rate", "speech_onset", "text_present", "text_new", "text_density", "cta", "persuasion",
    "loudness", "loud_var", "loud_jump", "silence", "keyword_repeat", "text_speech_overlap",
    "number", "hook_window",
)
RETENTION_PARAMS: tuple[str, ...] = ("hook_drop", "base_drop", "k_attention", "cut_bonus")
SEVERITIES = ("high", "medium", "low")
BRAIN_PROVIDERS = ("tribe", "heuristic", "mock")


class ConfigError(ValueError):
    """Configuration invalide (message en français, liste des problèmes)."""


@dataclass(frozen=True)
class BrainConfig:
    provider: str
    model: str
    device: str
    timeout_s: float
    no_text: bool
    retry_without_text: bool
    low_context_s: float
    low_context_end_s: float = 3.0


@dataclass(frozen=True)
class SpeechConfig:
    model: str
    language: str | None          # None = détection automatique
    device: str
    compute_type: str
    beam_size: int
    vad_filter: bool
    cpu_threads: int


@dataclass(frozen=True)
class SamplingConfig:
    frames_fps: float
    ocr_fps: float
    frame_width: int


@dataclass(frozen=True)
class RetentionParams:
    hook_seconds: float
    hook_drop: float
    base_drop: float
    k_attention: float
    cut_bonus: float
    allow_increase: bool

    def as_dict(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in RETENTION_PARAMS}


@dataclass
class Config:
    weights: dict[str, Any]           # weights.yaml effectif (calibration appliquée)
    roi_mapping: dict[str, Any]
    rules: list[dict[str, Any]]
    raw: dict[str, Any]               # {"weights", "roi_mapping", "rules"} : YAML bruts
    brain: BrainConfig
    speech: SpeechConfig
    sampling: SamplingConfig
    retention: RetentionParams
    calibration: dict[str, Any] | None = None
    config_dir: Path = field(default_factory=lambda: paths.CONFIG_DIR)

    @property
    def index_names(self) -> tuple[str, ...]:
        return INDEX_NAMES

    def section(self, name: str) -> dict[str, Any]:
        return self.weights.get(name, {}) or {}

    def with_brain_provider(self, provider: str) -> "Config":
        """Copie de la config avec un autre fournisseur cérébral (CLI --brain)."""
        if provider not in BRAIN_PROVIDERS:
            raise ConfigError(f"Fournisseur cérébral inconnu : {provider}")
        new = copy.copy(self)
        new.weights = copy.deepcopy(self.weights)
        new.weights["brain"]["provider"] = provider
        new.brain = _brain(new.weights["brain"])
        return new

    def effective_dict(self) -> dict[str, Any]:
        """Config telle qu'utilisée (copiée dans summary.json)."""
        return {"weights": copy.deepcopy(self.weights), "roi_mapping": copy.deepcopy(self.roi_mapping),
                "rules": copy.deepcopy(self.raw["rules"])}


# ----------------------------------------------------------------------------- validation
class _V:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def err(self, msg: str) -> None:
        self.errors.append(msg)

    def get(self, d: Any, key: str, where: str) -> Any:
        if not isinstance(d, dict) or key not in d:
            self.err(f"{where}.{key} manquant")
            return None
        return d[key]

    def num(self, d: Any, key: str, where: str, lo: float | None = None, hi: float | None = None) -> float:
        v = self.get(d, key, where)
        if v is None:
            return 0.0
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            self.err(f"{where}.{key} doit être un nombre (reçu {v!r})")
            return 0.0
        if lo is not None and v < lo:
            self.err(f"{where}.{key} = {v} < {lo}")
        if hi is not None and v > hi:
            self.err(f"{where}.{key} = {v} > {hi}")
        return float(v)

    def boolean(self, d: Any, key: str, where: str) -> bool:
        v = self.get(d, key, where)
        if v is None:
            return False
        if not isinstance(v, bool):
            self.err(f"{where}.{key} doit être true/false (reçu {v!r})")
        return bool(v)

    def string(self, d: Any, key: str, where: str, choices: tuple[str, ...] | None = None) -> str:
        v = self.get(d, key, where)
        if v is None:
            return ""
        if not isinstance(v, str):
            self.err(f"{where}.{key} doit être une chaîne (reçu {v!r})")
            return str(v)
        if choices and v not in choices:
            self.err(f"{where}.{key} = {v!r} ∉ {list(choices)}")
        return v

    def mapping(self, d: Any, key: str, where: str) -> dict:
        v = self.get(d, key, where)
        if v is None:
            return {}
        if not isinstance(v, dict):
            self.err(f"{where}.{key} doit être un dictionnaire")
            return {}
        return v


def _validate_weights(w: Any, v: _V) -> None:
    if not isinstance(w, dict):
        v.err("weights.yaml : la racine doit être un dictionnaire")
        return
    s = v.mapping(w, "sampling", "weights")
    v.num(s, "frames_fps", "sampling", 0.5, 30)
    ocr_fps = v.num(s, "ocr_fps", "sampling", 0.1, 30)
    if s and isinstance(s.get("frames_fps"), (int, float)) and ocr_fps > s["frames_fps"]:
        v.err("sampling.ocr_fps doit être ≤ sampling.frames_fps")
    v.num(s, "frame_width", "sampling", 64, 4096)
    sc = v.mapping(w, "scenes", "weights")
    v.string(sc, "detector", "scenes", ("content", "ffmpeg"))
    v.num(sc, "threshold", "scenes", 0)
    v.num(sc, "min_scene_len_s", "scenes", 0)
    v.num(sc, "ffmpeg_threshold", "scenes", 0, 1)
    sp = v.mapping(w, "speech", "weights")
    v.string(sp, "model", "speech")
    v.string(sp, "language", "speech")
    v.string(sp, "device", "speech")
    v.string(sp, "compute_type", "speech")
    v.num(sp, "beam_size", "speech", 1, 20)
    v.boolean(sp, "vad_filter", "speech")
    v.num(sp, "cpu_threads", "speech", 0)
    v.num(v.mapping(w, "faces", "weights"), "min_confidence", "faces", 0, 1)
    o = v.mapping(w, "ocr", "weights")
    v.num(o, "min_score", "ocr", 0, 1)
    v.num(o, "min_chars", "ocr", 0)
    v.num(o, "same_text_similarity", "ocr", 0, 1)
    v.num(o, "det_limit_side_len", "ocr", 32)
    v.string(o, "det_limit_type", "ocr", ("max", "min"))
    v.boolean(o, "use_angle_cls", "ocr")
    v.num(o, "reuse_if_diff_below", "ocr", 0)
    a = v.mapping(w, "audio", "weights")
    v.num(a, "window_s", "audio", 0.01, 1)
    v.num(a, "silence_dbfs", "audio", -120, 0)
    v.num(a, "silence_rel", "audio", 0, 1)
    f = v.mapping(w, "features", "weights")
    for k in ("cut_tau_s", "motion_ref", "speech_rate_ref", "text_density_ref", "face_close_area",
              "loud_var_ref_db", "loud_jump_ref_db", "keyword_repeat_ref"):
        v.num(f, k, "features", 1e-6)
    for k in ("speech_rate_window_s", "loud_var_window_s", "keyword_min_len", "overlap_window_s"):
        v.num(f, k, "features", 0)
    scl = v.mapping(f, "since_cut_long", "features")
    a0 = v.num(scl, "start_s", "features.since_cut_long", 0)
    a1 = v.num(scl, "full_s", "features.since_cut_long", 0)
    if scl and a1 <= a0:
        v.err("features.since_cut_long.full_s doit être > start_s")
    for key in ("cta_keywords", "persuasion_keywords", "stopwords", "number_words"):
        lst = v.get(w, key, "weights")
        if lst is not None and (not isinstance(lst, list) or not all(isinstance(x, str) for x in lst)):
            v.err(f"weights.{key} doit être une liste de chaînes")
    ind = v.mapping(w, "indices", "weights")
    for name in INDEX_NAMES:
        d = v.mapping(ind, name, "indices")
        if not d:
            continue
        v.num(d, "brain_weight", f"indices.{name}", 0, 1)
        v.num(d, "baseline", f"indices.{name}", -1, 1)
        b = v.mapping(d, "boosters", f"indices.{name}")
        for feat, wt in b.items():
            if feat not in FEATURE_NAMES:
                v.err(f"indices.{name}.boosters : feature inconnue {feat!r} (connues : {', '.join(FEATURE_NAMES)})")
            if isinstance(wt, bool) or not isinstance(wt, (int, float)):
                v.err(f"indices.{name}.boosters.{feat} doit être un nombre")
    for extra in set(ind) - set(INDEX_NAMES):
        v.err(f"indices : indice inconnu {extra!r}")
    n = v.mapping(w, "normalization", "weights")
    v.num(n, "sigmoid_gain", "normalization", 1e-6)
    v.num(n, "sigmoid_center", "normalization")
    v.string(n, "brain_roi_source", "normalization", ("corpus", "video"))
    v.num(n, "mad_scale", "normalization", 1e-6)
    v.num(n, "reference_min_reels", "normalization", 1)
    v.num(n, "reference_max_seconds_per_reel", "normalization", 1)
    v.num(n, "hrf_shift_s", "normalization", -30, 30)
    sm = v.get(n, "smoothing", "normalization")
    if sm is not None and (not isinstance(sm, list) or not all(isinstance(x, (int, float)) for x in sm)):
        v.err("normalization.smoothing doit être une liste de nombres")
    br = v.mapping(w, "brain", "weights")
    v.string(br, "provider", "brain", BRAIN_PROVIDERS)
    v.string(br, "model", "brain")
    v.string(br, "device", "brain", ("auto", "cpu", "mps"))
    v.num(br, "timeout_s", "brain", 1)
    v.boolean(br, "no_text", "brain")
    v.boolean(br, "retry_without_text", "brain")
    v.num(br, "low_context_s", "brain", 0)
    sc = v.mapping(w, "score", "weights")
    sw = v.mapping(sc, "weights", "score")
    for k, wt in sw.items():
        if k not in INDEX_NAMES and k != "retention":
            v.err(f"score.weights : clé inconnue {k!r}")
        if isinstance(wt, bool) or not isinstance(wt, (int, float)) or wt < 0:
            v.err(f"score.weights.{k} doit être un nombre ≥ 0")
    if sw and sum(float(x) for x in sw.values() if isinstance(x, (int, float))) <= 0:
        v.err("score.weights : la somme des poids doit être > 0")
    lo = v.num(sc, "floor", "score", 0, 1)
    hi = v.num(sc, "ceil", "score", 0, 1)
    if sc and hi <= lo:
        v.err("score.ceil doit être > score.floor")
    h = v.mapping(sc, "hook", "score")
    for k in ("window_s", "text", "face", "speech_start", "speech_start_max_s", "motion_or_cut", "motion_min", "max"):
        v.num(h, k, "score.hook", 0)
    dp = v.mapping(sc, "dips", "score")
    v.string(dp, "index", "score.dips", INDEX_NAMES)
    v.num(dp, "threshold", "score.dips", 0, 1)
    v.num(dp, "min_duration_s", "score.dips", 1)
    v.num(dp, "points_per_second", "score.dips", 0)
    v.num(dp, "max", "score.dips", 0)
    r = v.mapping(w, "retention", "weights")
    v.num(r, "hook_seconds", "retention", 0)
    for k in RETENTION_PARAMS:
        v.num(r, k, "retention", 0, 1)
    v.boolean(r, "allow_increase", "retention")
    bounds = v.mapping(r, "calibration_bounds", "retention")
    for k in RETENTION_PARAMS:
        b = bounds.get(k)
        if not (isinstance(b, list) and len(b) == 2 and all(isinstance(x, (int, float)) for x in b) and b[0] <= b[1]):
            v.err(f"retention.calibration_bounds.{k} doit être [min, max]")
    v.num(r, "calibration_ridge", "retention", 0)


def _validate_roi_mapping(m: Any, v: _V) -> None:
    if not isinstance(m, dict):
        v.err("roi_mapping.yaml : la racine doit être un dictionnaire")
        return
    v.string(m, "atlas", "roi_mapping")
    v.string(m, "hemispheres", "roi_mapping", ("both", "left", "right"))
    rois = v.mapping(m, "rois", "roi_mapping")
    for name, labels in rois.items():
        if not isinstance(labels, list) or not labels or not all(isinstance(x, str) for x in labels):
            v.err(f"roi_mapping.rois.{name} doit être une liste non vide de labels")
    ind = v.mapping(m, "indices", "roi_mapping")
    for name in INDEX_NAMES:
        d = ind.get(name)
        if not isinstance(d, dict) or not d:
            v.err(f"roi_mapping.indices.{name} manquant ou vide")
            continue
        for roi, wt in d.items():
            if roi not in rois:
                v.err(f"roi_mapping.indices.{name} : ROI inconnue {roi!r}")
            if isinstance(wt, bool) or not isinstance(wt, (int, float)):
                v.err(f"roi_mapping.indices.{name}.{roi} doit être un nombre")
        if sum(abs(float(x)) for x in d.values() if isinstance(x, (int, float))) <= 0:
            v.err(f"roi_mapping.indices.{name} : somme des |poids| nulle")
    conf = m.get("confidence", {})
    if not isinstance(conf, dict):
        v.err("roi_mapping.confidence doit être un dictionnaire")


def _validate_rules(r: Any, v: _V) -> list[dict]:
    if not isinstance(r, dict) or not isinstance(r.get("rules"), list):
        v.err("rules.yaml : la racine doit contenir une liste `rules`")
        return []
    out = []
    seen = set()
    for i, rule in enumerate(r["rules"]):
        where = f"rules[{i}]"
        if not isinstance(rule, dict):
            v.err(f"{where} doit être un dictionnaire")
            continue
        name = v.string(rule, "name", where)
        if name in seen:
            v.err(f"{where} : règle {name!r} en double")
        seen.add(name)
        v.boolean(rule, "enabled", where)
        v.string(rule, "severity", where, SEVERITIES)
        params = rule.get("params", {})
        if not isinstance(params, dict):
            v.err(f"{where}.params doit être un dictionnaire")
        msg = v.string(rule, "message", where)
        if msg and "{t" not in msg and "{start" not in msg:
            v.err(f"{where}.message doit contenir {{t}}")
        out.append(rule)
    return out


# ----------------------------------------------------------------------------- construction
def _brain(b: dict) -> BrainConfig:
    return BrainConfig(provider=b["provider"], model=b["model"], device=b["device"],
                       timeout_s=float(b["timeout_s"]), no_text=bool(b["no_text"]),
                       retry_without_text=bool(b["retry_without_text"]),
                       low_context_s=float(b["low_context_s"]),
                       low_context_end_s=float(b.get("low_context_end_s", 3)))


def _read_yaml(path: Path) -> Any:
    if not path.exists():
        raise ConfigError(f"Fichier de configuration introuvable : {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML invalide dans {path.name} : {e}") from e


def load_calibration(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    params = data.get("params") if isinstance(data, dict) else None
    if not isinstance(params, dict):
        return None
    clean = {k: float(v) for k, v in params.items()
             if k in RETENTION_PARAMS and isinstance(v, (int, float)) and 0 <= v <= 1}
    if not clean:
        return None
    data["params"] = clean
    return data


_DEFAULT = "default"


def load_config(config_dir: Path | None = None,
                calibration_file: Path | None | str = _DEFAULT) -> Config:
    """Charge et valide la configuration. Lève `ConfigError` (message FR) si invalide.

    config_dir : défaut `paths.CONFIG_DIR` ; calibration_file : défaut `paths.CALIBRATION_FILE`,
    None = ignorer la calibration.
    """
    config_dir = Path(config_dir) if config_dir is not None else paths.CONFIG_DIR
    if isinstance(calibration_file, str) and calibration_file == _DEFAULT:
        calibration_file = paths.CALIBRATION_FILE
    weights = _read_yaml(config_dir / "weights.yaml")
    roi_mapping = _read_yaml(config_dir / "roi_mapping.yaml")
    rules_raw = _read_yaml(config_dir / "rules.yaml")
    v = _V()
    _validate_weights(weights, v)
    _validate_roi_mapping(roi_mapping, v)
    rules = _validate_rules(rules_raw, v)
    if v.errors:
        raise ConfigError("Configuration invalide :\n- " + "\n- ".join(v.errors))

    raw = {"weights": copy.deepcopy(weights), "roi_mapping": copy.deepcopy(roi_mapping),
           "rules": copy.deepcopy(rules_raw)}
    eff = copy.deepcopy(weights)
    calibration = load_calibration(calibration_file)
    if calibration:
        eff["retention"].update(calibration["params"])

    sp = eff["speech"]
    lang = str(sp["language"]).strip().lower()
    speech = SpeechConfig(model=sp["model"], language=None if lang in ("", "auto") else lang,
                          device=sp["device"], compute_type=sp["compute_type"],
                          beam_size=int(sp["beam_size"]), vad_filter=bool(sp["vad_filter"]),
                          cpu_threads=int(sp["cpu_threads"]))
    s = eff["sampling"]
    sampling = SamplingConfig(frames_fps=float(s["frames_fps"]), ocr_fps=float(s["ocr_fps"]),
                              frame_width=int(s["frame_width"]))
    r = eff["retention"]
    retention = RetentionParams(hook_seconds=float(r["hook_seconds"]), hook_drop=float(r["hook_drop"]),
                                base_drop=float(r["base_drop"]), k_attention=float(r["k_attention"]),
                                cut_bonus=float(r["cut_bonus"]), allow_increase=bool(r["allow_increase"]))
    return Config(weights=eff, roi_mapping=copy.deepcopy(roi_mapping), rules=copy.deepcopy(rules),
                  raw=raw, brain=_brain(eff["brain"]), speech=speech, sampling=sampling,
                  retention=retention, calibration=calibration, config_dir=config_dir)
