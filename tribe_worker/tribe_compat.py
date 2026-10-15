"""Contournements « Mac / sans CUDA » pour tribev2 @ af58661 + neuralset 0.0.2 + transformers 5.x.

Tout ce qui touche aux internes de tribev2/neuralset est regroupé ici, pour qu'une mise à jour de
tribev2 ne casse qu'un seul fichier. Aucune modification des paquets installés.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Modèles que l'on accepte de charger. `facebook/tribev2` (LLaMA-3.2-3B + V-JEPA2 ViT-g) dépasse la
# RAM d'une machine de 16 Go : refusé sauf variable TRIBE_ALLOW_FULL=1.
FULL_MODEL = "facebook/tribev2"
TRIBE_FILES = ["config.yaml", "best.ckpt"]
# Fichiers téléchargés pour chaque backbone (on évite les `original/*.pth` en double).
BACKBONE_PATTERNS = ["*.json", "*.safetensors", "tokenizer.model"]
BACKBONE_IGNORE = ["original/*", "*.pth", "*.pt", "*.bin", "*.ipynb", "*.msgpack", "*.h5", "onnx/*"]


class HFAccessError(RuntimeError):
    def __init__(self, repo: str, detail: str = ""):
        super().__init__(f"accès Hugging Face refusé pour {repo} {detail}".strip())
        self.repo = repo


# --------------------------------------------------------------------------------------------
# Cache Hugging Face (lecture pure du système de fichiers : aucun appel réseau, aucun import HF)
# --------------------------------------------------------------------------------------------
def hf_hub_cache() -> Path:
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var]).expanduser()
    home = os.environ.get("HF_HOME") or os.path.join(
        os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "huggingface")
    return Path(home).expanduser() / "hub"


def cached_path(repo: str, filename: str, revision: str = "main") -> Path | None:
    d = hf_hub_cache() / ("models--" + repo.replace("/", "--"))
    ref = d / "refs" / revision
    if not ref.is_file():
        return None
    p = d / "snapshots" / ref.read_text().strip() / filename
    return p if p.exists() else None


def backbone_cached(repo: str, need_tokenizer: bool = False) -> bool:
    if cached_path(repo, "config.json") is None:
        return False
    if cached_path(repo, "model.safetensors") is None and cached_path(repo, "model.safetensors.index.json") is None:
        return False
    if need_tokenizer and cached_path(repo, "tokenizer.json") is None and cached_path(repo, "tokenizer.model") is None:
        return False
    return True


def backbones_from_config(config_path: Path) -> dict[str, str]:
    """{'text': repo, 'audio': repo, 'video': repo} lus dans le config.yaml du checkpoint TRIBE."""
    import yaml

    class _SafeIgnorePython(yaml.SafeLoader):
        """SafeLoader qui remplace les tags !!python/* (PosixPath, tuple) par None au lieu de les exécuter."""

    _SafeIgnorePython.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)
    with open(config_path) as f:
        cfg = yaml.load(f, Loader=_SafeIgnorePython)  # noqa: S506 (SafeLoader dérivé)
    data = cfg["data"]
    out = {}
    for mod in data.get("features_to_use", ["text", "audio", "video"]):
        feat = data.get(f"{mod}_feature") or {}
        name = feat.get("model_name") or (feat.get("image") or {}).get("model_name")
        if name:
            out[mod] = name
    return out


def required_repos(model: str, use_text: bool, use_audio: bool = True,
                   use_video: bool = True) -> tuple[dict[str, str], list[str]]:
    """Retourne (backbones, dépôts manquants dans le cache local)."""
    missing = []
    cfg = cached_path(model, "config.yaml")
    if cfg is None or cached_path(model, "best.ckpt") is None:
        missing.append(model)
        return {}, missing
    bb = backbones_from_config(cfg)
    for mod, repo in bb.items():
        if (mod == "text" and not use_text) or (mod == "audio" and not use_audio) or (mod == "video" and not use_video):
            continue
        if not backbone_cached(repo, need_tokenizer=(mod == "text")):
            missing.append(repo)
    return bb, missing


def _is_access_error(exc: BaseException) -> bool:
    try:
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError, RepositoryNotFoundError
    except Exception:  # pragma: no cover
        return False
    if isinstance(exc, (GatedRepoError, RepositoryNotFoundError)):
        return True
    if isinstance(exc, HfHubHTTPError):
        status = getattr(getattr(exc, "response", None), "status_code", None)
        return status in (401, 403)
    return False


def download_repos(model: str, use_text: bool, progress, use_audio: bool = True,
                   use_video: bool = True) -> dict[str, str]:
    """Télécharge checkpoint TRIBE + backbones (idempotent). Lève HFAccessError si gated/sans token."""
    from huggingface_hub import hf_hub_download, snapshot_download

    def fetch(repo, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if _is_access_error(e):
                raise HFAccessError(repo, f"({type(e).__name__})") from e
            raise

    progress(0.02, f"Téléchargement du checkpoint {model}")
    cfg = fetch(model, lambda: hf_hub_download(model, "config.yaml"))
    fetch(model, lambda: hf_hub_download(model, "best.ckpt"))
    bb = backbones_from_config(Path(cfg))
    order = [m for m in ("audio", "video", "text") if m in bb and (use_text or m != "text")
             and (use_audio or m != "audio") and (use_video or m != "video")]
    for i, mod in enumerate(order):
        repo = bb[mod]
        progress(0.1 + 0.85 * i / max(1, len(order)), f"Téléchargement du modèle {mod} ({repo})")
        fetch(repo, lambda: snapshot_download(repo, allow_patterns=BACKBONE_PATTERNS, ignore_patterns=BACKBONE_IGNORE))
    progress(1.0, "Modèles téléchargés")
    return bb


def access_repo_from_exception(exc: BaseException) -> str | None:
    """Cherche dans la chaîne d'exceptions une erreur d'accès HF (gated / 401 / 403) et le dépôt concerné."""
    seen = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, HFAccessError):
            return e.repo
        msg = str(e)
        if _is_access_error(e) or re.search(r"gated repo|401 Client Error|403 Client Error|Access to model .* is restricted", msg):
            m = re.search(r"huggingface\.co/(?:api/models/)?([\w.\-]+/[\w.\-]+)", msg)
            return m.group(1).rstrip(".") if m else "inconnu"
        e = e.__cause__ or e.__context__
    return None


def is_oom(exc: BaseException) -> bool:
    seen = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        if isinstance(e, MemoryError):
            return True
        msg = str(e).lower()
        if any(k in msg for k in ("out of memory", "cannot allocate memory", "bad_alloc",
                                  "not enough memory", "failed to allocate", "invalid buffer size")):
            return True
        e = e.__cause__ or e.__context__
    return False


# --------------------------------------------------------------------------------------------
# Patches neuralset : progression, device MPS, dtype
# --------------------------------------------------------------------------------------------
class _ProgressIter:
    def __init__(self, it, total, cb):
        self._it, self._total, self._cb, self._k = it, max(1, int(total)), cb, 0

    def __iter__(self):
        for x in self._it:
            yield x
            self._k += 1
            self._cb(self._k / self._total)

    def __len__(self):
        return self._total

    def __getattr__(self, name):  # set_description, close, ...
        return getattr(self._it, name)


def install_progress_hooks(callbacks: dict[str, callable]) -> None:
    """`callbacks` : {desc tqdm: f(fraction)} ; ex. {"Encoding video": ..., "Computing word embeddings": ...}."""
    import neuralset.extractors.text as nst
    import neuralset.extractors.video as nsv
    from tqdm import tqdm as _tqdm

    def wrapped(iterable=None, *args, **kwargs):
        desc = kwargs.get("desc")
        kwargs.setdefault("mininterval", 5.0)  # moins de bruit dans les logs
        kwargs.setdefault("file", sys.stderr)
        it = _tqdm(iterable, *args, **kwargs)
        cb = callbacks.get(desc)
        if cb is None:
            return it
        total = kwargs.get("total")
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                return it
        return _ProgressIter(it, total, cb)

    nsv.tqdm = wrapped
    nst.tqdm = wrapped


TEXT_STATE: dict = {}  # device -> nom du dtype du LLM texte


def set_text_dtype(dtype_for_device: dict[str, str]) -> None:
    """dtype du LLM texte selon son device (transformers 5 chargerait le dtype du checkpoint, bf16 pour
    LLaMA-3.2 ; transformers 4 chargeait en fp32, régime de l'entraînement de TRIBE)."""
    import torch
    from neuralset.extractors.text import HuggingFaceText

    TEXT_STATE.update(dtype_for_device)
    orig = HuggingFaceText._load_model
    if getattr(orig, "_patched", False):
        return

    def _load_model(self, **kwargs):
        dtype = getattr(torch, TEXT_STATE.get(self.device, "float32"))
        if self.device != "mps" or "dtype" in kwargs:
            kwargs.setdefault("dtype", dtype)
            return orig(self, **kwargs)
        # Sur MPS : chargement sur CPU dans le dtype du checkpoint (bf16 pour LLaMA-3.2 : fichier mmappé,
        # pas de copie), puis transfert + conversion paramètre par paramètre vers MPS. Valeurs identiques à
        # un chargement direct dans le dtype cible, mais sans copie complète en RAM CPU en plus de celle sur
        # MPS (pic transitoire mesuré à 12 Go d'empreinte sur un Reel de 41 s avec le chargement d'origine
        # en fp32). `device_map="mps"` exigerait le paquet accelerate.
        object.__setattr__(self, "device", "cpu")  # l'original fait `model.to(self.device)` en fin de chargement
        try:
            model = orig(self, dtype="auto", **kwargs)
        finally:
            object.__setattr__(self, "device", "mps")
        model = model.to(device="mps", dtype=dtype)
        torch.mps.empty_cache()
        return model

    _load_model._patched = True
    HuggingFaceText._load_model = _load_model


def patch_sdpa_gqa_mps() -> None:
    """LLaMA-3.2 (GQA : 32 têtes de requête, 8 de clé/valeur) : transformers 5 appelle la SDPA de torch avec
    `enable_gqa=True` quand il n'y a pas de masque. Sur MPS (torch 2.6, bf16), au-delà d'environ 200 tokens de
    contexte, le noyau MPSGraph échoue (« mps.matmul … incompatible dimensions », tensor<1x32x220x64> ×
    tensor<1x8x64x220>) par un LLVM ERROR qui tue le process (SIGABRT, aucun repli CPU possible) — mesuré le
    12/10/2026 sur une voix off de 190 mots. On désactive la GQA native sur MPS : transformers répète alors les
    têtes clé/valeur (`repeat_kv`) avant la SDPA — calcul mathématiquement identique."""
    try:
        import transformers.integrations.sdpa_attention as sa
    except Exception:  # noqa: BLE001
        return
    orig = sa.use_gqa_in_sdpa
    if getattr(orig, "_patched", False):
        return

    def use_gqa_in_sdpa(attention_mask, key, value):
        if getattr(key, "device", None) is not None and key.device.type == "mps":
            return False
        return orig(attention_mask, key, value)

    use_gqa_in_sdpa._patched = True
    sa.use_gqa_in_sdpa = use_gqa_in_sdpa


_ATTN_CHUNK = int(os.environ.get("TRIBE_ATTN_CHUNK", "1024"))


def _chunked_sdpa(module, query, key, value, attention_mask=None, scaling=None, dropout=0.0,
                  is_causal=False, **kwargs):
    """Attention exacte (non causale, sans masque) calculée par blocs de requêtes.

    V-JEPA 2 traite 64 images 256x256 -> 8192 tokens : la SDPA de torch 2.6 sur MPS matérialise la
    matrice d'attention complète (16 têtes x 8192² -> « Invalid buffer size: 12.00 GB »). Par blocs de
    1024 requêtes, le résultat est mathématiquement identique (écart mesuré ~1e-6 relatif vs CPU)."""
    import torch
    import torch.nn.functional as F

    if attention_mask is not None or is_causal:
        raise RuntimeError("chunked_sdpa : masque/causal non supportés")
    if scaling is None:
        scaling = query.shape[-1] ** -0.5
    outs = []
    for s in range(0, query.shape[2], _ATTN_CHUNK):
        q = query[:, :, s:s + _ATTN_CHUNK]
        if q.device.type == "mps" and q.dtype == torch.float16:
            # Sur MPS en fp16, matmul + softmax explicites : -25 % de temps vs la SDPA de torch 2.6
            # (même erreur max vs référence fp32 : 1.2e-4 sur données aléatoires).
            w = torch.softmax((q * scaling) @ key.transpose(-1, -2), dim=-1)
            outs.append(w @ value)
            del w
        else:
            outs.append(F.scaled_dot_product_attention(q, key, value, dropout_p=0.0, scale=scaling))
    return torch.cat(outs, dim=2).transpose(1, 2).contiguous(), None


FRAME_CACHE_SIZE = 80  # > 64 images par clip + 8 nouvelles par pas
PROC_CACHE = None  # OrderedDict (id clip, instant) -> (clip, tenseur 3x256x256) ; initialisé plus bas
STEP_KEYS: list = []  # clés des images lues depuis le dernier appel au modèle
VJEPA_STATE = {"load_dtype": "float32"}  # dtype de chargement du prochain V-JEPA 2 (selon son device)


def patch_vjepa2(dtype_for_device: dict[str, str]) -> None:
    global PROC_CACHE
    """Patch de `neuralset.extractors.video._HFVideoModel` pour V-JEPA 2 :
    - `skip_predictor=True` : TRIBE n'utilise que les états cachés de l'ENCODEUR ; le prédicteur JEPA
      (12 couches sur 8192 tokens) était calculé pour rien (sorties de l'encodeur identiques) ;
    - attention par blocs (voir `_chunked_sdpa`) ;
    - dtype par device (fp16 sur MPS : ~x3 plus rapide que fp32, corrélation 0.99999998 des features),
      chargé directement dans ce dtype (évite une copie fp32 de 1,3 Go) ; repli fp32 si débordement ;
    - états cachés rapatriés sur CPU avant la moyenne sur les tokens (accumulation fp32 garantie).
    """
    import torch
    import transformers
    import neuralset.extractors.video as nsv
    from neuralset.extractors.image import _fix_pixel_values
    from transformers import AttentionInterface

    from collections import OrderedDict as _OD

    if PROC_CACHE is None:
        PROC_CACHE = _OD()
    cls = nsv._HFVideoModel
    for dev, dt in dtype_for_device.items():
        VJEPA_STATE[f"dtype_{dev}"] = dt
    if getattr(cls, "_tribe_worker_patched", False):
        return
    AttentionInterface.register("chunked_sdpa", _chunked_sdpa)
    orig_init = cls.__init__
    orig_predict = cls.predict
    orig_hidden = cls.predict_hidden_states

    def __init__(self, model_name, *args, **kwargs):
        VJEPA_STATE["active"] = "vjepa2" in model_name
        PROC_CACHE.clear()
        STEP_KEYS.clear()
        if "vjepa2" not in model_name:
            return orig_init(self, model_name, *args, **kwargs)
        orig_fp = transformers.AutoModel.from_pretrained

        def from_pretrained(*a, **k):
            k.setdefault("dtype", getattr(torch, VJEPA_STATE["load_dtype"]))
            return orig_fp(*a, **k)

        transformers.AutoModel.from_pretrained = from_pretrained
        try:
            orig_init(self, model_name, *args, **kwargs)
        finally:
            transformers.AutoModel.from_pretrained = orig_fp

    def _prepare(self):
        if getattr(self, "_tw_ready", False):
            return
        dtype = getattr(torch, dtype_for_device.get(self.model.device.type, "float32"))
        if self.model.dtype != dtype:
            self.model.to(dtype=dtype)
        try:
            self.model.set_attn_implementation("chunked_sdpa")
        except Exception:  # noqa: BLE001
            self.model.config._attn_implementation = "chunked_sdpa"
        self._tw_ready = True

    def _forward(self, pv):
        with torch.inference_mode():
            out = self.model(pixel_values_videos=pv.to(self.model.dtype), skip_predictor=True)
            if VJEPA_STATE.get("token_mean", False):
                # TRIBE moyenne les 8192 tokens (token_aggregation="mean") APRÈS sélection linéaire des
                # couches : les deux opérations commutent, on moyenne donc ici, en fp32, sur le GPU, et on
                # ne rapatrie que (L+1) x 1 x D au lieu de (L+1) x 8192 x D. Résultat identique.
                states = torch.stack([h[0].float().mean(0, keepdim=True) for h in out.hidden_states]).cpu()
            else:
                states = torch.stack([h[0] for h in out.hidden_states]).cpu().float()
        del out
        if pv.device.type == "mps":
            # sans cela le cache de l'allocateur MPS gonfle l'empreinte de plusieurs Go
            torch.mps.empty_cache()
        return states

    def _pixel_values(self, images):
        """(1, 64, 3, 256, 256) : images prétraitées, en ne traitant que celles absentes du cache."""
        keys = STEP_KEYS[-len(images):] if len(STEP_KEYS) >= len(images) else None
        STEP_KEYS.clear()
        if keys is None:  # appel hors du flux neuralset : traitement direct de toutes les images
            inputs = self.processor(videos=list(images), return_tensors="pt")
            _fix_pixel_values(inputs)
            return inputs["pixel_values_videos"]
        missing = [i for i, (k, v) in enumerate(keys) if k not in PROC_CACHE or PROC_CACHE[k][0] is not v]
        if missing:
            done = {}
            for i in missing:  # images identiques (début de vidéo : t borné à 0) traitées une fois
                done.setdefault(keys[i][0], i)
            idx = list(done.values())
            pv = self.processor(videos=[images[idx]], return_tensors="pt")["pixel_values_videos"][0]
            for j, i in enumerate(idx):
                PROC_CACHE[keys[i][0]] = (keys[i][1], pv[j].clone())
            while len(PROC_CACHE) > FRAME_CACHE_SIZE:
                PROC_CACHE.popitem(last=False)
        return torch.stack([PROC_CACHE[k][1] for k, _ in keys])[None]

    def predict(self, images, audio=None):
        if "vjepa2" not in self.model_name:
            return orig_predict(self, images, audio)
        _prepare(self)
        pv = _pixel_values(self, images).to(self.model.device)
        states = _forward(self, pv)
        if not torch.isfinite(states).all():
            if self.model.dtype == torch.float32:
                raise RuntimeError("V-JEPA 2 : états cachés non finis")
            dtype = self.model.dtype  # débordement fp16 : ce clip est recalculé en fp32
            self.model.to(dtype=torch.float32)
            states = _forward(self, pv)
            self.model.to(dtype=dtype)
        return states

    def predict_hidden_states(self, images, audio=None):
        if "vjepa2" not in self.model_name:
            return orig_hidden(self, images, audio)
        return self.predict(images, audio).unsqueeze(0)  # B=1 x L x N x D (comme l'original)

    cls.__init__ = __init__
    cls.predict = predict
    cls.predict_hidden_states = predict_hidden_states
    cls._tribe_worker_patched = True

    # Cache des images PRÉTRAITÉES. Chaque pas de 0,5 s lit 64 images couvrant les 4 s précédentes
    # (grille de 1/16 s), dont 56 déjà lues et prétraitées au pas précédent ; sans cache, moviepy revient
    # en arrière et redécode tout, et le préprocesseur convertit 64 images pleine résolution en float
    # (~1,6 Go en 1080p). Le prétraitement V-JEPA 2 (redimensionnement, recadrage, normalisation) est
    # image par image : prétraiter seulement les images nouvelles donne exactement le même tenseur
    # (vérifié : écart 0.0). Pour une image déjà en cache, `_read` renvoie une image factice (noire, même
    # taille) sans décoder ; `predict` remplace ces lignes par la version prétraitée en cache.
    import numpy as np
    from PIL import Image

    vi = nsv._VideoImage
    orig_read = vi._read
    dummies: dict = {}

    def _read(self):
        if not VJEPA_STATE.get("active"):
            return orig_read(self)
        key = (id(self.video), round(float(self.time), 4))
        STEP_KEYS.append((key, self.video))
        hit = PROC_CACHE.get(key)
        if hit is not None and hit[0] is self.video:  # référence forte : l'id du clip ne peut être réutilisé
            PROC_CACHE.move_to_end(key)
            w, h = self.video.size
            if (w, h) not in dummies:
                dummies.clear()
                dummies[(w, h)] = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
            return dummies[(w, h)]
        return orig_read(self)

    vi._read = _read


def set_extractor_device(extractor, device: str) -> None:
    """neuralset n'accepte que auto|cpu|cuda|accelerate à la validation (Literal pydantic) ; sans
    `validate_assignment`, on peut réassigner après construction. Le device est exclu de l'uid de cache."""
    target = extractor.image if hasattr(extractor, "image") else extractor
    try:
        target.device = device
    except Exception:  # noqa: BLE001
        # exca « gèle » l'extracteur dès que son cache est utilisé ; `device` est exclu de l'uid de
        # cache, on peut donc le modifier sans risque en contournant le gel (repli MPS -> CPU).
        object.__setattr__(target, "device", device)
    VJEPA_STATE["load_dtype"] = VJEPA_STATE.get(f"dtype_{device}", "float32")


def free_extractor(extractor) -> None:
    """Libère le backbone après mise en cache de ses features (équivalent tribev2.main._free_extractor_model + MPS)."""
    import gc

    import torch

    for target in (extractor, getattr(extractor, "image", None)):
        if target is not None and isinstance(getattr(target, "_model", None), torch.nn.Module):
            try:
                delattr(target, "_model")
            except Exception:  # noqa: BLE001
                pass
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# --------------------------------------------------------------------------------------------
# Mémoire : sur Apple Silicon, les buffers Metal (MPS) n'apparaissent pas dans ru_maxrss ; le
# « phys_footprint » (ce que montre le Moniteur d'activité, et `/usr/bin/time -l` « peak memory
# footprint ») les inclut.
# --------------------------------------------------------------------------------------------
def peak_footprint_mb() -> float | None:
    if sys.platform != "darwin":
        return None
    try:
        import ctypes

        names = ("ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups ri_pageins ri_wired_size "
                 "ri_resident_size ri_phys_footprint ri_proc_start_abstime ri_proc_exit_abstime "
                 "ri_child_user_time ri_child_system_time ri_child_pkg_idle_wkups ri_child_interrupt_wkups "
                 "ri_child_pageins ri_child_elapsed_abstime ri_diskio_bytesread ri_diskio_byteswritten "
                 "ri_cpu_time_qos_default ri_cpu_time_qos_maintenance ri_cpu_time_qos_background "
                 "ri_cpu_time_qos_utility ri_cpu_time_qos_legacy ri_cpu_time_qos_user_initiated "
                 "ri_cpu_time_qos_user_interactive ri_billed_system_time ri_serviced_system_time "
                 "ri_logical_writes ri_lifetime_max_phys_footprint ri_instructions ri_cycles ri_billed_energy "
                 "ri_serviced_energy ri_interval_max_phys_footprint ri_runnable_time").split()

        class RUsageInfoV4(ctypes.Structure):
            _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in names]

        buf = RUsageInfoV4()
        lib = ctypes.CDLL("/usr/lib/libproc.dylib")
        if lib.proc_pid_rusage(os.getpid(), 4, ctypes.byref(buf)) != 0:  # RUSAGE_INFO_V4
            return None
        return round(buf.ri_lifetime_max_phys_footprint / 2**20, 1)
    except Exception:  # noqa: BLE001
        return None


def patch_w2vbert_relative_key() -> None:
    """w2v-BERT 2.0 (`position_embeddings_type="relative_key"`) matérialise un tenseur (L, L, 64) de
    plongements de distance par couche : 1,1 Go pour 41 s d'audio, 2,3 Go pour un morceau de 60 s, d'où une
    empreinte de 8 Go mesurée à l'étape audio. Les distances étant bornées à [-64, 8], il n'y a que 73
    plongements distincts : q·E[idx(l,r)] = (q·Eᵀ)[..., idx(l,r)]. Même résultat (à l'ordre des sommes
    près), sans le tenseur (L, L, 64)."""
    import torch
    import transformers.models.wav2vec2_bert.modeling_wav2vec2_bert as w2v

    if getattr(w2v, "_tribe_worker_patched", False):
        return

    def _apply_relative_key_position_encoding(module, query, key):
        lq, lk = query.shape[2], key.shape[2]
        left, right = module.left_max_position_embeddings, module.right_max_position_embeddings
        pos_l = torch.arange(lq, dtype=torch.long, device=query.device).view(-1, 1)
        pos_r = torch.arange(lk, dtype=torch.long, device=query.device).view(1, -1)
        idx = torch.clamp(pos_r - pos_l, -left, right) + left                     # (lq, lk)
        table = module.distance_embedding.weight.to(dtype=query.dtype)            # (left+right+1, d)
        q_table = torch.matmul(query, table.t())                                  # (b, h, lq, n_dist)
        b, h = q_table.shape[:2]
        bias = torch.gather(q_table, 3, idx.view(1, 1, lq, lk).expand(b, h, lq, lk))
        return query, bias * module.scaling

    w2v._apply_relative_key_position_encoding = _apply_relative_key_position_encoding
    w2v._tribe_worker_patched = True
