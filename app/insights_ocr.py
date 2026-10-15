"""Lecture de captures d'écran Instagram Insights (RapidOCR, 100 % local).

Boîtes texte -> libellés FR/EN connus -> valeur numérique la plus proche (même ligne à droite,
sinon juste en dessous). Résultat TOUJOURS à valider à la main dans l'UI.

Unités renvoyées : compteurs en nombre, durées en secondes, `skip_rate` en pourcentage (0..100).
"""
from __future__ import annotations

import io
import re
import threading
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

# ---------------------------------------------------------------------------- libellés
# kind : count | duration | percent
FIELDS: dict[str, dict[str, Any]] = {
    "views": {"kind": "count", "label": "Vues",
              "syn": ["vues", "views", "lectures", "plays", "nombre de vues", "vues totales"]},
    "reach": {"kind": "count", "label": "Comptes touchés",
              "syn": ["comptes touches", "accounts reached", "comptes atteints", "couverture", "reach"]},
    "watch_time_s": {"kind": "duration", "label": "Durée de visionnage",
                     "syn": ["duree de visionnage", "watch time", "temps de visionnage", "duree totale de visionnage",
                             "total watch time", "duree de lecture"]},
    "avg_watch_s": {"kind": "duration", "label": "Durée moyenne de visionnage",
                    "syn": ["duree moyenne de visionnage", "duree de visionnage moyenne", "average watch time",
                            "avg watch time", "temps de visionnage moyen", "temps moyen de visionnage",
                            "duree moyenne de lecture"]},
    "skip_rate": {"kind": "percent", "label": "Taux de skip",
                  "syn": ["taux de skip", "skip rate", "taux de saut", "taux de skips"]},
    "likes": {"kind": "count", "label": "J'aime", "syn": ["j'aime", "jaime", "mentions j'aime", "likes"]},
    "comments": {"kind": "count", "label": "Commentaires", "syn": ["commentaires", "comments"]},
    "shares": {"kind": "count", "label": "Partages", "syn": ["partages", "shares"]},
    "saves": {"kind": "count", "label": "Enregistrements", "syn": ["enregistrements", "saves", "sauvegardes"]},
    "follows": {"kind": "count", "label": "Abonnements", "syn": ["abonnements", "follows", "nouveaux abonnements"]},
    "interactions": {"kind": "count", "label": "Interactions",
                     "syn": ["interactions", "interactions avec le contenu", "content interactions"]},
}

_SPACES = re.compile(r"[\s    ]+")
_APOS = re.compile(r"[’‘`´ʼ′]")


def normalize(text: str) -> str:
    """minuscules, sans accents, apostrophes uniformisées, espaces simples, ponctuation de bord retirée."""
    t = unicodedata.normalize("NFKD", text or "")
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = _APOS.sub("'", t).lower()
    t = _SPACES.sub(" ", t).strip()
    return t.strip(" :;•·-–—()[]|>")


# ---------------------------------------------------------------------------- parsing des nombres
_SUFFIX = {"k": 1e3, "mille": 1e3, "m": 1e6, "mio": 1e6, "million": 1e6, "millions": 1e6, "md": 1e9,
           "mds": 1e9, "b": 1e9, "milliard": 1e9, "milliards": 1e9}
_COUNT_RE = re.compile(r"^(\d[\d .,']*?)\s*(k|mille|mio|millions?|mds?|milliards?|m|b)?\.?$")


def _clean_num_text(s: str) -> str:
    s = _SPACES.sub(" ", (s or "").strip()).replace("’", "'")
    return s.strip()


def parse_number(digits: str, has_suffix: bool = False) -> float | None:
    """'1 234' -> 1234 ; '1.234' -> 1234 ; '12,3' -> 12.3 ; '1,234.5' -> 1234.5 ; '1.234,5' -> 1234.5."""
    s = digits.strip().replace("'", " ")
    if not s or not s[0].isdigit():
        return None
    s = re.sub(r"(?<=\d) (?=\d)", " ", s)
    groups = s.split(" ")
    # Espaces : toujours séparateurs de milliers (« 1 234 ») si les groupes suivants font 3 chiffres
    if len(groups) > 1:
        tail = groups[1:]
        if not all(re.fullmatch(r"\d{3}([.,]\d+)?", g) for g in tail):
            return None
        s = "".join(groups)
    if re.fullmatch(r"\d+", s):
        return float(s)
    dots, commas = s.count("."), s.count(",")
    if dots and commas:  # le dernier séparateur est décimal
        dec = "." if s.rfind(".") > s.rfind(",") else ","
        thou = "," if dec == "." else "."
        s = s.replace(thou, "")
        if s.count(dec) != 1:
            return None
        s = s.replace(dec, ".")
    else:
        sep = "." if dots else ","
        n = dots or commas
        parts = s.split(sep)
        if not all(p.isdigit() for p in parts):
            return None
        if n > 1:  # « 1.234.567 »
            if not all(len(p) == 3 for p in parts[1:]):
                return None
            s = "".join(parts)
        elif len(parts[1]) == 3 and not has_suffix:  # « 1.234 » / « 1,234 » sans suffixe -> milliers
            s = "".join(parts)
        else:
            s = parts[0] + "." + parts[1]
    try:
        return float(s)
    except ValueError:
        return None


def parse_count(text: str) -> float | None:
    """'12,3 k' -> 12300 ; '1,2 M' -> 1200000 ; '1 234' -> 1234 ; '1.234' -> 1234. Refuse '%', '+12'."""
    s = _clean_num_text(text).lower()
    if not s or "%" in s or s[0] in "+-−" or ":" in s:
        return None
    m = _COUNT_RE.match(s)
    if not m:
        return None
    num, suf = m.group(1).strip(), m.group(2)
    v = parse_number(num, has_suffix=bool(suf))
    if v is None:
        return None
    if suf:
        v *= _SUFFIX[suf]
    return float(round(v))


_DUR_UNITS = [
    (re.compile(r"^(jours?|days?|j|d)$"), 86400.0),
    (re.compile(r"^(heures?|hours?|hrs?|h)$"), 3600.0),
    (re.compile(r"^(minutes?|mins?|mn|m)$"), 60.0),
    (re.compile(r"^(secondes?|seconds?|secs?|s)$"), 1.0),
]
_DUR_TOKEN = re.compile(r"(\d+(?:[.,]\d+)?)\s*([a-z]+)\.?")


def parse_duration(text: str) -> tuple[float | None, bool]:
    """Renvoie (secondes, avec_unité). '2:35' -> 155 ; '1 h 20 min' -> 4800 ; '35 s' -> 35 ; '12' -> (12, False)."""
    s = normalize(text)
    if not s or "%" in s or s[0] in "+-":
        return None, False
    m = re.fullmatch(r"(\d+):(\d{2})(?::(\d{2}))?", s)
    if m:
        a, b, c = m.group(1), m.group(2), m.group(3)
        if c is not None:
            return float(int(a) * 3600 + int(b) * 60 + int(c)), True
        return float(int(a) * 60 + int(b)), True
    compact = s.replace(" ", "")
    pos, total, n_tok = 0, 0.0, 0
    for tok in _DUR_TOKEN.finditer(compact):
        if tok.start() != pos:
            return None, False
        num = float(tok.group(1).replace(",", "."))
        unit = tok.group(2)
        mult = next((k for rx, k in _DUR_UNITS if rx.match(unit)), None)
        if mult is None:
            return None, False
        total += num * mult
        pos = tok.end()
        n_tok += 1
    if n_tok and pos == len(compact):
        return total, True
    m = re.fullmatch(r"\d+(?:[.,]\d+)?", s)
    if m:
        return float(s.replace(",", ".")), False
    return None, False


def parse_percent(text: str) -> float | None:
    """'12 %' -> 12.0 ; '45,2%' -> 45.2. Refuse les variations signées ('+5 %')."""
    s = _clean_num_text(text)
    m = re.fullmatch(r"(\d{1,3}(?:[.,]\d+)?)\s*%", s)
    if not m:
        return None
    v = float(m.group(1).replace(",", "."))
    return v if 0 <= v <= 100 else None


_PERCENT_IN = re.compile(r"(?<![+\-−\d])(\d{1,3}(?:[.,]\d+)?)\s*%")


def parse_value(kind: str, text: str) -> tuple[float | None, float]:
    """(valeur, facteur de confiance du format)."""
    if kind == "count":
        return parse_count(text), 1.0
    if kind == "duration":
        v, unit = parse_duration(text)
        return v, (1.0 if unit else 0.6)
    if kind == "percent":
        return parse_percent(text), 1.0
    return None, 0.0


# ---------------------------------------------------------------------------- géométrie
@dataclass
class Box:
    text: str
    conf: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def h(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def norm(self) -> str:
        return normalize(self.text)

    def as_dict(self) -> dict[str, Any]:
        return {"text": self.text, "conf": round(self.conf, 3),
                "box": [round(self.x1), round(self.y1), round(self.x2), round(self.y2)]}


def boxes_from_rapidocr(result: Any) -> list[Box]:
    out = []
    for item in result or []:
        try:
            pts, text, conf = item[0], str(item[1]), float(item[2])
            xs = [float(p[0]) for p in pts]
            ys = [float(p[1]) for p in pts]
        except (TypeError, ValueError, IndexError):
            continue
        if text.strip():
            out.append(Box(text, conf, min(xs), min(ys), max(xs), max(ys)))
    return out


def raw_text(boxes: list[Box]) -> str:
    """Texte brut ligne par ligne (regroupement par centre vertical)."""
    lines: list[list[Box]] = []
    for b in sorted(boxes, key=lambda b: (b.cy, b.x1)):
        if lines and abs(lines[-1][-1].cy - b.cy) <= 0.5 * max(b.h, lines[-1][-1].h):
            lines[-1].append(b)
        else:
            lines.append([b])
    return "\n".join("  ".join(x.text for x in sorted(line, key=lambda b: b.x1)) for line in lines)


# ---------------------------------------------------------------------------- appariement
def match_label(norm_text: str) -> tuple[str, float, str] | None:
    """(champ, score 0..1, reste du texte après le libellé) pour le meilleur libellé reconnu."""
    best: tuple[str, float, str, int] | None = None
    t = norm_text
    t_noapos = t.replace("'", "")
    for fid, spec in FIELDS.items():
        for syn in spec["syn"]:
            cand: tuple[float, str] | None = None
            syn_noapos = syn.replace("'", "")
            for text, s in ((t, syn), (t_noapos, syn_noapos)):
                if text == s:
                    cand = (1.0, "")
                    break
                if text.startswith(s + " ") or text.startswith(s + ":"):
                    cand = (0.97, text[len(s):].strip(" :"))
                    break
            if cand is None and spec["kind"] == "percent" and syn in t:
                # phrase du type « Le taux de skip de ce reel est de 45 % »
                rest = t.split(syn, 1)[1]
                if _PERCENT_IN.search(rest):
                    cand = (0.9, rest.strip(" :"))
            if cand is None and len(syn) >= 6 and abs(len(t) - len(syn)) <= 4:
                r = SequenceMatcher(None, t, syn).ratio()
                if r >= 0.84:
                    cand = (0.95 * r, "")
            if cand is None:
                continue
            key = (cand[0], len(syn))
            if best is None or key > (best[1], best[3]):
                best = (fid, cand[0], cand[1], len(syn))
    if best is None:
        return None
    return best[0], best[1], best[2]


def extract_fields(boxes: list[Box]) -> dict[str, dict[str, Any]]:
    """Pour chaque libellé reconnu : valeur la plus proche (même ligne à droite, sinon en dessous)."""
    found: dict[str, list[dict[str, Any]]] = {}
    labels: list[tuple[Box, str, float, str]] = []
    for b in boxes:
        m = match_label(b.norm)
        if m:
            labels.append((b, *m))
    label_ids = {id(b) for b, *_ in labels}

    for lb, fid, lscore, rest in labels:
        kind = FIELDS[fid]["kind"]
        cands: list[dict[str, Any]] = []
        # 1) valeur dans la même boîte (« Vues 12,3 k », phrase contenant « 45 % »)
        if rest:
            v, f = parse_value(kind, rest)
            if v is None and kind == "percent":
                pm = _PERCENT_IN.search(rest)
                v, f = (parse_percent(pm.group(0)), 0.9) if pm else (None, 0)
            if v is not None:
                cands.append({"value": v, "value_text": rest, "pos": "inline",
                              "conf": lscore * lb.conf * f, "dist": 0.0})
            else:
                # « Vues de non-abonnés » : probablement un autre libellé -> voisins moins fiables
                lscore *= 0.6
        for vb in boxes:
            if vb is lb or id(vb) in label_ids:
                continue
            v, f = parse_value(kind, vb.text)
            if v is None:
                continue
            hmax = max(lb.h, vb.h)
            same_line = abs(vb.cy - lb.cy) <= 0.6 * hmax and vb.x1 >= lb.x2 - 0.3 * lb.h
            below = (vb.y1 >= lb.cy and (vb.y1 - lb.y2) <= 3.0 * lb.h
                     and vb.x1 <= lb.x2 + lb.h and vb.x2 >= lb.x1 - lb.h)
            if same_line:
                pos, pf, dist = "right", 1.0, vb.x1 - lb.x2
            elif below:
                pos, pf, dist = "below", 0.85, 10000.0 + (vb.y1 - lb.y2) + 0.1 * abs(vb.x1 - lb.x1)
            else:
                continue
            cands.append({"value": v, "value_text": vb.text, "pos": pos, "dist": dist,
                          "conf": lscore * (lb.conf ** 0.5) * vb.conf * pf * f})
        if not cands:
            continue
        cands.sort(key=lambda c: c["dist"])
        best = cands[0]
        found.setdefault(fid, []).append({
            "value": best["value"], "confidence": round(min(1.0, best["conf"]), 3),
            "label_text": lb.text, "value_text": best["value_text"], "position": best["pos"],
        })

    out: dict[str, dict[str, Any]] = {}
    for fid, items in found.items():
        items.sort(key=lambda it: -it["confidence"])
        best = dict(items[0])
        alts = sorted({it["value"] for it in items[1:] if it["value"] != best["value"]})
        best["alternatives"] = alts
        if alts:  # plusieurs valeurs différentes pour le même libellé : on baisse la confiance
            best["confidence"] = round(best["confidence"] * 0.8, 3)
        best["level"] = confidence_level(best["confidence"])
        best["label"] = FIELDS[fid]["label"]
        best["kind"] = FIELDS[fid]["kind"]
        out[fid] = best
    return out


def confidence_level(c: float) -> str:
    return "haute" if c >= 0.8 else ("moyenne" if c >= 0.55 else "faible")


def guess_window(text: str) -> str | None:
    t = normalize(text)
    if re.search(r"7 (derniers )?jours|last 7 days|7 days|7 j\b", t):
        return "7d"
    if re.search(r"48 ?h|48 heures|2 (derniers )?jours|last 2 days", t):
        return "48h"
    return None


# ---------------------------------------------------------------------------- moteur OCR
_engine = None
_engine_lock = threading.Lock()


def get_engine():
    global _engine
    with _engine_lock:
        if _engine is None:
            from rapidocr_onnxruntime import RapidOCR

            _engine = RapidOCR()
        return _engine


def _to_array(image: bytes | Any):
    import numpy as np
    from PIL import Image, ImageOps

    if isinstance(image, (bytes, bytearray)):
        img = Image.open(io.BytesIO(image))
    elif isinstance(image, Image.Image):
        img = image
    else:
        img = Image.open(image)
    img = ImageOps.exif_transpose(img).convert("RGB")
    # Les captures iPhone font ~1170x2532 : on borne la taille pour la vitesse
    if max(img.size) > 2600:
        img.thumbnail((2600, 2600))
    # Mode sombre d'Instagram : le modèle lit mieux du texte foncé sur fond clair
    # (ex. « 90 » blanc sur noir lu « 06 ») -> inversion si le fond est sombre.
    if float(np.asarray(img.convert("L")).mean()) < 110:
        img = ImageOps.invert(img)
    return np.asarray(img)[:, :, ::-1].copy()  # BGR, comme OpenCV


_run_lock = threading.Lock()


def ocr_boxes(image: bytes | Any) -> list[Box]:
    arr = _to_array(image)
    engine = get_engine()
    with _run_lock:  # une inférence à la fois (mémoire)
        result, _elapsed = engine(arr)
    return boxes_from_rapidocr(result)


def read_insights(image: bytes | Any) -> dict[str, Any]:
    """OCR d'UNE capture -> {fields, raw_text, boxes, window_guess}."""
    boxes = ocr_boxes(image)
    text = raw_text(boxes)
    return {"fields": extract_fields(boxes), "raw_text": text, "boxes": [b.as_dict() for b in boxes],
            "window_guess": guess_window(text)}


def merge_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Fusionne plusieurs captures : pour chaque champ, la lecture la plus sûre l'emporte."""
    fields: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(results):
        for fid, f in r.get("fields", {}).items():
            f = dict(f, image=i)
            cur = fields.get(fid)
            if cur is None or f["confidence"] > cur["confidence"]:
                if cur is not None and cur["value"] != f["value"]:
                    f["alternatives"] = sorted(set(f.get("alternatives", [])) | {cur["value"]})
                fields[fid] = f
            elif cur["value"] != f["value"]:
                cur["alternatives"] = sorted(set(cur.get("alternatives", [])) | {f["value"]})
    windows = [r.get("window_guess") for r in results if r.get("window_guess")]
    raw = "\n\n".join(f"--- Capture {i + 1} ---\n{r.get('raw_text', '')}" for i, r in enumerate(results))
    return {"fields": fields, "raw_text": raw, "window_guess": windows[0] if windows else None,
            "images": [{"raw_text": r.get("raw_text", ""), "n_boxes": len(r.get("boxes", []))} for r in results],
            "engine": "rapidocr"}
