"""Normalisation de texte et recherche de mots-clés (insensible à la casse et aux accents).

Deux modes de recherche :
- par tokens (parole et texte à l'écran) : « clique sur le lien » contient « lien » ;
- « collé » (texte à l'écran uniquement, `find_in_text`) : l'OCR fusionne souvent les mots d'une même
  ligne (« PRENDREMONGUIDE », « LIENENBIO »). On cherche alors le mot-clé, espaces retirés, À L'INTÉRIEUR
  d'un token plus long. Pour limiter les faux positifs, ce mode est réservé aux mots-clés d'au moins
  `GLUED_MIN_LEN` lettres (« lien », « dm » ou « like » collés matcheraient « ALIEN », « ADMIN »,
  « LIKELY »), et une occurrence est rejetée quand elle fait partie d'un mot courant qui la contient
  (« apPRENDRE », « comPRENDS », « surPRENDRE », « ABONNEment » : liste `GLUED_CONFUSABLES`).
"""
from __future__ import annotations

import re
import unicodedata

_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)

GLUED_MIN_LEN = 5
# Mots courants qui contiennent un mot-clé d'appel à l'action sans en être un (forme normalisée).
GLUED_CONFUSABLES = (
    "apprendre", "apprends", "comprendre", "comprends", "reprendre", "reprends", "surprendre", "surprends",
    "entreprendre", "entreprends", "meprendre", "abonnement", "abonnements", "photocopie", "photocopies",
    "partagee", "partagees", "telechargement", "telechargements", "reservation", "reservations",
    "shareholder", "shareholders",
)


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


def tokens(s: str) -> list[str]:
    """Mots normalisés (minuscules, sans accents) ; « abonne-toi » -> [abonne, toi]."""
    return _TOKEN.findall(strip_accents(s.lower()))


def squash(s: str) -> str:
    """Texte normalisé sans espaces ni ponctuation : « Lien en bio ! » -> « lienenbio »."""
    return "".join(tokens(s))


def keyword_tokens(keywords: list[str]) -> list[tuple[str, tuple[str, ...]]]:
    out = []
    for k in keywords:
        tk = tuple(tokens(k))
        if tk:
            out.append((k, tk))
    return out


def find_in_tokens(toks: list[str], keywords: list[tuple[str, tuple[str, ...]]]) -> list[tuple[int, str]]:
    """Positions (index du 1er token) des mots-clés trouvés dans une séquence de tokens."""
    hits = []
    n = len(toks)
    for label, kt in keywords:
        m = len(kt)
        for i in range(n - m + 1):
            if tuple(toks[i:i + m]) == kt:
                hits.append((i, label))
    return hits


def _inside_confusable(token: str, key: str, start: int) -> bool:
    for word in GLUED_CONFUSABLES:
        k = word.find(key)
        if word != key and k >= 0 and start - k >= 0 and token[start - k:start - k + len(word)] == word:
            return True
    return False


def _glued_match(token: str, key: str) -> bool:
    """`key` (sans espaces) apparaît dans `token` (plus long) hors d'un mot courant qui le contient."""
    start = token.find(key)
    while start >= 0:
        if not _inside_confusable(token, key, start):
            return True
        start = token.find(key, start + 1)
    return False


def find_glued(text: str, keywords: list[tuple[str, tuple[str, ...]]],
               min_len: int = GLUED_MIN_LEN) -> list[str]:
    """Mots-clés (≥ `min_len` lettres une fois les espaces retirés) collés à d'autres mots dans un token."""
    toks = [tk for tk in tokens(text) if len(tk) >= min_len]
    out = set()
    for label, kt in keywords:
        key = "".join(kt)
        if len(key) < min_len:
            continue
        if any(len(tk) >= len(key) and _glued_match(tk, key) for tk in toks):
            out.add(label)
    return sorted(out)


def find_in_text(text: str, keywords: list[tuple[str, tuple[str, ...]]],
                 glued_min_len: int | None = GLUED_MIN_LEN) -> list[str]:
    """Mots-clés présents dans un texte à l'écran (tokens + mode « collé » de l'OCR si `glued_min_len`)."""
    found = {label for _, label in find_in_tokens(tokens(text), keywords)}
    if glued_min_len:
        found.update(find_glued(text, keywords, glued_min_len))
    return sorted(found)
