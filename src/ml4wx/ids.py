"""Identifier normalization and title matching for deduplication."""

from __future__ import annotations

import re
import unicodedata

# YYMM.NNNNN with a real month, not inside a longer number (so DOIs like 10.1098/x.2020.0782 fail).
_ARXIV_NEW = re.compile(r"(?<![\d.])(\d{2}(?:0[1-9]|1[0-2])\.\d{4,5})(v\d+)?(?![\d])")
_ARXIV_OLD = re.compile(r"(?<![\w-])([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?(?!\d)")
_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(.+)$", re.I)


def norm_arxiv(s: str | None) -> str | None:
    """'http://arxiv.org/abs/2212.12794v2', 'arXiv:2212.12794' or the arXiv DOI -> '2212.12794'.

    Other DOIs never give an arXiv id.
    """
    if not s:
        return None
    s = s.strip()
    if m := _ARXIV_DOI.search(s):
        s = m.group(1)
    elif re.search(r"(^|/)10\.\d{4,}/", s):
        return None
    if m := _ARXIV_NEW.search(s):
        return m.group(1)
    if m := _ARXIV_OLD.search(s):
        return m.group(1)
    return None


def norm_doi(s: str | None) -> str | None:
    """'https://doi.org/10.1126/Science.ADI2336' -> '10.1126/science.adi2336'."""
    if not s:
        return None
    s = s.strip().lower()
    s = re.sub(r"^(https?://)?(dx\.)?doi\.org/", "", s)
    s = s.removeprefix("doi:")
    return s if s.startswith("10.") else None


def norm_openalex(s: str | None) -> str | None:
    if not s:
        return None
    s = s.rsplit("/", 1)[-1].upper()
    return s if re.fullmatch(r"W\d+", s) else None


def keys_for(arxiv: str | None, doi: str | None, openalex: str | None) -> list[str]:
    """All lookup keys of a work, e.g. ['arxiv:2212.12794', 'doi:10.1126/...']."""
    keys = []
    if arxiv:
        keys.append(f"arxiv:{arxiv}")
        keys.append(f"doi:10.48550/arxiv.{arxiv.lower()}")
    if doi:
        keys.append(f"doi:{doi}")
    if openalex:
        keys.append(f"openalex:{openalex}")
    return keys


def paper_id(arxiv: str | None, doi: str | None, openalex: str | None) -> str:
    """File-safe stable id. arXiv first, then DOI, then OpenAlex."""
    if arxiv:
        return "arxiv-" + arxiv.replace("/", "-")
    if doi and not doi.startswith("10.48550/"):
        return "doi-" + re.sub(r"[^a-z0-9.]+", "-", doi).strip("-")
    if openalex:
        return "openalex-" + openalex
    raise ValueError("work has no identifier")


_RETRACTED = re.compile(r"^\s*retracted(\s+article)?\s*:\s*", re.I)
# British/American variants that differ between preprint and journal titles.
_SPELLING = [(r"parametri[sz]ation|parameteri[sz]ation", "parameterization"), (r"modell", "model")]


def is_retracted_title(t: str) -> bool:
    return bool(_RETRACTED.match(t))


def norm_title(t: str) -> str:
    t = re.sub(r"[‐-―−]", "-", t)  # Unicode dashes (Wiley uses U+2010)
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode().lower()
    t = _RETRACTED.sub("", t)
    for pattern, repl in _SPELLING:
        t = re.sub(pattern, repl, t)
    return re.sub(r"[^a-z0-9]+", " ", t).strip()


def title_key(t: str) -> str:
    """Letters and digits only: 'data set' == 'dataset', 'heat wave' == 'heatwave'."""
    return norm_title(t).replace(" ", "")


def title_words(t: str) -> set[str]:
    """Words for overlap checks. Numbers always count, so 'EPT-1.5' != 'EPT-2'."""
    return {w for w in norm_title(t).split() if len(w) > 2 or w.isdigit()}


def word_overlap(a: str, b: str) -> float:
    wa, wb = title_words(a), title_words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def same_title(a: str, b: str, threshold: float = 0.8) -> bool:
    """Same title key, or high word overlap (preprint vs journal titles differ a bit)."""
    if title_key(a) == title_key(b):
        return True
    if len(title_words(a)) < 4 or len(title_words(b)) < 4:
        return False
    return word_overlap(a, b) >= threshold


def surname(name: str) -> str:
    parts = norm_title(name).split()
    return parts[-1] if parts else ""
