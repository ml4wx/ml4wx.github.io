"""Semantic Scholar Graph API: batch metadata and citation snowballing. Uses $S2_API_KEY.

With a key the limit is 1 request per second; without one, a busy shared pool.
"""

from __future__ import annotations

import datetime as dt
import logging
import os
from collections.abc import Iterator

from .. import ids
from ..http import Client
from ..models import Candidate, Ids, Links

log = logging.getLogger(__name__)

API = "https://api.semanticscholar.org/graph/v1"
FIELDS = (
    "paperId,externalIds,title,abstract,year,venue,publicationDate,authors,citationCount,"
    "openAccessPdf,publicationVenue,journal"
)
_client = Client(min_interval=1.05, max_retries=8)


def _headers() -> dict:
    key = os.environ.get("S2_API_KEY")
    return {"x-api-key": key} if key else {}


def s2_id(key: str) -> str | None:
    """'arxiv:2212.12794' -> 'ARXIV:2212.12794'; 'doi:10.x/y' -> 'DOI:10.x/y'."""
    kind, _, val = key.partition(":")
    prefix = {"arxiv": "ARXIV", "doi": "DOI"}.get(kind)
    return f"{prefix}:{val}" if prefix and val else None


def parse_paper(p: dict, source: str) -> Candidate | None:
    if not p or not p.get("title"):
        return None
    ext = p.get("externalIds") or {}
    arxiv = ids.norm_arxiv(ext.get("ArXiv"))
    doi = ids.norm_doi(ext.get("DOI"))
    if doi and doi.startswith("10.48550/"):
        doi = None
    if not (arxiv or doi):
        return None  # without arXiv id or DOI we cannot merge reliably
    date = p.get("publicationDate") or (f"{p['year']}-01-01" if p.get("year") else None)
    if not date:
        return None
    pv = p.get("publicationVenue") or {}
    vtype = {"journal": "journal", "conference": "conference"}.get(pv.get("type") or "", "")
    venue = (p.get("journal") or {}).get("name") or pv.get("name") or p.get("venue") or None
    if not vtype:
        vtype = "preprint" if arxiv and (not venue or "arxiv" in venue.lower()) else "other"
    if vtype == "preprint":
        venue = None
    return Candidate(
        ids=Ids(arxiv=arxiv, doi=doi),
        title=" ".join(p["title"].split()),
        authors=[a.get("name") or "" for a in p.get("authors") or []],
        date=dt.date.fromisoformat(date),
        abstract=p.get("abstract"),
        abstract_license_ok=bool(arxiv),
        venue=venue,
        venue_type=vtype,
        links=Links(
            paper=f"https://arxiv.org/abs/{arxiv}" if arxiv else f"https://doi.org/{doi}",
            pdf=(p.get("openAccessPdf") or {}).get("url") or None,
        ),
        cited_by=p.get("citationCount"),
        source=source,
    )


def batch(keys: list[str], source: str = "s2") -> Iterator[tuple[str, dict | None]]:
    """(key, raw paper or None) for each key, 500 per request."""
    pairs = [(k, s2_id(k)) for k in keys]
    pairs = [(k, s) for k, s in pairs if s]
    for i in range(0, len(pairs), 500):
        chunk = pairs[i : i + 500]
        r = _client.post(
            f"{API}/paper/batch", json={"ids": [s for _, s in chunk]}, params={"fields": FIELDS}, headers=_headers()
        )
        for (k, _), p in zip(chunk, r.json(), strict=True):
            yield k, p


def _edges(paper_id: str, direction: str, source: str, max_n: int) -> Iterator[Candidate]:
    field = "citingPaper" if direction == "citations" else "citedPaper"
    offset = 0
    while offset < max_n:
        r = _client.get(
            f"{API}/paper/{paper_id}/{direction}",
            params={"fields": FIELDS, "limit": 1000, "offset": offset},
            headers=_headers(),
        )
        if r.status_code == 404:
            return
        data = r.json()
        for e in data.get("data") or []:
            if c := parse_paper(e.get(field) or {}, source):
                yield c
        if data.get("next") is None:
            return
        offset = data["next"]


def citations(paper_id: str, source: str = "snowball", max_n: int = 9000) -> Iterator[Candidate]:
    """Papers that cite `paper_id` (an S2 paperId or 'ARXIV:...' / 'DOI:...')."""
    yield from _edges(paper_id, "citations", source, max_n)


def references(paper_id: str, source: str = "snowball", max_n: int = 1000) -> Iterator[Candidate]:
    yield from _edges(paper_id, "references", source, max_n)
