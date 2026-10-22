"""Crossref: metadata by DOI. Free, no key; used as a fallback for DOI-only papers."""

from __future__ import annotations

import datetime as dt
import logging
import re
from collections.abc import Iterator

from .. import ids
from ..http import Client
from ..models import Candidate, Ids, Links

log = logging.getLogger(__name__)

API = "https://api.crossref.org/works/"
_client = Client(min_interval=0.2)


def _date(m: dict) -> dt.date | None:
    for k in ("published-online", "published-print", "issued", "created"):
        parts = (m.get(k) or {}).get("date-parts") or [[]]
        p = parts[0]
        if p and p[0]:
            return dt.date(p[0], p[1] if len(p) > 1 else 1, p[2] if len(p) > 2 else 1)
    return None


def parse(m: dict, source: str) -> Candidate | None:
    doi = ids.norm_doi(m.get("DOI"))
    title = " ".join(" ".join(m.get("title") or []).split())
    date = _date(m)
    if not (doi and title and date):
        return None
    kind = m.get("type", "")
    abstract = m.get("abstract")
    if abstract:
        abstract = " ".join(re.sub(r"<[^>]+>", " ", abstract).split())
    licenses = [x.get("URL", "").lower() for x in m.get("license") or []]
    return Candidate(
        ids=Ids(doi=doi),
        title=title,
        authors=[f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author") or []],
        date=date,
        abstract=abstract,
        abstract_license_ok=any("creativecommons.org/licenses/by" in u or "publicdomain" in u for u in licenses),
        venue=" ".join(m.get("container-title") or []) or None,
        venue_type="conference" if "proceedings" in kind else "journal" if "journal" in kind else "other",
        links=Links(paper=f"https://doi.org/{doi}"),
        affiliations=[a.get("name", "") for au in m.get("author") or [] for a in au.get("affiliation") or []],
        cited_by=m.get("is-referenced-by-count"),
        source=source,
    )


def lookup(dois: list[str], source: str = "crossref") -> Iterator[Candidate]:
    for doi in dois:
        r = _client.get(API + doi, params={"mailto": "ml4wx@users.noreply.github.com"})
        if r.status_code != 200:
            log.warning("crossref %s: HTTP %d", doi, r.status_code)
            continue
        if c := parse(r.json().get("message", {}), source):
            yield c
