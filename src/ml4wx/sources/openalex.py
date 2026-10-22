"""OpenAlex works API: journals, affiliations, citations. Uses $OPENALEX_API_KEY if set."""

from __future__ import annotations

import datetime as dt
import logging
import os
from collections.abc import Iterator

from .. import ids
from ..http import Client
from ..models import Candidate, Ids, Links

log = logging.getLogger(__name__)

API = "https://api.openalex.org/works"
SELECT = (
    "id,doi,title,publication_date,type,primary_location,best_oa_location,locations,"
    "authorships,abstract_inverted_index,cited_by_count"
)
OPEN_LICENSES = ("cc-by", "cc0", "public-domain")
_client = Client(min_interval=0.12)


def _params(**kw) -> dict:
    p = {k: v for k, v in kw.items() if v is not None}
    if key := os.environ.get("OPENALEX_API_KEY"):
        p["api_key"] = key
    return p


def abstract_from_index(inv: dict[str, list[int]] | None) -> str | None:
    if not inv:
        return None
    words: dict[int, str] = {}
    for w, positions in inv.items():
        for i in positions:
            words[i] = w
    return " ".join(words[i] for i in sorted(words)) or None


def _arxiv_from_locations(w: dict) -> str | None:
    for loc in [w.get("primary_location"), *(w.get("locations") or [])]:
        if not loc:
            continue
        for url in (loc.get("landing_page_url"), loc.get("pdf_url")):
            if url and "arxiv.org/" in url and (a := ids.norm_arxiv(url.split("arxiv.org/", 1)[1])):
                return a
    doi = w.get("doi") or ""
    return ids.norm_arxiv(doi) if "10.48550/" in doi.lower() else None


def parse_work(w: dict, source: str) -> Candidate | None:
    title = " ".join((w.get("title") or "").split())
    date = w.get("publication_date")
    if not title or not date:
        return None
    doi = ids.norm_doi(w.get("doi"))
    arxiv = _arxiv_from_locations(w)
    if doi and doi.startswith("10.48550/"):
        doi = None  # arXiv DOI: covered by the arXiv id
    loc = w.get("primary_location") or {}
    src = loc.get("source") or {}
    src_type = src.get("type")
    venue = src.get("display_name")
    venue_type = {"journal": "journal", "conference": "conference", "repository": "preprint"}.get(
        src_type or "", "other"
    )
    if venue_type == "preprint" and arxiv:
        venue = None
    oa = w.get("best_oa_location") or {}
    license_ = (oa.get("license") or loc.get("license") or "").lower()
    authorships = w.get("authorships") or []
    return Candidate(
        ids=Ids(arxiv=arxiv, doi=doi, openalex=ids.norm_openalex(w.get("id"))),
        title=title,
        authors=[(a.get("author") or {}).get("display_name") or "" for a in authorships],
        date=dt.date.fromisoformat(date),
        abstract=abstract_from_index(w.get("abstract_inverted_index")),
        abstract_license_ok=bool(arxiv) or license_.startswith(OPEN_LICENSES),
        venue=venue,
        venue_type=venue_type,
        links=Links(
            paper=f"https://doi.org/{doi}" if doi else loc.get("landing_page_url"),
            pdf=oa.get("pdf_url"),
        ),
        institutions=sorted(
            {
                x
                for a in authorships
                for i in a.get("institutions") or []
                for x in (
                    i["id"].rsplit("/", 1)[-1].upper() if i.get("id") else None,
                    "ror:" + i["ror"].rsplit("/", 1)[-1].lower() if i.get("ror") else None,
                )
                if x
            }
        ),
        affiliations=[s for a in authorships for s in a.get("raw_affiliation_strings") or []],
        cited_by=w.get("cited_by_count"),
        source=source,
    )


def _paged(filter_: str, search: str | None, source: str, limit: int | None) -> Iterator[Candidate]:
    cursor, n = "*", 0
    while cursor:
        r = _client.get(
            API, params=_params(filter=filter_, search=search, select=SELECT, per_page=200, cursor=cursor)
        )
        data = r.json()
        for w in data.get("results", []):
            if c := parse_work(w, source):
                yield c
                n += 1
                if limit and n >= limit:
                    return
        cursor = data.get("meta", {}).get("next_cursor")
        log.info("openalex %s: %d / %s", source, n, data.get("meta", {}).get("count"))
        if not data.get("results"):
            break


def search(
    expr: str,
    start: dt.date,
    end: dt.date,
    types: list[str] | None = None,
    source: str = "openalex",
    limit: int | None = None,
) -> Iterator[Candidate]:
    """Boolean search over title and abstract, by publication date."""
    f = [
        f"title_and_abstract.search:{' '.join(expr.split())}",
        f"from_publication_date:{start.isoformat()}",
        f"to_publication_date:{end.isoformat()}",
    ]
    if types:
        f.append("type:" + "|".join(types))
    yield from _paged(",".join(f), None, source, limit)


def count(expr: str, start: dt.date, end: dt.date, types: list[str] | None = None) -> int:
    f = [
        f"title_and_abstract.search:{' '.join(expr.split())}",
        f"from_publication_date:{start.isoformat()}",
        f"to_publication_date:{end.isoformat()}",
    ]
    if types:
        f.append("type:" + "|".join(types))
    r = _client.get(API, params=_params(filter=",".join(f), per_page=1, select="id"))
    return int(r.json()["meta"]["count"])


def lookup(keys: list[str], source: str = "openalex") -> Iterator[Candidate]:
    """Fetch works by 'doi:...', 'arxiv:...' or 'openalex:W...' keys, 50 per request."""
    dois = [k[4:] for k in keys if k.startswith("doi:")]
    dois += [f"10.48550/arxiv.{k[6:]}" for k in keys if k.startswith("arxiv:")]
    oas = [k[9:] for k in keys if k.startswith("openalex:")]
    for field, vals in (("doi", dois), ("openalex", oas)):
        for i in range(0, len(vals), 50):
            chunk = vals[i : i + 50]
            yield from _paged(f"{field}:" + "|".join(chunk), None, source, None)


def citing(openalex_id: str, source: str = "snowball", limit: int | None = None) -> Iterator[Candidate]:
    """Works that cite the given work (forward snowballing)."""
    yield from _paged(f"cites:{openalex_id}", None, source, limit)
