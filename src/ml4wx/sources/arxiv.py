"""arXiv API (Atom). Rules: one request every 3 seconds, at most 2000 results per page."""

from __future__ import annotations

import datetime as dt
import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterator

from .. import ids
from ..http import Client
from ..models import Candidate, Ids, Links

log = logging.getLogger(__name__)

API = "https://export.arxiv.org/api/query"
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
}
# arXiv throttles busy shared addresses (CI runners) with 429 for minutes: be patient (~25 min).
_client = Client(min_interval=3.1, max_retries=9, max_delay=300)


def _text(el: ET.Element, path: str) -> str | None:
    x = el.find(path, NS)
    return " ".join(x.text.split()) if x is not None and x.text else None


def parse_entry(e: ET.Element, source: str) -> Candidate | None:
    arxiv_id = ids.norm_arxiv(_text(e, "a:id"))
    title = _text(e, "a:title")
    if not arxiv_id or not title:
        return None
    published = _text(e, "a:published") or ""
    journal_ref = _text(e, "arxiv:journal_ref")
    return Candidate(
        ids=Ids(arxiv=arxiv_id, doi=ids.norm_doi(_text(e, "arxiv:doi"))),
        title=title,
        authors=[_text(a, "a:name") or "" for a in e.findall("a:author", NS)],
        date=dt.date.fromisoformat(published[:10]),
        abstract=_text(e, "a:summary"),
        abstract_license_ok=True,  # arXiv metadata, including abstracts, is CC0
        venue=journal_ref,
        venue_type="journal" if journal_ref else "preprint",
        links=Links(
            paper=f"https://arxiv.org/abs/{arxiv_id}", pdf=f"https://arxiv.org/pdf/{arxiv_id}"
        ),
        affiliations=[
            x.text.strip() for x in e.findall("a:author/arxiv:affiliation", NS) if x.text
        ],
        comment=_text(e, "arxiv:comment"),
        source=source,
    )


def search(
    query: str, start: dt.date, end: dt.date, source: str = "arxiv", page_size: int = 1000
) -> Iterator[Candidate]:
    """All entries matching `query` submitted between start and end (inclusive)."""
    q = " ".join(query.split())
    full = f"({q}) AND submittedDate:[{start:%Y%m%d}0000 TO {end:%Y%m%d}2359]"
    offset = 0
    while True:
        r = _client.get(
            API,
            params={
                "search_query": full,
                "start": offset,
                "max_results": page_size,
                "sortBy": "submittedDate",
                "sortOrder": "ascending",
            },
        )
        root = ET.fromstring(r.content)
        total = int(_text(root, "os:totalResults") or 0)
        entries = root.findall("a:entry", NS)
        for e in entries:
            if c := parse_entry(e, source):
                yield c
        offset += len(entries)
        log.info("arxiv %s: %d / %d", source, offset, total)
        if not entries or offset >= total:
            break


def lookup(arxiv_ids: list[str], source: str = "arxiv") -> Iterator[Candidate]:
    """Fetch specific papers by id, 100 per request."""
    for i in range(0, len(arxiv_ids), 100):
        chunk = arxiv_ids[i : i + 100]
        r = _client.get(API, params={"id_list": ",".join(chunk), "max_results": len(chunk)})
        for e in ET.fromstring(r.content).findall("a:entry", NS):
            if c := parse_entry(e, source):
                yield c


_GITHUB = re.compile(r"https?://github\.com/[\w.-]+/[\w.-]+", re.I)


def code_link(*texts: str | None) -> str | None:
    for t in texts:
        if t and (m := _GITHUB.search(t)):
            return m.group(0).rstrip(".)")
    return None


def html_front_matter(arxiv_id: str, max_chars: int = 20000) -> str:
    """Plain text of the start of the arXiv HTML version: title block with affiliations."""
    r = _client.get(f"https://arxiv.org/html/{arxiv_id}")
    if r.status_code != 200:
        return ""
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", r.text[: max_chars * 5], flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())[:max_chars]
