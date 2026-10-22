"""Find and merge duplicate papers already in the archive (preprint vs journal version)."""

from __future__ import annotations

import logging
from collections import defaultdict

from . import ids
from .config import Config
from .ingest import finalize
from .models import Candidate, Paper
from .store import Archive

log = logging.getLogger(__name__)

# DOIs of preprint servers: a journal DOI is preferred over these.
PREPRINT_DOI_PREFIXES = (
    "10.48550/",  # arXiv
    "10.2139/",  # SSRN
    "10.22541/",  # Authorea / ESS Open Archive
    "10.1002/essoar",  # ESS Open Archive (older)
    "10.31223/",  # EarthArXiv
    "10.23689/",  # fidgeo
    "10.20944/",  # Preprints.org
    "10.21203/",  # Research Square
    "10.5194/egusphere",  # EGUsphere
    "10.36227/",  # TechRxiv
)


def is_preprint_doi(doi: str | None) -> bool:
    return bool(doi) and doi.startswith(PREPRINT_DOI_PREFIXES)


def _rank(p: Paper) -> tuple:
    """Lower is better: keep the arXiv-id record (stable id), then journal, then more labs."""
    return (not p.ids.arxiv, p.venue_type != "journal", -len(p.orgs), p.added, p.id)


def groups(archive: Archive, min_overlap: float = 0.85, max_years: int = 2) -> list[list[Paper]]:
    """Groups of papers that are the same work: same title key, or same first author with a
    very similar title and close years."""
    papers = list(archive.papers.values())
    parent = {p.id: p.id for p in papers}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        parent[find(a)] = find(b)

    by_key: dict[str, list[Paper]] = defaultdict(list)
    by_first: dict[str, list[Paper]] = defaultdict(list)
    for p in papers:
        by_key[ids.title_key(p.title)].append(p)
        if p.authors:
            by_first[ids.surname(p.authors[0])].append(p)
    for same in by_key.values():
        for p in same[1:]:
            union(p.id, same[0].id)
    for same in by_first.values():
        for i, a in enumerate(same):
            for b in same[i + 1 :]:
                if abs(a.year - b.year) <= max_years and ids.word_overlap(a.title, b.title) >= min_overlap:
                    union(a.id, b.id)
    out: dict[str, list[Paper]] = defaultdict(list)
    for p in papers:
        out[find(p.id)].append(p)
    return [sorted(g, key=_rank) for g in out.values() if len(g) > 1]


def merge_into(keep: Paper, other: Paper) -> None:
    """Move what `other` knows into `keep` (ids, journal venue, labs, citations, links)."""
    if other.ids.arxiv and not keep.ids.arxiv:
        keep.ids.arxiv = other.ids.arxiv
    if other.ids.doi and (not keep.ids.doi or (is_preprint_doi(keep.ids.doi) and not is_preprint_doi(other.ids.doi))):
        keep.ids.doi = other.ids.doi
    if other.ids.openalex and not keep.ids.openalex:
        keep.ids.openalex = other.ids.openalex
    if other.venue_type == "journal" and keep.venue_type != "journal":
        keep.venue, keep.venue_type = other.venue, "journal"
        if other.links.paper:
            keep.links.paper = other.links.paper
    for k in type(keep.links).model_fields:
        if not getattr(keep.links, k) and getattr(other.links, k):
            setattr(keep.links, k, getattr(other.links, k))
    keep.orgs = sorted(set(keep.orgs) | set(other.orgs))
    keep.cited_by = max(keep.cited_by or 0, other.cited_by or 0) or None
    keep.abstract = keep.abstract or other.abstract
    keep.classic = keep.classic or other.classic
    if other.date < keep.date:
        keep.date, keep.year = other.date, other.date.year
    if other.significance > keep.significance:
        keep.significance = other.significance
        keep.highlight_reason = other.highlight_reason or keep.highlight_reason
    keep.highlight_reason = keep.highlight_reason or other.highlight_reason


def run(archive: Archive, cfg: Config, dry_run: bool = False) -> tuple[int, int]:
    """Merge duplicates and drop retracted works. Returns (merged, retracted) counts."""
    merged = retracted = 0
    gone: set[str] = set()
    for group in groups(archive):
        if any(ids.is_retracted_title(p.title) for p in group):
            for p in group:
                log.info("retracted: %s %s", p.id, p.title[:80])
                if not dry_run:
                    remove(archive, p)
                gone.add(p.id)
                retracted += 1
            continue
        keep, rest = group[0], group[1:]
        for other in rest:
            log.info("merge %s <- %s | %s", keep.id, other.id, other.title[:70])
            if not dry_run:
                old_path = archive.paper_path(keep)
                merge_into(keep, other)
                remove(archive, other, reject=False)
                finalize(keep, cfg)
                if archive.paper_path(keep) != old_path:
                    old_path.unlink(missing_ok=True)
                archive.save_paper(keep)
            merged += 1
    for p in [p for p in archive.papers.values() if ids.is_retracted_title(p.title) and p.id not in gone]:
        log.info("retracted: %s %s", p.id, p.title[:80])
        if not dry_run:
            remove(archive, p)
        retracted += 1
    if not dry_run:
        archive.save_state()
    return merged, retracted


def remove(archive: Archive, p: Paper, reject: bool = True) -> None:
    archive.paper_path(p).unlink(missing_ok=True)
    archive.papers.pop(p.id, None)
    if reject:  # never add this work again
        archive.reject(Candidate(ids=p.ids, title=p.title, date=p.date))
