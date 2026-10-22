"""From candidates to archived papers: merge, prefilter, classify, enrich, score, save."""

from __future__ import annotations

import datetime as dt
import functools
import logging
import re
from dataclasses import dataclass, field

from . import classify, ids, llm
from .config import Config
from .models import Candidate, Links, Paper
from .sources import arxiv
from .store import Archive

log = logging.getLogger(__name__)


# --- merging -------------------------------------------------------------------
def merge(a: Candidate, b: Candidate) -> Candidate:
    """Combine two records of the same work (e.g. arXiv + OpenAlex)."""
    journal = a if a.venue_type == "journal" else b if b.venue_type == "journal" else None
    abstract_src = next((x for x in (a, b) if x.abstract and x.abstract_license_ok), None) or (
        a if a.abstract else b
    )
    return Candidate(
        ids=a.ids.model_copy(
            update={k: v for k, v in b.ids.model_dump().items() if v and not getattr(a.ids, k)}
        ),
        title=a.title,
        authors=a.authors if len(a.authors) >= len(b.authors) else b.authors,
        date=min(a.date, b.date),
        abstract=abstract_src.abstract,
        abstract_license_ok=abstract_src.abstract_license_ok,
        venue=journal.venue if journal else a.venue or b.venue,
        venue_type=journal.venue_type if journal else a.venue_type,
        links=Links(**{k: getattr(a.links, k) or getattr(b.links, k) for k in Links.model_fields}),
        institutions=sorted(set(a.institutions) | set(b.institutions)),
        affiliations=list(dict.fromkeys(a.affiliations + b.affiliations)),
        comment=a.comment or b.comment,
        cited_by=max(a.cited_by or 0, b.cited_by or 0) or None,
        source=a.source or b.source,
    )


def dedupe(cands: list[Candidate]) -> list[Candidate]:
    out: list[Candidate] = []
    by_key: dict[str, int] = {}
    by_title: dict[str, int] = {}
    for c in cands:
        hit = next((by_key[k] for k in c.ids.keys() if k in by_key), None)
        if hit is None:
            hit = by_title.get(ids.norm_title(c.title))
        if hit is None:
            out.append(c)
            hit = len(out) - 1
        else:
            out[hit] = merge(out[hit], c)
        for k in out[hit].ids.keys():
            by_key[k] = hit
        by_title[ids.norm_title(out[hit].title)] = hit
    return out


# --- filters and enrichment ----------------------------------------------------
@functools.cache
def _pattern(alternatives: tuple[str, ...]) -> re.Pattern:
    return re.compile("|".join(f"(?:{a})" for a in alternatives), re.I)


def text_passes(text: str, cfg: Config) -> bool:
    """At least one ML pattern and one weather/climate pattern (config/sources.yaml)."""
    ml = _pattern(tuple(cfg.sources["ml_patterns"]))
    wx = _pattern(tuple(cfg.sources["wx_patterns"]))
    return bool(ml.search(text) and wx.search(text))


def prefilter(c: Candidate, cfg: Config) -> bool:
    return text_passes(f"{c.title} {c.abstract or ''}", cfg)


def detect_orgs(c: Candidate, cfg: Config, extra_text: str = "") -> list[str]:
    found = []
    affil = " | ".join(c.affiliations).lower() + " | " + extra_text.lower()
    for o in cfg.orgs:
        if set(c.institutions) & o.institution_keys or any(a in affil for a in o.aliases):
            found.append(o.slug)
    return found


def score(p: Paper, cfg: Config) -> int:
    w = cfg.settings["highlight"]["weights"]
    tiers = {cfg.org_by_slug[o].tier for o in p.orgs if o in cfg.org_by_slug}
    s = w["org_tier1"] if 1 in tiers else w["org_tier2"] if 2 in tiers else 0
    if p.venue and p.venue.lower() in {v.lower() for v in cfg.settings["top_venues"]}:
        s += w["top_venue"]
    if p.links.code or p.links.weights:
        s += w["open_code"]
    if p.status == "operational":
        s += w["operational"]
    for threshold, pts in sorted(cfg.settings["highlight"].get("citations", []), reverse=True):
        if (p.cited_by or 0) >= threshold:
            s += pts
            break
    s += w["significance"] * p.significance
    return min(100, s)


def finalize(p: Paper, cfg: Config) -> None:
    """Set score and highlight flag. A highlight keeps the judge's reason (or the summary)."""
    h = cfg.settings["highlight"]
    p.score = score(p, cfg)
    p.highlight = (p.score >= h["threshold"] and p.significance >= h["min_significance"]) or (
        p.significance >= h["always_significance"]
    )
    if not p.highlight:
        p.highlight_reason = None


def maybe_highlight(p: Paper, cfg: Config) -> bool:
    """Worth a second opinion from the judge model? (The judge may raise or lower significance.)"""
    h = cfg.settings["highlight"]
    return score(p, cfg) >= h["threshold"] - 15 or p.significance >= h["min_significance"] + 1


JUDGE_SYSTEM = """You judge how important a new paper is for the machine learning for weather \
and climate community. Be strict: 9-10 only for landmark work (new state of the art global \
model, major benchmark, operational deployment at a major centre); 7-8 major contribution that \
experts should read this week; 4-6 solid; 0-3 incremental. Give one short sentence (max 25 words) \
saying why it matters, for a researcher in the field."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "significance": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": ["id", "significance", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}


def judge(papers: list[Paper], cfg: Config) -> None:
    """Second opinion from a stronger model on possible highlights (updates in place)."""
    by_id = {p.id: p for p in papers}
    for i in range(0, len(papers), 10):
        chunk = papers[i : i + 10]
        prompt = "\n".join(
            f"[{p.id}]\nTitle: {p.title}\nOrgs: {', '.join(p.orgs) or 'unknown'}\n"
            f"Venue: {p.venue or 'preprint'} ({p.year})\nAbstract: {(p.abstract or p.summary)[:1500]}\n"
            for p in chunk
        )
        try:
            out = llm.structured(prompt, JUDGE_SYSTEM, JUDGE_SCHEMA, cfg.settings["llm"]["judge_model"])
        except llm.LLMError as e:
            log.warning("judge failed: %s", e)
            continue
        for r in out["results"]:
            if p := by_id.get(r["id"].strip("[]")):
                p.significance = max(0, min(10, int(r["significance"])))
                p.highlight_reason = r["reason"].strip()


def to_paper(c: Candidate, r: dict, orgs: list[str], today: dt.date, source: str) -> Paper:
    links = c.links.model_copy()
    links.code = links.code or arxiv.code_link(c.comment, c.abstract)
    return Paper(
        id=ids.paper_id(c.ids.arxiv, c.ids.doi, c.ids.openalex),
        title=c.title,
        authors=c.authors[:50],
        date=c.date,
        year=c.date.year,
        venue=c.venue,
        venue_type=c.venue_type,
        ids=c.ids,
        links=links,
        orgs=orgs,
        topic=r["topic"],
        topics=r["topics"],
        type=r["type"],
        method=r["method"],
        timescale=[t for t in r["timescale"] if t != "none"],
        spatial=r["spatial"],
        data=r["data"],
        status=r["status"],
        summary=r["summary"].strip(),
        abstract=c.abstract if c.abstract_license_ok else None,
        significance=r["significance"],
        cited_by=c.cited_by,
        added=today,
        source=source,
    )


def update_existing(p: Paper, c: Candidate, cfg: Config) -> bool:
    """New info about an archived paper (journal version, DOI, orgs). True if changed."""
    before = p.model_dump()
    for k, v in c.ids.model_dump().items():
        if v and not getattr(p.ids, k):
            setattr(p.ids, k, v)
    if c.venue_type == "journal" and p.venue_type != "journal":
        p.venue, p.venue_type = c.venue, "journal"
        if c.links.paper:
            p.links.paper = c.links.paper
    p.orgs = sorted(set(p.orgs) | set(detect_orgs(c, cfg)))
    if c.cited_by and c.cited_by > (p.cited_by or 0):
        p.cited_by = c.cited_by
    if p.model_dump() != before:
        finalize(p, cfg)
        return True
    return False


# --- main entry ----------------------------------------------------------------
@dataclass
class Report:
    added: list[Paper] = field(default_factory=list)
    updated: list[Paper] = field(default_factory=list)
    parked: int = 0
    rejected: int = 0
    filtered: int = 0
    known: int = 0
    failed: int = 0


def ingest(
    cands: list[Candidate],
    archive: Archive,
    cfg: Config,
    *,
    source: str,
    today: dt.date | None = None,
    use_prefilter: bool = True,
    html_affiliations: bool = False,
    judge_highlights: bool = True,
    force_topic: dict[str, str] | None = None,
    classic_keys: set[str] | None = None,
) -> Report:
    """Classify new candidates and save in-scope ones.

    force_topic: candidate key -> topic; these are always archived (curated seeds).
    """
    today = today or dt.date.today()
    force_topic = force_topic or {}
    rep = Report()
    todo: list[Candidate] = []
    for c in dedupe(cands):
        if p := archive.find(c):
            if update_existing(p, c, cfg):
                archive.save_paper(p)
                rep.updated.append(p)
            else:
                rep.known += 1
        elif archive.status(c) and c.key not in force_topic:
            rep.known += 1
        elif use_prefilter and not prefilter(c, cfg):
            rep.filtered += 1
        else:
            todo.append(c)
    log.info("%d new candidates to classify (%d known, %d filtered)", len(todo), rep.known, rep.filtered)

    results = classify.classify(todo, cfg)
    new: list[Paper] = []
    for c in todo:
        r = results.get(c.key)
        if r is not None and (topic := force_topic.get(c.key)):
            r["decision"], r["topic"] = "in", topic
            r["topics"] = [t for t in r["topics"] if t != topic]
        if r is None:
            rep.failed += 1
        elif r["decision"] == "out":
            archive.reject(c)
            rep.rejected += 1
        elif r["decision"] == "later":
            archive.park(c, r["topic"])
            rep.parked += 1
        else:
            extra = ""
            if html_affiliations and c.ids.arxiv and not detect_orgs(c, cfg):
                extra = arxiv.html_front_matter(c.ids.arxiv)
            p = to_paper(c, r, detect_orgs(c, cfg, extra), today, source)
            p.classic = c.key in (classic_keys or ())
            new.append(p)

    if judge_highlights:
        if maybe := [p for p in new if maybe_highlight(p, cfg)]:
            judge(maybe, cfg)
    for p in new:
        finalize(p, cfg)
        archive.save_paper(p)
        rep.added.append(p)
    archive.save_state()
    log.info(
        "ingest: +%d papers, %d updated, %d parked, %d rejected, %d failed",
        len(rep.added), len(rep.updated), rep.parked, rep.rejected, rep.failed,
    )  # fmt: skip
    return rep
