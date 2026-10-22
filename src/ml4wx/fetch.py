"""Collect candidates from the sources for a date range."""

from __future__ import annotations

import datetime as dt
import logging
import os

from .config import Config
from .http import BudgetExceeded
from .models import Candidate
from .sources import arxiv, openalex

log = logging.getLogger(__name__)

SOURCES = ("arxiv", "openalex")


class SourceFailed(RuntimeError):
    pass


def from_source(cfg: Config, source: str, start: dt.date, end: dt.date) -> list[Candidate]:
    """All candidates of one source for [start, end]; raises SourceFailed if the source is down."""
    out: list[Candidate] = []
    try:
        if source == "arxiv":
            for q in cfg.sources["arxiv"]["queries"]:
                out += arxiv.search(q["query"], start, end, source=f"arxiv:{q['name']}")
        else:
            oa = cfg.sources["openalex"]
            for q in oa["queries"]:
                out += openalex.search(q["search"], start, end, oa.get("types"), source=f"openalex:{q['name']}")
    except (RuntimeError, BudgetExceeded) as e:  # RuntimeError: retries exhausted (e.g. arXiv 429)
        raise SourceFailed(f"{source}: {e}") from e
    return out


def default_sources(use_openalex: bool | None = None) -> list[str]:
    if use_openalex is None:
        use_openalex = bool(os.environ.get("OPENALEX_API_KEY"))
    return ["arxiv", "openalex"] if use_openalex else ["arxiv"]


def recent(cfg: Config, start: dt.date, end: dt.date, use_openalex: bool | None = None) -> list[Candidate]:
    """Candidates from all sources. One source failing is logged; all failing raises."""
    out: list[Candidate] = []
    sources = default_sources(use_openalex)
    failed = []
    for s in sources:
        try:
            out += from_source(cfg, s, start, end)
        except SourceFailed as e:
            log.warning("source skipped: %s", e)
            failed.append(s)
    if len(failed) == len(sources):
        raise SourceFailed(f"all sources failed: {', '.join(failed)}")
    log.info("fetched %d candidates for %s..%s", len(out), start, end)
    return out
