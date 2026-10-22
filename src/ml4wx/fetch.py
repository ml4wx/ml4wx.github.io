"""Collect candidates from all sources for a date range."""

from __future__ import annotations

import datetime as dt
import logging
import os

from .config import Config
from .http import BudgetExceeded
from .models import Candidate
from .sources import arxiv, openalex

log = logging.getLogger(__name__)


def recent(cfg: Config, start: dt.date, end: dt.date, use_openalex: bool | None = None) -> list[Candidate]:
    """arXiv queries plus (if a key is set, or forced) OpenAlex queries for [start, end]."""
    out: list[Candidate] = []
    for q in cfg.sources["arxiv"]["queries"]:
        out += arxiv.search(q["query"], start, end, source=f"arxiv:{q['name']}")
    if use_openalex is None:
        use_openalex = bool(os.environ.get("OPENALEX_API_KEY"))
    if use_openalex:
        oa = cfg.sources["openalex"]
        try:
            for q in oa["queries"]:
                out += openalex.search(q["search"], start, end, oa.get("types"), source=f"openalex:{q['name']}")
        except BudgetExceeded as e:
            log.warning("OpenAlex skipped: %s", e)
    log.info("fetched %d candidates for %s..%s", len(out), start, end)
    return out
