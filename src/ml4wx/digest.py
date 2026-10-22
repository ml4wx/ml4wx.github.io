"""Weekly digest: digests/<YYYY-Www>.md, used as the GitHub issue body and as a site page."""

from __future__ import annotations

import datetime as dt
import logging
from collections import defaultdict
from pathlib import Path

from . import llm
from .config import Config
from .export import env, sorted_papers
from .models import Paper
from .store import Archive

log = logging.getLogger(__name__)

INTRO_SYSTEM = """You write the opening of a weekly digest of new machine learning papers for \
weather and climate. Readers are researchers in the field. Write 2-4 plain sentences: the main \
themes of the week and the one or two papers most worth reading, by name. No hype, no lists, \
no greeting. Use simple, direct English."""

INTRO_SCHEMA = {
    "type": "object",
    "properties": {"intro": {"type": "string"}},
    "required": ["intro"],
    "additionalProperties": False,
}


def week_id(d: dt.date) -> str:
    y, w, _ = d.isocalendar()
    return f"{y}-W{w:02d}"


def intro(papers: list[Paper], cfg: Config) -> str:
    if not papers:
        return "No new papers this week."
    lines = "\n".join(
        f"- {'[HIGHLIGHT] ' if p.highlight else ''}{p.title} ({', '.join(p.orgs) or 'n/a'}): {p.summary}"
        for p in sorted(papers, key=lambda p: -p.score)[:40]
    )
    try:
        return llm.structured(
            f"This week's papers:\n{lines}", INTRO_SYSTEM, INTRO_SCHEMA, cfg.settings["llm"]["digest_model"]
        )["intro"].strip()
    except llm.LLMError as e:
        log.warning("digest intro failed: %s", e)
        return f"{len(papers)} new papers this week."


def build(
    archive: Archive,
    cfg: Config,
    since: dt.date,
    today: dt.date | None = None,
    releases: list[dict] | None = None,
    pr_url: str | None = None,
    write_intro: bool = True,
) -> tuple[Path, str]:
    """Write the digest for papers added on or after `since`. Returns (path, markdown)."""
    today = today or dt.date.today()
    papers = sorted_papers(p for p in archive.papers.values() if p.added >= since and not p.classic)
    by_topic: dict[str, list[Paper]] = defaultdict(list)
    for p in papers:
        by_topic[p.topic].append(p)
    week = week_id(today)
    md = env(cfg).get_template("digest.md.j2").render(
        week=week,
        since=since,
        today=today,
        intro=intro(papers, cfg) if write_intro else "",
        papers=papers,
        highlights=sorted((p for p in papers if p.highlight), key=lambda p: -p.score),
        by_topic=[(t, by_topic[t.slug]) for t in cfg.topics if by_topic.get(t.slug)],
        orgs=cfg.org_by_slug,
        releases=releases or [],
        pr_url=pr_url,
        site=cfg.settings["site"]["url"],
    )
    path = archive.root / "digests" / f"{week}.md"
    path.parent.mkdir(exist_ok=True)
    path.write_text(md)
    return path, md
