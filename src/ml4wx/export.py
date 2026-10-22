"""Files generated from the archive and committed to the repo: README, lists/, exports/."""

from __future__ import annotations

import csv
import datetime as dt
import io
import re
from collections import Counter
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from . import ids
from .config import Config
from .models import Paper
from .store import Archive

TEMPLATES = Path(__file__).parent / "site" / "templates"


def env(cfg: Config) -> Environment:
    """Jinja environment for the Markdown templates (README, lists, digests)."""
    e = Environment(loader=FileSystemLoader(TEMPLATES), keep_trailing_newline=True)
    orgs = cfg.org_by_slug
    e.filters["authors_short"] = authors_short
    e.filters["md_escape"] = lambda s: (s or "").replace("|", "\\|").replace("\n", " ")
    e.filters["org_names"] = lambda slugs: ", ".join(orgs[s].name for s in slugs if s in orgs)
    return e


def authors_short(authors: list[str], n: int = 3) -> str:
    if not authors:
        return ""
    if len(authors) <= n:
        return ", ".join(authors)
    return ", ".join(authors[:n]) + " et al."


def sorted_papers(papers) -> list[Paper]:
    return sorted(papers, key=lambda p: (p.date, p.id), reverse=True)


def bibtex_key(p: Paper, used: set[str]) -> str:
    first = ids.surname(p.authors[0]) if p.authors else "anon"
    word = next((w for w in ids.norm_title(p.title).split() if len(w) > 3), "paper")
    base = re.sub(r"[^a-z0-9]", "", f"{first}{p.year}{word}") or p.id
    key, n = base, 1
    while key in used:
        n += 1
        key = f"{base}{chr(96 + n)}"
    used.add(key)
    return key


def _bib_escape(s: str) -> str:
    return s.replace("{", "\\{").replace("}", "\\}").replace("&", "\\&").replace("%", "\\%")


def bibtex(papers: list[Paper]) -> str:
    used: set[str] = set()
    out = []
    for p in sorted(papers, key=lambda p: p.id):
        fields = {
            "title": "{" + _bib_escape(p.title) + "}",
            "author": " and ".join(p.authors) or "Unknown",
            "year": str(p.year),
        }
        if p.venue_type == "journal" and p.venue:
            kind = "article"
            fields["journal"] = _bib_escape(p.venue)
        elif p.venue_type == "conference" and p.venue:
            kind = "inproceedings"
            fields["booktitle"] = _bib_escape(p.venue)
        else:
            kind = "misc"
        if p.ids.doi:
            fields["doi"] = p.ids.doi
        if p.ids.arxiv:
            fields["eprint"] = p.ids.arxiv
            fields["archiveprefix"] = "arXiv"
        if url := p.links.paper:
            fields["url"] = url
        body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items())
        out.append(f"@{kind}{{{bibtex_key(p, used)},\n{body}\n}}\n")
    return "\n".join(out)


def csv_text(papers: list[Paper]) -> str:
    cols = [
        "id", "title", "authors", "date", "year", "venue", "topic", "topics", "type", "orgs",
        "method", "doi", "arxiv", "url", "code", "highlight", "score", "summary",
    ]  # fmt: skip
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(cols)
    for p in sorted_papers(papers):
        w.writerow([
            p.id, p.title, "; ".join(p.authors), p.date.isoformat(), p.year, p.venue or "",
            p.topic, "; ".join(p.topics), p.type, "; ".join(p.orgs), "; ".join(p.method),
            p.ids.doi or "", p.ids.arxiv or "", p.links.paper or "", p.links.code or "",
            int(p.highlight), p.score, p.summary,
        ])  # fmt: skip
    return buf.getvalue()


def stats(archive: Archive, cfg: Config) -> dict:
    papers = list(archive.papers.values())
    by_topic = Counter(p.topic for p in papers)
    hl_topic = Counter(p.topic for p in papers if p.highlight)
    return {
        "n_papers": len(papers),
        "n_highlights": sum(p.highlight for p in papers),
        "n_parked": len(archive.parked),
        "by_topic": by_topic,
        "hl_topic": hl_topic,
        "years": sorted(Counter(p.year for p in papers).items()),
        "updated": max((p.added for p in papers), default=dt.date.today()),
    }


def write_all(archive: Archive, cfg: Config) -> list[Path]:
    root = archive.root
    e = env(cfg)
    papers = sorted_papers(archive.papers.values())
    st = stats(archive, cfg)
    groups = cfg.taxonomy["groups"]
    written = []

    (root / "lists").mkdir(exist_ok=True)
    for t in cfg.topics:
        if not t.enabled:
            continue
        sel = [p for p in papers if p.topic == t.slug]
        also = [p for p in papers if t.slug in p.topics]
        path = root / "lists" / f"{t.slug}.md"
        path.write_text(
            e.get_template("list.md.j2").render(topic=t, papers=sel, also=also, cfg=cfg, orgs=cfg.org_by_slug)
        )
        written.append(path)

    (root / "exports").mkdir(exist_ok=True)
    (root / "exports" / "papers.bib").write_text(bibtex(papers))
    (root / "exports" / "papers.csv").write_text(csv_text(papers))
    written += [root / "exports" / "papers.bib", root / "exports" / "papers.csv"]

    readme = e.get_template("README.md.j2").render(
        cfg=cfg,
        st=st,
        groups=groups,
        topics=cfg.topics,
        highlights=[p for p in papers if p.highlight][:12],
        orgs=cfg.org_by_slug,
    )
    (root / "README.md").write_text(readme)
    written.append(root / "README.md")
    return written
