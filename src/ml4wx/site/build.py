"""Static site: Jinja2 templates -> _site/, then Pagefind builds the search index."""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from email.utils import format_datetime
from pathlib import Path
from xml.sax.saxutils import escape

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt

from .. import resources
from ..config import Config
from ..export import authors_short, sorted_papers, stats
from ..models import Paper
from ..store import Archive
from . import catalog as catalog_page

log = logging.getLogger(__name__)

HERE = Path(__file__).parent


def _env(cfg: Config) -> Environment:
    e = Environment(
        loader=FileSystemLoader(HERE / "templates"),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    e.filters["authors_short"] = authors_short
    e.filters["compact"] = lambda n: f"{n / 1000:.1f}k".replace(".0k", "k") if n >= 1000 else str(n)
    e.globals.update(
        site=cfg.settings["site"],
        topics=cfg.topic_by_slug,
        orgs=cfg.org_by_slug,
        facets=cfg.facets,
        now=dt.date.today(),
    )
    return e


def _write(out: Path, rel: str, text: str) -> None:
    path = out / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _digests(root: Path) -> list[dict]:
    md = MarkdownIt("commonmark", {"linkify": False}).enable("table")
    out = []
    for f in sorted((root / "digests").glob("*.md"), reverse=True):
        text = f.read_text(encoding="utf-8")
        meta: dict = {}
        if m := re.match(r"^---\n(.*?)\n---\n", text, re.S):
            meta = dict(line.split(": ", 1) for line in m.group(1).splitlines() if ": " in line)
            text = text[m.end():]
        out.append({"week": f.stem, "meta": meta, "html": md.render(text)})
    return out


def _rss(cfg: Config, papers: list[Paper], title: str, path: str) -> str:
    base = cfg.settings["site"]["url"]
    items = []
    for p in papers[:50]:
        when = dt.datetime.combine(p.added, dt.time(8), tzinfo=dt.UTC)
        items.append(
            f"<item><title>{escape(p.title)}</title><link>{base}/papers/{p.id}/</link>"
            f"<guid isPermaLink=\"false\">{p.id}</guid><pubDate>{format_datetime(when)}</pubDate>"
            f"<description>{escape(p.summary)}</description></item>"
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0"><channel>'
        f"<title>{escape(title)}</title><link>{base}/</link>"
        f"<description>{escape(cfg.settings['site']['tagline'])}</description>"
        + "".join(items)
        + "</channel></rss>\n"
    )


def build(archive: Archive, cfg: Config, out: Path, index: bool = True) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    shutil.copytree(HERE / "static", out / "static")
    e = _env(cfg)
    papers = sorted_papers(archive.papers.values())
    st = stats(archive, cfg)
    highlights = [p for p in papers if p.highlight]
    by_topic: dict[str, list[Paper]] = defaultdict(list)
    by_org: dict[str, list[Paper]] = defaultdict(list)
    for p in papers:
        by_topic[p.topic].append(p)
        for o in p.orgs:
            by_org[o].append(p)
    groups = cfg.taxonomy["groups"]
    digests = _digests(archive.root)
    latest_added = max((p.added for p in papers), default=None)

    # Resources catalog: kind/slug -> archived paper id, and the reverse for paper pages.
    catalog = resources.Catalog(archive.root)
    key_to_paper = {k: p.id for p in papers for k in p.ids.keys()}
    res_by_paper = catalog.for_papers(key_to_paper)
    paper_ids = {f"{r.kind}/{r.slug}": pid for pid, rs in res_by_paper.items() for r in rs}
    res_vocab = resources.vocabulary()
    e.globals.update(res_vocab=res_vocab, github=catalog.github, paper_ids=paper_ids, res_pages=resources.PAGES)
    # Topic pages: the most important resources of each kind for that topic.
    res_by_topic: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for kind in resources.DIRS:
        for r in catalog.ordered(kind):
            if kind == "dataset" and (r.get("access") or "open") not in catalog_page.PUBLIC_ACCESS:
                continue
            for t in r.topics:
                res_by_topic[t][kind].append(r)

    def render(rel: str, tpl: str, **kw) -> None:
        _write(out, rel, e.get_template(tpl).render(**kw))

    render(
        "index.html", "index.html",
        st=st, highlights=highlights[:6], latest=[p for p in papers if not p.classic][:12],
        groups=groups, by_topic=by_topic, latest_added=latest_added,
        digest=digests[0] if digests else None,
    )  # fmt: skip
    render("explore/index.html", "explore.html")
    render("topics/index.html", "topics.html", groups=groups, by_topic=by_topic, st=st)
    for t in cfg.topics:
        if t.enabled:
            ps = by_topic.get(t.slug, [])
            also = [p for p in papers if t.slug in p.topics]
            years = sorted(Counter(p.year for p in ps).items())
            render(f"topics/{t.slug}/index.html", "topic.html", topic=t, papers=ps, also=also, years=years,
                   resources=res_by_topic.get(t.slug, []))  # fmt: skip
    for p in papers:
        render(f"papers/{p.id}/index.html", "paper.html", p=p, resources=res_by_paper.get(p.id, []))
    for kind, path in resources.PAGES.items():
        render(f"{path}/index.html", "catalog.html", pg=catalog_page.page(kind, catalog, res_vocab, cfg))
    render("resources/index.html", "redirect.html", target="/datasets/")  # old URL of the combined page
    org_list = sorted(
        (o for o in cfg.orgs if by_org.get(o.slug)), key=lambda o: (o.tier, -len(by_org[o.slug]))
    )
    render("orgs/index.html", "orgs.html", org_list=org_list, by_org=by_org)
    for o in org_list:
        render(f"orgs/{o.slug}/index.html", "org.html", org=o, papers=by_org[o.slug])
    render("highlights/index.html", "highlights.html", papers=highlights)
    render("digests/index.html", "digests.html", digests=digests)
    for d in digests:
        render(f"digests/{d['week']}/index.html", "digest.html", d=d)
    render("about/index.html", "about.html", cfg=cfg, st=st,
           scope=" ".join(cfg.taxonomy["scope_rule"].split()))  # fmt: skip
    render("404.html", "404.html")
    _write(out, "feed.xml", _rss(cfg, [p for p in papers if not p.classic], "ML4WX: new papers", "feed.xml"))
    _write(out, "highlights.xml", _rss(cfg, highlights, "ML4WX: highlights", "highlights.xml"))
    _write(out, ".nojekyll", "")
    data = [
        p.model_dump(mode="json", exclude={"abstract"}, exclude_none=True) for p in papers
    ]
    _write(out, "data/papers.json", json.dumps(data, ensure_ascii=False, separators=(",", ":")))
    res_data = [
        r.model_dump(mode="json", exclude_none=True) | {"github": catalog.github.get(r.repo)}
        for r in catalog.items.values()
    ]
    _write(out, "data/resources.json", json.dumps(res_data, ensure_ascii=False, separators=(",", ":")))
    log.info("site: %d papers, %d topics, %d orgs, %d digests", len(papers), len(by_topic), len(org_list), len(digests))

    if index:
        subprocess.run([sys.executable, "-m", "pagefind", "--site", str(out)], check=True)
