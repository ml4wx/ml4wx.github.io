"""Static site: Jinja2 templates -> _site/, then Pagefind builds the search index."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from email.utils import format_datetime
from pathlib import Path
from xml.sax.saxutils import escape

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markdown_it import MarkdownIt

from .. import insights, resources
from ..config import Config
from ..export import authors_short, bibtex, sorted_papers, stats
from ..models import Paper
from ..store import Archive
from . import catalog as catalog_page
from . import charts

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


# Data facet values -> where the matching datasets are on the Datasets tab.
DATA_LINKS = {
    "era5": "/datasets/#era5",
    "cmip": "/datasets/#cmip6",
    "radar": "/datasets/?type=observations-radar",
    "satellite": "/datasets/?type=observations-satellite",
    "stations": "/datasets/?type=observations-in-situ",
    "nwp-analysis": "/datasets/?type=forecast-archive",
    "km-scale-sim": "/datasets/?type=km-scale-simulation",
}


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
    # Cache busting: static files are linked as /static/<file>?v=<content hash>, so browsers fetch the
    # new version right after a deploy instead of using a cached old one (Pages caches for 10 min).
    versions = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:10] for f in (HERE / "static").iterdir() if f.is_file()
    }
    e.globals["asset"] = lambda name: f"/static/{name}?v={versions.get(name, '0')}"
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

    # Numbers for the home page, topic starter kits and Trends.
    today = latest_added or dt.date.today()
    first_year = cfg.settings["start_year"]
    tstats = insights.topic_stats(papers, today, first_year)
    tr = insights.trends(papers, today, first_year)
    enabled = [t for t in cfg.topics if t.enabled]
    growing = sorted(
        (tstats[t.slug] for t in enabled if t.slug in tstats and tstats[t.slug].growth is not None),
        key=lambda s: -s.growth,
    )
    year_ago = today - dt.timedelta(days=365)
    n_res = {k: len(catalog_page.page(k, catalog, res_vocab, cfg)["rows"]) for k in resources.DIRS}

    render(
        "index.html", "index.html",
        st=st, tr=tr, highlights=highlights[:6], n_res=n_res, latest_added=latest_added, groups=groups,
        by_topic=by_topic, digest=digests[0] if digests else None, growing=growing[:6],
        week=sorted((p for p in papers if p.date > today - dt.timedelta(days=14) and not p.classic),
                    key=lambda p: (-p.score, p.id)),
        cited=sorted((p for p in papers if p.date > year_ago and p.cited_by), key=lambda p: -p.cited_by)[:8],
    )  # fmt: skip
    render("explore/index.html", "explore.html")
    render("topics/index.html", "topics.html", groups=groups, by_topic=by_topic, st=st)
    for t in enabled:
        ts = tstats.get(t.slug) or insights.TopicStats(slug=t.slug)
        chart = charts.columns([(f"'{y % 100:02d}", n) for y, n in ts.per_year.items()], partial_last=True,
                               label=f"{t.name}: papers per year")  # fmt: skip
        render(f"topics/{t.slug}/index.html", "topic.html", topic=t, st=ts, chart=chart,
               also=[p for p in papers if t.slug in p.topics], resources=res_by_topic.get(t.slug, []))  # fmt: skip
        _write(out, f"topics/{t.slug}/feed.xml",
               _rss(cfg, [p for p in by_topic.get(t.slug, []) if not p.classic], f"ML4WX: {t.name}", ""))  # fmt: skip

    # Paper pages: related papers, BibTeX, and links from the data used to the Datasets tab.
    try:
        related = insights.related(papers)
    except ImportError:  # scikit-learn is in the `site` extra
        log.warning("scikit-learn missing: no related papers")
        related = {}
    by_id = {p.id: p for p in papers}
    for p in papers:
        render(f"papers/{p.id}/index.html", "paper.html", p=p, resources=res_by_paper.get(p.id, []),
               related=[by_id[i] for i in related.get(p.id, []) if i in by_id], bibtex=bibtex([p]).strip(),
               data_links=DATA_LINKS)  # fmt: skip

    for kind, path in resources.PAGES.items():
        render(f"{path}/index.html", "catalog.html", pg=catalog_page.page(kind, catalog, res_vocab, cfg))
    model_rows = catalog_page.page("model", catalog, res_vocab, cfg)["rows"]
    render("models/compare/index.html", "compare.html", rows=model_rows)
    _write(out, "data/resources-index.json", json.dumps([
        {"name": r.name, "kind": "tool" if r.kind == "software" else r.kind,
         "url": f"/{resources.PAGES[r.kind]}/#{r.slug}", "key": f"{r.name} {r.slug}".lower().replace("-", " ")}
        for k in resources.DIRS for r in catalog.ordered(k)
    ], separators=(",", ":")))  # fmt: skip

    # Trends page charts.
    labels = {"cnn": "CNN / U-Net", "rnn": "RNN / LSTM", "transformer": "Transformer", "gnn": "Graph neural network",
              "diffusion": "Diffusion / flow matching", "neural-operator": "Neural operator", "llm": "Language model",
              "classical-ml": "Classical ML"}  # fmt: skip
    ymax = charts.nice_ticks(max(max(v.values()) for v in tr.method_share.values()), 3)[-1]
    trend_charts = {
        "per_year": charts.columns([(str(y), n) for y, n in tr.per_year.items()], partial_last=True,
                                   label="Papers per year") + charts.table(["Year", "Papers"],
                                   [[y, n] for y, n in tr.per_year.items()], "Papers per year"),
        "methods": [{"label": labels[m], "svg": charts.mini_line(tr.method_share[m], ymax=ymax, unit="%",
                                                                 label=f"{labels[m]}: share of papers per year",
                                                                 partial_last=True)}
                    for m in sorted(insights.METHODS, key=lambda m: -tr.method_share[m][tr.years[-1]])],
        "methods_table": charts.table(["Method"] + [str(y) for y in tr.years],
                                      [[labels[m]] + [f"{tr.method_share[m][y]:.1f}%" for y in tr.years]
                                       for m in insights.METHODS], "Share of papers per method and year"),
        "labs": charts.hbars([(cfg.org_by_slug[o].name, n, f"/orgs/{o}/") for o, n in tr.top_labs
                              if o in cfg.org_by_slug], label="Most active labs"),
        "venues": charts.hbars([(v, n, None) for v, n in tr.top_venues], label="Top venues"),
    }  # fmt: skip
    growth_tiles = []
    for t in sorted(enabled, key=lambda t: -(tstats[t.slug].last12 if t.slug in tstats else 0)):
        ts = tstats.get(t.slug)
        if ts:
            growth_tiles.append({"slug": t.slug, "last12": ts.last12, "growth": ts.growth,
                                 "spark": charts.sparkline(list(ts.per_year.values()), label=f"{t.name} per year")})
    render("trends/index.html", "trends.html", tr=tr, charts=trend_charts, growth=growth_tiles, total=len(papers),
           fastest=growing[0] if growing else None)
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
