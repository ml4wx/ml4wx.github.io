"""Daily check: important new papers and lab blog posts become `highlight` issues.

Nothing is written to the archive here; the weekly run adds the papers. An HTML comment with
the paper id in each issue body prevents duplicates.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
import subprocess

from . import classify, fetch, ingest
from .config import Config
from .export import authors_short
from .models import Paper
from .sources import arxiv, feeds
from .store import Archive

log = logging.getLogger(__name__)

LABEL = "highlight"
MARK = re.compile(r"<!-- ml4wx:id=(\S+) -->")


def _gh(*args: str, input_: str | None = None) -> str:
    return subprocess.run(
        ["gh", *args], input=input_, capture_output=True, text=True, check=True
    ).stdout


def alerted_ids(repo: str) -> set[str]:
    out = _gh("issue", "list", "-R", repo, "--label", LABEL, "--state", "all", "--limit", "500",
              "--json", "body")  # fmt: skip
    return {m for issue in json.loads(out) for m in MARK.findall(issue.get("body") or "")}


def paper_issue(p: Paper, cfg: Config) -> tuple[str, str]:
    orgs = ", ".join(cfg.org_by_slug[o].name for o in p.orgs if o in cfg.org_by_slug)
    links = [f"[Paper]({p.links.paper})"] if p.links.paper else []
    if p.links.pdf:
        links.append(f"[PDF]({p.links.pdf})")
    if p.links.code:
        links.append(f"[Code]({p.links.code})")
    topic = cfg.topic_by_slug[p.topic].name
    title = f"⭐ {p.title}" + (f" ({orgs})" if orgs else "")
    body = f"""**{p.title}**
{authors_short(p.authors, 5)}{" · " + orgs if orgs else ""} · {p.venue or "preprint"} · {p.date}

> {p.highlight_reason or p.summary}

{p.summary}

Topic: {topic} · Score: {p.score}/100 · {" · ".join(links)}

<sub>Daily alert from ML4WX. The paper is added to the archive in the next weekly update.</sub>
<!-- ml4wx:id={p.id} -->
"""
    return title[:250], body


def post_issue(post: dict) -> tuple[str, str]:
    title = f"📰 {post['org_name']}: {post['title']}"
    body = f"""**[{post['title']}]({post['url']})**
{post['org_name']} blog · {post['date']}

{post['summary']}

<!-- ml4wx:id={post['url']} -->
"""
    return title[:250], body


def find(cfg: Config, archive: Archive, days: int, today: dt.date | None = None) -> list[Paper]:
    """Highlight-worthy papers from the last `days` days that are not archived yet."""
    today = today or dt.date.today()
    cands = ingest.dedupe(fetch.recent(cfg, today - dt.timedelta(days=days), today))
    cands = [c for c in cands if not archive.status(c) and ingest.prefilter(c, cfg)]
    results = classify.classify(cands, cfg)
    papers = []
    for c in cands:
        r = results.get(c.key)
        if not r or r["decision"] != "in":
            continue
        extra = arxiv.html_front_matter(c.ids.arxiv) if c.ids.arxiv and not ingest.detect_orgs(c, cfg) else ""
        papers.append(ingest.to_paper(c, r, ingest.detect_orgs(c, cfg, extra), today, "daily"))
    maybe = [p for p in papers if ingest.maybe_highlight(p, cfg)]
    if maybe:
        ingest.judge(maybe, cfg)
    for p in maybe:
        ingest.finalize(p, cfg)
    return [p for p in maybe if p.highlight]


def run(cfg: Config, archive: Archive, days: int, dry_run: bool = False) -> int:
    repo = cfg.settings["site"]["repo"]
    since = dt.date.today() - dt.timedelta(days=days)
    issues = [(p.id, *paper_issue(p, cfg)) for p in find(cfg, archive, days)]
    issues += [(post["url"], *post_issue(post)) for post in feeds.blog_posts(cfg, since)]
    done = set() if dry_run else alerted_ids(repo)
    n = 0
    for key, title, body in issues:
        if key in done:
            continue
        if dry_run:
            print(f"--- {title}\n{body}")
        else:
            _gh("issue", "create", "-R", repo, "--title", title, "--label", LABEL, "--body-file", "-",
                input_=body)  # fmt: skip
        done.add(key)
        n += 1
    log.info("alerts: %d new issues", n)
    return n
