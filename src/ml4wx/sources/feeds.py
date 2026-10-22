"""Lab blogs (RSS/Atom) and GitHub releases of watched repos."""

from __future__ import annotations

import datetime as dt
import logging
import os
import re

import feedparser

from ..config import Config
from ..http import Client

log = logging.getLogger(__name__)
_client = Client(min_interval=0.5, max_retries=2)


def _date(entry) -> dt.date | None:
    for k in ("published_parsed", "updated_parsed"):
        if t := entry.get(k):
            return dt.date(t.tm_year, t.tm_mon, t.tm_mday)
    return None


def blog_posts(cfg: Config, since: dt.date) -> list[dict]:
    """Posts since `since` that mention both an ML term and a weather/climate term."""
    from ..ingest import text_passes

    posts = []
    for org in cfg.watchlist.get("orgs") or []:
        for url in org.get("blog_feeds") or []:
            try:
                feed = feedparser.parse(_client.get(url).content)
            except Exception as e:  # noqa: BLE001 - one broken feed must not stop the others
                log.warning("feed %s: %s", url, e)
                continue
            for e in feed.entries:
                d = _date(e)
                if not d or d < since:
                    continue
                summary = re.sub(r"<[^>]+>", " ", e.get("summary", ""))
                title = " ".join(re.sub(r"<[^>]+>", " ", e.get("title", "")).split())  # ECMWF puts <a> in titles
                if text_passes(f"{title} {summary}", cfg):
                    posts.append({
                        "org": org["slug"], "org_name": org["name"], "title": title,
                        "url": e.get("link"), "date": d.isoformat(), "summary": " ".join(summary.split())[:400],
                    })  # fmt: skip
    return posts


def releases(cfg: Config, since: dt.date) -> list[dict]:
    """Non-draft, non-prerelease GitHub releases of watched repos since `since`."""
    headers = {}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    out = []
    for org in cfg.watchlist.get("orgs") or []:
        for repo in org.get("github_repos") or []:
            try:
                r = _client.get(
                    f"https://api.github.com/repos/{repo}/releases", params={"per_page": 5}, headers=headers
                )
            except Exception as e:  # noqa: BLE001
                log.warning("releases %s: %s", repo, e)
                continue
            if r.status_code != 200:
                log.warning("releases %s: HTTP %d", repo, r.status_code)
                continue
            for rel in r.json():
                d = dt.date.fromisoformat((rel.get("published_at") or "1970-01-01")[:10])
                if d >= since and not rel.get("draft") and not rel.get("prerelease"):
                    out.append({
                        "repo": repo, "tag": rel.get("tag_name"), "name": rel.get("name"),
                        "url": rel.get("html_url"), "date": d.isoformat(), "org": org["slug"],
                    })  # fmt: skip
    return sorted(out, key=lambda r: r["date"], reverse=True)
