"""Resources catalog: datasets, models with public code or weights, and software.

Curated by hand: one YAML file per resource in resources/<kind dir>/<slug>.yaml.
Generated: resources/_github.json (stars, activity, latest release), refreshed weekly.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from . import ids
from .config import Config, repo_root
from .http import Client

log = logging.getLogger(__name__)

DIRS = {"dataset": "datasets", "model": "models", "software": "software"}
GITHUB_FILE = "_github.json"


class PaperRef(BaseModel):
    arxiv: str | None = None
    doi: str | None = None

    def keys(self) -> list[str]:
        return ids.keys_for(ids.norm_arxiv(self.arxiv), ids.norm_doi(self.doi), None)


class Resource(BaseModel):
    """Common fields. Kind-specific fields (coverage, weights, install, ...) are kept as extras."""

    model_config = ConfigDict(extra="allow")

    slug: str
    name: str
    kind: Literal["dataset", "model", "software"]
    category: str
    description: str
    provider: str = ""
    url: str = ""
    docs: str = ""
    repo: str = ""
    paper: PaperRef | None = None
    license: str = ""
    topics: list[str] = []
    added: dt.date | None = None

    def get(self, field: str, default=None):
        return getattr(self, field, None) or (self.model_extra or {}).get(field) or default


def _dump(r: Resource) -> str:
    d = r.model_dump(mode="json", exclude_none=True)
    d = {k: v for k, v in d.items() if v not in ("", [], {}, None)}
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=100)


class Catalog:
    def __init__(self, root: Path | None = None):
        self.root = (root or repo_root()) / "resources"
        self.items: dict[str, Resource] = {}
        self.invalid: list[tuple[Path, str]] = []
        for f in sorted(self.root.glob("*/*.yaml")):
            try:
                r = Resource.model_validate(yaml.safe_load(f.read_text(encoding="utf-8")))
            except (yaml.YAMLError, ValueError) as e:
                self.invalid.append((f, str(e).splitlines()[0]))
                continue
            self.items[f"{r.kind}/{r.slug}"] = r
        gh = self.root / GITHUB_FILE
        self.github: dict[str, dict] = json.loads(gh.read_text()) if gh.exists() else {}

    # --- lookups ------------------------------------------------------------------
    def by_kind(self) -> dict[str, list[Resource]]:
        out: dict[str, list[Resource]] = defaultdict(list)
        for r in sorted(self.items.values(), key=lambda r: r.name.lower()):
            out[r.kind].append(r)
        return out

    def by_topic(self) -> dict[str, list[Resource]]:
        out: dict[str, list[Resource]] = defaultdict(list)
        for r in sorted(self.items.values(), key=lambda r: (list(DIRS).index(r.kind), r.name.lower())):
            for t in r.topics:
                out[t].append(r)
        return out

    def for_papers(self, key_to_paper: dict[str, str]) -> dict[str, list[Resource]]:
        """paper id -> resources that cite it as their reference paper."""
        out: dict[str, list[Resource]] = defaultdict(list)
        for r in self.items.values():
            if r.paper:
                pid = next((key_to_paper[k] for k in r.paper.keys() if k in key_to_paper), None)
                if pid:
                    out[pid].append(r)
        return out

    def path(self, r: Resource) -> Path:
        return self.root / DIRS[r.kind] / f"{r.slug}.yaml"

    def save(self, r: Resource) -> None:
        p = self.path(r)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(_dump(r), encoding="utf-8")
        self.items[f"{r.kind}/{r.slug}"] = r

    # --- checks -------------------------------------------------------------------
    def validate(self, cfg: Config, vocab: dict) -> list[str]:
        errors = [f"{p.relative_to(self.root.parent)}: cannot load ({m})" for p, m in self.invalid]
        topics = set(cfg.enabled_slugs)
        for r in self.items.values():
            where = f"resources/{DIRS[r.kind]}/{r.slug}.yaml"
            if not self.path(r).exists():
                errors.append(f"{where}: file name or folder does not match slug/kind")
            if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", r.slug):
                errors.append(f"{where}: slug must be kebab-case")
            if r.category not in vocab["kinds"][r.kind]["categories"]:
                errors.append(f"{where}: unknown category '{r.category}'")
            errors += [f"{where}: unknown topic '{t}'" for t in r.topics if t not in topics]
            if not r.topics:
                errors.append(f"{where}: needs at least one topic")
            for field in ("url", "docs", "access_url", "weights"):
                v = r.get(field)
                if v and not str(v).startswith(("http://", "https://")):
                    errors.append(f"{where}: {field} is not a URL")
            if r.repo and not re.fullmatch(r"[\w.-]+/[\w.-]+", r.repo):
                errors.append(f"{where}: repo must be owner/name")
            if r.kind == "dataset" and r.get("access") and r.get("access") not in vocab["access"]:
                errors.append(f"{where}: unknown access '{r.get('access')}'")
            if not (r.url or r.repo):
                errors.append(f"{where}: needs url or repo")
        return errors

    # --- GitHub metadata ----------------------------------------------------------
    def refresh_github(self) -> int:
        """Stars, last push, archived flag and latest release for every repo. Returns repos updated."""
        get = _github_getter()
        repos = sorted({r.repo for r in self.items.values() if r.repo})
        out: dict[str, dict] = {}
        for repo in repos:
            d = get(f"repos/{repo}")
            if d is None:
                log.warning("github %s: not found or not reachable", repo)
                continue
            info = {
                "full_name": d.get("full_name"),
                "stars": d.get("stargazers_count", 0),
                "pushed_at": (d.get("pushed_at") or "")[:10],
                "archived": bool(d.get("archived")),
            }
            if (j := get(f"repos/{repo}/releases/latest")) is not None:
                info["release"] = {
                    "tag": j.get("tag_name"),
                    "name": j.get("name") or "",
                    "date": (j.get("published_at") or "")[:10],
                    "url": j.get("html_url"),
                }
            out[repo] = info
        (self.root / GITHUB_FILE).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        self.github = out
        return len(out)

    def releases_since(self, since: dt.date) -> list[dict]:
        """Releases published on or after `since`, newest first (for the weekly digest)."""
        out = []
        names = {r.repo: r.name for r in self.items.values() if r.repo}
        for repo, info in self.github.items():
            rel = info.get("release") or {}
            if rel.get("date") and dt.date.fromisoformat(rel["date"]) >= since:
                out.append({"repo": repo, "name": names.get(repo, repo), "tag": rel["tag"],
                            "url": rel["url"], "date": rel["date"]})  # fmt: skip
        return sorted(out, key=lambda r: r["date"], reverse=True)


def _github_getter():
    """GET a GitHub API path -> JSON dict, or None on 404/errors.

    In CI the workflow token ($GITHUB_TOKEN) is used. Locally, without a token, `gh api` is used,
    so the user's credentials stay inside the gh CLI.
    """
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token and shutil.which("gh"):

        def get_gh(path: str) -> dict | None:
            p = subprocess.run(["gh", "api", path], capture_output=True, text=True, check=False)
            return json.loads(p.stdout) if p.returncode == 0 else None

        return get_gh
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    client = Client(min_interval=0.1, max_retries=3)

    def get_http(path: str) -> dict | None:
        try:
            r = client.get(f"https://api.github.com/{path}", headers=headers)
        except RuntimeError:
            return None
        return r.json() if r.status_code == 200 else None

    return get_http


def vocabulary() -> dict:
    with open(repo_root() / "config" / "resources.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def import_list(path: Path, catalog: Catalog, today: dt.date | None = None) -> tuple[int, int]:
    """Add resources from a YAML list (e.g. a research batch). Existing slugs are kept as they are."""
    added = skipped = 0
    for raw in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
        raw.setdefault("added", (today or dt.date.today()).isoformat())
        r = Resource.model_validate(raw)
        if f"{r.kind}/{r.slug}" in catalog.items:
            skipped += 1
            continue
        catalog.save(r)
        added += 1
    return added, skipped
