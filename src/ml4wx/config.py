"""Load the YAML files in config/ once, with the repo root as the anchor for all paths."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import yaml


def repo_root() -> Path:
    """Repo root: $ML4WX_ROOT, else the first parent of the cwd that has config/taxonomy.yaml."""
    if env := os.environ.get("ML4WX_ROOT"):
        return Path(env).resolve()
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "config" / "taxonomy.yaml").exists():
            return p
    raise FileNotFoundError("run inside the ml4wx repo (config/taxonomy.yaml not found)")


def _load(name: str) -> dict:
    with open(repo_root() / "config" / name, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class Topic:
    slug: str
    name: str
    group: str
    enabled: bool
    description: str
    examples: tuple[str, ...]


@dataclass(frozen=True)
class Org:
    slug: str
    name: str
    tier: int
    kind: str
    openalex_ids: tuple[str, ...]
    ror: tuple[str, ...]
    aliases: tuple[str, ...]

    @property
    def institution_keys(self) -> set[str]:
        """Keys as stored in Candidate.institutions: 'I123...' and 'ror:0abc...'."""
        return set(self.openalex_ids) | {f"ror:{r}" for r in self.ror}


@dataclass(frozen=True)
class Config:
    settings: dict
    taxonomy: dict
    sources: dict
    watchlist: dict

    @property
    def topics(self) -> list[Topic]:
        return [
            Topic(
                slug=t["slug"],
                name=t["name"],
                group=t["group"],
                enabled=bool(t.get("enabled", True)),
                description=" ".join(str(t.get("description", "")).split()),
                examples=tuple(t.get("examples") or ()),
            )
            for t in self.taxonomy["topics"]
        ]

    @property
    def topic_by_slug(self) -> dict[str, Topic]:
        return {t.slug: t for t in self.topics}

    @property
    def enabled_slugs(self) -> list[str]:
        return [t.slug for t in self.topics if t.enabled]

    @property
    def later_slugs(self) -> list[str]:
        return [t.slug for t in self.topics if not t.enabled]

    @property
    def facets(self) -> dict[str, dict[str, str]]:
        return self.taxonomy["facets"]

    @property
    def orgs(self) -> list[Org]:
        return [
            Org(
                slug=o["slug"],
                name=o["name"],
                tier=int(o.get("tier", 2)),
                kind=o.get("kind", ""),
                openalex_ids=tuple(_short_openalex(i) for i in o.get("openalex_ids") or ()),
                ror=tuple(_listify(o.get("ror"))),
                aliases=tuple(a.lower() for a in o.get("aliases") or ()),
            )
            for o in self.watchlist.get("orgs") or []
        ]

    @property
    def org_by_slug(self) -> dict[str, Org]:
        return {o.slug: o for o in self.orgs}


def _short_openalex(i: str) -> str:
    return i.rsplit("/", 1)[-1].upper()


def _listify(v) -> list[str]:
    """ROR ids as short lowercase ids; accepts a string, a list, or nothing."""
    vals = [v] if isinstance(v, str) else (v or [])
    return [str(x).rsplit("/", 1)[-1].lower() for x in vals if x]


@cache
def load() -> Config:
    root = repo_root()
    watch = root / "config" / "watchlist.yaml"
    return Config(
        settings=_load("settings.yaml"),
        taxonomy=_load("taxonomy.yaml"),
        sources=_load("sources.yaml"),
        watchlist=_load("watchlist.yaml") if watch.exists() else {},
    )
