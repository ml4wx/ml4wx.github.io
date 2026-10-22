"""Data models: a Candidate is a raw work from a source; a Paper is an archived entry."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field

from . import ids


class Ids(BaseModel):
    arxiv: str | None = None
    doi: str | None = None
    openalex: str | None = None

    def keys(self) -> list[str]:
        return ids.keys_for(self.arxiv, self.doi, self.openalex)


class Links(BaseModel):
    paper: str | None = None
    pdf: str | None = None
    code: str | None = None
    weights: str | None = None
    project: str | None = None
    data: str | None = None


class Candidate(BaseModel):
    """A work found by a source, before classification."""

    ids: Ids
    title: str
    authors: list[str] = []
    date: dt.date
    abstract: str | None = None
    abstract_license_ok: bool = False  # may we republish the abstract? (arXiv, CC-BY, CC0)
    venue: str | None = None
    venue_type: str = "preprint"  # journal | conference | preprint | other
    links: Links = Links()
    institutions: list[str] = []  # OpenAlex institution ids
    affiliations: list[str] = []  # raw affiliation strings
    comment: str | None = None  # arXiv comment field (often has code links / venue)
    cited_by: int | None = None
    retracted: bool = False
    source: str = ""  # arxiv:<query> | openalex:<query> | seed | snowball | feed

    @property
    def key(self) -> str:
        return self.ids.keys()[0]


class Paper(BaseModel):
    """An archived paper; stored as papers/<year>/<id>.yaml."""

    id: str
    title: str
    authors: list[str] = []  # empty for a few records; the YAML then has no authors key
    date: dt.date
    year: int
    venue: str | None = None
    venue_type: str = "preprint"
    ids: Ids
    links: Links = Links()
    orgs: list[str] = []
    topic: str
    topics: list[str] = []
    type: str
    method: list[str] = []
    timescale: list[str] = []
    spatial: list[str] = []
    data: list[str] = []
    status: str = "research"
    summary: str
    abstract: str | None = None
    significance: int = Field(default=0, ge=0, le=10)
    cited_by: int | None = None  # snapshot when added or last updated
    score: int = 0
    highlight: bool = False
    highlight_reason: str | None = None
    classic: bool = False
    added: dt.date
    source: str = ""
