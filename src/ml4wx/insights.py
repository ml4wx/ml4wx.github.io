"""Numbers behind the topic starter kits, Trends, Essentials and the home page, plus related papers."""

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .models import Paper

METHODS = ["cnn", "rnn", "transformer", "gnn", "diffusion", "neural-operator", "llm", "classical-ml"]


def _cited(p: Paper) -> int:
    return p.cited_by or 0


@dataclass
class TopicStats:
    slug: str
    total: int = 0
    last12: int = 0
    prev12: int = 0
    per_year: dict[int, int] = field(default_factory=dict)
    start_here: list[Paper] = field(default_factory=list)  # reviews, most cited first
    key_papers: list[Paper] = field(default_factory=list)  # highlights and landmark papers
    most_cited_recent: list[Paper] = field(default_factory=list)
    newest: list[Paper] = field(default_factory=list)

    @property
    def growth(self) -> float | None:
        """Papers in the last 12 months vs the 12 months before, in percent."""
        return 100 * (self.last12 - self.prev12) / self.prev12 if self.prev12 >= 10 else None


def topic_stats(papers: list[Paper], today: dt.date, first_year: int = 2018) -> dict[str, TopicStats]:
    year_ago, two_years_ago = today - dt.timedelta(days=365), today - dt.timedelta(days=730)
    by_topic: dict[str, list[Paper]] = defaultdict(list)
    for p in papers:
        by_topic[p.topic].append(p)
    out = {}
    for slug, ps in by_topic.items():
        st = TopicStats(slug=slug, total=len(ps))
        st.last12 = sum(p.date > year_ago for p in ps)
        st.prev12 = sum(two_years_ago < p.date <= year_ago for p in ps)
        years = Counter(p.year for p in ps if p.year >= first_year)
        st.per_year = {y: years.get(y, 0) for y in range(first_year, today.year + 1)}
        st.start_here = sorted((p for p in ps if p.type == "review"), key=lambda p: (-_cited(p), -p.score))[:6]
        st.key_papers = sorted(
            (p for p in ps if p.highlight or p.classic), key=lambda p: (-p.score, -_cited(p))
        )[:9]
        st.most_cited_recent = sorted((p for p in ps if p.date > year_ago and _cited(p)), key=lambda p: -_cited(p))[:6]
        st.newest = sorted(ps, key=lambda p: (p.date, p.id), reverse=True)[:40]
        out[slug] = st
    return out


@dataclass
class Trends:
    years: list[int]
    per_year: dict[int, int]
    method_share: dict[str, dict[int, float]]  # method -> year -> % of that year's papers
    top_labs: list[tuple[str, int]]  # (org slug, papers in the last 12 months)
    top_venues: list[tuple[str, int]]
    last12: int
    prev12: int


def trends(papers: list[Paper], today: dt.date, first_year: int = 2018) -> Trends:
    years = list(range(first_year, today.year + 1))
    by_year: dict[int, list[Paper]] = defaultdict(list)
    for p in papers:
        if p.year >= first_year:
            by_year[p.year].append(p)
    method_share = {
        m: {y: 100 * sum(m in p.method for p in by_year[y]) / max(1, len(by_year[y])) for y in years} for m in METHODS
    }
    year_ago, two_years_ago = today - dt.timedelta(days=365), today - dt.timedelta(days=730)
    recent = [p for p in papers if p.date > year_ago]
    labs = Counter(o for p in recent for o in p.orgs)
    venues = Counter(p.venue for p in recent if p.venue and p.venue_type in ("journal", "conference"))
    return Trends(
        years=years,
        per_year={y: len(by_year[y]) for y in years},
        method_share=method_share,
        top_labs=labs.most_common(15),
        top_venues=venues.most_common(12),
        last12=len(recent),
        prev12=sum(two_years_ago < p.date <= year_ago for p in papers),
    )


def citations_per_year(p: Paper, today: dt.date) -> float:
    """Citations per year since publication; the age counts as at least one year."""
    return _cited(p) / max(1.0, (today - p.date).days / 365.25)


def essential_papers(papers: list[Paper], today: dt.date, rules: dict) -> tuple[list[Paper], list[Paper]]:
    """(recent, earlier) candidates, most important first. The first rules["size"] recent ones are the
    essentials for all topics; earlier ones (before rules["since"]) are the foundations.

    Candidates: highlights, classics and papers with significance >= landmark, all with significance
    >= min_significance; sorted by citations per year. `pin` adds papers, `exclude` removes them."""
    pin, exclude = set(rules.get("pin") or []), set(rules.get("exclude") or [])
    floor, landmark = rules.get("min_significance", 0), rules.get("landmark", 11)
    ranked = sorted(
        (p for p in papers if p.id not in exclude and (p.id in pin or (
            p.significance >= floor and (p.highlight or p.classic or p.significance >= landmark)))),
        key=lambda p: (-citations_per_year(p, today), p.id),
    )  # fmt: skip
    recent = [p for p in ranked if p.year >= rules.get("since", 0) or p.id in pin]
    earlier = [p for p in ranked if p.year < rules.get("since", 0) and p.id not in pin]
    # Pinned papers are always in the top `size`; they keep their place by citations per year.
    free = rules.get("size", 50) - len(pin & {p.id for p in recent})
    top = pin | set([p.id for p in recent if p.id not in pin][: max(0, free)])
    return [p for p in recent if p.id in top] + [p for p in recent if p.id not in top], earlier


def related(papers: list[Paper], k: int = 6) -> dict[str, list[str]]:
    """paper id -> ids of the k most similar papers (TF-IDF on title, summary and abstract)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if len(papers) < 3:
        return {}
    docs = [f"{p.title} {p.title} {p.summary} {p.abstract or ''}" for p in papers]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True)
    x = vec.fit_transform(docs)
    out: dict[str, list[str]] = {}
    for start in range(0, len(papers), 1000):
        sims = (x[start : start + 1000] @ x.T).toarray()
        for i, row in enumerate(sims, start=start):
            row[i] = -1  # not itself
            best = row.argsort()[::-1][:k]
            out[papers[i].id] = [papers[j].id for j in best if row[j] > 0.08]
    return out
