"""Datasets, Models and Tools pages: filter groups and rows for templates/catalog.html."""

from __future__ import annotations

import re
from collections import Counter

from ..config import Config
from ..resources import PAGES, Catalog, Resource

_GLOBAL = re.compile(r"\b(global|worldwide|world)\b", re.I)
_CLOUD = re.compile(
    r"zarr|amazonaws|registry\.opendata\.aws|googleapis|storage\.cloud\.google|gs://|s3://|"
    r"planetarycomputer|source\.coop|huggingface",
    re.I,
)

# Filter groups per kind: (key, label). Values and their labels come from facet_values().
FACETS = {
    "dataset": [("type", "Type"), ("use", "Used for"), ("access", "Access"), ("scope", "Coverage"),
                ("cloud", "Cloud-ready")],
    "model": [("type", "Type"), ("use", "Used for"), ("runs", "Runs in"), ("open", "Open"), ("lab", "Lab"),
              ("status", "Status")],
    "software": [("task", "What for"), ("language", "Language")],
}  # fmt: skip

PAGE_TEXT = {
    "dataset": ("Datasets", "Public datasets for ML in weather and climate: reanalyses, climate model "
                "output, forecast archives, observations and ML-ready benchmarks."),
    "model": ("Models", "ML weather and climate models with public code or weights, ready to run or "
              "fine-tune."),
    "software": ("Tools", "Open-source tools to load data, train and run models, evaluate forecasts and "
                 "make plots."),
}  # fmt: skip

PUBLIC_ACCESS = {"open", "registration"}


def first_sentence(text: str, limit: int = 170) -> str:
    s = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
    return s if len(s) <= limit else s[: limit - 1].rsplit(" ", 1)[0] + "…"


def facet_values(r: Resource, vocab: dict, cfg: Config) -> dict[str, list[tuple[str, str]]]:
    """facet key -> [(value, label)] for one resource."""
    cats = vocab["kinds"][r.kind]["categories"]
    topics = cfg.topic_by_slug
    use = [(t, topics[t].name) for t in r.topics if t in topics]
    if r.kind == "dataset":
        text = f"{r.get('coverage', '')} {r.name} {r.description}"
        cloud_src = f"{' '.join(r.get('formats') or [])} {r.get('access_url', '')}"
        return {
            "type": [(r.category, cats.get(r.category, r.category))],
            "use": use,
            "access": [(a, vocab["access"][a]) for a in [r.get("access") or "open"]],
            "scope": [("global", "Global") if _GLOBAL.search(text) else ("regional", "Regional")],
            "cloud": [("yes", "Cloud-ready (Zarr or cloud bucket)")] if _CLOUD.search(cloud_src) else [],
        }
    if r.kind == "model":
        runners = vocab.get("runners", {})
        org = cfg.org_by_slug.get(r.get("org") or "")
        has_weights = "weights" in (r.get("open") or []) or bool(r.get("weights"))
        return {
            "type": [(r.category, cats.get(r.category, r.category))],
            "use": use,
            "runs": [(x, runners.get(x, x)) for x in r.get("runs_in") or []],
            "open": [("weights", "Code + weights")] if has_weights else [("code", "Code only")],
            "lab": [(org.slug, org.name)] if org else [("other", "Other")],
            "status": [("operational", "Operational")] if r.get("operational") else [],
        }
    tasks = vocab.get("tasks", {})
    lang = (r.get("language") or "Other").split("/")[0].strip()
    return {
        "task": [(t, tasks.get(t, t)) for t in r.get("tasks") or [r.category]],
        "language": [(lang.lower().replace(" ", "-"), lang)],
    }


def page(kind: str, catalog: Catalog, vocab: dict, cfg: Config) -> dict:
    """Everything catalog.html needs for one kind."""
    items = catalog.ordered(kind)
    if kind == "dataset":
        items = [r for r in items if (r.get("access") or "open") in PUBLIC_ACCESS]
    rows = []
    counts: dict[str, Counter] = {key: Counter() for key, _ in FACETS[kind]}
    labels: dict[str, dict[str, str]] = {key: {} for key, _ in FACETS[kind]}
    for order, r in enumerate(items):
        values = facet_values(r, vocab, cfg)
        for key, vals in values.items():
            for v, label in vals:
                counts[key][v] += 1
                labels[key][v] = label
        gh = catalog.github.get(r.repo, {}) if r.repo else {}
        rows.append({
            "r": r, "order": order, "facets": {k: [v for v, _ in vals] for k, vals in values.items()},
            "summary": first_sentence(r.description), "stars": gh.get("stars", 0),
            "release": (gh.get("release") or {}), "pushed": gh.get("pushed_at", ""), "archived": gh.get("archived"),
            "citations": catalog.citations.get(f"{r.kind}/{r.slug}", 0),
        })  # fmt: skip
    # Value order: vocabulary order for type/task, otherwise most common first.
    vocab_order = {
        "type": list(vocab["kinds"][kind]["categories"]),
        "task": list(vocab.get("tasks", {})),
        "access": list(vocab["access"]),
    }
    facets = []
    for key, label in FACETS[kind]:
        vals = list(counts[key])
        if key in vocab_order:
            vals.sort(key=lambda v: vocab_order[key].index(v) if v in vocab_order[key] else 99)
        else:
            vals.sort(key=lambda v: (-counts[key][v], labels[key][v]))
        if len(vals) > 1 or key in ("cloud", "status"):
            facets.append({"key": key, "label": label, "values": [(v, labels[key][v], counts[key][v]) for v in vals]})
    title, intro = PAGE_TEXT[kind]
    return {
        "kind": kind, "path": PAGES[kind], "title": title, "intro": intro, "rows": rows, "facets": facets,
        "popular": [row for row in rows if row["r"].rank and row["r"].rank <= 10],
    }  # fmt: skip
