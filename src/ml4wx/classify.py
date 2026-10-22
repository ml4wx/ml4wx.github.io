"""Relevance, topic and facets for candidates, many abstracts per `claude -p` call."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from . import llm
from .config import Config
from .models import Candidate

log = logging.getLogger(__name__)

MAX_ABSTRACT = 1800


def system_prompt(cfg: Config) -> str:
    topics = "\n".join(
        f"- {t.slug}: {t.description}" + (f" (e.g. {', '.join(t.examples)})" if t.examples else "")
        for t in cfg.topics
        if t.enabled
    )
    later = "\n".join(f"- {t.slug}: {t.description}" for t in cfg.topics if not t.enabled)
    facets = "\n".join(
        f"- {name}: " + "; ".join(f"{k} = {v}" for k, v in vocab.items())
        for name, vocab in cfg.facets.items()
    )
    scope = " ".join(cfg.taxonomy["scope_rule"].split())
    return f"""You classify research papers for ML4WX, an archive of machine learning research \
for weather and climate. You get several papers (title, venue, abstract). For EVERY paper return \
one result with the same key.

SCOPE
{scope}
decision = "in", "later" or "out".

TOPICS (primary `topic`: exactly one; `topics`: 0-2 secondary topics, only if clearly relevant)
{topics}
LATER topics (use as `topic` only when decision = "later"):
{later}
For decision = "out", set topic = "none" and fill the other fields with your best guess.

FACETS (lists may be empty when the abstract does not say)
{facets}

summary: one sentence, at most 30 words, plain English. Say what is new and the main result, \
with a number if the abstract gives one. Do not start with "This paper" or "We".

significance (0-10): importance for the ML weather/climate field, judged from the abstract only.
0-3 incremental, narrow or regional study; 4-6 solid contribution; 7-8 major new model, \
benchmark or result likely to be widely used or cited; 9-10 landmark (GraphCast-level). \
Most papers are 2-5. Be strict.
"""


def schema(cfg: Config) -> dict:
    f = cfg.facets
    enum_list = lambda vals: {"type": "array", "items": {"type": "string", "enum": list(vals)}}  # noqa: E731
    item = {
        "type": "object",
        "properties": {
            "key": {"type": "string"},
            "decision": {"type": "string", "enum": ["in", "later", "out"]},
            "topic": {"type": "string", "enum": [t.slug for t in cfg.topics] + ["none"]},
            "topics": enum_list(cfg.enabled_slugs),
            "type": {"type": "string", "enum": list(f["type"])},
            "method": enum_list(f["method"]),
            "timescale": enum_list(f["timescale"]),
            "spatial": enum_list(f["spatial"]),
            "data": enum_list(f["data"]),
            "status": {"type": "string", "enum": list(f["status"])},
            "summary": {"type": "string"},
            "significance": {"type": "integer"},
        },
        "required": [
            "key", "decision", "topic", "topics", "type", "method", "timescale",
            "spatial", "data", "status", "summary", "significance",
        ],  # fmt: skip
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"results": {"type": "array", "items": item}},
        "required": ["results"],
        "additionalProperties": False,
    }


def _render(c: Candidate) -> str:
    abstract = (c.abstract or "(no abstract)")[:MAX_ABSTRACT]
    venue = c.venue or ("arXiv preprint" if c.ids.arxiv else "unknown venue")
    return f"[{c.key}]\nTitle: {c.title}\nVenue: {venue} ({c.date.year})\nAbstract: {abstract}\n"


def _clean(r: dict, cfg: Config) -> dict:
    r["significance"] = max(0, min(10, int(r.get("significance", 0))))
    r["topics"] = [t for t in dict.fromkeys(r.get("topics", [])) if t != r["topic"]][:2]
    for k in ("method", "timescale", "spatial", "data"):
        r[k] = list(dict.fromkeys(r.get(k, [])))
    if r["decision"] == "in" and r["topic"] not in cfg.enabled_slugs:
        r["decision"] = "later" if r["topic"] in cfg.later_slugs else "out"
    if r["decision"] == "later" and r["topic"] not in cfg.later_slugs:
        r["decision"] = "in" if r["topic"] in cfg.enabled_slugs else "out"
    return r


def _batch(cands: list[Candidate], cfg: Config, model: str) -> dict[str, dict]:
    prompt = "Classify these papers:\n\n" + "\n".join(_render(c) for c in cands)
    keys = {c.key for c in cands}
    try:
        out = llm.structured(
            prompt, system_prompt(cfg), schema(cfg), model, cfg.settings["llm"]["timeout_s"]
        )
    except llm.LLMError as e:
        if len(cands) == 1:
            log.error("classification failed for %s: %s", cands[0].key, e)
            return {}
        log.warning("batch of %d failed (%s); splitting", len(cands), e)
        mid = len(cands) // 2
        return _batch(cands[:mid], cfg, model) | _batch(cands[mid:], cfg, model)
    got = {r["key"].strip("[]"): _clean(r, cfg) for r in out.get("results", [])}
    got = {k: v for k, v in got.items() if k in keys}
    missing = [c for c in cands if c.key not in got]
    if missing and len(missing) < len(cands):
        log.warning("%d results missing; retrying them", len(missing))
        got |= _batch(missing, cfg, model)
    elif missing and len(cands) > 1:
        # Nothing usable came back (e.g. keys rewritten): split, like a failed call.
        log.warning("no usable results for a batch of %d; splitting", len(cands))
        mid = len(cands) // 2
        got |= _batch(cands[:mid], cfg, model) | _batch(cands[mid:], cfg, model)
    elif missing:
        log.error("no result for %s", cands[0].key)
    return got


def classify(
    cands: list[Candidate], cfg: Config, model: str | None = None, workers: int = 3
) -> dict[str, dict]:
    """key -> classification dict. Candidates that fail twice are left out."""
    model = model or cfg.settings["llm"]["classify_model"]
    size = cfg.settings["llm"]["batch_size"]
    batches = [cands[i : i + size] for i in range(0, len(cands), size)]
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, got in enumerate(pool.map(lambda b: _batch(b, cfg, model), batches), 1):
            results |= got
            log.info("classified batch %d/%d (%d results)", i, len(batches), len(results))
    return results
