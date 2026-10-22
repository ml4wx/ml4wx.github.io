"""ml4wx command line. Run `uv run ml4wx --help`."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

from . import alerts, digest, export, fetch, ingest
from .config import load
from .models import Candidate
from .sources import arxiv, crossref, feeds, s2
from .store import Archive

log = logging.getLogger("ml4wx")


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


def _gh_output(**kw) -> None:
    """Expose values to later GitHub Actions steps."""
    if path := os.environ.get("GITHUB_OUTPUT"):
        with open(path, "a") as f:
            for k, v in kw.items():
                f.write(f"{k}={v}\n")


# --- commands --------------------------------------------------------------------
def cmd_validate(args) -> int:
    cfg, archive = load(), Archive()
    errors = []
    slugs = {t.slug for t in cfg.topics}
    keys: dict[str, str] = {}
    for p in archive.papers.values():
        where = f"{p.year}/{p.id}"
        if p.topic not in cfg.enabled_slugs:
            errors.append(f"{where}: topic '{p.topic}' is not an enabled topic")
        errors += [f"{where}: unknown secondary topic '{t}'" for t in p.topics if t not in slugs]
        for facet in ("method", "timescale", "spatial", "data"):
            vocab = cfg.facets[facet]
            errors += [f"{where}: unknown {facet} '{v}'" for v in getattr(p, facet) if v not in vocab]
        if p.type not in cfg.facets["type"]:
            errors.append(f"{where}: unknown type '{p.type}'")
        if p.status not in cfg.facets["status"]:
            errors.append(f"{where}: unknown status '{p.status}'")
        errors += [f"{where}: org '{o}' not in watchlist" for o in p.orgs if o not in cfg.org_by_slug]
        for k in p.ids.keys():
            if k in keys and keys[k] != p.id:
                errors.append(f"{where}: id {k} also used by {keys[k]}")
            keys[k] = p.id
        if archive.paper_path(p).name != f"{p.id}.yaml":
            errors.append(f"{where}: file name does not match id")
    for e in errors:
        print(f"ERROR {e}")
    print(f"{len(archive.papers)} papers, {len(archive.parked)} parked, {len(errors)} errors")
    return 1 if errors else 0


def cmd_fetch(args) -> int:
    cands = fetch.recent(load(), args.start, args.end, use_openalex=args.openalex)
    with open(args.out, "w") as f:
        for c in cands:
            f.write(c.model_dump_json() + "\n")
    print(f"{len(cands)} candidates -> {args.out}")
    return 0


def cmd_ingest(args) -> int:
    cfg, archive = load(), Archive()
    cands = [Candidate.model_validate_json(line) for line in open(args.file) if line.strip()]
    rep = ingest.ingest(cands, archive, cfg, source=args.source, use_prefilter=not args.no_prefilter)
    print(f"+{len(rep.added)} papers ({sum(p.highlight for p in rep.added)} ★), {rep.parked} parked, "
          f"{rep.rejected} rejected, {rep.failed} failed")  # fmt: skip
    return 0


def cmd_seed(args) -> int:
    """Add curated seed papers. Classification labels them; the seed file's topic wins."""
    cfg, archive = load(), Archive()
    seeds = json.loads(Path(args.file).read_text())
    by_key = {}
    for s in seeds:
        key = f"arxiv:{s['arxiv']}" if s.get("arxiv") else f"doi:{s['doi'].lower()}"
        by_key[key] = s
    cands: list[Candidate] = list(arxiv.lookup([s["arxiv"] for s in seeds if s.get("arxiv")], source="seed"))
    found = {c.key for c in cands}
    dois = [k for k in by_key if k.startswith("doi:")]
    if os.environ.get("S2_API_KEY"):  # without a key the shared pool is too busy to wait for
        try:
            for key, raw in s2.batch(dois, source="seed"):
                if c := s2.parse_paper(raw or {}, "seed"):
                    cands.append(c)
                    found.add(key)
        except RuntimeError as e:
            log.warning("Semantic Scholar unavailable (%s); using Crossref", e)
    # DOI-only seeds that Semantic Scholar did not return (or no arXiv id): Crossref.
    for c in crossref.lookup([k[4:] for k in dois if k not in found], source="seed"):
        cands.append(c)
        found.add(c.key)
    # Journal versions of arXiv seeds: add the DOI so the record merges with OpenAlex later.
    for c in cands:
        s = by_key.get(c.key) or {}
        if s.get("doi") and not c.ids.doi:
            c.ids.doi = s["doi"].lower()
    missing = [k for k in by_key if k not in found]
    if missing:
        log.warning("seeds not found: %s", ", ".join(missing))
    forced = {c.key: by_key.get(c.key, by_key.get(f"doi:{c.ids.doi}", {})).get("topic") for c in cands}
    classic = {c.key for c in cands if by_key.get(c.key, {}).get("kind") == "classic"}
    rep = ingest.ingest(
        cands, archive, cfg, source="seed", use_prefilter=False,
        force_topic={k: v for k, v in forced.items() if v}, classic_keys=classic,
    )  # fmt: skip
    print(f"seeds: +{len(rep.added)} papers, {len(rep.updated)} updated, {len(missing)} not found")
    return 0


def cmd_backfill(args) -> int:
    """Month by month over arXiv (+ OpenAlex with a key). Progress is saved after each month."""
    cfg, archive = load(), Archive()
    state_file = archive.root / "state" / "backfill.json"
    done = set(json.loads(state_file.read_text())) if state_file.exists() else set()
    month = dt.date(args.start.year, args.start.month, 1)
    n_months = 0
    deadline = time.monotonic() + 60 * args.max_minutes
    while month <= args.end and n_months < args.max_months and time.monotonic() < deadline:
        tag = month.strftime("%Y-%m")
        nxt = (month + dt.timedelta(days=32)).replace(day=1)
        if tag not in done:
            cands = fetch.recent(cfg, month, min(nxt - dt.timedelta(days=1), args.end), use_openalex=args.openalex)
            rep = ingest.ingest(cands, archive, cfg, source="backfill", judge_highlights=False)
            print(f"{tag}: {len(cands)} candidates -> +{len(rep.added)} papers, {rep.parked} parked, "
                  f"{rep.rejected} rejected, {rep.failed} failed", flush=True)  # fmt: skip
            if rep.failed <= max(5, len(cands) // 20):
                done.add(tag)
                state_file.parent.mkdir(exist_ok=True)
                state_file.write_text(json.dumps(sorted(done), indent=0))
            n_months += 1
        month = nxt
    return 0


def cmd_snowball(args) -> int:
    """Papers citing at least `min_hits` archived papers (Semantic Scholar), since start_year."""
    cfg, archive = load(), Archive()
    start = dt.date(cfg.settings["start_year"], 1, 1)
    seeds = sorted(archive.papers.values(), key=lambda p: -p.score)[: args.max_seeds]
    hits: Counter[str] = Counter()
    pool: dict[str, Candidate] = {}
    for i, p in enumerate(seeds, 1):
        sid = f"ARXIV:{p.ids.arxiv}" if p.ids.arxiv else f"DOI:{p.ids.doi}" if p.ids.doi else None
        if not sid:
            continue
        for c in s2.citations(sid, source="snowball", max_n=args.max_citations):
            if c.date >= start:
                hits[c.key] += 1
                pool.setdefault(c.key, c)
        log.info("snowball %d/%d %s: pool %d", i, len(seeds), p.id, len(pool))
    cands = [pool[k] for k, n in hits.most_common() if n >= args.min_hits]
    cands = [c for c in cands if not archive.find(c) and not archive.status(c)]
    print(f"snowball: {len(pool)} citing papers, {len(cands)} new ones cite >= {args.min_hits} archived papers")
    # Chunks, so a time limit or a crash keeps what is done (archive is saved per chunk).
    deadline = time.monotonic() + 60 * args.max_minutes
    for i in range(0, len(cands), 400):
        if time.monotonic() > deadline:
            print("time budget used; run again to continue")
            break
        rep = ingest.ingest(cands[i : i + 400], archive, cfg, source="snowball", judge_highlights=False)
        print(f"chunk {i // 400 + 1}: +{len(rep.added)} papers, {rep.parked} parked, "
              f"{rep.rejected} rejected, {rep.failed} failed", flush=True)  # fmt: skip
    return 0


def cmd_judge(args) -> int:
    """Re-score highlights for papers added by a source (e.g. after a backfill)."""
    cfg, archive = load(), Archive()
    sel = [p for p in archive.papers.values() if not args.source or p.source == args.source]
    maybe = [p for p in sel if ingest.maybe_highlight(p, cfg)]
    ingest.judge(maybe, cfg)
    for p in maybe:
        ingest.finalize(p, cfg)
        archive.save_paper(p)
    print(f"judged {len(maybe)} papers; {sum(p.highlight for p in maybe)} highlights")
    return 0


def cmd_weekly(args) -> int:
    cfg, archive = load(), Archive()
    today = dt.date.today()
    since = today - dt.timedelta(days=cfg.settings["weekly_lookback_days"])
    cands = fetch.recent(cfg, since, today)
    rep = ingest.ingest(cands, archive, cfg, source="weekly", today=today, html_affiliations=True)
    export.write_all(archive, cfg)
    rels = feeds.releases(cfg, today - dt.timedelta(days=7))
    path, _ = digest.build(
        archive, cfg, since=today, today=today, releases=rels, pr_url=args.pr_url, window_start=since
    )
    n_hl = sum(p.highlight for p in rep.added)
    print(f"weekly: +{len(rep.added)} papers ({n_hl} ★), {len(rep.updated)} updated, digest {path.name}")
    _gh_output(added=len(rep.added), highlights=n_hl, updated=len(rep.updated),
               digest=str(path.relative_to(archive.root)), week=digest.week_id(today))  # fmt: skip
    return 0


def cmd_alerts(args) -> int:
    n = alerts.run(load(), Archive(), args.days, dry_run=args.dry_run)
    print(f"{n} alerts")
    return 0


def cmd_export(args) -> int:
    for p in export.write_all(Archive(), load()):
        log.info("wrote %s", p)
    return 0


def cmd_digest(args) -> int:
    cfg, archive = load(), Archive()
    path, md = digest.build(archive, cfg, since=args.since, pr_url=args.pr_url, write_intro=not args.no_intro)
    print(path)
    return 0


def cmd_site(args) -> int:
    from .site.build import build

    build(Archive(), load(), Path(args.out), index=not args.no_index)
    return 0


# --- parser ----------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ml4wx", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("validate", help="check papers against taxonomy and watchlist").set_defaults(fn=cmd_validate)

    p = sub.add_parser("fetch", help="collect candidates for a date range into a JSONL file")
    p.add_argument("--start", type=_date, required=True)
    p.add_argument("--end", type=_date, required=True)
    p.add_argument("--openalex", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("-o", "--out", default="candidates.jsonl")
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("ingest", help="classify and archive candidates from a JSONL file")
    p.add_argument("file")
    p.add_argument("--source", default="manual")
    p.add_argument("--no-prefilter", action="store_true")
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("seed", help="add curated seed papers from a JSON file")
    p.add_argument("file")
    p.set_defaults(fn=cmd_seed)

    p = sub.add_parser("backfill", help="month-by-month historical search (resumable)")
    p.add_argument("--start", type=_date, default=dt.date(2018, 1, 1))
    p.add_argument("--end", type=_date, default=dt.date.today())
    p.add_argument("--max-months", type=int, default=1000)
    p.add_argument("--max-minutes", type=float, default=300, help="stop starting new months after this")
    p.add_argument("--openalex", action=argparse.BooleanOptionalAction, default=None)
    p.set_defaults(fn=cmd_backfill)

    p = sub.add_parser("snowball", help="add papers that cite archived papers (Semantic Scholar)")
    p.add_argument("--max-seeds", type=int, default=300)
    p.add_argument("--min-hits", type=int, default=2)
    p.add_argument("--max-citations", type=int, default=3000)
    p.add_argument("--max-minutes", type=float, default=300, help="stop starting new chunks after this")
    p.set_defaults(fn=cmd_snowball)

    p = sub.add_parser("judge", help="second-opinion significance for likely highlights")
    p.add_argument("--source", default=None)
    p.set_defaults(fn=cmd_judge)

    p = sub.add_parser("weekly", help="weekly crawl: fetch, ingest, export, digest")
    p.add_argument("--pr-url", default=None)
    p.set_defaults(fn=cmd_weekly)

    p = sub.add_parser("alerts", help="daily highlight check; opens GitHub issues")
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_alerts)

    sub.add_parser("export", help="write README, lists/ and exports/").set_defaults(fn=cmd_export)

    p = sub.add_parser("digest", help="write the digest for papers added since a date")
    p.add_argument("--since", type=_date, default=dt.date.today() - dt.timedelta(days=6))
    p.add_argument("--pr-url", default=None)
    p.add_argument("--no-intro", action="store_true")
    p.set_defaults(fn=cmd_digest)

    p = sub.add_parser("site", help="build the static site")
    p.add_argument("--out", default="_site")
    p.add_argument("--no-index", action="store_true", help="skip the Pagefind search index")
    p.set_defaults(fn=cmd_site)

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
