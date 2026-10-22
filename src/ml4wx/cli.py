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
from .http import BudgetExceeded
from .models import Candidate
from .sources import arxiv, crossref, feeds, openalex, s2
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
    errors = [f"{path.relative_to(archive.root)}: cannot load ({msg})" for path, msg in archive.invalid]
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
    # Journal versions of arXiv seeds: add the DOI so the arXiv record merges with the journal one.
    for c in cands:
        s = by_key.get(c.key) or {}
        if s.get("doi") and not c.ids.doi:
            c.ids.doi = s["doi"].lower()
    # Crossref for every DOI not already fetched: venue and citations (dedupe merges the records).
    fetched = {c.ids.doi for c in cands if c.source == "seed" and c.venue_type == "journal"}
    all_dois = [s["doi"].lower() for s in seeds if s.get("doi")]
    for c in crossref.lookup([d for d in all_dois if d not in fetched], source="seed"):
        cands.append(c)
        found.add(c.key)
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


def _load_backfill_state(path: Path) -> dict[str, list[str]]:
    """month -> sources done. The old format (a list of months) meant both sources."""
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {m: list(fetch.SOURCES) for m in raw} if isinstance(raw, list) else raw


def cmd_backfill(args) -> int:
    """Month by month over arXiv and OpenAlex, resumable per month and per source.

    A source that fails (e.g. arXiv answering 429 to shared CI addresses) is skipped for the
    rest of the run; its part of each month stays open for a later run.
    """
    cfg, archive = load(), Archive()
    state_file = archive.root / "state" / "backfill.json"
    done = _load_backfill_state(state_file)
    sources = fetch.default_sources(args.openalex)
    if args.only:
        sources = [s for s in sources if s == args.only]
    # Sources found down earlier in the same CI job (the workflow calls backfill once per chunk).
    down_file = Path(os.environ["ML4WX_DOWN_FILE"]) if os.environ.get("ML4WX_DOWN_FILE") else None
    down: set[str] = set(down_file.read_text().split()) if down_file and down_file.exists() else set()
    month = dt.date(args.start.year, args.start.month, 1)
    n_months = 0
    deadline = time.monotonic() + 60 * args.max_minutes
    while month <= args.end and n_months < args.max_months and time.monotonic() < deadline:
        tag = month.strftime("%Y-%m")
        nxt = (month + dt.timedelta(days=32)).replace(day=1)
        month_end = min(nxt - dt.timedelta(days=1), args.end)
        need = [s for s in sources if s not in done.get(tag, []) and s not in down]
        if need:
            cands, got = [], []
            for s in need:
                try:
                    cands += fetch.from_source(cfg, s, month, month_end)
                    got.append(s)
                except fetch.SourceFailed as e:
                    log.warning("%s: %s; skipping this source for the rest of the run", tag, e)
                    down.add(s)
                    if down_file:
                        down_file.write_text(" ".join(sorted(down)))
            if got:
                rep = ingest.ingest(cands, archive, cfg, source="backfill", judge_highlights=False)
                print(f"{tag} [{'+'.join(got)}]: {len(cands)} candidates -> +{len(rep.added)} papers, "
                      f"{rep.parked} parked, {rep.rejected} rejected, {rep.failed} failed", flush=True)  # fmt: skip
                if rep.failed <= max(5, len(cands) // 20):
                    done[tag] = sorted(set(done.get(tag, [])) | set(got))
                    state_file.parent.mkdir(exist_ok=True)
                    state_file.write_text(json.dumps(dict(sorted(done.items())), indent=0))
                n_months += 1
            if down >= set(sources):
                print("all sources are down; stopping (run again later)")
                break
        month = nxt
    open_parts = sum(
        1
        for m in _months(args.start, args.end)
        for s in sources
        if s not in done.get(m, [])
    )
    print(f"backfill: {open_parts} month-source parts still open in this range", flush=True)
    return 0


def _months(start: dt.date, end: dt.date) -> list[str]:
    out, m = [], dt.date(start.year, start.month, 1)
    while m <= end:
        out.append(m.strftime("%Y-%m"))
        m = (m + dt.timedelta(days=32)).replace(day=1)
    return out


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


def cmd_enrich(args) -> int:
    """Labs, journal versions and citation counts from OpenAlex for archived papers."""
    cfg, archive = load(), Archive()
    since = dt.date.today() - dt.timedelta(days=args.days) if args.days else None
    papers = [p for p in archive.papers.values() if since is None or p.added >= since]
    if args.missing_orgs:
        papers = [p for p in papers if not p.orgs]
    # The arXiv preprint and the journal version are often separate OpenAlex works: ask for both.
    keys = [
        k
        for p in papers
        for k in (
            f"arxiv:{p.ids.arxiv}" if p.ids.arxiv else None,
            f"doi:{p.ids.doi}" if p.ids.doi else None,
            f"openalex:{p.ids.openalex}" if p.ids.openalex and not (p.ids.arxiv or p.ids.doi) else None,
        )
        if k
    ]
    changed = 0
    try:
        for c in openalex.lookup(keys, source="enrich"):
            if (p := archive.find(c)) and ingest.update_existing(p, c, cfg):
                archive.save_paper(p)
                changed += 1
    except BudgetExceeded as e:
        log.warning("stopped early: %s", e)
    print(f"enrich: {len(papers)} papers looked up, {changed} updated")
    return 0


def cmd_reclassify(args) -> int:
    """Classify papers again with abstracts fetched from OpenAlex and arXiv.

    For papers first classified from the title only (Semantic Scholar often has no abstract).
    Out-of-scope papers are removed and rejected, 'later' ones are parked.
    """
    from . import classify, dedupe

    cfg, archive = load(), Archive()
    papers = [p for p in archive.papers.values() if p.source in args.source and (args.all or not p.abstract)]
    by_key = {k: p for p in papers for k in p.ids.keys()}
    fetched: dict[str, Candidate] = {}

    def keep(c: Candidate) -> None:
        p = next((by_key[k] for k in c.ids.keys() if k in by_key), None)
        if p and c.abstract and (p.id not in fetched or c.abstract_license_ok):
            fetched[p.id] = c

    for c in arxiv.lookup([p.ids.arxiv for p in papers if p.ids.arxiv], source="reclassify"):
        keep(c)
    try:
        keys = [k for p in papers if p.id not in fetched for k in p.ids.keys()[:2]]
        for c in openalex.lookup(keys, source="reclassify"):
            keep(c)
    except BudgetExceeded as e:
        log.warning("OpenAlex stopped early: %s", e)
    cands = []
    for p in papers:
        c = fetched.get(p.id) or Candidate(ids=p.ids, title=p.title, authors=p.authors, date=p.date)
        cands.append(c)
    print(f"reclassify: {len(papers)} papers, {len(fetched)} with an abstract now")
    results = classify.classify(cands, cfg)
    removed = parked = updated = 0
    for p, c in zip(papers, cands, strict=True):
        r = results.get(c.key)
        if r is None:
            continue
        if r["decision"] == "out":
            dedupe.remove(archive, p)
            removed += 1
        elif r["decision"] == "later":
            dedupe.remove(archive, p, reject=False)
            archive.park(c, r["topic"])
            parked += 1
        else:
            for k in ("topic", "topics", "type", "method", "spatial", "data", "status", "significance"):
                setattr(p, k, r[k])
            p.timescale = [t for t in r["timescale"] if t != "none"]
            p.summary = r["summary"].strip()
            if c.abstract and c.abstract_license_ok:
                p.abstract = c.abstract
            ingest.finalize(p, cfg)
            archive.save_paper(p)
            updated += 1
    archive.save_state()
    print(f"reclassify: {updated} kept (labels updated), {removed} removed as out of scope, {parked} parked")
    return 0


def cmd_dedupe(args) -> int:
    from . import dedupe

    merged, retracted = dedupe.run(Archive(), load(), dry_run=args.dry_run)
    print(f"dedupe: {merged} duplicates merged, {retracted} retracted papers removed"
          + (" (dry run)" if args.dry_run else ""))  # fmt: skip
    return 0


def cmd_judge(args) -> int:
    """Re-score highlights for papers added by a source (e.g. after a backfill)."""
    cfg, archive = load(), Archive()
    sel = [p for p in archive.papers.values() if not args.source or p.source in args.source]
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
    from . import dedupe

    dedupe.run(archive, cfg)  # journal versions of preprints that slipped through matching
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
    p.add_argument("--only", choices=fetch.SOURCES, help="only this source (e.g. arxiv from a local machine)")
    p.set_defaults(fn=cmd_backfill)

    p = sub.add_parser("snowball", help="add papers that cite archived papers (Semantic Scholar)")
    p.add_argument("--max-seeds", type=int, default=300)
    p.add_argument("--min-hits", type=int, default=2)
    p.add_argument("--max-citations", type=int, default=3000)
    p.add_argument("--max-minutes", type=float, default=300, help="stop starting new chunks after this")
    p.set_defaults(fn=cmd_snowball)

    p = sub.add_parser("enrich", help="labs, journal versions, citations from OpenAlex")
    p.add_argument("--days", type=int, default=None, help="only papers added in the last N days")
    p.add_argument("--missing-orgs", action="store_true", help="only papers without labs")
    p.set_defaults(fn=cmd_enrich)

    p = sub.add_parser("reclassify", help="classify again with abstracts (title-only papers)")
    p.add_argument("--source", nargs="+", default=["snowball"])
    p.add_argument("--all", action="store_true", help="also papers that already have an abstract")
    p.set_defaults(fn=cmd_reclassify)

    p = sub.add_parser("dedupe", help="merge duplicate papers, remove retracted ones")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(fn=cmd_dedupe)

    p = sub.add_parser("judge", help="second-opinion significance for likely highlights")
    p.add_argument("--source", nargs="*", default=None, help="only papers from these sources")
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
