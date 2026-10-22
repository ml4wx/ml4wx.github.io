"""Read and write the archive: papers/, parked/ and state/rejected.txt."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from . import ids
from .config import repo_root
from .models import Candidate, Paper


def _dump_paper(p: Paper) -> str:
    d = p.model_dump(mode="json", exclude_none=True)
    for k in [k for k, v in d.items() if v in ([], {}) or (k == "classic" and v is False)]:
        del d[k]
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True, width=100)


class Archive:
    def __init__(self, root: Path | None = None):
        self.root = root or repo_root()
        self.papers: dict[str, Paper] = {}
        for f in sorted((self.root / "papers").glob("*/*.yaml")):
            with open(f, encoding="utf-8") as fh:
                p = Paper.model_validate(yaml.safe_load(fh))
            self.papers[p.id] = p
        self.parked: dict[str, dict] = {}
        for f in sorted((self.root / "parked").glob("*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.parked[rec["key"]] = rec
        rej = self.root / "state" / "rejected.txt"
        self.rejected: set[str] = (
            {x.strip() for x in rej.read_text().splitlines() if x.strip()} if rej.exists() else set()
        )
        self._reindex()

    def _reindex(self) -> None:
        self._by_key: dict[str, str] = {}
        self._by_title: dict[str, list[str]] = {}
        self._by_first: dict[str, list[str]] = {}
        for p in self.papers.values():
            self._index(p)
        self._parked_keys = {k for rec in self.parked.values() for k in rec.get("keys", [rec["key"]])}

    def _index(self, p: Paper) -> None:
        for k in p.ids.keys():
            self._by_key[k] = p.id
        self._by_title.setdefault(ids.norm_title(p.title)[:40], []).append(p.id)
        if p.authors:
            self._by_first.setdefault(ids.surname(p.authors[0]), []).append(p.id)

    # --- lookup ----------------------------------------------------------------
    def find(self, c: Candidate) -> Paper | None:
        """Same work already archived? Match on any id, then on title + first-author surname."""
        for k in c.ids.keys():
            if k in self._by_key:
                return self.papers[self._by_key[k]]
        first = ids.surname(c.authors[0]) if c.authors else ""
        for pid in self._by_title.get(ids.norm_title(c.title)[:40], []):
            p = self.papers[pid]
            if ids.same_title(p.title, c.title):
                return p
        if not first:
            return None
        # Titles that changed between preprint and journal: compare against same first author.
        for pid in self._by_first.get(first, []):
            if ids.same_title(self.papers[pid].title, c.title):
                return self.papers[pid]
        return None

    def status(self, c: Candidate) -> str | None:
        """'archived' | 'parked' | 'rejected' | None."""
        if self.find(c):
            return "archived"
        keys = set(c.ids.keys())
        if keys & self._parked_keys:
            return "parked"
        if keys & self.rejected:
            return "rejected"
        return None

    # --- updates ---------------------------------------------------------------
    def paper_path(self, p: Paper) -> Path:
        return self.root / "papers" / str(p.year) / f"{p.id}.yaml"

    def save_paper(self, p: Paper) -> None:
        path = self.paper_path(p)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_dump_paper(p), encoding="utf-8")
        self.papers[p.id] = p
        self._index(p)

    def park(self, c: Candidate, topic_guess: str) -> None:
        rec = {
            "key": c.key,
            "keys": c.ids.keys(),
            "ids": c.ids.model_dump(exclude_none=True),
            "title": c.title,
            "authors": c.authors[:5],
            "date": c.date.isoformat(),
            "venue": c.venue,
            "topic_guess": topic_guess,
            "source": c.source,
        }
        self.parked[c.key] = rec
        self._parked_keys.update(rec["keys"])

    def reject(self, c: Candidate) -> None:
        self.rejected.update(c.ids.keys())

    def save_state(self) -> None:
        by_year: dict[str, list[dict]] = {}
        for rec in self.parked.values():
            by_year.setdefault(rec["date"][:4], []).append(rec)
        (self.root / "parked").mkdir(exist_ok=True)
        for year, recs in by_year.items():
            recs.sort(key=lambda r: (r["date"], r["key"]))
            lines = [json.dumps(r, ensure_ascii=False, sort_keys=True) for r in recs]
            (self.root / "parked" / f"{year}.jsonl").write_text("\n".join(lines) + "\n")
        (self.root / "state").mkdir(exist_ok=True)
        (self.root / "state" / "rejected.txt").write_text("".join(f"{k}\n" for k in sorted(self.rejected)))
