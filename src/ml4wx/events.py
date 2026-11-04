"""Events: conferences, workshops, schools and competitions, with their deadlines.

Curated by hand: one YAML file per event in events/<slug>.yaml. An event is one of
- an edition with dates (start, end, deadlines), checked on its official page;
- a machine learning conference read from ccf-deadlines (`ccf`: the file in that repo), whose next
  edition, dates and deadlines are copied every week;
- a series to watch (`usual`, no dates yet): its next edition is not announced.
Generated weekly: events/_status.json (`ml4wx events refresh`): the ccf-deadlines editions, the dates
found on each event page, and the changes since last week (new or moved deadlines, changed pages).
"""

from __future__ import annotations

import datetime as dt
import html
import json
import logging
import re
from pathlib import Path
from typing import Literal

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field

from .config import Config, repo_root
from .http import Client

log = logging.getLogger(__name__)

STATUS_FILE = "_status.json"
MAX_CHANGES = 40
CCF_RAW = "https://raw.githubusercontent.com/ccfddl/ccf-deadlines/main/{}"
CCF_REPO = "https://github.com/ccfddl/ccf-deadlines"
# ccf-deadlines timeline keys we show: submissions only (rebuttals and notifications are not deadlines to meet).
CCF_LABELS = {"abstract_deadline": "Abstracts", "deadline": "Papers"}
TZ_HOURS = {
    "AoE": -12, "UTC": 0, "GMT": 0, "CET": 1, "CEST": 2, "BST": 1, "EST": -5, "EDT": -4, "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6, "PST": -8, "PDT": -7, "JST": 9, "KST": 9, "AEST": 10, "AEDT": 11,
}  # fmt: skip
MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov",
                                      "dec"], start=1)}  # fmt: skip
EUROPE = ("Austria", "Belgium", "Croatia", "Czech", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
          "Hungary", "Iceland", "Ireland", "Italy", "Latvia", "Lithuania", "Luxembourg", "Netherlands", "Norway",
          "Poland", "Portugal", "Romania", "Serbia", "Slovakia", "Slovenia", "Spain", "Sweden", "UK", "United Kingdom",
          "Europe")  # fmt: skip
MIN_DAYS_BETWEEN_CHECKS = 6  # the pages are checked once a week (weekly workflow); a rerun the same week skips
KINDS = {"conference": "Conference", "workshop": "Workshop", "school": "School", "hackathon": "Hackathon",
         "competition": "Competition"}  # fmt: skip


class Deadline(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    date: str  # "YYYY-MM-DD" or "YYYY-MM-DD HH:MM"; without a time, the end of the day
    tz: str = "AoE"  # as written on the event page: AoE, CET, UTC+8, ...
    was: str | None = None  # the date before an extension, shown as "extended from"


class Event(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str
    name: str
    kind: Literal["conference", "workshop", "school", "hackathon", "competition"]
    start: dt.date | None = None
    end: dt.date | None = None
    location: str = ""
    online: bool = False
    url: str = ""
    deadlines_url: str = ""
    deadlines: list[Deadline] = Field(default_factory=list)
    ccf: str = ""  # file in ccfddl/ccf-deadlines, e.g. conference/AI/nips.yml
    usual: str = ""  # a series to watch: when the next edition usually happens
    note: str = ""
    added: dt.date


def utc(date: str, tz: str = "AoE") -> dt.datetime:
    """A deadline as an aware UTC time. Without a time it is the end of that day in its time zone."""
    has_time = len(date.strip()) > 10
    local = dt.datetime.fromisoformat(date.strip() if has_time else f"{date.strip()} 23:59:59")
    m = re.fullmatch(r"(?:UTC|GMT)\s*([+-])\s*(\d{1,2})(?::?(\d\d))?", tz.strip())
    if m:
        hours = (1 if m.group(1) == "+" else -1) * (int(m.group(2)) + int(m.group(3) or 0) / 60)
    elif tz.strip() in TZ_HOURS:
        hours = TZ_HOURS[tz.strip()]
    else:
        raise ValueError(f"unknown time zone {tz!r}")
    return (local - dt.timedelta(hours=hours)).replace(tzinfo=dt.UTC)


def regions(location: str, online: bool = False) -> list[str]:
    """Filter tags for the page: switzerland (also europe), europe, online, elsewhere."""
    out = []
    if "Switzerland" in location:
        out += ["switzerland", "europe"]
    elif any(c in location for c in EUROPE):
        out.append("europe")
    if online or location.strip().lower() == "online":
        out.append("online")
    return out or ["elsewhere"]


def date_range(start: dt.date | None, end: dt.date | None) -> str:
    """'7-11 Dec 2026', '26 Jan - 4 Feb 2027', '31 Dec 2026 - 2 Jan 2027'."""
    if not start:
        return ""
    end = end or start
    if start == end:
        return f"{start.day} {start:%b %Y}"
    if (start.year, start.month) == (end.year, end.month):
        return f"{start.day}–{end.day} {end:%b %Y}"
    if start.year == end.year:
        return f"{start.day} {start:%b} – {end.day} {end:%b %Y}"
    return f"{start.day} {start:%b %Y} – {end.day} {end:%b %Y}"


def parse_dates_text(text: str) -> tuple[dt.date | None, dt.date | None]:
    """First and last day in a free-text date like 'July 31 - August 6, 2027' or
    'December 8-10 (Sydney), December 9-11 (Atlanta and Paris), 2026'."""
    years = re.findall(r"\b(20\d\d)\b", text)
    if not years:
        return None, None
    year, days = int(years[-1]), []
    for m in re.finditer(r"\b([A-Z][a-z]{2,8})\.?\s+(\d{1,2})(?:\s*[-–]\s*(\d{1,2})\b(?!\s*[A-Z]))?", text):
        month = MONTHS.get(m.group(1)[:3].lower())
        if not month:
            continue
        for d in filter(None, (m.group(2), m.group(3))):
            try:
                days.append(dt.date(year, month, int(d)))
            except ValueError:
                pass
    return (min(days), max(days)) if days else (None, None)


DATE_RE = re.compile(
    r"\b(?:(?P<d1>\d{1,2})(?:st|nd|rd|th)?\s+(?P<m1>[A-Z][a-z]{2,8})\.?,?\s+(?P<y1>20\d\d)"
    r"|(?P<m2>[A-Z][a-z]{2,8})\.?\s+(?P<d2>\d{1,2})(?:st|nd|rd|th)?,?\s+(?P<y2>20\d\d)"
    r"|(?P<y3>20\d\d)-(?P<m3>\d\d)-(?P<d3>\d\d))\b"
)


def page_dates(page: str, today: dt.date) -> list[str]:
    """The dates written on a page (ISO, sorted), within about a year before and two years after today.
    Comparing them week to week shows new, moved or extended deadlines without parsing each site."""
    t = re.sub(r"<script.*?</script>|<style.*?</style>|<!--.*?-->", " ", page, flags=re.S | re.I)
    text = " ".join(html.unescape(re.sub(r"<[^>]+>", " ", t)).split())
    out = set()
    for m in DATE_RE.finditer(text):
        try:
            if m.group("y3"):
                d = dt.date(int(m.group("y3")), int(m.group("m3")), int(m.group("d3")))
            else:
                month = MONTHS.get((m.group("m1") or m.group("m2"))[:3].lower())
                if not month:
                    continue
                d = dt.date(int(m.group("y1") or m.group("y2")), month, int(m.group("d1") or m.group("d2")))
        except ValueError:
            continue
        if today - dt.timedelta(days=400) <= d <= today + dt.timedelta(days=800):
            out.add(d.isoformat())
    return sorted(out)


def ccf_edition(text: str, today: dt.date) -> dict:
    """The next edition in a ccf-deadlines file: the first whose deadlines or dates are not over yet."""
    conf = yaml.safe_load(text)[0]
    editions = sorted(conf["confs"], key=lambda c: c.get("year", 0))
    pick = editions[-1]
    for c in editions:
        end = parse_dates_text(str(c.get("date") or ""))[1]
        if (end and end >= today) or any(d["utc"][:10] >= today.isoformat() for d in _ccf_deadlines(c)):
            pick = c
            break
    start, end = parse_dates_text(str(pick.get("date") or ""))
    return {
        "title": conf["title"], "full": conf.get("description", ""), "year": pick.get("year"),
        "date_text": pick.get("date", ""), "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None, "place": pick.get("place", ""), "link": pick.get("link", ""),
        "deadlines": _ccf_deadlines(pick),
        "tbd": any(str(r.get("deadline", "")).upper() == "TBD" for r in pick.get("timeline") or []),
        "comment": next((r["comment"] for r in pick.get("timeline") or [] if r.get("comment")), ""),
    }  # fmt: skip


def _ccf_deadlines(c: dict) -> list[dict]:
    out = []
    rounds = c.get("timeline") or []
    tz = str(c.get("timezone") or "AoE").replace("UTC-12", "AoE")
    for i, r in enumerate(rounds):
        # Several rounds: name them by their comment ("Research track"), else by number.
        which = str(r.get("comment") or "").split(";")[0].strip() or f"round {i + 1}"
        for key, label in CCF_LABELS.items():
            v = str(r.get(key) or "")
            if not re.match(r"\d{4}-\d\d-\d\d", v):  # "TBD"
                continue
            if key == "abstract_deadline" and v == str(r.get("deadline")):
                continue  # the same day as the paper deadline
            out.append({"label": label + (f" ({which})" if len(rounds) > 1 else ""), "date": v[:16], "tz": tz,
                        "utc": utc(v[:19], tz).isoformat()})  # fmt: skip
    return out


class Catalog:
    def __init__(self, root: Path | None = None):
        self.root = (root or repo_root()) / "events"
        self.items: dict[str, Event] = {}
        self.invalid: list[tuple[Path, str]] = []
        for f in sorted(self.root.glob("*.yaml")):
            try:
                e = Event.model_validate(yaml.safe_load(f.read_text(encoding="utf-8")))
            except (yaml.YAMLError, ValueError) as err:
                self.invalid.append((f, str(err).splitlines()[0]))
                continue
            self.items[e.slug] = e
        status = self.root / STATUS_FILE
        self.status: dict = json.loads(status.read_text()) if status.exists() else {}

    def validate(self, cfg: Config) -> list[str]:
        errors = [f"events/{p.name}: cannot load ({m})" for p, m in self.invalid]
        for e in self.items.values():
            where = f"events/{e.slug}.yaml"
            if not (self.root / f"{e.slug}.yaml").exists():
                errors.append(f"{where}: file name does not match slug")
            if sum(map(bool, (e.start, e.ccf, e.usual))) != 1:
                errors.append(f"{where}: needs exactly one of start (an edition), ccf or usual (a series)")
            if e.start and e.end and e.end < e.start:
                errors.append(f"{where}: end before start")
            for field in ("url", "deadlines_url"):
                v = getattr(e, field)
                if v and not v.startswith(("http://", "https://")):
                    errors.append(f"{where}: {field} is not a URL")
            if not e.url and not e.ccf:
                errors.append(f"{where}: needs a url")
            for d in e.deadlines:
                try:
                    utc(d.date, d.tz)
                    if d.was:
                        utc(d.was, d.tz)
                except ValueError as err:
                    errors.append(f"{where}: deadline '{d.label}': {err}")
        return errors

    def refresh(self, today: dt.date | None = None, client: Client | None = None, force: bool = False) -> list[str]:
        """Copy the ccf-deadlines editions and read the dates on every event page, at most once a week.
        Returns this run's changes."""
        today = today or dt.date.today()
        last = self.status.get("checked")
        if not force and last and (today - dt.date.fromisoformat(last)).days < MIN_DAYS_BETWEEN_CHECKS:
            log.info("events: already checked on %s; once a week is enough (use force to check again)", last)
            return []
        client = client or Client(min_interval=0.5, max_retries=2)
        old = self.status
        ccf, pages = dict(old.get("ccf", {})), dict(old.get("pages", {}))
        changes: list[str] = []
        for e in sorted(self.items.values(), key=lambda e: e.slug):
            if e.ccf:
                try:
                    r = client.get(CCF_RAW.format(e.ccf))
                    r.raise_for_status()
                    new = ccf_edition(r.text, today)
                except Exception as err:  # keep last week's copy
                    log.warning("%s: ccf-deadlines not read (%s)", e.slug, err)
                    continue
                changes += _ccf_changes(f"{e.name} {new['year']}", ccf.get(e.slug), new)
                ccf[e.slug] = new
                continue
            for url in dict.fromkeys(u for u in (e.url, e.deadlines_url) if u):
                last_day = e.end or e.start
                if last_day and last_day < today - dt.timedelta(days=30):
                    continue  # over: nothing will change that matters
                status, dates = _read_page(client, url, today)
                prev = pages.get(url)
                pages[url] = {"checked": today.isoformat(), "status": status,
                              "dates": dates if dates is not None else (prev or {}).get("dates", [])}  # fmt: skip
                if prev and status == 404 and prev.get("status") != 404:
                    changes.append(f"{e.name}: the page is gone ({url})")
                # Also from no dates to some: for a series, that is the next edition being announced.
                if prev and prev.get("status") == 200 and dates is not None and dates != prev.get("dates", []):
                    added = [d for d in dates if d not in prev.get("dates", [])]
                    gone = [d for d in prev.get("dates", []) if d not in dates]
                    pages[url]["changed"] = today.isoformat()
                    parts = ([f"new {_list(added)}"] if added else []) + ([f"gone {_list(gone)}"] if gone else [])
                    changes.append(f"{e.name}: dates on its page changed ({'; '.join(parts)}), check the deadlines")
        used = {u for e in self.items.values() for u in (e.url, e.deadlines_url) if u}
        pages = {u: p for u, p in pages.items() if u in used}  # pages no event links to any more
        ccf = {k: v for k, v in ccf.items() if k in self.items}
        log_ = [{"date": today.isoformat(), "text": t} for t in changes] + old.get("changes", [])
        self.status = {"checked": today.isoformat(), "ccf": ccf, "pages": pages, "changes": log_[:MAX_CHANGES]}
        (self.root / STATUS_FILE).write_text(json.dumps(self.status, indent=1, ensure_ascii=False) + "\n")
        return changes

    def is_new(self, e: Event, today: dt.date) -> bool:
        """Added in the last 30 days, after the first batch (which would make everything "new")."""
        first = min(x.added for x in self.items.values())
        return e.added > first and (today - e.added).days <= 30

    def editions(self, today: dt.date) -> list[dict]:
        """Every event with dates (curated, or the next ccf-deadlines edition) for the page and calendar."""
        out = []
        for e in self.items.values():
            if e.usual:
                continue
            if e.ccf:
                c = self.status.get("ccf", {}).get(e.slug)
                if not c:
                    continue
                start = dt.date.fromisoformat(c["start"]) if c.get("start") else None
                end = dt.date.fromisoformat(c["end"]) if c.get("end") else None
                dls = [{"label": d["label"], "date": d["date"], "tz": d["tz"], "utc": d["utc"], "was": None}
                       for d in c["deadlines"]]  # fmt: skip
                item = {"name": f"{e.name} {c['year']}", "location": c["place"], "url": c["link"] or e.url,
                        # Several venues ("December 8-10 (Sydney), December 9-11 (Atlanta and Paris)"): their text.
                        "dates_text": c["date_text"] if "(" in str(c["date_text"]) else date_range(start, end),
                        "tbd": c["tbd"], "source": CCF_REPO,
                        "online": False, "regions": regions(c["place"])}  # fmt: skip
            else:
                start, end = e.start, e.end or e.start
                dls = [{"label": d.label, "date": d.date, "tz": d.tz, "utc": utc(d.date, d.tz).isoformat(),
                        "was": d.was} for d in e.deadlines]  # fmt: skip
                item = {"name": e.name, "location": e.location, "url": e.url, "dates_text": date_range(start, end),
                        "tbd": False, "source": "", "online": e.online,
                        "regions": regions(e.location, e.online)}  # fmt: skip
            now = dt.datetime.combine(today, dt.time(), tzinfo=dt.UTC).isoformat()
            for d in dls:
                d["open"] = d["utc"] >= now
                d["when"] = _when(d)
            state = "upcoming" if not start or start > today else "now" if (end or start) >= today else "past"
            out.append({**item, "slug": e.slug, "kind": e.kind, "kind_label": KINDS[e.kind], "start": start, "end": end,
                        "state": state, "note": e.note, "deadlines": dls, "deadlines_url": e.deadlines_url,
                        "next": next((d for d in sorted(dls, key=lambda d: d["utc"]) if d["open"]), None),
                        "new": self.is_new(e, today)})  # fmt: skip
        return sorted(out, key=lambda x: (x["start"] or dt.date.max, x["name"]))

    def page(self, today: dt.date) -> dict:
        eds = self.editions(today)
        deadlines = sorted(({**d, "event": x} for x in eds for d in x["deadlines"] if d["open"]),
                           key=lambda d: d["utc"])  # fmt: skip
        pages = self.status.get("pages", {})
        series = [{"e": e, "changed": pages.get(e.url, {}).get("changed"), "regions": regions(e.location, e.online)}
                  for e in sorted(self.items.values(), key=lambda e: e.name) if e.usual]  # fmt: skip
        new = sorted(({"date": e.added.isoformat(), "text": f"New on this page: {e.name}"}
                      for e in self.items.values() if self.is_new(e, today)), key=lambda c: c["date"], reverse=True)
        return {
            "deadlines": deadlines,
            "upcoming": [x for x in eds if x["state"] != "past"],
            "past": sorted((x for x in eds if x["state"] == "past" and x["end"] and (today - x["end"]).days <= 365),
                           key=lambda x: x["end"], reverse=True),
            "series": series,
            "changes": sorted(self.status.get("changes", []) + new, key=lambda c: c["date"], reverse=True)[:15],
        }  # fmt: skip

    def digest(self, today: dt.date, days: int = 30) -> dict:
        """Deadlines in the next `days` days and this week's changes, for the weekly digest."""
        soon = (today + dt.timedelta(days=days)).isoformat()
        p = self.page(today)
        week = (today - dt.timedelta(days=7)).isoformat()
        return {"deadlines": [d for d in p["deadlines"] if d["utc"][:10] <= soon],
                "changes": [c for c in self.status.get("changes", []) if c["date"] >= week]}


def _when(d: dict) -> str:
    day = dt.date.fromisoformat(d["date"][:10])
    time = f", {d['date'][11:16]} {d['tz']}" if len(d["date"]) > 10 else ""
    return f"{day.day} {day:%b %Y}{time}"


def _list(dates: list[str], n: int = 4) -> str:
    shown = [f"{dt.date.fromisoformat(d).day} {dt.date.fromisoformat(d):%b %Y}" for d in dates[:n]]
    return ", ".join(shown) + (f" and {len(dates) - n} more" if len(dates) > n else "")


def _read_page(client: Client, url: str, today: dt.date) -> tuple[int | str, list[str] | None]:
    """(status, dates on the page). Sites that block robots (AMS, AGU) give 'blocked' and no dates."""
    try:
        r = client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; ml4wx-bot; +https://ml4wx.github.io)"})
    except httpx.HTTPStatusError as err:
        code = err.response.status_code
        return ("blocked" if code in (401, 403, 429) else code), None
    except Exception as err:
        log.warning("%s: not read (%s)", url, err)
        return "error", None
    if r.status_code in (401, 403, 429):
        return "blocked", None
    if r.status_code >= 400:
        return r.status_code, None
    return r.status_code, page_dates(r.text, today)


def _ccf_changes(name: str, old: dict | None, new: dict) -> list[str]:
    if not old:
        return []
    if old.get("year") != new["year"]:
        return [f"{name}: the next edition is listed, {new['date_text']} in {new['place']}"]
    before = {d["label"]: d for d in old.get("deadlines", [])}
    out = []
    for d in new["deadlines"]:
        b = before.get(d["label"])
        if not b:
            out.append(f"{name}: {d['label'].lower()} deadline announced, {_when(d)}")
        elif b["utc"] != d["utc"]:
            out.append(f"{name}: {d['label'].lower()} deadline moved from {_when(b)} to {_when(d)}")
    return out


def ics(editions: list[dict], today: dt.date) -> str:
    """An iCalendar file with the upcoming events (all-day) and their open deadlines."""
    def esc(s: str) -> str:
        return s.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")

    stamp = f"{today:%Y%m%d}T000000Z"
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//ML4WX//Events//EN", "CALSCALE:GREGORIAN",
             "X-WR-CALNAME:ML4WX events and deadlines"]  # fmt: skip
    for x in editions:
        if x["state"] == "past":
            continue
        if x["start"]:
            lines += ["BEGIN:VEVENT", f"UID:{x['slug']}-{x['start']:%Y%m%d}@ml4wx.github.io", f"DTSTAMP:{stamp}",
                      f"DTSTART;VALUE=DATE:{x['start']:%Y%m%d}",
                      f"DTEND;VALUE=DATE:{(x['end'] or x['start']) + dt.timedelta(days=1):%Y%m%d}",
                      f"SUMMARY:{esc(x['name'])}", f"LOCATION:{esc(x['location'])}", f"URL:{x['url']}",
                      "END:VEVENT"]  # fmt: skip
        for d in x["deadlines"]:
            if not d["open"]:
                continue
            t = dt.datetime.fromisoformat(d["utc"])
            summary = esc(f"Deadline: {x['name']}, {d['label']}")
            lines += ["BEGIN:VEVENT", f"UID:{x['slug']}-{_slug(d['label'])}-{t:%Y%m%d}@ml4wx.github.io",
                      f"DTSTAMP:{stamp}", f"DTSTART:{t:%Y%m%dT%H%M%SZ}", f"DTEND:{t:%Y%m%dT%H%M%SZ}",
                      f"SUMMARY:{summary}", f"URL:{x['url']}", "END:VEVENT"]  # fmt: skip
    lines.append("END:VCALENDAR")
    return "".join(_fold(line) + "\r\n" for line in lines)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _fold(line: str) -> str:
    """iCalendar lines are at most 75 octets; longer ones continue on lines that start with a space."""
    out, cur = [], ""
    for ch in line:
        if len((cur + ch).encode()) > 74:
            out.append(cur)
            cur = " "
        cur += ch
    return "\r\n".join(out + [cur])
