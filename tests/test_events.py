import datetime as dt
import json

import pytest

from ml4wx import events

TODAY = dt.date(2026, 11, 4)

CCF = """
- title: CVPR
  description: Computer Vision and Pattern Recognition
  confs:
    - year: 2026
      date: June 5-7, 2026
      place: Denver, USA
      timezone: UTC-12
      link: https://cvpr.thecvf.com/Conferences/2026
      timeline:
        - abstract_deadline: '2025-11-07 23:59:59'
          deadline: '2025-11-13 23:59:59'
    - year: 2027
      date: June 20-25, 2027
      place: Seattle, WA, USA
      timezone: UTC-12
      link: https://cvpr.thecvf.com/Conferences/2027
      timeline:
        - abstract_deadline: '2026-11-10 23:59:59'
          deadline: '2026-11-16 23:59:59'
          comment: Main track
        - deadline: '2027-01-15 23:59:59'
          abstract_deadline: '2027-01-15 23:59:59'
          comment: Workshops; on OpenReview
"""


def test_dates_time_zones_and_regions():
    assert events.utc("2027-01-12 13:00", "CET").isoformat() == "2027-01-12T12:00:00+00:00"
    assert events.utc("2026-08-29 23:59", "AoE").isoformat() == "2026-08-30T11:59:00+00:00"
    assert events.utc("2026-06-26").isoformat() == "2026-06-27T11:59:59+00:00"  # a date only: end of day, AoE
    assert events.utc("2026-05-07 11:59", "UTC+0") == events.utc("2026-05-07 13:59", "UTC+2")
    with pytest.raises(ValueError):
        events.utc("2026-01-01", "Mars time")
    assert events.date_range(dt.date(2026, 12, 7), dt.date(2026, 12, 11)) == "7–11 Dec 2026"
    assert events.date_range(dt.date(2026, 11, 12), dt.date(2027, 2, 4)) == "12 Nov 2026 – 4 Feb 2027"
    assert events.parse_dates_text("July 31 - August 6, 2027") == (dt.date(2027, 7, 31), dt.date(2027, 8, 6))
    assert events.parse_dates_text("TBD") == (None, None)
    assert events.regions("Fribourg, Switzerland") == ["switzerland", "europe"]
    assert events.regions("Vienna, Austria", online=True) == ["europe", "online"]
    assert events.regions("Denver, CO, USA") == ["elsewhere"]
    page = ("<p>Abstracts due <b>12 January 2027</b>, 13:00 CET; meeting 4–9 April 2027.</p>"
            "<script>var x = '1 January 2027';</script><p>© 2003-2026; founded March 3, 1990; 2027-03-08 early</p>")
    assert events.page_dates(page, TODAY) == ["2027-01-12", "2027-03-08", "2027-04-09"]  # not 1990, not scripts


def test_ccf_edition_picks_the_next_one():
    c = events.ccf_edition(CCF, TODAY)
    assert (c["title"], c["year"], c["start"], c["end"]) == ("CVPR", 2027, "2027-06-20", "2027-06-25")
    labels = ["Abstracts (Main track)", "Papers (Main track)", "Papers (Workshops)"]  # same-day abstracts left out
    assert [d["label"] for d in c["deadlines"]] == labels
    assert c["deadlines"][0]["tz"] == "AoE" and c["deadlines"][0]["utc"] == "2026-11-11T11:59:59+00:00"
    assert not c["tbd"]


@pytest.fixture
def event_root(root):
    d = root / "events"
    d.mkdir()
    (d / "cvpr.yaml").write_text("slug: cvpr\nname: CVPR\nkind: conference\nccf: conference/AI/cvpr.yml\n"
                                 "added: 2026-11-04\n")
    (d / "egu-2027.yaml").write_text(
        "slug: egu-2027\nname: EGU General Assembly 2027\nkind: conference\nstart: 2027-04-04\nend: 2027-04-09\n"
        "location: Vienna, Austria\nonline: true\nurl: https://egu27.eu/\n"
        "deadlines:\n- {label: Abstracts, date: '2027-01-12 13:00', tz: CET}\n"
        "- {label: Support applications, date: '2026-12-01 13:00', tz: CET, was: '2026-11-20 13:00'}\n"
        "added: 2026-11-06\n")  # fmt: skip
    (d / "sgm-2025.yaml").write_text(
        "slug: sgm-2025\nname: Swiss Geoscience Meeting 2025\nkind: conference\nstart: 2025-11-14\n"
        "location: Geneva, Switzerland\nurl: https://example.org/sgm25\nadded: 2026-11-04\n")
    (d / "ci.yaml").write_text(
        "slug: ci\nname: Climate Informatics\nkind: conference\nurl: https://ci.example.org/\n"
        "usual: Every spring.\nadded: 2026-11-04\n")
    return root


class Resp:
    def __init__(self, text, status=200):
        self.text, self.status_code = text, status

    def raise_for_status(self):
        pass


def _client(pages, ccf=CCF):
    calls = []

    class Client:
        def get(self, url, params=None, headers=None):
            calls.append(url)
            if url == events.CCF_RAW.format("conference/AI/cvpr.yml"):
                return Resp(ccf)
            return Resp(pages[url])

    return Client(), calls


def test_refresh_once_a_week_and_changes(event_root):
    cat = events.Catalog(event_root)
    client, calls = _client({"https://egu27.eu/": "<p>Abstracts: 12 January 2027</p>",
                             "https://ci.example.org/": "<p>Join our mailing list</p>"})  # fmt: skip
    assert cat.refresh(today=TODAY, client=client) == []  # the first check records what is there
    assert "https://example.org/sgm25" not in calls  # long over: not checked
    assert cat.refresh(today=TODAY + dt.timedelta(days=3), client=client) == [] and len(calls) == 3  # same week

    moved = CCF.replace("2026-11-16 23:59:59", "2026-11-18 23:59:59")
    client, calls = _client({"https://egu27.eu/": "<p>Abstracts: 15 January 2027 (extended)</p>",
                             "https://ci.example.org/": "<p>Climate Informatics 2027: 3-6 May 2027</p>"}, moved)
    changes = cat.refresh(today=TODAY + dt.timedelta(days=7), client=client)
    assert changes == [
        "Climate Informatics: dates on its page changed (new 6 May 2027), check the deadlines",
        "CVPR 2027: papers (main track) deadline moved from 16 Nov 2026, 23:59 AoE to 18 Nov 2026, 23:59 AoE",
        "EGU General Assembly 2027: dates on its page changed (new 15 Jan 2027; gone 12 Jan 2027), check the deadlines",
    ]
    status = json.loads((event_root / "events" / "_status.json").read_text())
    assert status["checked"] == "2026-11-11" and status["pages"]["https://ci.example.org/"]["changed"] == "2026-11-11"

    p = events.Catalog(event_root).page(TODAY + dt.timedelta(days=7))
    assert [(d["event"]["slug"], d["label"]) for d in p["deadlines"]] == [
        ("cvpr", "Abstracts (Main track)"), ("cvpr", "Papers (Main track)"), ("egu-2027", "Support applications"),
        ("egu-2027", "Abstracts"), ("cvpr", "Papers (Workshops)")]  # in time order; same-day abstracts left out
    assert [x["slug"] for x in p["upcoming"]] == ["egu-2027", "cvpr"]  # by start date
    assert [x["slug"] for x in p["past"]] == ["sgm-2025"] and p["series"][0]["changed"] == "2026-11-11"
    egu = p["upcoming"][0]
    assert egu["regions"] == ["europe", "online"] and egu["next"]["label"] == "Support applications"
    assert egu["new"] and not p["upcoming"][1]["new"]  # added after the first batch; CVPR came with it
    assert {"date": "2026-11-06", "text": "New on this page: EGU General Assembly 2027"} in p["changes"]
    assert egu["deadlines"][1]["was"] == "2026-11-20 13:00"

    cal = events.ics(events.Catalog(event_root).editions(TODAY), TODAY)
    assert cal.startswith("BEGIN:VCALENDAR\r\n") and "DTSTART;VALUE=DATE:20270404" in cal
    assert "DTEND;VALUE=DATE:20270410" in cal and "DTSTART:20270112T120000Z" in cal and "Swiss Geoscience" not in cal
    assert all(len(line.encode()) <= 75 for line in cal.split("\r\n"))


def test_validation(event_root):
    from ml4wx.config import load

    (event_root / "events" / "bad.yaml").write_text(
        "slug: bad\nname: Bad\nkind: conference\nstart: 2027-01-02\nend: 2027-01-01\nusual: also a series\n"
        "url: ftp://x\ndeadlines:\n- {label: Papers, date: '2026-12-01', tz: Mars}\nadded: 2026-11-04\n")
    (event_root / "events" / "typo.yaml").write_text(
        "slug: typo\nname: T\nkind: talk\nurl: https://x\nadded: 2026-11-04\n")
    errors = events.Catalog(event_root).validate(load())
    for msg in ("needs exactly one of start", "end before start", "url is not a URL", "unknown time zone 'Mars'"):
        assert any("bad.yaml: " in e and msg in e for e in errors), msg
    assert any("typo.yaml: cannot load" in e for e in errors)  # unknown kind


def test_benchmarks_are_copied_once_a_week(root):
    from ml4wx import benchmarks

    (root / "benchmarks").mkdir()
    (root / "benchmarks" / "_leaderboards.json").write_text(json.dumps({"checked": "2026-11-01", "benchmarks": {}}))

    class NoClient:
        def get(self, *a, **k):
            raise AssertionError("fetched although checked this week")

    assert benchmarks.Catalog(root).refresh(today=TODAY, client=NoClient()) == []


def test_events_page_and_digest(event_root):
    from conftest import make_paper

    from ml4wx import digest
    from ml4wx.config import load
    from ml4wx.site.build import build
    from ml4wx.store import Archive

    client, _ = _client({"https://egu27.eu/": "<p>12 January 2027</p>", "https://ci.example.org/": ""})
    cat = events.Catalog(event_root)
    cat.refresh(today=dt.date.today(), client=client, force=True)
    a = Archive()
    a.save_paper(make_paper())
    build(a, load(), event_root / "_site", index=False)
    html = (event_root / "_site" / "events" / "index.html").read_text()
    assert "Next deadlines" in html and 'id="egu-2027"' in html and "Series to watch" in html
    assert 'data-regions="europe online"' in html and "via ccf-deadlines" in html
    assert (event_root / "_site" / "events" / "events.ics").read_text().startswith("BEGIN:VCALENDAR")
    assert 'href="/events/"' in (event_root / "_site" / "index.html").read_text()
    quick = json.loads((event_root / "_site" / "data" / "quick.json").read_text())
    assert any(q["k"] == "event" and q["u"] == "/events/#egu-2027" for q in quick)

    _, md = digest.build(a, load(), since=TODAY, today=TODAY, write_intro=False,
                         events=events.Catalog(event_root).digest(dt.date(2026, 12, 20)))  # fmt: skip
    assert "## 📅 Events" in md and "EGU General Assembly 2027, abstracts" in md
