import datetime as dt
import json

from conftest import make_paper

from ml4wx import insights
from ml4wx.config import load
from ml4wx.models import Ids, Links
from ml4wx.site import charts
from ml4wx.store import Archive

TODAY = dt.date(2026, 10, 26)


def _papers():
    ps = []
    for i in range(30):  # 20 in the last 12 months, 10 in the 12 months before
        d = TODAY - dt.timedelta(days=30 + 10 * i) if i < 20 else TODAY - dt.timedelta(days=400 + 5 * i)
        ps.append(make_paper(id=f"arxiv-2501.{i:05d}", ids=Ids(arxiv=f"2501.{i:05d}"), date=d, year=d.year,
                             title=f"Diffusion ensemble paper {i}", method=["diffusion"] if i % 2 else ["cnn"],
                             links=Links(code="https://github.com/x/y") if i < 6 else Links(),
                             cited_by=i, type="review" if i == 3 else "model", highlight=i == 5))  # fmt: skip
    return ps


def test_topic_stats_growth_and_lists():
    st = insights.topic_stats(_papers(), TODAY)["global-medium-range"]
    assert (st.total, st.last12, st.prev12) == (30, 20, 10) and st.growth == 100
    assert [p.id for p in st.start_here] == ["arxiv-2501.00003"]
    assert st.most_cited_recent[0].cited_by == 19 and st.key_papers[0].id == "arxiv-2501.00005"
    assert st.per_year[2018] == 0 and st.per_year[2026] > 0


def test_essential_papers():
    rules = {"size": 2, "min_significance": 5, "landmark": 8, "pin": [], "exclude": []}

    def paper(id, age, **kw):  # age in years
        d = TODAY - dt.timedelta(days=round(365.25 * age))
        return make_paper(id=id, date=d, year=d.year, **kw)

    ps = [
        paper("a", 4, cited_by=400, highlight=True, significance=6),  # 100 per year
        paper("b", 0.2, cited_by=50, significance=8),  # landmark; the age counts as 1 year
        paper("c", 2, cited_by=60, highlight=True, significance=6),  # 30 per year
        paper("d", 4, cited_by=4000, classic=True, significance=3),  # below the floor
        paper("e", 4, cited_by=4000, significance=7),  # not a highlight, classic or landmark
        paper("f", 1, cited_by=0, significance=2),
    ]
    ids = lambda rules: [[p.id for p in ps] for ps in insights.essential_papers(ps, TODAY, rules)]  # noqa: E731
    assert ids(rules) == [["a", "b", "c"], []]
    assert ids(rules | {"exclude": ["a"]}) == [["b", "c"], []]
    assert ids(rules | {"pin": ["f"]}) == [["a", "f", "b", "c"], []]  # pinned: always in the top `size`
    assert ids(rules | {"since": TODAY.year - 2}) == [["b", "c"], ["a"]]  # older ones are listed apart


def test_trends_and_related():
    tr = insights.trends(_papers(), TODAY)
    assert tr.last12 == 20 and tr.prev12 == 10
    assert round(sum(tr.method_share["diffusion"][y] * tr.per_year[y] for y in tr.years) / 100) == 15
    radar = make_paper(id="arxiv-2502.00001", ids=Ids(arxiv="2502.00001"),
                       title="Radar nowcasting with optical flow", summary="Radar echoes extrapolated.")
    ps = _papers() + [radar]
    rel = insights.related(ps)
    assert "arxiv-2502.00001" not in rel["arxiv-2501.00001"]  # unrelated text is not "related"
    assert rel["arxiv-2501.00001"] and "arxiv-2501.00001" not in rel["arxiv-2501.00001"]


def test_charts_render():
    svg = charts.columns([("2025", 10), ("2026", 4)], partial_last=True, label="x")
    assert svg.startswith("<svg") and "viz-bar partial" in svg and "2026: 4 (so far)" in svg
    assert "<title>A: 50%</title>" in charts.hbars([("A", 50, None)], unit="%")
    assert "viz-line" in charts.mini_line({2024: 1, 2025: 3}, ymax=5) and "viz-dot" in charts.sparkline([1, 2, 3])
    assert charts.nice_ticks(23) == [0, 10, 20, 30]


def test_new_pages_build(root):
    from ml4wx.site.build import build

    a = Archive()
    for p in _papers():
        a.save_paper(p)
    build(a, load(), root / "_site", index=False)
    site = root / "_site"
    trends = (site / "trends" / "index.html").read_text()
    assert "Which methods papers use" in trends and "<svg class='viz" in trends and "Table view" in trends
    topic = (site / "topics" / "global-medium-range" / "index.html").read_text()
    assert "Start here" in topic and "Most cited, last 12 months" in topic and "+100%" in topic
    assert (site / "topics" / "global-medium-range" / "feed.xml").read_text().startswith("<?xml")
    paper = (site / "papers" / "arxiv-2501.00001" / "index.html").read_text()
    assert "Copy BibTeX" in paper and "@misc{" in paper and "Related papers" in paper
    assert (site / "models" / "compare" / "index.html").exists()
    home = (site / "index.html").read_text()
    assert "Fastest-growing topics" in home and 'class="ess-band"' in home and "data-quick" in home
    ess = (site / "essentials" / "index.html").read_text()
    assert 'id="ess-topic"' in ess and 'href="/papers/arxiv-2501.00005/"' in ess and '"arxiv-2501.00005": "@' in ess
    quick = json.loads((site / "data" / "quick.json").read_text())
    assert quick[0]["k"] == "paper" and quick[0]["u"].startswith("/papers/") and quick[0]["s"] == quick[0]["s"].lower()
    assert 'class="badge-ess"' in paper and "Essentials for this topic" in topic
