import datetime as dt

from conftest import make_paper

from ml4wx import dedupe, ids
from ml4wx.config import load
from ml4wx.models import Ids, Links
from ml4wx.store import Archive


def test_title_matching_rules():
    assert ids.same_title("WeatherBench: A benchmark dataset", "WeatherBench: A Benchmark Data Set")
    assert ids.same_title("Could ML break the convection parametrization deadlock?",
                          "Could ML break the convection parameterization deadlock?")  # fmt: skip
    assert not ids.same_title("EPT-1.5 Technical Report", "EPT-2 Technical Report")
    assert ids.is_retracted_title("RETRACTED ARTICLE: Machine learning prediction of the MJO")
    assert ids.title_key("RETRACTED ARTICLE: Machine learning prediction") == ids.title_key(
        "Machine learning prediction"
    )


def test_dedupe_merges_preprint_server_copy_into_arxiv_record(root):
    a = Archive()
    a.save_paper(make_paper(
        id="arxiv-2002.00469", ids=Ids(arxiv="2002.00469"), venue=None, venue_type="preprint", orgs=[],
        title="WeatherBench: A benchmark dataset for data-driven weather forecasting",
    ))  # fmt: skip
    a.save_paper(make_paper(
        id="doi-10.1029-2020ms002203", title="WeatherBench: A Benchmark Data Set for Data-Driven Weather Forecasting",
        ids=Ids(doi="10.1029/2020ms002203"), venue="JAMES", venue_type="journal", orgs=["ecmwf"],
        links=Links(paper="https://doi.org/10.1029/2020ms002203"), cited_by=900, date=dt.date(2020, 11, 1),
    ))  # fmt: skip
    a.save_paper(make_paper(id="arxiv-2401.00001", ids=Ids(arxiv="2401.00001"), title="EPT-2 Technical Report"))
    a.save_paper(make_paper(id="arxiv-2301.00001", ids=Ids(arxiv="2301.00001"), title="EPT-1.5 Technical Report"))
    merged, retracted = dedupe.run(a, load())
    assert (merged, retracted) == (1, 0)
    b = Archive()
    assert sorted(b.papers) == ["arxiv-2002.00469", "arxiv-2301.00001", "arxiv-2401.00001"]
    p = b.papers["arxiv-2002.00469"]
    assert p.ids.doi == "10.1029/2020ms002203" and p.venue == "JAMES" and p.orgs == ["ecmwf"] and p.cited_by == 900


def test_dedupe_drops_retracted_work_with_its_copies(root):
    a = Archive()
    title = "Machine learning prediction of the Madden-Julian Oscillation"
    a.save_paper(make_paper(id="arxiv-2101.00001", ids=Ids(arxiv="2101.00001"), title=title))
    a.save_paper(make_paper(id="doi-10.1-x", ids=Ids(doi="10.1/x"), title=f"RETRACTED ARTICLE: {title}"))
    assert dedupe.run(a, load()) == (0, 2)
    b = Archive()
    assert not b.papers and "doi:10.1/x" in b.rejected
