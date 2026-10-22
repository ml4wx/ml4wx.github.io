import datetime as dt

from conftest import make_candidate, make_paper

from ml4wx import export, ingest
from ml4wx.config import load
from ml4wx.models import Ids
from ml4wx.store import Archive


def test_archive_roundtrip_and_find(root):
    a = Archive()
    a.save_paper(make_paper())
    b = Archive()
    assert list(b.papers) == ["arxiv-2212.12794"]
    # Found by DOI only
    assert b.find(make_candidate(ids=Ids(doi="10.1126/science.adi2336"))) is not None
    # Found by changed journal title + same first author
    c = make_candidate(ids=Ids(openalex="W999"), title="Learning skillful medium-range global weather forecasting")
    assert b.find(c) is not None
    assert b.find(make_candidate(ids=Ids(arxiv="2301.00001"), title="Something else entirely about rain")) is None


def test_park_and_reject_persist(root):
    a = Archive()
    a.park(make_candidate(ids=Ids(arxiv="2301.00001"), title="Ocean model"), "ocean-seaice")
    a.reject(make_candidate(ids=Ids(arxiv="2301.00002"), title="Image deraining"))
    a.save_state()
    b = Archive()
    assert b.status(make_candidate(ids=Ids(arxiv="2301.00001"), title="Ocean model")) == "parked"
    assert b.status(make_candidate(ids=Ids(arxiv="2301.00002"), title="Image deraining")) == "rejected"


def test_dedupe_merges_arxiv_and_openalex(root):
    arx = make_candidate()
    oa = make_candidate(
        ids=Ids(arxiv="2212.12794", doi="10.1126/science.adi2336", openalex="W1"),
        venue="Science", venue_type="journal", abstract=None, abstract_license_ok=False,
        institutions=["ror:00971b260"],
    )  # fmt: skip
    (m,) = ingest.dedupe([arx, oa])
    assert m.ids.doi == "10.1126/science.adi2336" and m.ids.openalex == "W1"
    assert m.venue == "Science" and m.abstract_license_ok
    assert ingest.detect_orgs(m, load()) == ["google-deepmind"]


def test_prefilter(root):
    cfg = load()
    assert ingest.prefilter(make_candidate(), cfg)
    assert not ingest.prefilter(make_candidate(title="A transformer for protein folding", abstract="Proteins."), cfg)


def test_score_and_highlight(root):
    cfg = load()
    p = make_paper(venue="Science")
    ingest.finalize(p, cfg)
    assert p.score == 100 and p.highlight
    q = make_paper(orgs=[], links=make_paper().links.model_copy(update={"code": None}), significance=4)
    ingest.finalize(q, cfg)
    assert q.score == 16 and not q.highlight and q.highlight_reason is None


def test_exports(root):
    a = Archive()
    a.save_paper(make_paper(venue="Science", venue_type="journal"))
    a.save_paper(make_paper(id="arxiv-2301.00001", ids=Ids(arxiv="2301.00001"), title="Rain {nowcasting} & more",
                            topic="nowcasting", orgs=[], date=dt.date(2023, 1, 2), year=2023))  # fmt: skip
    written = export.write_all(a, load())
    bib = (root / "exports" / "papers.bib").read_text()
    assert "@article{lam2022graphcast," in bib and "\\{nowcasting\\} \\& more" in bib
    assert "| [Nowcasting](lists/nowcasting.md) | 1 | 0 |" in (root / "README.md").read_text()
    assert (root / "lists" / "global-medium-range.md") in written


def test_site_build(root):
    from ml4wx.site.build import build

    a = Archive()
    a.save_paper(make_paper(highlight=True, highlight_reason="Landmark model."))
    (root / "digests" / "2026-W43.md").write_text("---\nweek: 2026-W43\npapers: 1\n---\n# Digest\n\nHello.\n")
    build(a, load(), root / "_site", index=False)
    page = (root / "_site" / "papers" / "arxiv-2212.12794" / "index.html").read_text()
    assert 'data-pagefind-filter="topic">Global medium-range NWP<' in page
    assert "Landmark model." in page
    assert (root / "_site" / "topics" / "global-medium-range" / "index.html").exists()
    assert (root / "_site" / "orgs" / "google-deepmind" / "index.html").exists()
    assert "<h1>Digest</h1>" in (root / "_site" / "digests" / "2026-W43" / "index.html").read_text()
