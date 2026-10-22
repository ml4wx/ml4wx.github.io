from ml4wx import ids


def test_norm_arxiv():
    assert ids.norm_arxiv("http://arxiv.org/abs/2212.12794v2") == "2212.12794"
    assert ids.norm_arxiv("https://doi.org/10.48550/arXiv.2212.12794") == "2212.12794"
    assert ids.norm_arxiv("physics/0601001v1") == "physics/0601001"
    assert ids.norm_arxiv("nothing here") is None


def test_norm_doi():
    assert ids.norm_doi("https://doi.org/10.1126/Science.ADI2336") == "10.1126/science.adi2336"
    assert ids.norm_doi("doi:10.1/x") == "10.1/x"
    assert ids.norm_doi("not a doi") is None


def test_paper_id_prefers_arxiv_then_doi():
    assert ids.paper_id("2212.12794", "10.1126/science.adi2336", None) == "arxiv-2212.12794"
    assert ids.paper_id(None, "10.1126/science.adi2336", "W1") == "doi-10.1126-science.adi2336"
    assert ids.paper_id(None, None, "W123") == "openalex-W123"


def test_same_title_tolerates_preprint_vs_journal():
    a = "GraphCast: Learning skillful medium-range global weather forecasting"
    b = "Learning skillful medium-range global weather forecasting"
    assert ids.same_title(a, b)
    pangu = "Pangu-Weather: A 3D high-resolution model for fast and accurate global weather forecast"
    assert not ids.same_title(a, pangu)
