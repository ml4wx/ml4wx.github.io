import datetime as dt
import shutil
from pathlib import Path

import pytest

from ml4wx import config
from ml4wx.models import Candidate, Ids, Links, Paper

REPO = Path(__file__).resolve().parents[1]

WATCHLIST = """
orgs:
  - slug: google-deepmind
    name: Google DeepMind
    tier: 1
    kind: company
    ror: [00971b260]
    aliases: [deepmind]
  - slug: eth-zurich
    name: ETH Zurich
    tier: 2
    kind: university
    openalex_ids: [I35440088]
    aliases: [eth zurich, eth zürich]
"""


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A throwaway repo root with the real config and a small watchlist."""
    shutil.copytree(REPO / "config", tmp_path / "config")
    (tmp_path / "config" / "watchlist.yaml").write_text(WATCHLIST)
    for d in ("papers", "parked", "state", "digests"):
        (tmp_path / d).mkdir()
    monkeypatch.setenv("ML4WX_ROOT", str(tmp_path))
    config.load.cache_clear()
    yield tmp_path
    config.load.cache_clear()


def make_candidate(**kw) -> Candidate:
    base = dict(
        ids=Ids(arxiv="2212.12794"),
        title="GraphCast: Learning skillful medium-range global weather forecasting",
        authors=["Remi Lam", "Alvaro Sanchez-Gonzalez"],
        date=dt.date(2022, 12, 24),
        abstract="We introduce a machine learning based weather forecasting method with a graph neural network.",
        abstract_license_ok=True,
        links=Links(paper="https://arxiv.org/abs/2212.12794"),
        source="test",
    )
    base.update(kw)
    return Candidate(**base)


def make_paper(**kw) -> Paper:
    base = dict(
        id="arxiv-2212.12794",
        title="GraphCast: Learning skillful medium-range global weather forecasting",
        authors=["Remi Lam", "Alvaro Sanchez-Gonzalez"],
        date=dt.date(2022, 12, 24),
        year=2022,
        ids=Ids(arxiv="2212.12794", doi="10.1126/science.adi2336"),
        links=Links(paper="https://arxiv.org/abs/2212.12794", code="https://github.com/google-deepmind/graphcast"),
        orgs=["google-deepmind"],
        topic="global-medium-range",
        type="model",
        method=["gnn"],
        timescale=["medium"],
        spatial=["global"],
        data=["era5"],
        summary="A graph neural network that beats HRES on 90% of 1380 targets.",
        significance=10,
        added=dt.date(2026, 10, 19),
    )
    base.update(kw)
    return Paper(**base)
