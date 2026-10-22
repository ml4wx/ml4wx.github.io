import datetime as dt
import json

import pytest
from conftest import make_candidate

from ml4wx import cli, fetch
from ml4wx.config import load
from ml4wx.models import Ids
from ml4wx.sources import arxiv, openalex

D = dt.date(2020, 1, 1)


def test_recent_survives_one_source_down(root, monkeypatch):
    def arxiv_down(*a, **k):
        raise RuntimeError("GET export.arxiv.org failed after 9 retries")

    monkeypatch.setattr(arxiv, "search", arxiv_down)
    monkeypatch.setattr(openalex, "search", lambda *a, **k: iter([make_candidate(ids=Ids(openalex="W1"))]))
    assert len(fetch.recent(load(), D, D, use_openalex=True)) == 4  # one per OpenAlex query
    with pytest.raises(fetch.SourceFailed):
        fetch.recent(load(), D, D, use_openalex=False)


def test_backfill_state_old_list_format(tmp_path):
    p = tmp_path / "backfill.json"
    p.write_text(json.dumps(["2018-01"]))
    assert cli._load_backfill_state(p) == {"2018-01": ["arxiv", "openalex"]}
    p.write_text(json.dumps({"2018-02": ["openalex"]}))
    assert cli._load_backfill_state(p) == {"2018-02": ["openalex"]}
