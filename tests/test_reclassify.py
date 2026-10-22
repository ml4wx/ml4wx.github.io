import argparse

from conftest import make_candidate, make_paper

from ml4wx import classify, cli
from ml4wx.models import Ids
from ml4wx.sources import arxiv, openalex
from ml4wx.store import Archive


def test_reclassify_keeps_removes_and_parks(root, monkeypatch):
    a = Archive()
    for i, title in enumerate(["Rain nowcasting net", "Data center cooling", "Ocean model emulator"]):
        a.save_paper(make_paper(id=f"arxiv-2501.0000{i}", ids=Ids(arxiv=f"2501.0000{i}"), title=title,
                                source="snowball", abstract=None))  # fmt: skip
    monkeypatch.setattr(
        arxiv, "lookup",
        lambda ids_, source="": iter([make_candidate(ids=Ids(arxiv=i), title="t", abstract="An abstract.")
                                      for i in ids_]),
    )  # fmt: skip
    monkeypatch.setattr(openalex, "lookup", lambda *a, **k: iter([]))
    base = {"topics": [], "type": "model", "method": [], "timescale": [], "spatial": [], "data": [],
            "status": "research", "summary": "New summary.", "significance": 3}  # fmt: skip
    decisions = {"arxiv:2501.00000": ("in", "nowcasting"), "arxiv:2501.00001": ("out", "none"),
                 "arxiv:2501.00002": ("later", "ocean-seaice")}  # fmt: skip
    monkeypatch.setattr(
        classify, "classify",
        lambda cands, cfg: {c.key: base | {"decision": decisions[c.key][0], "topic": decisions[c.key][1]}
                            for c in cands},
    )  # fmt: skip
    cli.cmd_reclassify(argparse.Namespace(source=["snowball"], all=False))
    b = Archive()
    assert list(b.papers) == ["arxiv-2501.00000"]
    p = b.papers["arxiv-2501.00000"]
    assert p.topic == "nowcasting" and p.summary == "New summary." and p.abstract == "An abstract."
    assert "arxiv:2501.00001" in b.rejected
    assert b.status(make_candidate(ids=Ids(arxiv="2501.00002"), title="Ocean model emulator")) == "parked"
