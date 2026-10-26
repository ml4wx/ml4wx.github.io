import datetime as dt
import json

import yaml
from conftest import REPO, make_paper

from ml4wx import resources
from ml4wx.config import load
from ml4wx.store import Archive

ITEMS = [
    {"slug": "graphcast", "name": "GraphCast", "kind": "model", "category": "global-deterministic",
     "description": "GNN weather model.", "provider": "Google DeepMind", "repo": "google-deepmind/weathernext",
     "weights": "https://example.org/w", "paper": {"arxiv": "2212.12794"}, "topics": ["global-medium-range"]},
    {"slug": "era5", "name": "ERA5", "kind": "dataset", "category": "reanalysis", "description": "Reanalysis.",
     "url": "https://example.org/era5", "access": "registration", "coverage": "global, 0.25°, hourly",
     "topics": ["global-medium-range", "downscaling"]},
]  # fmt: skip


def _setup(root, items=ITEMS):
    (root / "config" / "resources.yaml").write_text((REPO / "config" / "resources.yaml").read_text())
    f = root / "batch.yaml"
    f.write_text(yaml.safe_dump(items))
    return resources.import_list(f, resources.Catalog(root), today=dt.date(2026, 10, 26))


def test_import_validate_and_lookups(root):
    assert _setup(root) == (2, 0)
    assert _setup(root) == (0, 2)  # existing slugs are kept
    cat = resources.Catalog(root)
    assert (root / "resources" / "models" / "graphcast.yaml").exists()
    assert cat.validate(load(), resources.vocabulary()) == []
    assert [r.slug for r in cat.by_topic()["global-medium-range"]] == ["era5", "graphcast"]  # datasets first
    assert cat.items["dataset/era5"].get("coverage") == "global, 0.25°, hourly"
    links = cat.for_papers({"arxiv:2212.12794": "arxiv-2212.12794"})
    assert [r.slug for r in links["arxiv-2212.12794"]] == ["graphcast"]


def test_validate_catches_bad_entries(root):
    bad = [{"slug": "Bad_Slug", "name": "X", "kind": "software", "category": "nope", "description": "x",
            "repo": "not a repo", "topics": ["no-such-topic"]}]  # fmt: skip
    _setup(root, bad)
    errs = resources.Catalog(root).validate(load(), resources.vocabulary())
    assert any("kebab-case" in e for e in errs) and any("unknown category" in e for e in errs)
    assert any("unknown topic" in e for e in errs) and any("owner/name" in e for e in errs)


def test_releases_since_and_site(root):
    from ml4wx.site.build import build

    _setup(root)
    (root / "resources" / "_github.json").write_text(json.dumps({
        "google-deepmind/weathernext": {"stars": 6000, "pushed_at": "2026-10-20", "archived": False,
                                        "release": {"tag": "v1.2", "date": "2026-10-22", "url": "https://x/r"}},
    }))  # fmt: skip
    cat = resources.Catalog(root)
    assert [r["tag"] for r in cat.releases_since(dt.date(2026, 10, 19))] == ["v1.2"]
    assert cat.releases_since(dt.date(2026, 10, 23)) == []
    a = Archive()
    a.save_paper(make_paper())
    build(a, load(), root / "_site", index=False)
    page = (root / "_site" / "resources" / "index.html").read_text()
    assert 'id="model-graphcast"' in page and "★ 6,000" in page and 'href="/papers/arxiv-2212.12794/"' in page
    paper = (root / "_site" / "papers" / "arxiv-2212.12794" / "index.html").read_text()
    assert "Resources from this paper" in paper and "/resources/#model-graphcast" in paper
    topic = (root / "_site" / "topics" / "global-medium-range" / "index.html").read_text()
    assert "Datasets, models &amp; tools" in topic and "ERA5" in topic
