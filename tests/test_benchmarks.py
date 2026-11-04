import datetime as dt
import json
import shutil

import numpy as np
import pytest
from conftest import REPO

from ml4wx import benchmarks

TODAY = dt.date(2026, 11, 4)


class Resp:
    def __init__(self, data=None, text="", content=b""):
        self._data, self.text, self.content = data, text, content
        self.status_code = 200

    def json(self):
        return self._data

    def raise_for_status(self):
        pass


def _owb(wn3_t2m_wins=29, extra_model=False):
    standings = [{"model": "wn3-ens-mean", "wins": wn3_t2m_wins, "mean": 1.1},
                 {"model": "hres", "wins": 30 - wn3_t2m_wins, "mean": 1.4}]  # fmt: skip
    if extra_model:
        standings.append({"model": "brand-new-model", "wins": 0, "mean": 1.5})
    mslp = [{"model": "hres", "wins": 20, "mean": 250.0}, {"model": "wn3-ens-mean", "wins": 10, "mean": 300.0}]
    return {"latestInit": "2026-10-26T12:00:00.000Z", "cycles": ["x"] * 30,
            "rankableVariables": ["2m_temperature", "mean_sea_level_pressure"],
            "variables": {"2m_temperature": {"standings": standings, "champions": ["wn3-ens-mean"]},
                          "mean_sea_level_pressure": {"standings": mslp, "champions": ["hres"]}}}  # fmt: skip


def _wb2_file(tmp_path, name, metric, z500):
    """A tiny WeatherBench 2 result file: global and tropics, 3 lead times, the 6 headline variables."""
    from scipy.io import netcdf_file

    path = tmp_path / name
    f = netcdf_file(path, "w")
    f.createDimension("region", 2)
    f.createDimension("string24", 24)
    f.createDimension("lead_time", 3)
    f.createDimension("level", 3)
    f.createVariable("region", "c", ("region", "string24"))[:] = np.array(
        [list(s.ljust(24, "\x00")) for s in ("global", "tropics")], dtype="S1")  # fmt: skip
    f.createVariable("lead_time", "i", ("lead_time",))[:] = [72, 120, 240]
    f.createVariable("level", "i", ("level",))[:] = [500, 700, 850]
    for var in ("geopotential", "temperature", "specific_humidity"):
        v = f.createVariable(f"{metric}.{var}", "d", ("region", "lead_time", "level"))
        v[:] = np.full((2, 3, 3), 0.001 if var == "specific_humidity" else 1.5)
    f.variables[f"{metric}.geopotential"][0, :, 0] = z500  # global, all leads, 500 hPa
    for var in ("2m_temperature", "10m_wind_speed", "mean_sea_level_pressure"):
        f.createVariable(f"{metric}.{var}", "d", ("region", "lead_time"))[:] = np.full((2, 3), 1.5)
    f.close()
    return path.read_bytes()


@pytest.fixture
def bench_root(root):
    shutil.copytree(REPO / "benchmarks", root / "benchmarks", ignore=shutil.ignore_patterns("_*.json"))
    return root


@pytest.fixture
def sources(tmp_path):
    files = {
        "graphcast_vs_era5_240x121_2020.nc": _wb2_file(tmp_path, "g.nc", "rmse", [120, 270, 700]),
        "hres_vs_era5_240x121_2020.nc": _wb2_file(tmp_path, "h.nc", "rmse", [130, 300, 800]),
        "climatology_vs_era5_240x121_2020.nc": _wb2_file(tmp_path, "c.nc", "rmse", [800, 800, 800]),
        "gencast_mean_vs_era5_240x121_2020.nc": _wb2_file(tmp_path, "gm.nc", "rmse", [110, 250, 600]),
        "gencast_single_member_vs_era5_240x121_2020.nc": b"never downloaded",
        "gencast_vs_era5_240x121_2020.nc": _wb2_file(tmp_path, "gp.nc", "crps", [50, 100, 300]),
        "ens_vs_era5_240x121_2020.nc": _wb2_file(tmp_path, "e.nc", "crps", [55, 110, 320]),
    }
    traces = [{"name": f"{m} vs. ERA5", "customdata": [metric], "x": list(range(1, 45)), "y": [v] * 44}
              for m, v in [("climatology", 80.0), ("graphcast", 108.0), ("ecmwf", 112.0)]
              for metric in ("rmse-z-500", "acc-z-500")]  # fmt: skip
    # plotly.js itself also contains "Plotly.newPlot(", so the data is in the last call.
    chaos = f'<script>Plotly.newPlot(gd, data);</script><script>Plotly.newPlot( "a1", {json.dumps(traces)}, {{}})'
    readme = ("# ClimateBench\n\n## Leaderboard\n\nText.\n\n"
              "| | ('tas', 'Total') | ('pr', 'Total') | ('pr90', 'Total') | ('diurnal_temperature_range', 'Total') |\n"
              "|-|-|-|-|-|\n| Neural Network | 0.3 | 3.1 | 4.3 | 16.7 |\n"
              "| Random Forest | 0.4 | 5.0 | 5.4 | 22.4 |\n")  # fmt: skip
    team = lambda name, models: {"Teamname": name, "Models": [  # noqa: E731
        {"Model Name": m, "Average RPSS": v, "tas_rpss": v, "mslp_rpss": v, "pr_rpss": None} for m, v in models]}
    periods = [("JJA 2026", 9, "Fri 04-Sep-2026 00:00:00 UTC"), ("SON 2026", 6, "Fri 23-Oct-2026 00:00:00 UTC")]
    aiwq = [{"Competition period": p, "Competition week": f"Competition week {w} (Thu 1-Oct-2026)", "week_index": w,
             "Forecast window": win, "Variable": "average", "RPSS type": "period", "Publication time": pub,
             "Teams": [team("Arcana", [("Spherical", 0.24 + w / 100)]),
                       team("Dynamical_S2SDatabase", [("ECMWF", 0.2)])]}
            for p, w, pub in periods for win in (1, 2)]  # fmt: skip
    weatherreal = ("# WeatherReal\n\n## Leaderboards\n\n### [Provisional] Global medium-range weather forecasting\n\n"
                   "| **WeatherReal-ISD** | 2-m temperature <br> (RMSE, K) | 6-hour precipitation > 1 mm <br> (ETS) |\n"
                   "|---|---|---|\n| Microsoft-Point | **2.258** | - |\n| ECMWF | 2.766 | **0.248** |\n\n"
                   "## About the data\n\n| not | a leaderboard |\n|---|---|\n| a | b |\n")  # fmt: skip
    nums = lambda x: "".join(f"<td><strong>{x + i}<strong></td>" for i in range(12))  # noqa: E731
    weather5k = ("<table><tr><th>Baselines</th><th>Lead Time</th><th>Temperature</th><th>Dewpoint</th>"
                 "<th>Wind Speed</th><th>Wind Direction</th><th>Sea Level Pressure</th><th>Overall</th></tr>"
                 "<tr><th></th><th></th>" + "<th>MAE</th><th>MSE</th>" * 6 + "</tr>"
                 f'<tr><td rowspan="2">🥇 1st Pyraformer</td><td>24</td>{nums(1)}</tr><tr><td>72</td>{nums(2)}</tr>'
                 f'<tr><td rowspan="2">iTransformer</td><td>24</td>{nums(1.5)}</tr><tr><td>72</td>{nums(1.8)}</tr>'
                 "</table>")  # fmt: skip
    head = "model,variable,metric,region,Historical (2005-2014),SSP2-4.5,Change (hist 2005),Percent Change (hist 2005)"
    scores = [("ACCESS-CM2", 1.0, 2.0), ("CanESM5", 2.0, 1.0), ("MIROC6", 1.5, 1.5)]  # model, 2005-2014, 2015-2024
    cb2 = "\n".join([head] + [f"{m},{v},{metric},{region},{hist},{test},0,0"
                               for m, hist, test in scores
                               for v in ("tas", "pr", "tos", "clt", "od550aer")
                               for metric, region in [("zonal_mean_rmse", "global"), ("zonal_mean_mae", "global"),
                                                      ("zonal_mean_rmse", "tropics")]])  # fmt: skip
    state = {"owb": _owb(), "fail": set()}

    class Client:
        def get(self, url, params=None, headers=None):
            for key in state["fail"]:
                if key in url:
                    raise RuntimeError(f"{key} is down")
            if url == benchmarks.OWB_API:
                return Resp(state["owb"])
            if url == benchmarks.WB2_LIST:
                return Resp({"items": [{"name": f"benchmark_results/{n}", "updated": "2026-10-30T00:00:00Z"}
                                       for n in files]})  # fmt: skip
            if url.startswith(benchmarks.WB2_FILE.format("")):
                name = url.rsplit("/", 1)[1]
                assert "single_member" not in name  # variants we do not show are not downloaded
                return Resp(content=files[name])
            if url == benchmarks.AIWQ_JSON:
                return Resp(aiwq)
            if url == benchmarks.CHAOS_HTML:
                return Resp(text=chaos)
            if url == benchmarks.CLIMATEBENCH_README:
                return Resp(text=readme)
            if url == benchmarks.WEATHERREAL_README:
                return Resp(text=weatherreal)
            if url == benchmarks.WEATHER5K_README:
                return Resp(text=f"# WEATHER-5K\n\n{weather5k}\n")
            if url == benchmarks.CLIMATEBENCH2_CSV:
                return Resp(text=cb2)
            if url.startswith("https://api.github.com/repos/") and url.endswith("/commits"):
                return Resp([{"commit": {"committer": {"date": "2026-01-30T10:00:00Z"}}}])
            raise AssertionError(url)

    return Client(), state


def test_refresh_reads_every_source_and_logs_changes(bench_root, sources):
    client, state = sources
    cat = benchmarks.Catalog(bench_root)
    assert cat.refresh(today=TODAY, client=client) == []  # the first refresh records what exists, no changes
    snap = cat.snapshot["benchmarks"]
    assert set(snap) == {"operational-weatherbench", "weatherbench2", "ai-weather-quest", "chaosbench", "climatebench",
                         "weatherreal", "weather-5k", "climatebench-2"}  # fmt: skip

    owb = snap["operational-weatherbench"]["boards"][0]
    wn3 = next(r for r in owb["rows"] if r["id"] == "wn3-ens-mean")
    assert (owb["id"], owb["leader"], owb["as_of"]) == ("global", "wn3-ens-mean", "2026-10-26")
    assert wn3["values"] == {"wins": 39, "led": 1, "2m_temperature": 1.1, "mean_sea_level_pressure": 3.0}  # hPa

    det, prob = snap["weatherbench2"]["boards"]
    rows = {r["id"]: r["values"] for r in det["rows"]}
    assert set(rows) == {"graphcast", "hres", "climatology", "gencast_mean"}
    assert rows["graphcast"]["z500_d5"] == 270 and rows["graphcast"]["t2m_d5"] == 1.5
    assert rows["hres"]["skill"] == 0 and rows["graphcast"]["skill"] < 0  # better than HRES
    # Ensemble means and baselines are shown but never lead the deterministic board.
    assert det["leader"] == "graphcast" and prob["leader"] == "gencast" and prob["id"] == "prob-2020"

    w1 = snap["ai-weather-quest"]["boards"][0]
    assert w1["title"].startswith("SON 2026") and w1["leader"] == "Arcana/Spherical"  # latest period only
    spherical = next(r for r in w1["rows"] if r["id"] == "Arcana/Spherical")["values"]
    assert spherical == pytest.approx({"avg": 0.3, "tas": 0.3, "mslp": 0.3})  # no precipitation score
    assert snap["chaosbench"]["boards"][0]["leader"] == "graphcast"  # climatology is a baseline
    cb = snap["climatebench"]["boards"][0]
    assert cb["leader"] == "Neural Network" and cb["rows"][0]["values"]["mean"] == pytest.approx(6.1)
    assert cb["as_of"] == "2026-01-30"  # the date of the last commit of the source file

    (wr,) = snap["weatherreal"]["boards"]  # the table under "## About the data" is not a leaderboard
    assert (wr["id"], wr["leader"]) == ("weatherreal-isd", "Microsoft-Point")
    assert [(c["key"], c["unit"], c["better"]) for c in wr["columns"]] == [
        ("2-m-temperature", "K", "lower"), ("6-hour-precipitation-1-mm", "", "higher")]  # ETS: higher is better
    assert wr["rows"][0]["values"] == {"2-m-temperature": 2.258}  # "-" is no value
    w5k = snap["weather-5k"]["boards"]
    assert [b["id"] for b in w5k] == ["lead-24", "lead-72"] and w5k[0]["leader"] == "Pyraformer"
    assert w5k[1]["rows"][1]["values"] == {"temperature": 1.8, "dewpoint": 3.8, "wind_speed": 5.8,
                                           "wind_direction": 7.8, "slp": 9.8, "overall": 11.8}  # MAE columns
    test, hist = snap["climatebench-2"]["boards"]
    assert (test["leader"], hist["leader"]) == ("CanESM5", "ACCESS-CM2")  # global RMSE; MAE and tropics ignored
    assert next(r for r in test["rows"] if r["id"] == "MIROC6")["values"]["skill"] == 0  # the median model

    # A week later: a new model, a new leader, and one source down (its last copy is kept).
    aiwq_before = snap["ai-weather-quest"]["boards"]
    state["owb"] = _owb(wn3_t2m_wins=5, extra_model=True)
    state["fail"] = {"aiweatherquest"}
    changes = cat.refresh(today=TODAY + dt.timedelta(days=7), client=client)
    assert "Brand new model is new on Operational WeatherBench (Global)" in changes
    assert "IFS HRES leads Operational WeatherBench (Global), ahead of WeatherNext 3 (ens. mean)" in changes
    snap = cat.snapshot
    assert snap["benchmarks"]["ai-weather-quest"]["error"].startswith("RuntimeError")
    assert snap["benchmarks"]["ai-weather-quest"]["boards"] == aiwq_before
    assert snap["changes"][0]["date"] == "2026-11-11"

    week = TODAY + dt.timedelta(days=7)
    page = benchmarks.page(benchmarks.Catalog(bench_root), {"graphcast": "/models/#graphcast"}, week)
    wb2 = next(x for x in page if x["b"].slug == "weatherbench2")
    det = wb2["boards"][0]
    assert [x["id"] for x in wb2["cards"]] == ["det-2020", "prob-2020"]
    gc = next(r for r in det["rows"] if r["id"] == "graphcast")
    assert gc["link"] == "/models/#graphcast" and gc["ranked"] and gc["name"] == "GraphCast"
    assert not next(r for r in det["rows"] if r["id"] == "gencast_mean")["ranked"]
    assert det["best"]["z500_d5"] == 250 and det["leader_row"]["id"] == "graphcast"
    chaos = next(x for x in page if x["b"].slug == "chaosbench")["boards"][0]
    assert chaos["baseline_row"]["name"] == "Climatology"  # climatology beats every model at weeks 3-4
    owb = next(x for x in page if x["b"].slug == "operational-weatherbench")["boards"][0]
    assert next(r for r in owb["rows"] if r["id"] == "brand-new-model")["new"]
    cb2 = next(x for x in page if x["b"].slug == "climatebench-2")["boards"][0]
    assert {r["kind"] for r in cb2["rows"]} == {"physics"} and cb2["rows"][0]["name"] == "CanESM5"  # default kind


def test_catalog_validation(bench_root):
    from ml4wx.config import load

    (bench_root / "benchmarks" / "bad.yaml").write_text(
        "slug: bad\nname: Bad\ntopics: [nowcasting]\ntask: x\nurl: https://x.org\n"
        "leaderboard: {source: owb, url: https://x.org, models: {m: {name: A (b, c)}}}\n"
    )  # the unquoted comma splits the name: an unknown key, so the file is rejected
    (bench_root / "benchmarks" / "wrong.yaml").write_text(
        "slug: wrong\nname: Wrong\ntopics: [no-such-topic]\ntask: x\nurl: ftp://x\ndataset: nope\n"
        "leaderboard: {source: nope, url: https://x.org, models: {m: {name: M, model: nope}}}\n"
    )
    cat = benchmarks.Catalog(bench_root)
    errors = cat.validate(load(), {"graphcast"}, {"weatherbench2"})
    assert any("bad.yaml: cannot load" in e for e in errors)
    for msg in ("unknown topic 'no-such-topic'", "url is not a URL", "unknown dataset 'nope'",
                "unknown leaderboard source 'nope'", "unknown model 'nope'"):  # fmt: skip
        assert any(f"wrong.yaml: {msg}" in e for e in errors), msg


def test_page_search_and_digest(bench_root, sources):
    from conftest import make_paper

    from ml4wx import digest
    from ml4wx.config import load
    from ml4wx.site.build import build
    from ml4wx.store import Archive

    client, state = sources
    cat = benchmarks.Catalog(bench_root)
    cat.refresh(today=TODAY, client=client)
    state["owb"] = _owb(wn3_t2m_wins=5, extra_model=True)
    cat.refresh(today=TODAY, client=client)
    a = Archive()
    a.save_paper(make_paper(type="benchmark", title="NewBench: a benchmark", added=TODAY, date=TODAY, year=2026))
    build(a, load(), bench_root / "_site", index=False)
    html = (bench_root / "_site" / "benchmarks" / "index.html").read_text()
    assert "Who is winning" in html and "WeatherNext 3 (ens. mean)" in html and 'id="lb-weatherbench2"' in html
    assert "Climatology does better" in html and "NewBench: a benchmark" in html and "is new on" in html
    assert 'id="tcbench"' in html  # catalog entries without a leaderboard are listed too
    assert 'href="/benchmarks/"' in (bench_root / "_site" / "index.html").read_text()
    quick = json.loads((bench_root / "_site" / "data" / "quick.json").read_text())
    assert {"n": "WeatherBench 2", "k": "benchmark", "u": "/benchmarks/#weatherbench2"}.items() <= next(
        q for q in quick if q["n"] == "WeatherBench 2" and q["k"] == "benchmark").items()

    _, md = digest.build(a, load(), since=TODAY, today=TODAY, write_intro=False,
                         leaderboard_changes=cat.snapshot["changes"][:2])  # fmt: skip
    assert "## 🏁 Leaderboards" in md and "is new on Operational WeatherBench" in md
