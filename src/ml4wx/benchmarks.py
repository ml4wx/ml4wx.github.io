"""Benchmarks catalog and leaderboards, copied from their official sources.

Curated by hand: one YAML file per benchmark in benchmarks/<slug>.yaml.
Generated weekly: benchmarks/_leaderboards.json (`ml4wx benchmarks refresh`), the leaderboards of the
benchmarks that publish their scores in a machine-readable form. We never run models ourselves.

A board is one ranked table: {id, title, note, as_of, columns, sort, card, rows}. A column is
{key, label, unit, better: lower|higher, digits, help}; a row is {id, name?, values: {key: number}}.
"""

from __future__ import annotations

import csv
import datetime as dt
import html
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict

from .config import Config, repo_root
from .http import Client
from .resources import PaperRef

log = logging.getLogger(__name__)

LEADERBOARDS_FILE = "_leaderboards.json"
NEW_DAYS = 28  # a model is "new" on a leaderboard this many days after it first appears
MAX_CHANGES = 40


class ModelInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")  # catches typos and unquoted commas in YAML flow mappings

    name: str
    model: str | None = None  # slug in resources/models
    kind: Literal["ml", "physics", "baseline"] = "ml"
    contender: bool = True  # False: shown, but never the leader (e.g. ensemble means on a deterministic board)
    note: str | None = None


class Leaderboard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str  # key of SOURCES
    url: str  # the official leaderboard page
    license: str = ""
    cards: list[str] = []  # board ids shown under "Who is winning" (default: the first board)
    default_kind: Literal["ml", "physics", "baseline"] = "ml"  # kind of the models not listed below
    models: dict[str, ModelInfo] = {}  # source model id -> display name, catalog link, kind
    exclude: list[str] = []  # regular expressions of source model ids to leave out


class Benchmark(BaseModel):
    model_config = ConfigDict(extra="forbid")

    slug: str
    name: str
    provider: str = ""
    topics: list[str]
    task: str
    truth: str = ""
    metrics: list[str] = []
    url: str = ""
    docs: str = ""
    repo: str = ""
    paper: PaperRef | None = None
    dataset: str | None = None  # slug in resources/datasets
    status: Literal["live", "active", "archived"] = "active"
    leaderboard: Leaderboard | None = None
    rank: int | None = None
    added: dt.date | None = None


class Catalog:
    def __init__(self, root: Path | None = None):
        self.root = (root or repo_root()) / "benchmarks"
        self.items: dict[str, Benchmark] = {}
        self.invalid: list[tuple[Path, str]] = []
        for f in sorted(self.root.glob("*.yaml")):
            try:
                b = Benchmark.model_validate(yaml.safe_load(f.read_text(encoding="utf-8")))
            except (yaml.YAMLError, ValueError) as e:
                self.invalid.append((f, str(e).splitlines()[0]))
                continue
            self.items[b.slug] = b
        snap = self.root / LEADERBOARDS_FILE
        self.snapshot: dict = json.loads(snap.read_text()) if snap.exists() else {}

    def ordered(self) -> list[Benchmark]:
        return sorted(self.items.values(), key=lambda b: (b.rank or 10_000, b.name.lower()))

    def validate(self, cfg: Config, model_slugs: set[str], dataset_slugs: set[str]) -> list[str]:
        errors = [f"benchmarks/{p.name}: cannot load ({m})" for p, m in self.invalid]
        for b in self.items.values():
            where = f"benchmarks/{b.slug}.yaml"
            if not (self.root / f"{b.slug}.yaml").exists():
                errors.append(f"{where}: file name does not match slug")
            errors += [f"{where}: unknown topic '{t}'" for t in b.topics if t not in cfg.enabled_slugs]
            if b.dataset and b.dataset not in dataset_slugs:
                errors.append(f"{where}: unknown dataset '{b.dataset}'")
            for field in ("url", "docs"):
                v = getattr(b, field)
                if v and not v.startswith(("http://", "https://")):
                    errors.append(f"{where}: {field} is not a URL")
            if not (b.url or b.repo):
                errors.append(f"{where}: needs url or repo")
            lb = b.leaderboard
            if lb:
                if lb.source not in SOURCES:
                    errors.append(f"{where}: unknown leaderboard source '{lb.source}'")
                errors += [f"{where}: unknown model '{m.model}'" for m in lb.models.values()
                           if m.model and m.model not in model_slugs]  # fmt: skip
        return errors

    def refresh(self, today: dt.date | None = None, only: str | None = None, client: Client | None = None) -> list[str]:
        """Fetch every leaderboard, keep the last good one when a source fails. Returns this run's changes."""
        today = today or dt.date.today()
        client = client or Client(min_interval=0.5, max_retries=3)
        old = self.snapshot
        boards_by = dict(old.get("benchmarks", {}))
        seen: dict[str, dict[str, str]] = {k: dict(v) for k, v in old.get("seen", {}).items()}
        changes: list[dict] = []
        for b in self.ordered():
            if not b.leaderboard or (only and b.slug != only):
                continue
            try:
                boards = SOURCES[b.leaderboard.source](client)
                if not any(board["rows"] for board in boards):
                    raise ValueError("no scores found")
            except Exception as e:  # one source failing must not stop the others
                log.warning("%s: leaderboard not updated (%s)", b.slug, e)
                boards_by.setdefault(b.slug, {"boards": []})["error"] = f"{type(e).__name__}: {e}"[:300]
                continue
            drop = [re.compile(p) for p in b.leaderboard.exclude]
            for board in boards:
                board["rows"] = [r for r in board["rows"] if not any(p.search(r["id"]) for p in drop)]
                board["leader"] = leader(board, b.leaderboard.models)
            changes += _changes(b, boards_by.get(b.slug, {}).get("boards", []), boards, seen, today)
            boards_by[b.slug] = {"fetched": today.isoformat(), "boards": boards, "error": None}
            log.info("%s: %d boards", b.slug, len(boards))
        log_ = (changes + old.get("changes", []))[:MAX_CHANGES]
        out = {"benchmarks": boards_by, "seen": seen, "changes": log_}
        (self.root / LEADERBOARDS_FILE).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
        self.snapshot = out
        return [c["text"] for c in changes]


def display_name(row: dict, models: dict[str, ModelInfo]) -> str:
    if row["id"] in models:
        return models[row["id"]].name
    if row.get("name"):
        return row["name"]
    if row["id"] != row["id"].lower():  # already a name ("iTransformer", "FIO-ESM-2-0")
        return row["id"]
    words = re.sub(r"[_-]+", " ", row["id"]).strip()  # a machine id ("astra_mean")
    return words[:1].upper() + words[1:]


def leader(board: dict, models: dict[str, ModelInfo]) -> str | None:
    """Id of the best row in the board's sort column. Baselines and non-contenders never lead."""
    col = next(c for c in board["columns"] if c["key"] == board["sort"])
    info = {k: m for k, m in models.items() if m.kind == "baseline" or not m.contender}
    rows = [r for r in board["rows"] if r["values"].get(col["key"]) is not None and r["id"] not in info]
    if not rows:
        return None
    sign = -1 if col["better"] == "higher" else 1
    return min(rows, key=lambda r: sign * r["values"][col["key"]])["id"]


def _changes(b: Benchmark, old: list[dict], new: list[dict], seen: dict, today: dt.date) -> list[dict]:
    """New models and new leaders since the last refresh. The first refresh records everything as known."""
    models = b.leaderboard.models
    first = b.slug not in seen
    known = seen.setdefault(b.slug, {})
    out = []
    old_by = {x["id"]: x for x in old}
    for board in new:
        for r in board["rows"]:
            if r["id"] not in known:
                known[r["id"]] = "initial" if first else today.isoformat()
                if not first:
                    out.append(f"{display_name(r, models)} is new on {b.name} ({board['title']})")
        prev = old_by.get(board["id"], {}).get("leader")
        if prev and board["leader"] and prev != board["leader"]:
            names = {r["id"]: display_name(r, models) for r in board["rows"]}
            out.append(f"{names.get(board['leader'])} leads {b.name} ({board['title']}), "
                       f"ahead of {names.get(prev, display_name({'id': prev}, models))}")  # fmt: skip
    return [{"date": today.isoformat(), "benchmark": b.slug, "text": t} for t in dict.fromkeys(out)]


def page(cat: Catalog, model_pages: dict[str, str], today: dt.date) -> list[dict]:
    """Leaderboards for the site: names, catalog links, best value per column, "new" flags."""
    out = []
    snap = cat.snapshot.get("benchmarks", {})
    seen = cat.snapshot.get("seen", {})
    for b in cat.ordered():
        data = snap.get(b.slug)
        if not b.leaderboard or not data or not data.get("boards"):
            continue
        models = b.leaderboard.models
        boards = []
        for board in data["boards"]:
            best = {}
            for c in board["columns"]:
                vals = [r["values"][c["key"]] for r in board["rows"] if r["values"].get(c["key"]) is not None]
                if vals:
                    best[c["key"]] = max(vals) if c["better"] == "higher" else min(vals)
            rows = []
            for r in board["rows"]:
                info = models.get(r["id"])
                since = seen.get(b.slug, {}).get(r["id"], "initial")
                new = since != "initial" and (today - dt.date.fromisoformat(since)).days <= NEW_DAYS
                kind = info.kind if info else b.leaderboard.default_kind
                ranked = kind != "baseline" and (info.contender if info else True)
                rows.append({**r, "name": display_name(r, models), "kind": kind,
                             "note": info.note if info else None, "new": new, "ranked": ranked,
                             "link": model_pages.get(info.model) if info and info.model else None})  # fmt: skip
            col = next(c for c in board["columns"] if c["key"] == board["sort"])
            sign = -1 if col["better"] == "higher" else 1
            rows.sort(key=lambda r: (r["values"].get(col["key"]) is None, sign * (r["values"].get(col["key"]) or 0)))
            lead = next((r for r in rows if r["id"] == leader(board, models)), None)
            # A baseline that beats every model (climatology at weeks 3-4) is worth saying next to the leader.
            base = next((r for r in rows if r["kind"] == "baseline" and r["values"].get(col["key"]) is not None), None)
            if not (lead and base and sign * base["values"][col["key"]] < sign * lead["values"][col["key"]]):
                base = None
            boards.append({**board, "rows": rows, "best": best, "leader_row": lead, "baseline_row": base})
        cards = b.leaderboard.cards or [boards[0]["id"]]
        out.append({"b": b, "boards": boards, "fetched": data.get("fetched"), "error": data.get("error"),
                    "cards": [x for x in boards if x["id"] in cards and x["leader_row"]]})  # fmt: skip
    return out


# --- sources ------------------------------------------------------------------------------------------
# Each source returns a list of boards. Values are converted to display units (hPa, g/kg).


def _col(key: str, label: str, unit: str = "", better: str = "lower", digits: int = 2, help: str = "") -> dict:
    return {"key": key, "label": label, "unit": unit, "better": better, "digits": digits, "help": help}


def _plain(cell: str) -> str:
    """Text of a Markdown or HTML table cell: no tags, bold markers or extra spaces."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", cell)).replace("**", "").split())


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _number(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _md_tables(text: str) -> list[list[list[str]]]:
    """The Markdown tables in a text, as rows of cells; the |---| separator rows are dropped."""
    tables, rows = [], []
    for line in text.splitlines() + [""]:
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                rows.append(cells)
        elif rows:
            tables.append(rows)
            rows = []
    return tables


def _last_commit(client: Client, repo: str, path: str) -> str:
    """Date of the last commit that changed a file, so a page can say how fresh a copied table is."""
    token = os.environ.get("GITHUB_TOKEN")
    try:
        r = client.get(f"https://api.github.com/repos/{repo}/commits", params={"path": path, "per_page": 1},
                       headers={"Authorization": f"Bearer {token}"} if token else None)  # fmt: skip
        return r.json()[0]["commit"]["committer"]["date"][:10] if r.status_code == 200 and r.json() else ""
    except Exception as e:  # the date is a nice-to-have
        log.warning("no commit date for %s/%s (%s)", repo, path, e)
        return ""


OWB_API = "https://owb.brightband.com/api/rankings"
OWB_REGIONS = {
    "global": "Global", "northern-hemisphere": "N. Hemisphere", "tropics": "Tropics",
    "southern-hemisphere": "S. Hemisphere", "europe": "Europe", "north-america": "North America",
}  # fmt: skip
# variable -> (label, unit, scale, digits)
OWB_VARS = {
    "geopotential_500": ("Z500", "m²/s²", 1, 1),
    "temperature_850": ("T850", "K", 1, 2),
    "2m_temperature": ("T2m", "K", 1, 2),
    "10m_wind_speed": ("10 m wind", "m/s", 1, 2),
    "wind_vector_850": ("Wind 850", "m/s", 1, 2),
    "mean_sea_level_pressure": ("MSLP", "hPa", 0.01, 2),
    "specific_humidity_700": ("Q700", "g/kg", 1000, 3),
}


def owb(client: Client) -> list[dict]:
    """Operational WeatherBench (Brightband): the dashboard's own rankings, last 30 daily runs."""
    boards = []
    for region, title in OWB_REGIONS.items():
        r = client.get(OWB_API, params={"region": region})
        r.raise_for_status()
        d = r.json()
        names = [v for v in d["rankableVariables"] if v in d["variables"] and v in OWB_VARS]
        runs = len(d.get("cycles") or [])
        rows: dict[str, dict] = {}
        for v in names:
            x = d["variables"][v]
            label, unit, scale, _ = OWB_VARS[v]
            for s in x["standings"]:
                row = rows.setdefault(s["model"], {"id": s["model"], "values": {"wins": 0, "led": 0}})
                row["values"]["wins"] += s["wins"]
                row["values"][v] = round(s["mean"] * scale, 5)
            for m in x.get("champions") or []:
                if m in rows:
                    rows[m]["values"]["led"] += 1
        columns = [
            _col("wins", "Days won", "", "higher", 0,
                 f"Runs in which the model was #1, summed over the {len(names)} variables ({runs} runs each)"),
            _col("led", "Variables led", "", "higher", 0, "Variables where the model is the 30-day champion"),
        ] + [_col(v, OWB_VARS[v][0], OWB_VARS[v][1], "lower", OWB_VARS[v][3], "RMSE") for v in names]
        latest = (d.get("latestInit") or "")[:10]
        boards.append({
            "id": region, "title": title, "as_of": latest, "columns": columns, "sort": "wins",
            "card": ["wins", "led"], "rows": list(rows.values()),
            "note": f"Last {runs} daily 12 UTC runs, up to {latest}. Variable columns: RMSE pooled over the ranked "
                    "lead times; each model against its own analysis (WeatherNext against the IFS analysis).",
        })  # fmt: skip
    return boards


WB2_LIST = "https://storage.googleapis.com/storage/v1/b/weatherbench2/o"
WB2_FILE = "https://storage.googleapis.com/weatherbench2/benchmark_results/{}"
WB2_SETS = [("2020", "240x121"), ("2022", "240x121")]  # (test year, grid); 240x121 = 1.5°
# key, label, variable, level, lead hours, unit, scale, digits
_WB2_VARS = [
    ("z500", "Z500", "geopotential", 500, "m²/s²", 1, 1),
    ("t850", "T850", "temperature", 850, "K", 1, 2),
    ("q700", "Q700", "specific_humidity", 700, "g/kg", 1000, 3),
    ("t2m", "T2m", "2m_temperature", None, "K", 1, 2),
    ("ws10", "10 m wind", "10m_wind_speed", None, "m/s", 1, 2),
    ("mslp", "MSLP", "mean_sea_level_pressure", None, "hPa", 0.01, 2),
]


def _wb2_cols(z500_days: tuple[int, ...]) -> list[tuple]:
    cols = [(f"z500_d{d}", f"Z500 day {d}", "geopotential", 500, 24 * d, "m²/s²", 1, 1) for d in z500_days]
    return cols + [(f"{k}_d5", f"{lab} day 5", v, lev, 120, u, s, dg) for k, lab, v, lev, u, s, dg in _WB2_VARS[1:]]


WB2_DET = _wb2_cols((3, 5, 10))
WB2_PROB = _wb2_cols((5, 10, 15))


def _wb2_values(raw: bytes) -> tuple[bool, dict[str, float]]:
    """(ensemble?, values) of one result file: CRPS for ensembles, RMSE otherwise."""
    from scipy.io import netcdf_file  # in the `site` extra

    f = netcdf_file(io.BytesIO(raw), "r", mmap=False)
    try:
        vs = f.variables
        ens = any(k.startswith("crps.") for k in vs)
        metric, cols = ("crps", WB2_PROB) if ens else ("rmse", WB2_DET)
        regions = [b"".join(r).decode().strip("\x00 ") for r in vs["region"][:]]
        leads = [int(x) for x in vs["lead_time"][:]]
        levels = [int(x) for x in vs["level"][:]] if "level" in vs else []
        g = regions.index("global")
        out = {}
        for key, _, var, level, hours, _, scale, _ in cols:
            v = vs.get(f"{metric}.{var}")
            if v is None or hours not in leads or (level and level not in levels):
                continue
            pos = {"region": g, "lead_time": leads.index(hours), "level": levels.index(level) if level else None}
            if any(pos.get(d) is None for d in v.dimensions):  # the dimension order differs between files
                continue
            x = float(v[tuple(pos[d] for d in v.dimensions)])
            if x == x:  # not NaN
                out[key] = round(x * scale, 5)
        return ens, out
    finally:
        f.close()


def wb2(client: Client) -> list[dict]:
    """WeatherBench 2: per-model result files in the public results bucket, global, against ERA5."""
    names, token = [], None
    while True:
        params = {"prefix": "benchmark_results/", "fields": "items(name,updated),nextPageToken", "maxResults": 1000}
        if token:
            params["pageToken"] = token
        r = client.get(WB2_LIST, params=params)
        r.raise_for_status()
        d = r.json()
        names += d.get("items", [])
        token = d.get("nextPageToken")
        if not token:
            break
    boards = []
    for year, grid in WB2_SETS:
        pat = re.compile(rf"benchmark_results/(.+)_vs_era5_{grid}_{year}\.nc$")
        det, prob, updated = [], [], []
        for item in names:
            m = pat.match(item["name"])
            if not m or re.search(r"_single_member$|100m_uv", m.group(1)):
                continue
            r = client.get(WB2_FILE.format(item["name"].split("/", 1)[1]))
            r.raise_for_status()
            ens, vals = _wb2_values(r.content)
            (prob if ens else det).append({"id": m.group(1), "values": vals})
            updated.append(item.get("updated", "")[:10])
        for rows, cols, metric, ref, ref_name in [(det, WB2_DET, "RMSE", "hres", "IFS HRES"),
                                                  (prob, WB2_PROB, "CRPS", "ens", "IFS ENS")]:  # fmt: skip
            if not rows:
                continue
            _relative_skill(rows, [c[0] for c in cols], ref)
            kind = "det" if metric == "RMSE" else "prob"
            boards.append({
                "id": f"{kind}-{year}", "as_of": max(updated),
                "title": f"{'Deterministic' if kind == 'det' else 'Ensembles'} · {year}",
                "columns": [_col(c[0], c[1], c[5], "lower", c[7]) for c in cols]
                + [_col("skill", f"vs {ref_name}", "%", "lower", 1,
                        f"Mean {metric} change against {ref_name} over the columns shown (negative = better); "
                        "computed by ML4WX from the WeatherBench 2 numbers")],
                "sort": "z500_d5", "card": ["z500_d5", "skill"], "rows": rows,
                "note": f"Global {metric}, test year {year}, 1.5° grid, against ERA5. Lower is better.",
            })  # fmt: skip
    return boards


def _relative_skill(rows: list[dict], keys: list[str], ref: str) -> None:
    """Mean relative change against the reference model, over columns both have (at least all but 2)."""
    base = next((r["values"] for r in rows if r["id"] == ref), None)
    if not base:
        return
    for r in rows:
        common = [k for k in keys if k in r["values"] and base.get(k)]
        if len(common) >= len(keys) - 2:
            r["values"]["skill"] = round(100 * sum(r["values"][k] / base[k] - 1 for k in common) / len(common), 2)


AIWQ_JSON = "https://aiweatherquest.ecmwf.int/wp-content/themes/picostrap5-child-base/js/results_page/response.json"
AIWQ_WINDOWS = {1: "Days 19–25", 2: "Days 26–32"}
AIWQ_COLS = [("avg", "RPSS, mean"), ("tas", "2 m temperature"), ("mslp", "Sea-level pressure"), ("pr", "Precipitation")]


def aiwq(client: Client) -> list[dict]:
    """ECMWF AI Weather Quest: period-aggregated RPSS of the current competition period."""
    r = client.get(AIWQ_JSON)
    r.raise_for_status()
    recs = [x for x in r.json() if x.get("RPSS type") == "period" and x.get("Variable") == "average"]

    def published(x: dict) -> dt.datetime:
        return dt.datetime.strptime(x["Publication time"].replace(" UTC", ""), "%a %d-%b-%Y %H:%M:%S")

    period = max(recs, key=published)["Competition period"]
    boards = []
    for window, title in AIWQ_WINDOWS.items():
        cand = [x for x in recs if x["Competition period"] == period and x["Forecast window"] == window]
        if not cand:
            continue
        rec = max(cand, key=lambda x: x["week_index"])
        rows = []
        for t in rec["Teams"]:
            for m in t["Models"]:
                vals = {"avg": m.get("Average RPSS"), "tas": m.get("tas_rpss"), "mslp": m.get("mslp_rpss"),
                        "pr": m.get("pr_rpss")}  # fmt: skip
                rows.append({"id": f"{t['Teamname']}/{m['Model Name']}", "name": f"{m['Model Name']} ({t['Teamname']})",
                             "values": {k: v for k, v in vals.items() if isinstance(v, int | float)}})  # fmt: skip
        week = re.sub(r"^Competition week (\d+) \((.+)\)$", r"week \1 (\2)", rec["Competition week"])
        boards.append({
            "id": f"window-{window}", "title": f"{period} · {title}", "as_of": published(rec).date().isoformat(),
            "columns": [_col(k, lab, "", "higher", 3) for k, lab in AIWQ_COLS],
            "sort": "avg", "card": ["avg"], "rows": rows,
            "note": f"{period}, scores aggregated over the period up to {week}. RPSS of weekly-mean quintile "
                    "forecasts; 0 = climatology, higher is better.",
        })  # fmt: skip
    return boards


CHAOS_HTML = "https://raw.githubusercontent.com/leap-stc/ChaosBench/main/website/html/control.html"
CHAOS_COLS = [("rmse-z-500", "Z500 RMSE", "lower", 1), ("rmse-t-850", "T850 RMSE", "lower", 2),
              ("rmse-q-700", "Q700 RMSE", "lower", 5), ("acc-z-500", "Z500 ACC", "higher", 3),
              ("acc-t-850", "T850 ACC", "higher", 3), ("acc-q-700", "Q700 ACC", "higher", 3)]  # fmt: skip


def plotly_traces(html: str) -> list[dict]:
    """The traces of the last Plotly.newPlot("id", [...], ...) call in a page written by plotly.py."""
    i = html.rfind("Plotly.newPlot(")
    m = re.match(r'Plotly\.newPlot\(\s*"[^"]+",\s*', html[i:])
    if i < 0 or not m:
        raise ValueError("no Plotly.newPlot call with data")
    traces, _ = json.JSONDecoder().raw_decode(html[i + m.end():])
    return traces


def chaosbench(client: Client) -> list[dict]:
    """ChaosBench: the numbers behind the interactive leaderboard (control forecasts), weeks 3-4."""
    r = client.get(CHAOS_HTML)
    r.raise_for_status()
    rows: dict[str, dict] = {}
    for t in plotly_traces(r.text):
        metric = t.get("customdata")
        while isinstance(metric, list) and metric:
            metric = metric[0]
        days = [y for x, y in zip(t["x"], t["y"], strict=True) if 15 <= x <= 28 and y is not None]
        if not isinstance(metric, str) or not days:
            continue
        mid = t["name"].removesuffix(" vs. ERA5")
        rows.setdefault(mid, {"id": mid, "values": {}})["values"][metric] = round(sum(days) / len(days), 5)
    return [{
        "id": "control", "title": "Control forecasts · weeks 3–4", "rows": list(rows.values()),
        "as_of": _last_commit(client, "leap-stc/ChaosBench", "website/html/control.html"),
        "columns": [_col(k, lab, "", better, digits) for k, lab, better, digits in CHAOS_COLS],
        "sort": "rmse-z-500", "card": ["rmse-z-500", "acc-z-500"],
        "note": "Mean over lead days 15–28 of the 2022 test year, against ERA5. Units as in ChaosBench.",
    }]  # fmt: skip


CLIMATEBENCH_README = "https://raw.githubusercontent.com/duncanwp/ClimateBench/main/README.md"
CLIMATEBENCH_VARS = {"tas": "Temperature", "diurnal_temperature_range": "Diurnal range", "pr": "Precipitation",
                     "pr90": "Extreme precip. (pr90)"}  # fmt: skip


def climatebench(client: Client) -> list[dict]:
    """ClimateBench v1.0: the leaderboard table in the README (total NRMSE, 2080-2100, SSP2-4.5)."""
    r = client.get(CLIMATEBENCH_README)
    r.raise_for_status()
    table = _md_tables(r.text.split("## Leaderboard", 1)[1])[0]
    totals = {}
    for i, h in enumerate(table[0]):
        m = re.fullmatch(r"\('(\w+)', 'Total'\)", h)
        if m and m.group(1) in CLIMATEBENCH_VARS:
            totals[m.group(1)] = i
    rows = []
    for cells in table[1:]:
        vals = {k: float(cells[i]) for k, i in totals.items()}
        vals["mean"] = round(sum(vals.values()) / len(vals), 5)
        rows.append({"id": cells[0], "values": vals})
    return [{
        "id": "nrmse", "title": "Total NRMSE · 2080–2100", "rows": rows,
        "as_of": _last_commit(client, "duncanwp/ClimateBench", "README.md"),
        "columns": [_col("mean", "Mean", "", "lower", 3)]
        + [_col(k, v, "", "lower", 3) for k, v in CLIMATEBENCH_VARS.items()],
        "sort": "mean", "card": ["mean"],
        "note": "Emulating NorESM2 under SSP2-4.5. Lower is better; ranked by the mean total NRMSE, as in the README.",
    }]  # fmt: skip


WEATHERREAL_README = "https://raw.githubusercontent.com/microsoft/WeatherReal-Benchmark/main/README.md"
HIGHER_IS_BETTER = {"ETS", "ACC", "CSI", "HSS", "R2", "R²"}


def weatherreal(client: Client) -> list[dict]:
    """WeatherReal: the leaderboard tables in the README, one board per observation dataset."""
    r = client.get(WEATHERREAL_README)
    r.raise_for_status()
    section = r.text.split("## Leaderboards", 1)[1].split("\n## ", 1)[0]
    task = re.search(r"^### (.+)$", section, flags=re.M)
    as_of = _last_commit(client, "microsoft/WeatherReal-Benchmark", "README.md")
    boards = []
    for table in _md_tables(section):
        name, cols = _plain(table[0][0]), []
        for h in table[0][1:]:
            label, _, inner = _plain(h).partition("(")
            metric, _, unit = inner.rstrip(")").partition(",")
            metric, label = metric.strip(), label.strip()
            better = "higher" if metric.upper() in HIGHER_IS_BETTER else "lower"
            cols.append(_col(_slug(label), label if metric == "RMSE" else f"{label} ({metric})", unit.strip(),
                             better, 3, metric))  # fmt: skip
        rows = []
        for cells in table[1:]:
            vals = {c["key"]: _number(_plain(v)) for c, v in zip(cols, cells[1:], strict=False)}
            rows.append({"id": _plain(cells[0]), "values": {k: v for k, v in vals.items() if v is not None}})
        boards.append({
            "id": _slug(name), "title": name, "as_of": as_of, "columns": cols, "rows": rows,
            "sort": cols[0]["key"], "card": [cols[0]["key"]],
            "note": f"{_plain(task.group(1)) if task else 'Leaderboard'}, against quality-controlled station "
                    "observations; scores as published in the README (mean over all forecasts and lead times).",
        })  # fmt: skip
    return boards


WEATHER5K_README = "https://raw.githubusercontent.com/taohan10200/WEATHER-5K/main/README.md"
# The README table: per variable an MAE and an MSE column, in this order.
WEATHER5K_VARS = [("temperature", "Temperature", "K"), ("dewpoint", "Dew point", "K"),
                  ("wind_speed", "Wind speed", "m/s"), ("wind_direction", "Wind direction", "°"),
                  ("slp", "Sea-level pressure", "hPa"), ("overall", "Overall", "")]  # fmt: skip


def weather5k(client: Client) -> list[dict]:
    """WEATHER-5K: the results table in the README (MAE per variable, one board per lead time)."""
    r = client.get(WEATHER5K_README)
    r.raise_for_status()
    table = r.text[r.text.index("<table>"):r.text.index("</table>")]
    head = [_plain(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", table, flags=re.S)[:8]]
    if head[2:] != ["Temperature", "Dewpoint", "Wind Speed", "Wind Direction", "Sea Level Pressure", "Overall"]:
        raise ValueError(f"unexpected WEATHER-5K table header {head}")
    by_lead: dict[int, list[dict]] = {}
    model = None
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", table, flags=re.S):
        cells = [_plain(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, flags=re.S)]
        if "rowspan" in tr and cells:  # a new model: "🥇 1st Pyraformer" -> "Pyraformer"
            model = re.sub(r"^\W*(\d+(st|nd|rd|th)\s+)?", "", cells.pop(0))
        if not model or len(cells) < 13 or not cells[0].isdigit():
            continue
        maes = [_number(cells[1 + 2 * i]) for i in range(len(WEATHER5K_VARS))]
        by_lead.setdefault(int(cells[0]), []).append(
            {"id": model, "values": {k: v for (k, _, _), v in zip(WEATHER5K_VARS, maes, strict=True) if v is not None}})
    as_of = _last_commit(client, "taohan10200/WEATHER-5K", "README.md")
    return [{
        "id": f"lead-{lead}", "title": f"{lead} h ahead", "as_of": as_of, "rows": rows,
        "columns": [_col(k, lab, unit, "lower", 2, "MAE") for k, lab, unit in WEATHER5K_VARS],
        "sort": "overall", "card": ["overall"],
        "note": f"MAE at {lead} h lead time from a 48 h input window, over 5,672 stations; as in the README.",
    } for lead, rows in sorted(by_lead.items())]  # fmt: skip


CLIMATEBENCH2_CSV = "https://raw.githubusercontent.com/climate-analytics-lab/ClimateBench_app/main/data/benchmark_results.csv"
# variable, label, unit, scale: CMIP6 units, precipitation from kg m-2 s-1 to mm/day
CLIMATEBENCH2_VARS = [("tas", "Air temperature", "K", 1), ("pr", "Precipitation", "mm/day", 86400),
                      ("tos", "Sea surface temp.", "K", 1), ("clt", "Cloud fraction", "%", 1),
                      ("od550aer", "Aerosol optical depth", "", 1)]  # fmt: skip
CLIMATEBENCH2_PERIODS = [("SSP2-4.5", "test", "2015–2024 · SSP2-4.5"), ("Historical (2005-2014)", "hist", "2005–2014")]


def climatebench2(client: Client) -> list[dict]:
    """ClimateBench 2.0: the scorecard data of its web app (global RMSE of zonal means against observations)."""
    r = client.get(CLIMATEBENCH2_CSV)
    r.raise_for_status()
    rows = [x for x in csv.DictReader(io.StringIO(r.text))
            if x["metric"] == "zonal_mean_rmse" and x["region"] == "global"]  # fmt: skip
    as_of = _last_commit(client, "climate-analytics-lab/ClimateBench_app", "data/benchmark_results.csv")
    scale = {k: sc for k, _, _, sc in CLIMATEBENCH2_VARS}
    keys = list(scale)
    boards = []
    for column, bid, title in CLIMATEBENCH2_PERIODS:
        models: dict[str, dict] = {}
        for x in rows:
            if x["variable"] in keys and _number(x[column]) is not None:
                models.setdefault(x["model"], {"id": x["model"], "values": {}})["values"][x["variable"]] = round(
                    float(x[column]) * scale[x["variable"]], 5)
        _relative_to_median(list(models.values()), keys)
        boards.append({
            "id": bid, "title": title, "as_of": as_of, "rows": list(models.values()),
            "columns": [_col("skill", "vs median model", "%", "lower", 1,
                             "Mean RMSE change against the median model over the variables shown (negative = better); "
                             "computed by ML4WX")]
            + [_col(k, lab, unit, "lower", 3, "RMSE of the zonal mean") for k, lab, unit, _ in CLIMATEBENCH2_VARS],
            "sort": "skill", "card": ["skill", "tas"],
            "note": "Preview scorecard of the ClimateBench 2.0 app: CMIP6 models against observations, global RMSE "
                    "of monthly zonal means.",
        })  # fmt: skip
    return boards


def _relative_to_median(rows: list[dict], keys: list[str]) -> None:
    """Mean relative change against the median model, over columns a model has (at least all but 1)."""
    med = {}
    for k in keys:
        vals = sorted(r["values"][k] for r in rows if r["values"].get(k))
        if vals:
            med[k] = vals[len(vals) // 2] if len(vals) % 2 else (vals[len(vals) // 2 - 1] + vals[len(vals) // 2]) / 2
    for r in rows:
        common = [k for k in keys if r["values"].get(k) and med.get(k)]
        if len(common) >= len(keys) - 1:
            r["values"]["skill"] = round(100 * sum(r["values"][k] / med[k] - 1 for k in common) / len(common), 2)


SOURCES = {"owb": owb, "wb2": wb2, "aiwq": aiwq, "chaosbench": chaosbench, "climatebench": climatebench,
           "weatherreal": weatherreal, "weather5k": weather5k, "climatebench2": climatebench2}  # fmt: skip
