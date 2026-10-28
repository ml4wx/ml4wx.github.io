// Explore page: full-text search + facet filters on top of the Pagefind JS API.
// State lives in the URL (?q=...&topic=...&lab=...&sort=...) so every view can be shared.

const FACETS = [
  ["highlight", "Importance"],
  ["topic", "Topic"],
  ["lab", "Lab / organization"],
  ["year", "Year"],
  ["method", "Method"],
  ["timescale", "Time scale"],
  ["type", "Paper type"],
  ["venue", "Venue"],
  ["data", "Training data"],
  ["domain", "Domain"],
  ["open", "Open"],
];
const PAGE = 20;
const OPEN_BY_DEFAULT = new Set(["highlight", "topic", "year"]);

const $ = (s) => document.querySelector(s);
const qInput = $("#q"), sortSel = $("#sort"), facetsEl = $("#facets");
const resultsEl = $("#results"), countEl = $("#count"), moreBtn = $("#more");

let pagefind;
try {
  pagefind = await import("/pagefind/pagefind.js");
  await pagefind.options({ excerptLength: 28 });
  await pagefind.init();
  // Load every filter index now; otherwise searches return counts only for filters in use.
  await pagefind.filters();
} catch (e) {
  document.querySelector("#results").textContent = `Search could not load: ${e?.message || e}`;
  throw e;
}

const selected = Object.fromEntries(FACETS.map(([k]) => [k, new Set()]));
const openState = new Map();
const wide = matchMedia("(min-width: 761px)").matches;
let current = [], shown = 0, runId = 0;

// --- URL state ---------------------------------------------------------------
function readUrl() {
  const u = new URLSearchParams(location.search);
  qInput.value = u.get("q") || "";
  sortSel.value = u.get("sort") || (qInput.value ? "relevance" : "date");
  for (const [k] of FACETS) for (const v of u.getAll(k)) selected[k].add(v);
}
function writeUrl() {
  const u = new URLSearchParams();
  if (qInput.value.trim()) u.set("q", qInput.value.trim());
  if (sortSel.value !== (qInput.value.trim() ? "relevance" : "date")) u.set("sort", sortSel.value);
  for (const [k] of FACETS) for (const v of selected[k]) u.append(k, v);
  history.replaceState(null, "", u.toString() ? `?${u}` : location.pathname);
}

// --- rendering ---------------------------------------------------------------
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

function renderFacets(counts) {
  facetsEl.innerHTML = "";
  for (const [key, label] of FACETS) {
    const values = counts[key] || {};
    let entries = Object.entries(values).filter(([v, n]) => n > 0 || selected[key].has(v));
    if (!entries.length) continue;
    entries.sort(key === "year" ? (a, b) => b[0].localeCompare(a[0]) : (a, b) => b[1] - a[1] || a[0].localeCompare(b[0]));
    const det = document.createElement("details");
    // Keep the user's open/closed choice across re-renders; start collapsed on small screens.
    det.open = openState.has(key) ? openState.get(key)
      : selected[key].size > 0 || (wide && OPEN_BY_DEFAULT.has(key));
    det.addEventListener("toggle", () => openState.set(key, det.open));
    det.innerHTML = `<summary>${esc(label)}${selected[key].size ? ` <span class="pill">${selected[key].size}</span>` : ""}</summary>`;
    const ul = document.createElement("ul");
    for (const [v, n] of entries) {
      const id = `f-${key}-${v}`.replace(/\W+/g, "-");
      const li = document.createElement("li");
      li.innerHTML = `<input type="checkbox" id="${esc(id)}" ${selected[key].has(v) ? "checked" : ""}>
        <label for="${esc(id)}">${esc(v)}</label><span class="n">${n}</span>`;
      li.querySelector("input").addEventListener("change", (e) => {
        e.target.checked ? selected[key].add(v) : selected[key].delete(v);
        run();
      });
      ul.append(li);
    }
    det.append(ul);
    facetsEl.append(det);
  }
}

function card(d) {
  const m = d.meta || {};
  const url = d.url.replace(/\/index\.html$/, "/").replace(/\.html$/, "/");
  const excerpt = qInput.value.trim() && d.excerpt ? `<p class="excerpt">…${d.excerpt}…</p>` : "";
  return `<article class="card${m.star ? " is-hl" : ""}">
    <h3 class="card-title">${m.star ? '<span class="star" title="Highlight">★</span>' : ""}<a href="${esc(url)}">${esc(m.title)}</a></h3>
    <p class="card-meta">${esc(m.authors)} <span class="dot">·</span> ${esc(m.venue)} <span class="dot">·</span> ${esc(m.date)}${m.orgs ? ` <span class="dot">·</span> <b>${esc(m.orgs)}</b>` : ""}</p>
    <p class="card-summary">${esc(m.summary)}</p>${excerpt}
  </article>`;
}

async function showMore() {
  const batch = current.slice(shown, shown + PAGE);
  const data = await Promise.all(batch.map((r) => r.data()));
  resultsEl.insertAdjacentHTML("beforeend", data.map(card).join(""));
  shown += batch.length;
  moreBtn.hidden = shown >= current.length;
}

// --- search ------------------------------------------------------------------
// Datasets, models and tools whose name matches the query are shown above the papers.
const resIndex = fetch("/data/resources-index.json").then((r) => r.json()).catch(() => []);
const resEl = $("#resmatch");
async function showResources(term) {
  const words = (term || "").toLowerCase().split(/\s+/).filter((w) => w.length > 1);
  const items = !words.length ? [] : (await resIndex).filter((r) =>
    words.some((w) => r.key.split(" ").some((k) => k.startsWith(w) || (w.length > 3 && k.includes(w)))));
  resEl.hidden = !items.length;
  resEl.innerHTML = items.length ? `<b>Datasets, models &amp; tools:</b> ` + items.slice(0, 8).map((r) =>
    `<a href="${esc(r.url)}">${esc(r.name)} <span class="muted">${esc(r.kind)}</span></a>`).join("") : "";
}

async function run() {
  const id = ++runId;
  writeUrl();
  const term = qInput.value.trim() || null;
  showResources(term);
  const filters = { kind: "paper" };
  for (const [k] of FACETS) if (selected[k].size) filters[k] = { any: [...selected[k]] };
  const sort = sortSel.value === "relevance" && term ? undefined : { [sortSel.value === "score" ? "score" : "date"]: "desc" };
  const search = await pagefind.debouncedSearch(term, { filters, sort }, 200);
  if (search === null || id !== runId) return; // superseded by a newer keystroke
  current = search.results;
  shown = 0;
  resultsEl.innerHTML = "";
  countEl.textContent = `${current.length} paper${current.length === 1 ? "" : "s"}`;
  renderFacets(search.filters);
  if (!current.length) resultsEl.innerHTML = `<p class="muted">No papers match. Try fewer filters or other words.</p>`;
  await showMore();
}

function fail(e) {
  console.error(e);
  facetsEl.innerHTML = "";
  countEl.textContent = "";
  resultsEl.innerHTML = `<p class="muted">Search could not load (${esc(e?.message || e)}). Browse by
    <a href="/topics/">topic</a> instead.</p>`;
}

readUrl();
const safeRun = () => run().catch(fail);
qInput.addEventListener("input", safeRun);
sortSel.addEventListener("change", safeRun);
moreBtn.addEventListener("click", () => showMore().catch(fail));
$("#clear").addEventListener("click", () => {
  for (const [k] of FACETS) selected[k].clear();
  qInput.value = "";
  safeRun();
});
safeRun();
