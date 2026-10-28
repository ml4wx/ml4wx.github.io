// Search suggestions for the header and home search boxes (forms with data-quick): the essentials
// before you type, then matching papers, models, datasets and tools. Enter without a pick searches
// all papers on Explore, as before. Data: /data/quick.json, most important of each kind first.

const KIND = { paper: "Paper", model: "Model", dataset: "Dataset", tool: "Tool" };
const POPULAR = 8, MAX = 7;

const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
// Same rule as _norm() in build.py: lower case words without punctuation.
const norm = (s) => s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();

let data;
const load = () => (data ??= fetch("/data/quick.json").then((r) => r.json())
  .then((items) => items.map((it, i) => ({ ...it, i, name: norm(it.n), words: it.s.split(" ") })))
  .catch(() => []));

// 0: the name starts with the query, 1: every query word starts a word of the name,
// 2: ... a word of the name or title, -1: no match.
function rank(it, q, words) {
  if (it.name.startsWith(q)) return 0;
  const starts = (keys) => words.every((w) => keys.some((k) => k.startsWith(w)));
  if (starts(it.name.split(" "))) return 1;
  return starts(it.words) ? 2 : -1;
}

for (const form of document.querySelectorAll("form[data-quick]")) {
  const input = form.querySelector("input[type=search]");
  const box = document.createElement("div");
  box.id = `${input.id}-suggest`;
  box.className = "suggest";
  box.setAttribute("role", "listbox");
  box.setAttribute("aria-label", "Suggestions");
  box.hidden = true;
  form.append(box);
  input.setAttribute("role", "combobox");
  input.setAttribute("aria-autocomplete", "list");
  input.setAttribute("aria-controls", box.id);
  input.setAttribute("aria-expanded", "false");
  let active = -1;

  const options = () => [...box.querySelectorAll("[role=option]")];
  function setActive(i) {
    const opts = options();
    active = i;
    opts.forEach((o, j) => o.setAttribute("aria-selected", String(j === i)));
    if (i < 0) return input.removeAttribute("aria-activedescendant");
    input.setAttribute("aria-activedescendant", opts[i].id);
    opts[i].scrollIntoView({ block: "nearest" });
  }
  function close() {
    box.hidden = true;
    input.setAttribute("aria-expanded", "false");
    setActive(-1);
  }

  async function render() {
    const items = await load();
    if (document.activeElement !== input) return;
    const raw = input.value.trim(), q = norm(raw), words = q.split(" ").filter(Boolean);
    const list = !q ? items.slice(0, POPULAR) : items.map((it) => [rank(it, q, words), it])
      .filter(([r]) => r >= 0).sort((a, b) => a[0] - b[0] || a[1].i - b[1].i).slice(0, MAX).map(([, it]) => it);
    let n = 0;
    const option = (url, html) => `<a role="option" id="${box.id}-${n++}" href="${esc(url)}" aria-selected="false" tabindex="-1">${html}</a>`;
    box.innerHTML = (list.length ? `<div class="suggest-head" role="presentation">${q ? "Suggestions" : "Essentials"}</div>` : "")
      + list.map((it) => option(it.u, `<span class="suggest-name">${esc(it.n)}</span>`
        + `<span class="suggest-kind">${KIND[it.k]}</span><span class="suggest-detail">${esc(it.d)}</span>`)).join("")
      + (q ? option(`/explore/?q=${encodeURIComponent(raw)}`, `<span class="suggest-name suggest-all">Search all papers for “${esc(raw)}” →</span>`)
           : option("/essentials/", `<span class="suggest-name suggest-all">All essentials, by topic →</span>`));
    box.hidden = false;
    input.setAttribute("aria-expanded", "true");
    setActive(-1);
  }

  input.addEventListener("focus", render);
  input.addEventListener("input", render);
  input.addEventListener("blur", close);
  box.addEventListener("mousedown", (e) => e.preventDefault()); // keep the focus, so the click lands
  input.addEventListener("keydown", (e) => {
    const opts = options();
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      if (box.hidden) return render();
      e.preventDefault();
      const n = opts.length;
      setActive(e.key === "ArrowDown" ? (active + 1) % n : active <= 0 ? n - 1 : active - 1);
    } else if (e.key === "Enter" && !box.hidden && active >= 0) {
      e.preventDefault();
      location.href = opts[active].href;
    } else if (e.key === "Escape" && !box.hidden) {
      e.preventDefault();
      close();
    }
  });
}
