// Datasets / Models / Tools pages: filter buttons (OR inside a group, AND across groups), search,
// sort. Each button shows how many rows it would give with the other active filters.
// State lives in the URL (?task=visualization&q=zarr&sort=stars), so a filtered view can be shared.
(() => {
  const $ = (s) => document.querySelector(s);
  const list = $("#crows");
  const rows = [...list.children];
  const groups = [...document.querySelectorAll(".fgroup")];
  const q = $("#cq"), sort = $("#csort"), count = $("#ccount"), none = $("#cnone"), clear = $("#cclear");
  const noun = count.textContent.replace(/^\d+\s*/, "");
  const selected = Object.fromEntries(groups.map((g) => [g.dataset.facet, new Set()]));
  const values = (row, key) => (row.dataset[`f${key[0].toUpperCase()}${key.slice(1)}`] || "").split(" ").filter(Boolean);

  function matches(row, skip) {
    const words = q.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.every((w) => row.dataset.text.includes(w))) return false;
    for (const [key, set] of Object.entries(selected)) {
      if (key === skip || !set.size) continue;
      if (!values(row, key).some((v) => set.has(v))) return false;
    }
    return true;
  }

  function apply() {
    let shown = 0;
    for (const r of rows) {
      r.hidden = !matches(r);
      if (!r.hidden) shown++;
    }
    for (const g of groups) {
      const key = g.dataset.facet;
      const base = rows.filter((r) => matches(r, key));
      for (const b of g.querySelectorAll(".fchip")) {
        const n = base.filter((r) => values(r, key).includes(b.dataset.value)).length;
        b.querySelector(".n").textContent = n;
        b.setAttribute("aria-pressed", String(selected[key].has(b.dataset.value)));
        b.disabled = n === 0 && !selected[key].has(b.dataset.value);
      }
    }
    const by = sort.value;
    const sorted = [...rows].sort((a, b) =>
      by === "name" ? a.dataset.name.localeCompare(b.dataset.name)
      : by === "stars" ? Number(b.dataset.stars) - Number(a.dataset.stars)
      : by === "release" ? (b.dataset.release || "").localeCompare(a.dataset.release || "")
      : Number(a.dataset.order) - Number(b.dataset.order));
    list.append(...sorted);
    count.textContent = `${shown} ${noun}`;
    none.hidden = shown > 0;
    const any = q.value.trim() || Object.values(selected).some((s) => s.size);
    clear.hidden = !any;
    const u = new URLSearchParams();
    for (const [key, set] of Object.entries(selected)) for (const v of set) u.append(key, v);
    if (q.value.trim()) u.set("q", q.value.trim());
    if (sort.value !== "order") u.set("sort", sort.value);
    history.replaceState(null, "", (u.toString() ? `?${u}` : location.pathname) + location.hash);
  }

  const u = new URLSearchParams(location.search);
  for (const key of Object.keys(selected)) for (const v of u.getAll(key)) selected[key].add(v);
  q.value = u.get("q") || "";
  if ([...sort.options].some((o) => o.value === u.get("sort"))) sort.value = u.get("sort");
  for (const g of groups) {
    for (const b of g.querySelectorAll(".fchip")) {
      b.addEventListener("click", () => {
        const set = selected[g.dataset.facet];
        set.has(b.dataset.value) ? set.delete(b.dataset.value) : set.add(b.dataset.value);
        apply();
      });
    }
  }
  // Long groups show the first 8 values; "Show all" opens the rest (also when one of them is selected).
  for (const g of groups) {
    const more = g.querySelector(".fmore");
    if (!more) continue;
    const extras = [...g.querySelectorAll(".fchip.extra")];
    const open = (yes) => { for (const b of extras) b.hidden = !yes; more.hidden = yes; };
    open(extras.some((b) => selected[g.dataset.facet].has(b.dataset.value)));
    more.addEventListener("click", () => open(true));
  }
  q.addEventListener("input", apply);
  sort.addEventListener("change", apply);
  clear.addEventListener("click", (e) => {
    e.preventDefault();
    for (const s of Object.values(selected)) s.clear();
    q.value = "";
    apply();
  });
  // Small screens: filters start folded.
  if (matchMedia("(max-width: 760px)").matches && !Object.values(selected).some((s) => s.size)) {
    $(".cat-filters-box").open = false;
  }
  apply();
  if (location.hash) {
    const target = document.getElementById(location.hash.slice(1));
    if (target) {
      if (target.hidden) { for (const s of Object.values(selected)) s.clear(); q.value = ""; apply(); }
      target.querySelector(".row-more").open = true;
      target.classList.add("target");
      target.scrollIntoView({ block: "center" });
    }
  }
})();
