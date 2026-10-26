// Resources page: filter cards by kind, topic, category and text; state in the URL (?kind=&topic=&cat=&q=).
(() => {
  const $ = (s) => document.querySelector(s);
  const cards = [...document.querySelectorAll(".res-grid .res")];
  const sections = [...document.querySelectorAll(".res-section")];
  const tabs = [...document.querySelectorAll(".res-tabs button")];
  const q = $("#rq"), topic = $("#rtopic"), cat = $("#rcat"), sort = $("#rsort"), count = $("#rcount");
  const categories = JSON.parse($("#rcats").textContent); // {kind: {slug: label}}
  let kind = "";

  function fillCategories() {
    const keep = cat.value;
    cat.innerHTML = '<option value="">All categories</option>';
    for (const [k, cats] of Object.entries(categories)) {
      if (kind && k !== kind) continue;
      for (const [slug, label] of Object.entries(cats)) {
        const o = new Option(kind ? label : `${label} (${k})`, `${k}:${slug}`);
        cat.append(o);
      }
    }
    cat.value = [...cat.options].some((o) => o.value === keep) ? keep : "";
  }

  function apply() {
    const words = q.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const [ck, cs] = cat.value ? cat.value.split(":") : ["", ""];
    let shown = 0;
    for (const c of cards) {
      const ok = (!kind || c.dataset.kind === kind)
        && (!topic.value || c.dataset.topics.split(" ").includes(topic.value))
        && (!cs || (c.dataset.kind === ck && c.dataset.category === cs))
        && words.every((w) => c.dataset.text.includes(w));
      c.hidden = !ok;
      if (ok) shown++;
    }
    for (const s of sections) {
      const grid = s.querySelector(".res-grid");
      const items = [...grid.children];
      items.sort(sort.value === "stars"
        ? (a, b) => Number(b.dataset.stars) - Number(a.dataset.stars)
        : (a, b) => a.querySelector(".card-title a").textContent.localeCompare(b.querySelector(".card-title a").textContent));
      grid.append(...items);
      s.hidden = !items.some((c) => !c.hidden);
    }
    count.textContent = `${shown} resource${shown === 1 ? "" : "s"}`;
    for (const t of tabs) t.setAttribute("aria-pressed", String(t.dataset.kind === kind));
    const u = new URLSearchParams();
    if (kind) u.set("kind", kind);
    if (topic.value) u.set("topic", topic.value);
    if (cat.value) u.set("cat", cat.value);
    if (q.value.trim()) u.set("q", q.value.trim());
    if (sort.value !== "name") u.set("sort", sort.value);
    history.replaceState(null, "", u.toString() ? `?${u}` : location.pathname + location.hash);
  }

  const u = new URLSearchParams(location.search);
  kind = u.get("kind") || "";
  topic.value = u.get("topic") || "";
  q.value = u.get("q") || "";
  sort.value = u.get("sort") || "name";
  fillCategories();
  cat.value = u.get("cat") || "";
  for (const t of tabs) t.addEventListener("click", () => { kind = t.dataset.kind; fillCategories(); apply(); });
  for (const el of [q, topic, cat, sort]) el.addEventListener(el === q ? "input" : "change", apply);
  apply();
  if (location.hash) document.querySelector(location.hash)?.scrollIntoView();
})();
