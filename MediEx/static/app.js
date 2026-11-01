"use strict";

const $ = (id) => document.getElementById(id);
const state = { all: [], selected: [], active: -1 };   // selected = [{id,label}]

function el(tag, cls, text) {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text !== undefined) n.textContent = text;       // textContent => no HTML injection
  return n;
}

/* ---------------------------------------------------------- symptoms ---- */
async function loadSymptoms() {
  try {
    const r = await fetch("/api/symptoms");
    if (!r.ok) return;
    state.all = (await r.json()).symptoms;
  } catch (e) { /* server not ready */ }
}

async function loadHistory() {
  const box = $("history");
  try {
    const r = await fetch("/api/history");
    (await r.json()).history.forEach((h) => {
      const lab = el("label");
      const cb = el("input");
      cb.type = "checkbox"; cb.value = h.id;
      lab.append(cb, el("span", "", h.label));
      box.append(lab);
    });
  } catch (e) { /* ignore */ }
}

function renderChips() {
  const box = $("chips");
  box.replaceChildren();
  state.selected.forEach((s) => {
    const chip = el("span", "chip");
    const rm = el("button", "", "×");
    rm.type = "button";
    rm.setAttribute("aria-label", "Remove " + s.label);
    rm.addEventListener("click", () => removeSymptom(s.id));
    chip.append(s.label, rm);
    box.append(chip);
  });
  $("chipHint").textContent = state.selected.length
    ? state.selected.length + " symptom(s) selected."
    : "No symptoms selected yet.";
}

function addSymptom(s) {
  if (!state.selected.some((x) => x.id === s.id)) state.selected.push(s);
  renderChips();
}
function removeSymptom(id) {
  state.selected = state.selected.filter((x) => x.id !== id);
  renderChips();
}

/* ------------------------------------------------------- autocomplete --- */
function matches(q) {
  q = q.trim().toLowerCase();
  if (!q) return [];
  const taken = new Set(state.selected.map((s) => s.id));
  const starts = [], contains = [];
  for (const s of state.all) {
    if (taken.has(s.id)) continue;
    const l = s.label.toLowerCase();
    if (l.startsWith(q)) starts.push(s);
    else if (l.includes(q)) contains.push(s);
  }
  return starts.concat(contains).slice(0, 8);
}

function showSuggestions() {
  const list = $("suggest"), input = $("symInput");
  const found = matches(input.value);
  list.replaceChildren();
  state.active = found.length ? 0 : -1;
  found.forEach((s, i) => {
    const li = el("li", "", s.label);
    li.setAttribute("role", "option");
    li.setAttribute("aria-selected", i === 0 ? "true" : "false");
    li.addEventListener("mousedown", (e) => { e.preventDefault(); pick(s); });
    list.append(li);
  });
  list.hidden = !found.length;
  input.setAttribute("aria-expanded", String(!!found.length));
}

function pick(s) {
  addSymptom(s);
  $("symInput").value = "";
  showSuggestions();
  $("symInput").focus();
}

function moveActive(delta) {
  const items = [...$("suggest").children];
  if (!items.length) return;
  state.active = (state.active + delta + items.length) % items.length;
  items.forEach((li, i) => li.setAttribute("aria-selected", String(i === state.active)));
}

$("symInput").addEventListener("input", showSuggestions);
$("symInput").addEventListener("blur", () => { $("suggest").hidden = true; });
$("symInput").addEventListener("keydown", (e) => {
  if (e.key === "ArrowDown") { e.preventDefault(); moveActive(1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); moveActive(-1); }
  else if (e.key === "Enter") {
    const found = matches($("symInput").value);
    if (found.length) { e.preventDefault(); pick(found[Math.max(state.active, 0)]); }
  } else if (e.key === "Escape") { $("suggest").hidden = true; }
});

/* --------------------------------------------------------- diagnosis ---- */
function formError(msg) {
  const p = $("formError");
  p.textContent = msg || "";
  p.hidden = !msg;
}

async function diagnose() {
  formError("");
  if (!state.selected.length) { formError("Please add at least one symptom."); return; }
  const body = {
    symptoms: state.selected.map((s) => s.id),
    age: $("age").value,
    history: [...document.querySelectorAll("#history input:checked")].map((c) => c.value),
  };
  $("empty").hidden = true; $("loading").hidden = false; $("go").disabled = true;
  try {
    const r = await fetch("/api/diagnose", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.error || "Something went wrong.");
    renderResults(data);
  } catch (err) {
    formError(err.message);
    $("empty").hidden = false;
  } finally {
    $("loading").hidden = true; $("go").disabled = false;
  }
}

function tags(items, cls) {
  const wrap = el("div", "tags");
  items.forEach((t) => wrap.append(el("span", "tag " + (cls || ""), t)));
  return wrap;
}

function renderResults(data) {
  const box = $("results");
  box.replaceChildren();
  if (!data.results.length) {
    box.append(el("div", "empty", "No disease rule matched these symptoms."));
  }
  data.results.forEach((d, i) => {
    const card = el("article", "result" + (i === 0 ? " top1" : ""));
    const head = el("div", "r-head");
    head.append(el("span", "r-name", (i + 1) + ". " + d.disease), el("span", "r-pct", d.certainty + "%"));
    const bar = el("div", "bar"); const fill = el("span");
    fill.style.width = Math.min(100, d.certainty) + "%"; bar.append(fill);
    card.append(head, bar);
    if (d.description) card.append(el("p", "r-desc", d.description));

    const det = el("details");
    det.append(el("summary", "", "Why this result?"));
    const dl = el("dl");
    const add = (t, node) => { dl.append(el("dt", "", t)); const dd = el("dd"); dd.append(node); dl.append(dd); };
    add("Rule fired", document.createTextNode(d.rule));
    add("Matched symptoms", tags(d.matched, "ok"));
    if (d.missing.length) add("Not reported", tags(d.missing));
    add("Certainty from symptoms", document.createTextNode(d.symptom_certainty + "%"));
    if (d.history_notes.length) add("History / age adjustments", document.createTextNode(d.history_notes.join("; ")));
    det.append(dl);
    card.append(det);

    if (i === 0 && d.precautions.length) {
      card.append(el("h3", "", "Suggested precautions"));
      const ul = el("ul", "prec");
      d.precautions.forEach((p) => ul.append(el("li", "", p)));
      card.append(ul);
    }
    box.append(card);
  });

  const fbox = $("follow"); fbox.replaceChildren();
  data.followups.forEach((f) => {
    const b = el("button", "chip suggestion", "+ " + f.label);
    b.type = "button";
    b.addEventListener("click", () => { addSymptom(f); diagnose(); });
    fbox.append(b);
  });
  $("followBox").hidden = !data.followups.length;
  $("resultsCard").scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function resetAll() {
  state.selected = []; renderChips(); formError("");
  $("age").value = "";
  document.querySelectorAll("#history input").forEach((c) => (c.checked = false));
  $("results").replaceChildren(); $("followBox").hidden = true; $("empty").hidden = false;
}

$("form").addEventListener("submit", (e) => { e.preventDefault(); diagnose(); });
$("reset").addEventListener("click", resetAll);

/* ----------------------------------------------------------- metrics ---- */
async function loadMetrics() {
  const box = $("metrics");
  try {
    const r = await fetch("/api/metrics");
    if (!r.ok) return;
    const m = await r.json();
    [[m.diseases, "diseases"], [m.symptoms, "symptoms"],
     [m.top1 + "%", "top-1 accuracy (hold-out)"], [m.top3 + "%", "top-3 accuracy (hold-out)"]]
      .forEach(([v, l]) => {
        const d = el("div", "metric"); d.append(el("b", "", String(v)), el("span", "", l)); box.append(d);
      });
  } catch (e) { /* ignore */ }
}

renderChips();
loadSymptoms();
loadHistory();
loadMetrics();
