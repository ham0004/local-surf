"""Write the human audit page for gate G2 from reports/v3_gate/audit_items.json.

The page embeds the 50 items (question, reference, candidate answer; the condition
T/TF and every judge score are hidden) and stores each label in the artifact's
database collection "labels" (doc id "i<item>": consistency 0-2, coverage 0-100,
optional note). scripts/v3_gate_analyze.py reads those labels back.

    python scripts/v3_audit_page.py   ->  reports/v3_gate/audit_page.html
"""

from __future__ import annotations

import json
from pathlib import Path

ITEMS = Path("reports/v3_gate/audit_items.json")
OUT = Path("reports/v3_gate/audit_page.html")

TEMPLATE = r"""<title>Answer Audit</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Literata:opsz,wght@7..72,400;7..72,600&family=Source+Sans+3:wght@400;600;700&family=JetBrains+Mono:wght@500&display=swap">
<style>
/* Layout: one reading column of item cards under a sticky progress bar; labels sit under each answer. */
:root {
  --bg: #f5f6f3; --card: #ffffff; --ink: #1f2a33; --muted: #5d6a73; --line: #d9ded9;
  --accent: #2f6f62; --accent-soft: #e2efe9; --warn: #9a5b13; --ref: #eef1ee;
  --f-read: "Literata", Georgia, serif; --f-ui: "Source Sans 3", "Segoe UI", sans-serif; --f-mono: "JetBrains Mono", Consolas, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #141a1d; --card: #1c2428; --ink: #e5ebe8; --muted: #9aa7a3; --line: #2e3a3f;
  --accent: #6cc0ab; --accent-soft: #1f3a34; --warn: #e0a35a; --ref: #222c30; color-scheme: dark } }
:root[data-theme="dark"] {
  --bg: #141a1d; --card: #1c2428; --ink: #e5ebe8; --muted: #9aa7a3; --line: #2e3a3f;
  --accent: #6cc0ab; --accent-soft: #1f3a34; --warn: #e0a35a; --ref: #222c30; color-scheme: dark }
*, *::before, *::after { box-sizing: border-box; }
.card, .card > div, .q, .ref, .ans { min-width: 0; overflow-wrap: anywhere; }
body { background: var(--bg); color: var(--ink); font-family: var(--f-ui); font-size: 15px; line-height: 1.5; padding: 0 16px 48px; }
header { position: sticky; top: env(safe-area-inset-top, 0px); z-index: 2; background: var(--bg); border-bottom: 1px solid var(--line);
  padding-block: 12px; margin: 0 -16px 20px; padding-inline: 16px; }
.wrap { max-width: 760px; margin: 0 auto; }
header .wrap { display: flex; flex-wrap: wrap; gap: 8px 20px; align-items: center; justify-content: space-between; }
h1 { font-size: 18px; margin: 0; }
.progress { display: flex; align-items: center; gap: 10px; font-variant-numeric: tabular-nums; color: var(--muted); }
.bar { width: 160px; height: 8px; border-radius: 4px; background: var(--line); overflow: hidden; }
.bar > i { display: block; height: 100%; background: var(--accent); width: 0; transition: width .2s; }
.status { font-size: 13px; color: var(--muted); min-height: 1.2em; }
.status.bad { color: var(--warn); }
.intro { color: var(--muted); max-width: 65ch; }
.intro b { color: var(--ink); }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 18px; margin-bottom: 18px; display: grid; gap: 12px; }
.card.done { border-color: var(--accent); }
.num { font-family: var(--f-mono); font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; }
.num .ok { color: var(--accent); }
.label { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); font-weight: 700; margin-bottom: 2px; }
.q { font-weight: 600; }
.ref { background: var(--ref); border-radius: 6px; padding: 10px 12px; font-family: var(--f-read); font-size: 14.5px; }
.ans { font-family: var(--f-read); font-size: 15.5px; white-space: pre-wrap; }
.scale { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.scale .name { width: 100%; font-size: 13px; color: var(--muted); }
.scale button { font: inherit; font-size: 14px; border: 1px solid var(--line); background: var(--card); color: var(--ink);
  border-radius: 6px; padding: 6px 12px; cursor: pointer; }
.scale button[aria-pressed="true"] { background: var(--accent); border-color: var(--accent); color: var(--card); font-weight: 700; }
.scale button:focus-visible, textarea:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
textarea { font: inherit; font-size: 14px; width: 100%; min-height: 2.4em; border: 1px solid var(--line); border-radius: 6px;
  background: var(--card); color: var(--ink); padding: 6px 8px; box-sizing: border-box; resize: vertical; }
.filter { display: flex; gap: 8px; align-items: center; font-size: 14px; color: var(--muted); }
@media (prefers-reduced-motion: reduce) { .bar > i { transition: none; } }
</style>

<header><div class="wrap">
  <h1>Answer audit · gate G2</h1>
  <div class="progress"><span id="count">0 / __N__ labelled</span><span class="bar"><i id="fill"></i></span></div>
  <label class="filter"><input type="checkbox" id="only-open"> Show unlabelled only</label>
  <div class="status" id="status">Connecting to the label store…</div>
</div></header>

<main class="wrap">
  <p class="intro">Each card is one answer from the frozen Qwen3-VL-2B to an EduVidQA lecture question, shown with the
  dataset's reference answer. Rate it on the same two scales the automatic judges use. <b>Consistency</b>: 2 = no contradiction
  with the reference and no clear factual error; 1 = a minor inaccuracy or partly unsupported claim; 0 = contradicts the reference,
  clearly wrong, or does not answer. <b>Coverage</b>: the share of the reference's key points the answer states. Judge meaning,
  not wording, and do not reward length. Which answers had video frames is hidden on purpose. Labels save as you click.</p>
  <div id="items"></div>
</main>

<script id="data" type="application/json">__DATA__</script>
<script>
const ITEMS = JSON.parse(document.getElementById("data").textContent);
const labels = {};            // item -> {consistency, coverage, note}
let db = null;
const $ = (s, el = document) => el.querySelector(s);
const statusEl = $("#status");
function setStatus(t, bad) { statusEl.textContent = t; statusEl.classList.toggle("bad", !!bad); }

function card(it) {
  const c = document.createElement("article");
  c.className = "card"; c.id = "item-" + it.item;
  c.innerHTML = `
    <div class="num"><span>Item ${it.item + 1} of ${ITEMS.length}</span><span class="saved"></span></div>
    <div><div class="label">Question</div><div class="q"></div></div>
    <div><div class="label">Reference answer</div><div class="ref"></div></div>
    <div><div class="label">Candidate answer</div><div class="ans"></div></div>
    <div class="scale" data-field="consistency"><span class="name">Consistency</span>
      ${[[2,"2 · consistent"],[1,"1 · minor issue"],[0,"0 · wrong / no answer"]].map(([v,t]) => `<button type="button" data-v="${v}" aria-pressed="false">${t}</button>`).join("")}</div>
    <div class="scale" data-field="coverage"><span class="name">Coverage of the reference's key points</span>
      ${[0,25,50,75,100].map(v => `<button type="button" data-v="${v}" aria-pressed="false">${v}%</button>`).join("")}</div>
    <div><label class="label" for="note-${it.item}">Note (optional)</label><textarea id="note-${it.item}" rows="1"></textarea></div>`;
  $(".q", c).textContent = it.question;
  $(".ref", c).textContent = it.reference;
  $(".ans", c).textContent = it.answer;
  c.querySelectorAll(".scale button").forEach(b => b.addEventListener("click", () => choose(it.item, b.parentElement.dataset.field, +b.dataset.v)));
  let t; $("textarea", c).addEventListener("input", e => { clearTimeout(t); t = setTimeout(() => choose(it.item, "note", e.target.value), 800); });
  return c;
}

function paint(item) {
  const c = document.getElementById("item-" + item), l = labels[item] || {};
  c.querySelectorAll(".scale").forEach(s => s.querySelectorAll("button").forEach(b =>
    b.setAttribute("aria-pressed", String(l[s.dataset.field] === +b.dataset.v))));
  const ta = $("textarea", c); if (document.activeElement !== ta) ta.value = l.note || "";
  const done = l.consistency !== undefined && l.coverage !== undefined;
  c.classList.toggle("done", done);
  $(".saved", c).innerHTML = done ? '<span class="ok">labelled</span>' : (l.consistency !== undefined || l.coverage !== undefined ? "one scale left" : "");
  c.hidden = $("#only-open").checked && done;
}

function summary() {
  const n = ITEMS.filter(it => { const l = labels[it.item]; return l && l.consistency !== undefined && l.coverage !== undefined; }).length;
  $("#count").textContent = `${n} / ${ITEMS.length} labelled`;
  $("#fill").style.width = (100 * n / ITEMS.length) + "%";
}

const writing = {};
async function choose(item, field, value) {
  labels[item] = { ...(labels[item] || {}), [field]: value };
  paint(item); summary();
  if (!db) { setStatus("Not saved: the label store is unavailable in this view. Open the page signed in to claude.ai.", true); return; }
  const body = { ...labels[item], item, at: new Date().toISOString() };
  while (writing[item]) await writing[item];          // one write at a time per document
  writing[item] = db.doc("labels/i" + String(item).padStart(2, "0")).set(body)
    .then(() => setStatus("Saved"), e => setStatus("Not saved (" + (e && e.code || "error") + "). Check you are signed in with edit access.", true))
    .finally(() => { writing[item] = null; });
}

const list = $("#items");
ITEMS.forEach(it => list.appendChild(card(it)));
ITEMS.forEach(it => paint(it.item));
summary();
$("#only-open").addEventListener("change", () => ITEMS.forEach(it => paint(it.item)));

(async () => {
  db = await window.claude?.use?.("db") ?? null;
  if (!db) { setStatus("Label store unavailable here: labels will not be saved. Open this page on claude.ai while signed in.", true); return; }
  setStatus("Connected. Labels save as you click.");
  db.collection("labels").onSnapshot(snap => {
    snap.docs.forEach(d => { const v = d.data(); if (v && Number.isInteger(v.item)) labels[v.item] = { consistency: v.consistency, coverage: v.coverage, note: v.note || "" }; });
    ITEMS.forEach(it => paint(it.item)); summary();
  }, e => setStatus("Live sync stopped (" + (e && e.code) + "); reload to resume.", true));
})();
</script>
"""


def main() -> None:
    items = json.loads(ITEMS.read_text(encoding="utf-8"))
    data = json.dumps(items).replace("</", "<\\/")
    OUT.write_text(TEMPLATE.replace("__DATA__", data).replace("__N__", str(len(items))), encoding="utf-8")
    print(f"wrote {OUT} with {len(items)} items")


if __name__ == "__main__":
    main()
