import { html, useEffect, useState } from "/v2/lib/html.js";
import { api, local } from "/v2/lib/api.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button } from "/v2/ui/Button.js";
import { Empty } from "/v2/ui/Reading.js";
import { RequestCard } from "/v2/ui/RequestCard.js";
import { when, plural } from "/v2/lib/format.js";
import { kindOf } from "/v2/state/kinds.js";
import { projects, projectName, childrenOf } from "/v2/state/projects.js";
import { openRequests, openTo } from "/v2/state/requests.js";
import { openProject } from "/v2/state/workspace.js";
import { newProject } from "/v2/wizards/index.js";

// Everything across every project on one page: what needs you (requests from subprojects,
// agents waiting, agents that failed), what is working now, what is built and what is still
// in progress, and the projects with their subprojects.

const DAY = 86_400_000;

export function Overview() {
  const [items, setItems] = useState(null);
  const [error, setError] = useState(null);
  const [seen, setSeen] = useState(() => local.get("seenFailures", {}));   // failures dismissed: id -> when
  useEffect(() => {
    let alive = true;
    const load = () => api("/api/items", { project: "*", kinds: ["agent", "graph", "chat"] })
      .then(r => { if (alive) { setItems(r.items || []); setError(null); } })
      .catch(e => alive && setError(e.message));
    load();
    const t = setInterval(() => !document.hidden && load(), 4000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  const all = items || [];
  const now = Date.now();
  const agents = all.filter(i => i.kind === "agent");
  const waiting = agents.filter(i => i.waiting || i.state === "waiting");
  const failed = agents.filter(i => i.state === "failed" && now - i.updated < 3 * DAY && seen[i.project + "/" + i.id] !== i.updated);
  const working = agents.filter(i => i.state === "running" && !i.waiting);
  const finished = agents.filter(i => i.state === "done" && now - i.updated < 3 * DAY).slice(0, 8);
  const built = all.filter(i => i.kind === "graph" && i.built);
  const progress = [
    ...all.filter(i => i.kind === "graph" && !i.built),
    ...all.filter(i => i.kind === "chat" && i.status !== "closed" && now - i.updated < 7 * DAY),
  ].sort((a, b) => b.updated - a.updated).slice(0, 12);
  const asks = openRequests.value;
  const needs = asks.length + waiting.length + failed.length;

  return html`<main class="main main-solo" id="main"><div class="page overview">
    <header class="page-head">
      <div>
        <p class="eyebrow">Studio</p>
        <h1 class="page-title">Overview</h1>
        <p class="page-lead">What needs you, what is working, and what is built, across ${plural(projects.value.length, "project")}.</p>
      </div>
    </header>
    ${error && html`<p class="error-line"><${Icon} name="warn" />${error}</p>`}

    <section class="card ov-needs" aria-labelledby="ov-needs">
      <h2 class="card-title" id="ov-needs">${needs ? html`<span class="live live-wait" />` : html`<${Icon} name="check" />`}Needs you
        <span class="count">${needs || ""}</span></h2>
      ${needs === 0 && html`<${Empty} title="Nothing needs you right now">Requests from subprojects and agents waiting for an answer show up here.<//>`}
      ${asks.length > 0 && html`<div class="req-list">${asks.map(q => html`<${RequestCard} key=${q.id} q=${q} />`)}</div>`}
      ${waiting.length > 0 && html`<${Rows} items=${waiting} sub=${() => "waits for your answer"} />`}
      ${failed.length > 0 && html`<${Rows} items=${failed} sub=${i => "failed " + when(i.updated)}
        dismiss=${i => setSeen(m => { const n = { ...m, [i.project + "/" + i.id]: i.updated }; local.set("seenFailures", n); return n; })} />`}
    </section>

    <div class="cards">
      <section class="card">
        <h2 class="card-title">${working.length ? html`<span class="live live-run" />` : null}Working now<span class="count">${working.length || ""}</span></h2>
        ${working.length ? html`<${Rows} items=${working} sub=${i => i.role || "agent"} /> `
          : html`<${Empty} title=${items ? "No agent is working" : "Loading…"} />`}
      </section>
      <section class="card">
        <h2 class="card-title">Built<span class="count">${built.length || ""}</span></h2>
        ${built.length ? html`<${Rows} items=${built} sub=${i => plural(i.nodes, "block")} />`
          : html`<${Empty} title="Nothing built yet">Networks show here once they are built.<//>`}
      </section>
      <section class="card">
        <h2 class="card-title">In progress<span class="count">${progress.length || ""}</span></h2>
        ${progress.length ? html`<${Rows} items=${progress} sub=${i => i.kind === "graph" ? "not built yet" : plural(i.messages, "message")} />`
          : html`<${Empty} title="Nothing open" />`}
      </section>
      ${finished.length > 0 && html`<section class="card">
        <h2 class="card-title">Finished lately</h2>
        <${Rows} items=${finished} sub=${i => i.role || "agent"} />
      </section>`}
    </div>

    <section class="card">
      <h2 class="card-title">Projects<${Button} variant="ghost" size="sm" icon="plus" label="New project" onClick=${() => newProject()} /></h2>
      <${Tree} items=${all} />
    </section>
  </div></main>`;
}

// items of every project: each opens in its project's tab
function Rows({ items, sub, dismiss }) {
  return html`<ul class="items ov-rows">${items.map(i => html`<li key=${i.kind + ":" + i.id} class="item" tabindex="0"
      onClick=${() => openProject(i.project, { route: { screen: i.kind, id: i.id } })}
      onKeyDown=${e => e.key === "Enter" && openProject(i.project, { route: { screen: i.kind, id: i.id } })}>
    <span class=${"item-ic k-" + i.kind}>${i.kind === "agent" && (i.state === "running" || i.waiting)
      ? html`<span class=${"live " + (i.waiting ? "live-wait" : "live-run")} />` : html`<${Icon} name=${kindOf(i.kind).icon} />`}</span>
    <span class="item-text"><span class="item-title">${i.title}</span>
      <span class="item-sub"><span class="chip">${projectName(i.project)}</span> ${sub(i)}</span></span>
    <span class="item-when">${when(i.updated)}</span>
    ${dismiss && html`<button type="button" class="mini" aria-label="Dismiss" title="Seen it; take it off this list"
      onClick=${e => { e.stopPropagation(); dismiss(i); }}><${Icon} name="close" size=${14} /></button>`}
  </li>`)}</ul>`;
}

// the projects as a tree from the top ones down; a subproject that is part of several shows
// under each, saying where else it belongs
function Tree({ items }) {
  const counts = new Map();
  for (const i of items) {
    const c = counts.get(i.project) || { working: 0, items: 0 };
    c.items++;
    if (i.kind === "agent" && (i.state === "running" || i.waiting)) c.working++;
    counts.set(i.project, c);
  }
  const tops = projects.value.filter(p => !(p.parents || []).length);
  const node = (p, from, seen) => {
    const c = counts.get(p.id) || { working: 0, items: 0 };
    const asks = openTo(p.id).length;
    const also = (p.parents || []).filter(x => x !== from).map(projectName);
    const kids = seen.has(p.id) ? [] : childrenOf(p.id);
    const next = new Set(seen).add(p.id);
    return html`<li key=${p.id}>
      <button type="button" class="tree-row" onClick=${() => openProject(p.id)}>
        <${Icon} name="project" size=${15} />
        <span class="tree-name">${p.name}</span>
        ${also.length > 0 && html`<span class="muted small">also in ${also.join(", ")}</span>`}
        <span class="grow" />
        ${asks > 0 && html`<span class="badge s-waiting">${plural(asks, "request")}</span>`}
        ${c.working > 0 && html`<span class="badge s-running">${c.working} working</span>`}
        <span class="muted small">${plural(c.items, "item")}</span>
      </button>
      ${kids.length > 0 && html`<ul class="tree">${kids.map(k => node(k, p.id, next))}</ul>`}
    </li>`;
  };
  return html`<ul class="tree tree-top">${tops.map(p => node(p, null, new Set()))}</ul>`;
}

