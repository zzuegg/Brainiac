import { html } from "/v2/lib/html.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button } from "/v2/ui/Button.js";
import { ItemList } from "/v2/ui/ItemList.js";
import { Empty } from "/v2/ui/Reading.js";
import { KINDS, SECTIONS } from "/v2/state/kinds.js";
import { byId } from "/v2/state/projects.js";
import { navigate } from "/v2/state/workspace.js";
import { renameProject, menuFor } from "/v2/state/actions.js";
import { openMenu } from "/v2/ui/overlay.js";
import { create } from "/v2/wizards/index.js";

// A project's front page: what it is for, what runs in it now, and its recent work.

export function Home({ tab }) {
  const p = byId.value.get(tab.project);
  const items = tab.items.value;
  const running = items.filter(i => i.kind === "agent" && (i.state === "running" || i.state === "waiting" || i.waiting));
  const projectItem = p && { kind: "project", id: p.id, title: p.name };
  return html`<div class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Project</p>
        <h1 class="page-title" onDblClick=${() => p && renameProject(p)} title="Double-click to rename">${p?.name ?? tab.project}</h1>
        ${p?.description ? html`<p class="page-lead">${p.description}</p>` : html`<p class="page-lead muted">No description yet.</p>`}
        ${p?.folder && html`<p class="page-meta"><${Icon} name="folder" size=${14} /><span class="mono">${p.folder}</span></p>`}
      </div>
      <div class="page-actions">
        ${projectItem && html`<${Button} variant="ghost" icon="more" label="Project actions" onClick=${e => openMenu(e.currentTarget, menuFor(tab, projectItem), p.name)} />`}
      </div>
    </header>

    <div class="quick">
      ${SECTIONS.map(k => html`<button type="button" class=${"quick-btn k-" + k} onClick=${() => create(k, tab)}>
        <span class="item-ic"><${Icon} name=${KINDS[k].icon} /></span>
        <span><strong>New ${KINDS[k].label.toLowerCase()}</strong><span class="quick-sub">${QUICK[k]}</span></span>
      </button>`)}
    </div>

    ${running.length > 0 && html`<section class="card">
      <h2 class="card-title"><span class="live live-run" />Working now</h2>
      <${ItemList} tab=${tab} items=${running} scope="running" label="Working now" />
    </section>`}

    <div class="cards">
      ${SECTIONS.map(k => {
        const of = items.filter(i => i.kind === k).slice(0, 6);
        return html`<section class="card">
          <h2 class="card-title">${KINDS[k].plural}<span class="count">${items.filter(i => i.kind === k).length || ""}</span>
            <${Button} variant="ghost" size="sm" icon="plus" label=${"New " + KINDS[k].label.toLowerCase()} onClick=${() => create(k, tab)} /></h2>
          ${of.length
            ? html`<${ItemList} tab=${tab} items=${of} compact scope="home" label=${KINDS[k].plural} />`
            : html`<${Empty} title=${"No " + KINDS[k].plural.toLowerCase() + " yet"}
                action=${html`<${Button} size="sm" onClick=${() => create(k, tab)}>New ${KINDS[k].label.toLowerCase()}<//>`} />`}
        </section>`;
      })}
    </div>
    ${tab.error.value && html`<p class="error-line"><${Icon} name="warn" />${tab.error.value}</p>`}
    <p class="muted small foot-note">Deleted something by mistake? <button type="button" class="link" onClick=${() => navigate(tab, { screen: "trash" })}>Open the trash</button>; it keeps items for 30 days.</p>
  </div>`;
}

const QUICK = { chat: "Talk to a model", agent: "Give a goal, get it done", graph: "Design a network" };
