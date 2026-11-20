import { html } from "/v2/lib/html.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button } from "/v2/ui/Button.js";
import { ItemList } from "/v2/ui/ItemList.js";
import { Empty } from "/v2/ui/Reading.js";
import { KINDS, SECTIONS } from "/v2/state/kinds.js";
import { byId, childrenOf, projectName, saveProject } from "/v2/state/projects.js";
import { navigate, openProject } from "/v2/state/workspace.js";
import { renameProject, menuFor, editParents, unlink } from "/v2/state/actions.js";
import { openTo } from "/v2/state/requests.js";
import { RequestCard } from "/v2/ui/RequestCard.js";
import { Choices } from "/v2/ui/Wizard.js";
import { fail } from "/v2/state/toasts.js";
import { openMenu } from "/v2/ui/overlay.js";
import { create, newProject } from "/v2/wizards/index.js";

// A project's front page: what it is for, what runs in it now, and its recent work.

export function Home({ tab }) {
  const p = byId.value.get(tab.project);
  const items = tab.items.value;
  const running = items.filter(i => i.kind === "agent" && (i.state === "running" || i.state === "waiting" || i.waiting));
  const projectItem = p && { kind: "project", id: p.id, title: p.name };
  const asks = openTo(tab.project);
  return html`<div class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">Project</p>
        <h1 class="page-title" onDblClick=${() => p && renameProject(p)} title="Double-click to rename">${p?.name ?? tab.project}</h1>
        ${p?.description ? html`<p class="page-lead">${p.description}</p>` : html`<p class="page-lead muted">No description yet.</p>`}
        ${p?.folder && html`<p class="page-meta"><${Icon} name="folder" size=${14} /><span class="mono">${p.folder}</span></p>`}
        ${p && p.id !== "default" && html`<${PartOf} p=${p} />`}
      </div>
      <div class="page-actions">
        <${Button} icon="settings" onClick=${() => navigate(tab, { screen: "config" })}>Project settings<//>
        ${projectItem && html`<${Button} variant="ghost" icon="more" label="Project actions" onClick=${e => openMenu(e.currentTarget, menuFor(tab, projectItem), p.name)} />`}
      </div>
    </header>

    <div class="quick">
      ${SECTIONS.map(k => html`<button type="button" class=${"quick-btn k-" + k} onClick=${() => create(k, tab)}>
        <span class="item-ic"><${Icon} name=${KINDS[k].icon} /></span>
        <span><strong>New ${KINDS[k].label.toLowerCase()}</strong><span class="quick-sub">${QUICK[k]}</span></span>
      </button>`)}
    </div>

    ${asks.length > 0 && html`<section class="card">
      <h2 class="card-title"><span class="live live-wait" />Requests from subprojects<span class="count">${asks.length}</span></h2>
      <div class="req-list">${asks.map(q => html`<${RequestCard} key=${q.id} q=${q} />`)}</div>
    </section>`}

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
    ${p && p.id !== "default" && html`<${Subprojects} p=${p} />`}
    ${tab.error.value && html`<p class="error-line"><${Icon} name="warn" />${tab.error.value}</p>`}
    <p class="muted small foot-note">Deleted something by mistake? <button type="button" class="link" onClick=${() => navigate(tab, { screen: "trash" })}>Open the trash</button>; it keeps items for 30 days.</p>
  </div>`;
}

// the projects this one is part of, each a link, with the way to change them
function PartOf({ p }) {
  const parents = p.parents || [];
  return html`<p class="page-meta part-of">
    <${Icon} name="project" size=${14} />
    ${parents.length ? html`<span>Part of ${parents.map((id, i) => html`${i ? ", " : ""}<span class="chip chip-link">
        <button type="button" class="link" onClick=${() => openProject(id)}>${projectName(id)}</button>
        <button type="button" class="chip-x" aria-label=${"Take it out of " + projectName(id)} title=${"Take it out of " + projectName(id)}
          onClick=${() => unlink(p.id, id)}><${Icon} name="close" size=${11} /></button></span>`)}</span>`
      : html`<span class="muted">Not part of another project</span>`}
    <button type="button" class="link small" onClick=${() => editParents(p.id)}>${parents.length ? "Change" : "Make it part of one…"}</button>
  </p>`;
}

// the projects that are part of this one, and who answers their requests
function Subprojects({ p }) {
  const kids = childrenOf(p.id);
  const setRequests = v => saveProject(p.id, { requests: v }).catch(fail);
  return html`<section class="card subprojects">
    <h2 class="card-title">Subprojects<span class="count">${kids.length || ""}</span>
      <${Button} variant="ghost" size="sm" icon="plus" label="New subproject" onClick=${() => newProject(p.id)} /></h2>
    ${kids.length ? html`<ul class="items items-compact">${kids.map(k => html`<li key=${k.id} class="item" tabindex="0"
        onClick=${() => openProject(k.id)} onKeyDown=${e => e.key === "Enter" && openProject(k.id)}>
      <span class="item-ic k-project"><${Icon} name="project" /></span>
      <span class="item-text"><span class="item-title">${k.name}</span>
        ${(k.parents || []).length > 1 && html`<span class="item-sub">also in ${k.parents.filter(x => x !== p.id).map(projectName).join(", ")}</span>`}</span>
      ${openTo(p.id).some(q => q.project === k.id) && html`<span class="badge s-waiting">asks</span>`}
    </li>`)}</ul>`
      : html`<p class="muted small side-pad">A subproject keeps a part of the work apart, with its own agents and limits. Agents here can work in it; it asks this project when it needs more.</p>`}
    <div class="field">
      <span class="field-label">Who answers requests from subprojects</span>
      <${Choices} name="Who answers requests" value=${p.requests || "user"} onChange=${setRequests} options=${[
        { value: "user", label: "Me", text: "They wait in your overview." },
        { value: "agents", label: "This project's agents", text: "Within what this project allows; you can still answer." },
      ]} />
    </div>
  </section>`;
}

const QUICK = { chat: "Talk to a model", agent: "Give a goal, get it done", graph: "Design a network" };
