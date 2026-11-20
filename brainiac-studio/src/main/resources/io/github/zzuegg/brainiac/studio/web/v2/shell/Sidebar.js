import { html, useRef, useState, cx } from "/v2/lib/html.js";
import { local } from "/v2/lib/api.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button } from "/v2/ui/Button.js";
import { ItemList } from "/v2/ui/ItemList.js";
import { KINDS, SECTIONS } from "/v2/state/kinds.js";
import { byId, childrenOf } from "/v2/state/projects.js";
import { navigate, openProject } from "/v2/state/workspace.js";
import { openTo } from "/v2/state/requests.js";
import { sidebarWidth, showSide } from "/v2/state/prefs.js";
import { create, newProject } from "/v2/wizards/index.js";

// The open project's own things: its home, then a section per kind with a "+" that starts its
// wizard, and a filter over all of them. The trash and settings sit at the bottom.

const CAP = 8;

export function Sidebar({ tab }) {
  const route = tab.route.value;
  const q = tab.filter.value.trim().toLowerCase();
  const items = tab.items.value.filter(i => !q || i.title.toLowerCase().includes(q));
  const collapsed = tab.collapsed.value;
  const toggle = k => (tab.collapsed.value = { ...collapsed, [k]: !collapsed[k] });
  const p = byId.value.get(tab.project);
  const kids = p ? childrenOf(p.id) : [];
  const asks = openTo(tab.project);
  // a long section shows its newest few, and the open item, until "Show all"
  const shown = (of, k) => q || collapsed["all:" + k] || of.length <= CAP ? of
    : of.filter((i, n) => n < CAP || (i.kind === route.screen && i.id === route.id));
  return html`<aside class="sidebar" aria-label="Project" style=${{ width: sidebarWidth.value + "px" }}>
    <div class="side-top">
      <div class="side-row">
        <button type="button" class=${cx("side-link", route.screen === "home" && "on")} onClick=${() => navigate(tab, { screen: "home" })}>
          <${Icon} name="home" /><span class="side-project">${p?.name ?? tab.project}</span>
        </button>
        <${Button} variant="ghost" icon="sidebar" label="Hide the sidebar (Ctrl+B)" onClick=${() => showSide(false)} />
      </div>
      <div class="search side-search"><${Icon} name="search" size=${14} />
        <input class="input" placeholder="Filter" value=${tab.filter.value} aria-label="Filter this project's items"
          onInput=${e => (tab.filter.value = e.target.value)} onKeyDown=${e => e.key === "Escape" && (tab.filter.value = "")} /></div>
    </div>
    <div class="side-scroll">
      ${SECTIONS.map(k => {
        const of = items.filter(i => i.kind === k);
        const shut = collapsed[k] && !q;
        return html`<section class="side-sec" key=${k}>
          <header class="side-head">
            <button type="button" class="side-toggle" aria-expanded=${!shut} onClick=${() => toggle(k)}>
              <${Icon} name=${shut ? "chevronRight" : "chevronDown"} size=${12} />${KINDS[k].plural}
              <span class="count">${of.length || ""}</span>
            </button>
            <${Button} variant="ghost" size="sm" icon="plus" label=${"New " + KINDS[k].label.toLowerCase()} onClick=${() => create(k, tab)} />
          </header>
          ${!shut && (of.length
            ? html`<${ItemList} tab=${tab} items=${shown(of, k)} current=${route} compact scope="side" label=${KINDS[k].plural} />
              ${of.length > CAP && !q && html`<button type="button" class="side-more" onClick=${() => toggle("all:" + k)}>
                ${collapsed["all:" + k] ? "Show fewer" : `Show all ${of.length}`}</button>`}`
            : html`<p class="side-empty">${q ? "No match" : "None yet"}</p>`)}
        </section>`;
      })}
      ${p && p.id !== "default" && html`<section class="side-sec">
        <header class="side-head">
          <button type="button" class="side-toggle" aria-expanded=${!collapsed.sub} onClick=${() => toggle("sub")}>
            <${Icon} name=${collapsed.sub ? "chevronRight" : "chevronDown"} size=${12} />Subprojects
            <span class="count">${kids.length || ""}</span>
          </button>
          <${Button} variant="ghost" size="sm" icon="plus" label="New subproject" onClick=${() => newProject(p.id)} />
        </header>
        ${!collapsed.sub && (kids.length
          ? html`<ul class="items items-compact" aria-label="Subprojects">${kids.map(k => html`<li key=${k.id} class="item" tabindex="0"
              onClick=${() => openProject(k.id)} onKeyDown=${e => e.key === "Enter" && openProject(k.id)}>
              <span class="item-ic k-project"><${Icon} name="project" /></span>
              <span class="item-text"><span class="item-title">${k.name}</span></span>
              ${asks.some(q => q.project === k.id) && html`<span class="live live-wait" title="Asks this project" />`}
            </li>`)}</ul>`
          : html`<p class="side-empty">None yet</p>`)}
      </section>`}
    </div>
    <div class="side-bottom">
      <${SideLinks} tab=${tab} route=${route} />
    </div>
    <${Resizer} />
  </aside>`;
}

// drag the sidebar's edge to resize it; double-click puts it back
function Resizer() {
  const start = useRef(null);
  const onDown = e => {
    start.current = { x: e.clientX, w: sidebarWidth.value };
    e.currentTarget.setPointerCapture(e.pointerId);
  };
  const onMove = e => {
    if (!start.current) return;
    sidebarWidth.value = Math.max(200, Math.min(480, start.current.w + e.clientX - start.current.x));
  };
  return html`<div class="resizer" role="separator" aria-orientation="vertical" aria-label="Resize the sidebar"
    onPointerDown=${onDown} onPointerMove=${onMove} onPointerUp=${() => (start.current = null)}
    onDblClick=${() => (sidebarWidth.value = 272)} />`;
}

// the pages beside the project's items: its own settings and trash, then the studio's pages,
// which fold away (remembered) to leave the lists more room
const PROJECT_LINKS = [["config", "edit", "Project settings"], ["trash", "trash", "Trash"]];
const STUDIO_LINKS = [["models", "layers", "Models"], ["machines", "server", "Machines"], ["agents", "agent", "Agent settings"],
  ["tools", "build", "Tools"], ["rl", "gauge", "Learning from rewards"], ["settings", "settings", "Studio settings"]];

function SideLinks({ tab, route }) {
  const [open, setOpen] = useState(() => local.get("sideStudio", true));
  const flip = () => { local.set("sideStudio", !open); setOpen(!open); };
  const link = ([screen, icon, label]) => html`<button type="button" key=${screen} class=${cx("side-link", route.screen === screen && "on")}
    onClick=${() => navigate(tab, { screen })}><${Icon} name=${icon} /><span>${label}</span></button>`;
  const here = STUDIO_LINKS.find(l => l[0] === route.screen);
  return html`${PROJECT_LINKS.map(link)}
    <button type="button" class="side-group" aria-expanded=${open} onClick=${flip}>
      <${Icon} name="chevronDown" size=${12} class=${open ? "" : "turned"} /><span>Studio</span></button>
    ${open ? STUDIO_LINKS.map(link) : here && link(here)}`;
}
