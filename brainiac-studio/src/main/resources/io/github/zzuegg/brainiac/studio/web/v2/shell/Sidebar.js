import { html, useRef, cx } from "/v2/lib/html.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button } from "/v2/ui/Button.js";
import { ItemList } from "/v2/ui/ItemList.js";
import { KINDS, SECTIONS } from "/v2/state/kinds.js";
import { byId } from "/v2/state/projects.js";
import { navigate } from "/v2/state/workspace.js";
import { sidebarWidth, sidebarOpen } from "/v2/state/prefs.js";
import { create } from "/v2/wizards/index.js";

// The open project's own things: its home, then a section per kind with a "+" that starts its
// wizard, and a filter over all of them. The trash and settings sit at the bottom.

export function Sidebar({ tab }) {
  const route = tab.route.value;
  const q = tab.filter.value.trim().toLowerCase();
  const items = tab.items.value.filter(i => !q || i.title.toLowerCase().includes(q));
  const collapsed = tab.collapsed.value;
  const toggle = k => (tab.collapsed.value = { ...collapsed, [k]: !collapsed[k] });
  const p = byId.value.get(tab.project);
  return html`<aside class="sidebar" aria-label="Project" style=${{ width: sidebarWidth.value + "px" }}>
    <div class="side-top">
      <div class="side-row">
        <button type="button" class=${cx("side-link", route.screen === "home" && "on")} onClick=${() => navigate(tab, { screen: "home" })}>
          <${Icon} name="home" /><span class="side-project">${p?.name ?? tab.project}</span>
        </button>
        <${Button} variant="ghost" icon="sidebar" label="Hide the sidebar (Ctrl+B)" onClick=${() => (sidebarOpen.value = false)} />
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
            ? html`<${ItemList} tab=${tab} items=${of} current=${route} compact scope="side" label=${KINDS[k].plural} />`
            : html`<p class="side-empty">${q ? "No match" : "None yet"}</p>`)}
        </section>`;
      })}
    </div>
    <div class="side-bottom">
      <button type="button" class=${cx("side-link", route.screen === "trash" && "on")} onClick=${() => navigate(tab, { screen: "trash" })}><${Icon} name="trash" /><span>Trash</span></button>
      <button type="button" class=${cx("side-link", route.screen === "settings" && "on")} onClick=${() => navigate(tab, { screen: "settings" })}><${Icon} name="settings" /><span>Settings</span></button>
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
