import { html, useState, cx } from "/v2/lib/html.js";
import { Icon } from "/v2/ui/Icon.js";
import { openMenu } from "/v2/ui/overlay.js";
import { tabs, activeKey, closeTab, closeOthers, moveTab, OVERVIEW, openOverview } from "/v2/state/workspace.js";
import { forUser, openTo } from "/v2/state/requests.js";
import { byId } from "/v2/state/projects.js";
import { renameProject, remove } from "/v2/state/actions.js";
import { pickProject } from "/v2/shell/ProjectPicker.js";

// Open projects as tabs across the top, like a browser: click to switch, middle-click or × to
// close, drag to reorder, right-click for the project's actions. A dot says when agents in a
// background project work, and turns amber when one waits for you.

export function TabBar() {
  const [drag, setDrag] = useState(null);
  const list = tabs.value;
  const onKey = (e, i) => {
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      const n = list[(i + (e.key === "ArrowRight" ? 1 : -1) + list.length) % list.length];
      activeKey.value = n.key;
      e.currentTarget.parentElement.querySelectorAll("[role=tab]")[list.indexOf(n)]?.focus();
      e.preventDefault();
    } else if (e.key === "Delete") {
      closeTab(list[i].key);
    }
  };
  return html`<nav class="tabbar" aria-label="Open projects">
    <div class="brand" title="Brainiac Studio"><img src="/v2/icon.svg" alt="" width="22" height="22" /></div>
    <button type="button" class=${cx("tab tab-overview", activeKey.value === OVERVIEW && "on")} aria-current=${activeKey.value === OVERVIEW ? "page" : undefined}
        title="Everything across your projects (Ctrl+Shift+O)" onClick=${openOverview}>
      <${Icon} name="home" size=${14} /><span class="tab-name">Overview</span>
      ${forUser.value.length > 0 && html`<span class="tab-count" aria-label=${forUser.value.length + " requests wait for you"}>${forUser.value.length}</span>`}
    </button>
    <div class="tabs" role="tablist">
      ${list.map((t, i) => {
        const p = byId.value.get(t.project);
        const s = t.status.value;
        const asks = openTo(t.project).filter(q => q.answeredBy === "user").length;
        const on = t.key === activeKey.value;
        return html`<div key=${t.key} role="tab" tabindex=${on ? 0 : -1} aria-selected=${on}
            class=${cx("tab", on && "on", drag === t.key && "dragging", !p && "missing")}
            draggable="true"
            onDragStart=${e => { setDrag(t.key); e.dataTransfer.effectAllowed = "move"; }}
            onDragEnd=${() => setDrag(null)}
            onDragOver=${e => { if (drag) e.preventDefault(); }}
            onDrop=${e => { e.preventDefault(); if (drag) moveTab(drag, i); setDrag(null); }}
            onClick=${() => (activeKey.value = t.key)}
            onAuxClick=${e => e.button === 1 && closeTab(t.key)}
            onKeyDown=${e => onKey(e, i)}
            onContextMenu=${e => {
              e.preventDefault();
              openMenu({ x: e.clientX, y: e.clientY }, [
                p && { label: "Rename project", icon: "edit", run: () => renameProject(p) },
                { label: "Open in new window", icon: "window", run: () => window.open("/v2/p/" + encodeURIComponent(t.project) + "/home", "_blank") },
                { separator: true },
                { label: "Close tab", icon: "close", run: () => closeTab(t.key) },
                list.length > 1 && { label: "Close other tabs", run: () => closeOthers(t.key) },
                p && p.id !== "default" && { separator: true },
                p && p.id !== "default" && { label: "Delete project", icon: "trash", danger: true, run: () => remove(null, { kind: "project", id: p.id, title: p.name }) },
              ], p?.name);
            }}>
          ${s.waiting + asks > 0 ? html`<span class="live live-wait" title=${s.waiting + asks + " waiting for you"} />`
            : s.running > 0 ? html`<span class="live live-run" title=${s.running + " working"} />`
            : html`<${Icon} name="project" size=${14} />`}
          <span class="tab-name">${p?.name ?? t.project}</span>
          <button type="button" class="tab-x" tabindex="-1" aria-label=${"Close " + (p?.name ?? t.project)}
            onClick=${e => { e.stopPropagation(); closeTab(t.key); }}><${Icon} name="close" size=${12} /></button>
        </div>`;
      })}
      <button type="button" class="tab-add" aria-label="Open a project" title="Open a project (Ctrl+P)"
        onClick=${e => pickProject(e.currentTarget)}><${Icon} name="plus" /></button>
    </div>
  </nav>`;
}
