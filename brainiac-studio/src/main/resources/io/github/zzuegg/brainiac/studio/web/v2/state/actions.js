import { html, useState } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { toast, fail } from "/v2/state/toasts.js";
import { kindOf } from "/v2/state/kinds.js";
import { keyOf, navigate, refreshAll, tabs, closeTab, openProject } from "/v2/state/workspace.js";
import { projects, loadProjects, projectName } from "/v2/state/projects.js";
import { openLayer } from "/v2/ui/overlay.js";
import { Layer } from "/v2/ui/Layer.js";
import { Button } from "/v2/ui/Button.js";
import { Icon } from "/v2/ui/Icon.js";

// The actions every item has, in one place: open, rename, duplicate, move to another project
// and delete with undo. Lists, headers and the keyboard all call these, so an item behaves
// the same wherever it shows.

export function open(tab, item) {
  if (item.kind === "project") return openProject(item.id);
  navigate(tab, { screen: item.kind, id: item.id });
}

/**
 * Starts renaming in place. An item can show in several places at once (sidebar, home, header),
 * so `scope` names the one that edits it.
 */
export function startRename(tab, item, scope = "head") {
  tab.renaming.value = scope + "|" + keyOf(item);
}

/** Whether an item is being renamed in a place. */
export const isRenaming = (tab, item, scope) => tab.renaming.value === scope + "|" + keyOf(item);

export async function rename(tab, item, title) {
  if (tab) tab.renaming.value = null;
  title = title.trim();
  if (!title || title === item.title) return;
  const before = item.title;
  tab?.patch(list => list.map(i => (keyOf(i) === keyOf(item) ? { ...i, title } : i)));
  try {
    const r = await api("/api/items/rename", { kind: item.kind, id: item.id, title });
    if (item.kind === "project") await loadProjects();
    // a network is named by its file, so its id follows its name
    if (r.item.id !== item.id && tab) {
      const route = tab.route.value;
      if (route.screen === item.kind && route.id === item.id) tab.route.value = { ...route, id: r.item.id };
    }
    await tab?.refresh();
  } catch (e) {
    tab?.patch(list => list.map(i => (keyOf(i) === keyOf(item) ? { ...i, title: before } : i)));
    fail(e);
  }
}

export async function duplicate(tab, item) {
  try {
    const r = await api("/api/items/duplicate", { kind: item.kind, id: item.id });
    await tab.refresh();
    toast(`${kindOf(item.kind).label} copied as “${r.item.title}”`);
    return r.item;
  } catch (e) {
    fail(e);
  }
}

export async function move(tab, item) {
  const to = await openLayer(MoveDialog, { item, from: tab.project });
  if (!to || to === tab.project) return;
  try {
    await api("/api/items/move", { kind: item.kind, id: item.id, project: to });
    leave(tab, item);
    await refreshAll();
    toast(`Moved to ${projectName(to)}`, {
      undo: async () => {
        await api("/api/items/move", { kind: item.kind, id: item.id, project: tab.project });
        await refreshAll();
      },
    });
  } catch (e) {
    fail(e);
  }
}

/** Deletes at once; the toast (and Ctrl+Z) brings it back from the trash. */
export async function remove(tab, item) {
  const k = kindOf(item.kind);
  if (item.kind === "agent" && (item.state === "running" || item.state === "waiting")) {
    return fail("Stop the agent before deleting it.");
  }
  tab?.patch(list => list.filter(i => keyOf(i) !== keyOf(item)));
  leave(tab, item);
  try {
    const r = await api("/api/items/delete", { kind: item.kind, id: item.id });
    if (item.kind === "project") {
      await loadProjects();
      tabs.value.filter(t => t.project === item.id).forEach(t => closeTab(t.key));
    }
    toast(`${k.label} “${item.title}” deleted`, {
      undo: async () => {
        await api("/api/trash/restore", { id: r.trash.id });
        if (item.kind === "project") {
          await loadProjects();
          openProject(item.id);
        }
        await refreshAll();
      },
    });
  } catch (e) {
    await tab?.refresh();
    fail(e);
  }
}

// a tab showing an item that went away goes back to its project's home
function leave(tab, item) {
  for (const t of tabs.value) {
    const r = t.route.value;
    if (r.screen === item.kind && r.id === item.id) t.route.value = { screen: "home" };
  }
}

/** The shared menu for an item, in the same order everywhere; entries that do not apply are left out. */
export function menuFor(tab, item, scope = "head") {
  const can = new Set(kindOf(item.kind).actions);
  return [
    can.has("open") && { label: "Open", icon: "open", hint: "Enter", run: () => open(tab, item) },
    item.kind === "project" && { label: "Open in new window", icon: "window", run: () => window.open("/v2/p/" + encodeURIComponent(item.id) + "/home", "_blank") },
    can.has("rename") && { label: "Rename", icon: "edit", hint: "F2", run: () => (item.kind === "project" ? renameProject(item) : startRename(tab, item, scope)) },
    can.has("duplicate") && { label: "Duplicate", icon: "copy", hint: "Ctrl+D", run: () => duplicate(tab, item) },
    can.has("move") && projects.value.length > 1 && { label: "Move to project…", icon: "move", run: () => move(tab, item) },
    can.has("delete") && { separator: true },
    can.has("delete") && {
      label: "Delete", icon: "trash", hint: "Del", danger: true,
      disabled: item.kind === "project" && item.id === "default",
      run: () => remove(tab, item),
    },
  ];
}

/** Renames a project from a dialog (its name shows in tabs, not in a list). */
export async function renameProject(project) {
  const title = await openLayer(RenameDialog, { title: "Rename project", value: project.name ?? project.title });
  if (title) await rename(null, { kind: "project", id: project.id, title: project.name ?? project.title }, title);
}

function RenameDialog({ title, value, close, top }) {
  const [v, setV] = useState(value);
  const done = () => close(v.trim() || undefined);
  return html`<${Layer} title=${title} close=${() => close()} top=${top}
      footer=${html`<${Button} data-close onClick=${() => close()}>Cancel<//><${Button} variant="primary" onClick=${done}>Rename<//>`}>
    <form onSubmit=${e => { e.preventDefault(); done(); }}>
      <input class="input" value=${v} onInput=${e => setV(e.target.value)} autofocus maxlength="200" aria-label="Name" />
    </form>
  <//>`;
}

function MoveDialog({ item, from, close, top }) {
  const others = projects.value.filter(p => p.id !== from);
  return html`<${Layer} title=${`Move “${item.title}”`} subtitle="Choose the project it belongs to." close=${() => close()} top=${top}>
    <ul class="pick-list">
      ${others.map(p => html`<li><button type="button" class="pick" onClick=${() => close(p.id)}>
        <${Icon} name="project" /><span class="pick-title">${p.name}</span>
        ${p.description && html`<span class="pick-sub">${p.description}</span>`}
      </button></li>`)}
    </ul>
  <//>`;
}
