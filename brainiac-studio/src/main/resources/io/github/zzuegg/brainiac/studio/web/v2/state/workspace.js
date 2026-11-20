import { signal, computed, effect, batch } from "/v2/lib/html.js";
import { api, local, pageState, remember } from "/v2/lib/api.js";
import { loaded as projectsLoaded, byId } from "/v2/state/projects.js";

// The open projects, one per tab. Each tab has a store of its own: where it is (its route), the
// project's items, its filter and what is being renamed, so switching tabs keeps every
// project exactly as it was left. Several tabs can show the same studio at once because the
// project travels with each API call instead of being a server-wide setting.

const SCREENS = ["home", "chat", "agent", "graph", "trash", "settings", "config", "training", "models", "machines", "agents", "rl", "tools"];

/** Item keys are "kind:id", unique across kinds. */
export const keyOf = item => item.kind + ":" + item.id;

function createTab({ key, project, route }) {
  const tab = {
    key: key || Math.random().toString(36).slice(2, 10),
    project,
    route: signal(route && SCREENS.includes(route.screen) ? route : { screen: "home" }),
    items: signal([]),
    loaded: signal(false),
    error: signal(null),
    filter: signal(""),
    renaming: signal(null),
    collapsed: signal({}),
    scroll: new Map(),
  };
  let inflight = null;
  tab.refresh = () => {
    if (inflight) return inflight;
    inflight = api("/api/items", { project }, { project })
      .then(r => batch(() => {
        tab.items.value = r.items || [];
        tab.loaded.value = true;
        tab.error.value = null;
      }))
      .catch(e => { tab.error.value = e.message; })
      .finally(() => { inflight = null; });
    return inflight;
  };
  /** Changes items in place before the studio confirms, so the list answers at once. */
  tab.patch = fn => { tab.items.value = fn(tab.items.value); };
  /** Running agents and agents waiting on the user, for the tab's status dot. */
  tab.status = computed(() => {
    let running = 0, waiting = 0;
    for (const i of tab.items.value) {
      if (i.kind === "training" && ["queued", "loading", "running"].includes(i.state)) { running++; continue; }
      if (i.kind !== "agent") continue;
      if (i.waiting || i.state === "waiting") waiting++;
      else if (i.state === "running") running++;
    }
    return { running, waiting };
  });
  return tab;
}

export const tabs = signal([]);
export const activeKey = signal(null);

/** The key that shows the overview across every project instead of a tab. */
export const OVERVIEW = "@overview";
export const showingOverview = computed(() => activeKey.value === OVERVIEW);
export const openOverview = () => { activeKey.value = OVERVIEW; };

export const activeTab = computed(() => activeKey.value === OVERVIEW ? null
  : tabs.value.find(t => t.key === activeKey.value) || tabs.value[0] || null);

/** Opens a project in a tab, or brings its tab forward. */
export function openProject(project, { route, focus = true, newTab = false } = {}) {
  let tab = newTab ? null : tabs.value.find(t => t.project === project);
  if (!tab) {
    tab = createTab({ project, route });
    const at = tabs.value.indexOf(activeTab.value);
    const list = [...tabs.value];
    list.splice(at < 0 ? list.length : at + 1, 0, tab);
    tabs.value = list;
    tab.refresh();
  } else if (route) {
    tab.route.value = route;
  }
  if (focus) activeKey.value = tab.key;
  return tab;
}

export function closeTab(key) {
  const list = tabs.value;
  const i = list.findIndex(t => t.key === key);
  if (i < 0) return;
  const next = list.filter(t => t.key !== key);
  batch(() => {
    tabs.value = next;
    if (activeKey.value === key) activeKey.value = (next[i] || next[i - 1] || null)?.key ?? null;
  });
}

export function closeOthers(key) {
  tabs.value = tabs.value.filter(t => t.key === key);
  activeKey.value = key;
}

export function moveTab(key, toIndex) {
  const list = [...tabs.value];
  const from = list.findIndex(t => t.key === key);
  if (from < 0) return;
  const [t] = list.splice(from, 1);
  list.splice(Math.max(0, Math.min(list.length, toIndex)), 0, t);
  tabs.value = list;
}

/** Shows a place in a tab: navigate(tab, {screen: "chat", id}). */
export function navigate(tab, route) {
  tab.route.value = route;
  if (tab.key !== activeKey.value) activeKey.value = tab.key;
}

/** Every tab reloads its items, after a change that may touch several projects. */
export const refreshAll = () => Promise.all(tabs.value.map(t => t.refresh()));

// ---------------------------------------------------------------- the address bar

export function urlOf(tab) {
  if (activeKey.value === OVERVIEW) return "/v2/overview";
  if (!tab) return "/v2/";
  const r = tab.route.value;
  return "/v2/p/" + encodeURIComponent(tab.project) + "/" + r.screen + (r.id != null ? "/" + encodeURIComponent(r.id) : "");
}

export function parseUrl(path = location.pathname) {
  if (/^\/v2\/overview\/?$/.test(path)) return { overview: true };
  const m = path.match(/^\/v2\/p\/([^/]+)(?:\/([a-z]+)(?:\/(.+))?)?\/?$/);
  if (!m) return null;
  const screen = SCREENS.includes(m[2]) ? m[2] : "home";
  return { project: decodeURIComponent(m[1]), route: m[3] ? { screen, id: decodeURIComponent(m[3]) } : { screen } };
}

// ---------------------------------------------------------------- keeping the tabs

/** Opens the tabs this browser (or else the studio) had open, then the one the address names. */
export function restore(fallbackProject) {
  const wanted = parseUrl();
  const saved = local.get("workspace", null) || pageState["v2.workspace"] || null;
  batch(() => {
    const list = (saved?.tabs || []).filter(t => t && typeof t.project === "string").map(createTab);
    tabs.value = list;
    activeKey.value = saved?.active === OVERVIEW || (saved?.active && list.some(t => t.key === saved.active))
      ? saved.active : list[0]?.key ?? null;
    if (wanted?.overview) activeKey.value = OVERVIEW;
    else if (wanted) openProject(wanted.project, { route: wanted.route });
    if (!tabs.value.length) openProject(fallbackProject, { focus: activeKey.value !== OVERVIEW });
  });
  tabs.value.forEach(t => t.refresh());
}

let lastUrl = null;
effect(() => {
  const list = tabs.value;
  const active = activeTab.value;
  const snapshot = {
    tabs: list.map(t => ({ key: t.key, project: t.project, route: t.route.value })),
    active: activeKey.value === OVERVIEW ? OVERVIEW : active?.key ?? null,
  };
  if (!list.length) return;
  local.set("workspace", snapshot);
  remember("v2.workspace", snapshot);
  const url = urlOf(active);
  if (url !== location.pathname) {
    // a new place is a step back can return to; the first one only replaces the address
    if (lastUrl === null) history.replaceState(null, "", url);
    else history.pushState(null, "", url);
  }
  lastUrl = url;
});

addEventListener("popstate", () => {
  const wanted = parseUrl();
  if (wanted) {
    lastUrl = location.pathname;
    if (wanted.overview) openOverview();
    else openProject(wanted.project, { route: wanted.route });
  }
});

// the open tab is kept fresh; the others often enough for their status dots
let tick = 0;
setInterval(() => {
  if (document.hidden) return;
  tick++;
  for (const t of tabs.value) {
    // a tab whose project was deleted elsewhere waits for the project to come back
    if (projectsLoaded.value && !byId.value.has(t.project)) continue;
    if (t === activeTab.value || tick % 4 === 0) t.refresh();
  }
}, 3000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshAll(); });
