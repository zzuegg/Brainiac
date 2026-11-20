import { html, useEffect, signal } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button } from "/v2/ui/Button.js";
import { MenuHost } from "/v2/ui/Menu.js";
import { LayerHost } from "/v2/ui/Layer.js";
import { Toasts } from "/v2/ui/Toasts.js";
import { TabBar } from "/v2/shell/TabBar.js";
import { Sidebar } from "/v2/shell/Sidebar.js";
import { pickProject } from "/v2/shell/ProjectPicker.js";
import { activeTab, tabs, closeTab, navigate } from "/v2/state/workspace.js";
import { byId, loaded } from "/v2/state/projects.js";
import { sidebarOpen } from "/v2/state/prefs.js";
import { undoLast } from "/v2/state/toasts.js";
import { layers, menu } from "/v2/ui/overlay.js";
import { Home } from "/v2/screens/Home.js";
import { Chat } from "/v2/screens/Chat.js";
import { Agent } from "/v2/screens/Agent.js";
import { Graph } from "/v2/screens/Graph.js";
import { Trash } from "/v2/screens/Trash.js";
import { Settings } from "/v2/screens/Settings.js";

const SCREENS = { home: Home, chat: Chat, agent: Agent, graph: Graph, trash: Trash, settings: Settings };

export function App() {
  useEffect(() => {
    const onKey = e => {
      const typing = e.target.closest?.("input, textarea, select, [contenteditable]");
      const mod = e.ctrlKey || e.metaKey;
      if (layers.value.length || menu.value) return;
      if (mod && !e.shiftKey && e.key.toLowerCase() === "b") sidebarOpen.value = !sidebarOpen.value;
      else if (mod && !e.shiftKey && e.key.toLowerCase() === "p") pickProject();
      else if (mod && !e.shiftKey && e.key.toLowerCase() === "z" && !typing) { if (!undoLast()) return; }
      else if (e.altKey && /^[1-9]$/.test(e.key)) {
        const t = tabs.value[+e.key - 1];
        if (t) navigate(t, t.route.value);
      } else return;
      e.preventDefault();
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, []);

  const tab = activeTab.value;
  return html`<div class="app">
    <${TabBar} />
    <div class="body">
      ${tab ? html`<${Workspace} key=${tab.key} tab=${tab} />` : html`<${NoTabs} />`}
    </div>
    <${StatusBar} />
    <${MenuHost} />
    <${LayerHost} />
    <${Toasts} />
  </div>`;
}

function Workspace({ tab }) {
  const gone = loaded.value && !byId.value.has(tab.project);
  if (gone) {
    return html`<main class="main"><div class="gone">
      <p>The project “${tab.project}” is not there any more. If it was deleted, the trash can bring it back.</p>
      <div class="row-btns"><${Button} onClick=${() => navigate(tab, { screen: "trash" })}>Open the trash<//>
        <${Button} variant="ghost" onClick=${() => closeTab(tab.key)}>Close this tab<//></div>
      ${tab.route.value.screen === "trash" && html`<${Trash} />`}
    </div></main>`;
  }
  const Screen = SCREENS[tab.route.value.screen] || Home;
  return html`
    ${sidebarOpen.value && html`<${Sidebar} tab=${tab} />`}
    <main class=${"main" + (sidebarOpen.value ? "" : " main-solo")} id="main">
      ${!sidebarOpen.value && html`<button type="button" class="side-flip btn btn-ghost btn-icon" aria-label="Show the sidebar"
        title="Show the sidebar (Ctrl+B)" onClick=${() => (sidebarOpen.value = true)}><${Icon} name="sidebar" /></button>`}
      <${Screen} tab=${tab} />
    </main>`;
}

function NoTabs() {
  return html`<main class="main"><div class="gone">
    <p>No project is open.</p>
    <${Button} variant="primary" icon="project" onClick=${() => pickProject()}>Open a project<//>
  </div></main>`;
}

// read-only: what runs where, and whether the studio answers
const online = signal(true);
setInterval(() => {
  if (!document.hidden) api("/api/health").then(() => (online.value = true)).catch(() => (online.value = false));
}, 10_000);

function StatusBar() {
  let running = 0, waiting = 0;
  for (const t of tabs.value) {
    const s = t.status.value;
    running += s.running;
    waiting += s.waiting;
  }
  return html`<footer class="statusbar">
    <span class=${"status-dot " + (online.value ? "ok" : "bad")} />${online.value ? "Studio running" : "Studio not answering"}
    ${running > 0 && html`<span class="status-item"><span class="live live-run" />${running} working</span>`}
    ${waiting > 0 && html`<span class="status-item"><span class="live live-wait" />${waiting} waiting for you</span>`}
    <span class="grow" />
    <span class="status-item muted">Studio v2 preview</span>
    <a class="status-item" href="/">Classic Studio</a>
  </footer>`;
}
