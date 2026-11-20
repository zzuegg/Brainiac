import { html, useEffect, useState } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { slug } from "/v2/lib/format.js";
import { openLayer } from "/v2/ui/overlay.js";
import { Wizard, Field, Choices, Summary } from "/v2/ui/Wizard.js";
import { projects, loadProjects } from "/v2/state/projects.js";
import { openProject, navigate } from "/v2/state/workspace.js";
import { toast, fail } from "/v2/state/toasts.js";

// The "New …" flows. Each one is a short wizard built from the shared parts; `create(kind, tab)`
// is what every "+" button calls.

export function create(kind, tab) {
  if (kind === "project") return newProject();
  if (kind === "chat") return newChat(tab);
  if (kind === "agent") return openLayer(NewAgent, { tab });
  if (kind === "graph") return openLayer(NewNetwork, { tab });
}

// ---------------------------------------------------------------- project

/** Starts the new-project wizard; with a parent, the project is made part of it. */
export async function newProject(parent) {
  const id = await openLayer(NewProject, { parent });
  if (id) openProject(id);
}

function NewProject({ parent, close, top }) {
  const parentName = parent && projects.value.find(p => p.id === parent)?.name;
  return html`<${Wizard} title=${parent ? "New subproject of " + (parentName || parent) : "New project"}
    draftKey=${parent ? "project:" + parent : "project"} close=${close} top=${top}
    initial=${{ name: "", description: "", folder: "", start: "empty" }}
    steps=${[
      {
        title: "Name and purpose",
        missing: d => (d.name.trim() ? null : "Give the project a name."),
        body: (d, set) => html`
          <${Field} label="Name"><input class="input" value=${d.name} autofocus maxlength="80"
              placeholder="Image classifier experiments" onInput=${e => set({ name: e.target.value })} /><//>
          <${Field} label="What is it for?" hint="Agents in this project get this as their brief. Optional.">
            <textarea class="input" rows="3" value=${d.description} onInput=${e => set({ description: e.target.value })}
              placeholder="Train and compare small CNNs on my photo folders." /><//>`,
      },
      {
        title: "Folder",
        body: (d, set) => html`
          <${Field} label="A folder on this PC its agents may read" hint=${parent ? "Leave empty to use the folder of the project it is part of." : "Leave empty for none; you can add one later in the project's settings."}>
            <input class="input mono" value=${d.folder} placeholder="D:\\Projects\\classifier" onInput=${e => set({ folder: e.target.value })} /><//>`,
      },
      {
        title: "Start from",
        body: (d, set) => html`<${Choices} name="Start from" value=${d.start} onChange=${v => set({ start: v })} options=${[
          { value: "empty", label: "Empty", icon: "project", text: "Nothing in it yet." },
          { value: "chat", label: "A first chat", icon: "chat", text: "Opens with a new conversation." },
          { value: "network", label: "A first network", icon: "graph", text: "Opens with an empty network to build." },
        ]} />`,
      },
    ]}
    summary=${(d, goTo) => html`<${Summary} goTo=${goTo}
      lead=${parent
        ? `Creates “${d.name.trim() || "…"}” as part of ${parentName || parent}, and opens it in a new tab. It can use only what ${parentName || parent} allows, and agents there can work in it.`
        : `Creates the project “${d.name.trim() || "…"}” and opens it in a new tab.`}
      rows=${[
        { label: "Name", value: d.name.trim(), step: 0 },
        { label: "Purpose", value: d.description.trim(), step: 0 },
        { label: "Folder", value: d.folder.trim(), step: 1 },
        { label: "Starts with", value: { empty: "nothing", chat: "a chat", network: "a network" }[d.start], step: 2 },
      ]} />`}
    finish=${async d => {
      const r = await api("/api/projects", {
        create: true,
        project: { id: slug(d.name), name: d.name.trim(), description: d.description.trim(), folder: d.folder.trim() || null,
          parents: parent ? [parent] : [] },
      });
      await loadProjects();
      const id = r.saved;
      if (d.start === "chat") await makeChat(id);
      if (d.start === "network") await api("/api/builder/graphs", { name: "network", graph: { nodes: [], edges: [] } }, { project: id });
      return id;
    }} />`;
}

// ---------------------------------------------------------------- chat

const chatId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);

async function makeChat(project) {
  const id = chatId();
  await api("/api/chats", { id, title: "New chat", messages: [], status: "open", updated: Date.now(), created: Date.now() }, { project });
  return id;
}

export async function newChat(tab) {
  try {
    const id = await makeChat(tab.project);
    await tab.refresh();
    navigate(tab, { screen: "chat", id });
    tab.renaming.value = "head|chat:" + id;
  } catch (e) {
    fail(e);
  }
}

// ---------------------------------------------------------------- agent

function NewAgent({ tab, close, top }) {
  const [roles, setRoles] = useState([]);
  useEffect(() => {
    api("/api/roles", undefined, { project: tab.project }).then(r => setRoles(r.roles || [])).catch(() => setRoles([]));
  }, []);
  const roleName = id => roles.find(r => r.id === id)?.name || id || "the default role";
  return html`<${Wizard} title="New agent" draftKey=${"agent:" + tab.project} close=${close} top=${top}
    initial=${{ goal: "", role: "", mode: "ask", maxSteps: 12 }}
    finishLabel="Start agent"
    steps=${[
      {
        title: "Goal",
        missing: d => (d.goal.trim() ? null : "Say what the agent should do."),
        body: (d, set) => html`<${Field} label="What should the agent do?" hint="One clear outcome works best. It can ask you when it is unsure.">
          <textarea class="input" rows="5" autofocus value=${d.goal} onInput=${e => set({ goal: e.target.value })}
            placeholder="Find three small image models that run on this PC and compare their speed." /><//>`,
      },
      {
        title: "Role",
        body: (d, set) => html`<${Choices} name="Role" value=${d.role} onChange=${v => set({ role: v })} options=${[
          { value: "", label: "Default", text: "The studio picks tools for the goal." },
          ...roles.slice(0, 8).map(r => ({ value: r.id, label: r.name || r.id, text: r.description || "" })),
        ]} />`,
      },
      {
        title: "Autonomy",
        body: (d, set) => html`
          <${Choices} name="Autonomy" value=${d.mode} onChange=${v => set({ mode: v })} options=${[
            { value: "plan", label: "Plan first", text: "It shows a plan and waits for your go-ahead." },
            { value: "ask", label: "Ask before acting", text: "It asks before anything that changes files or costs money." },
            { value: "auto", label: "Automatic", text: "It acts on its own within the project's limits." },
          ]} />
          <${Field} label="Step budget" hint="It stops after this many steps and reports back.">
            <input class="input narrow" type="number" min="1" max="200" value=${d.maxSteps}
              onInput=${e => set({ maxSteps: Math.max(1, Math.min(200, +e.target.value || 12)) })} /><//>`,
      },
    ]}
    summary=${(d, goTo) => html`<${Summary} goTo=${goTo}
      lead=${"Starts an agent in this project as " + roleName(d.role) + "."}
      rows=${[
        { label: "Goal", value: d.goal.trim(), step: 0 },
        { label: "Role", value: roleName(d.role), step: 1 },
        { label: "Autonomy", value: { plan: "plan first", ask: "ask before acting", auto: "automatic" }[d.mode] + `, up to ${d.maxSteps} steps`, step: 2 },
      ]} />`}
    finish=${async d => {
      const r = await api("/api/agents/start", { goal: d.goal.trim(), role: d.role, mode: d.mode, maxSteps: d.maxSteps }, { project: tab.project });
      await tab.refresh();
      navigate(tab, { screen: "agent", id: r.id });
      return r.id;
    }} />`;
}

// ---------------------------------------------------------------- network

function NewNetwork({ tab, close, top }) {
  return html`<${Wizard} title="New network" draftKey=${"graph:" + tab.project} close=${close} top=${top}
    initial=${{ name: "", start: "blank" }}
    steps=${[
      {
        title: "Name",
        missing: d => (/^[A-Za-z0-9][A-Za-z0-9 ._-]{0,63}$/.test(d.name.trim()) ? null
          : "Use 1 to 64 letters, digits, spaces, dots, dashes or underscores, starting with a letter or digit."),
        body: (d, set) => html`<${Field} label="Name" hint="The built model is saved under this name too.">
          <input class="input" value=${d.name} autofocus maxlength="64" placeholder="digits-cnn" onInput=${e => set({ name: e.target.value })} /><//>`,
      },
      {
        title: "Start from",
        body: (d, set) => html`<${Choices} name="Start from" value=${d.start} onChange=${v => set({ start: v })} options=${[
          { value: "blank", label: "Blank", icon: "graph", text: "An empty canvas." },
        ]} />
        <p class="muted small">Templates and sizing from a dataset come with the Experiments step.</p>`,
      },
    ]}
    summary=${(d, goTo) => html`<${Summary} goTo=${goTo} lead=${`Creates the network “${d.name.trim()}” in this project.`}
      rows=${[{ label: "Name", value: d.name.trim(), step: 0 }, { label: "Start from", value: "blank", step: 1 }]} />`}
    finish=${async d => {
      const name = d.name.trim();
      const existing = await api("/api/builder/graphs", undefined, { project: tab.project });
      if ((existing.graphs || []).some(g => g.name === name)) throw new Error(`There is already a network called “${name}”.`);
      await api("/api/builder/graphs", { name, graph: { nodes: [], edges: [] } }, { project: tab.project });
      await tab.refresh();
      navigate(tab, { screen: "graph", id: name });
      toast(`Network “${name}” created`);
      return name;
    }} />`;
}

export { projects };
