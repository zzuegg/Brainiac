import { html, useEffect, useState, signal, cx } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button } from "/v2/ui/Button.js";
import { Choices, Field } from "/v2/ui/Wizard.js";
import { byId, projectName, saveProject } from "/v2/state/projects.js";
import { modelInfo, loadModels, placeOf } from "/v2/state/models.js";
import { editParents, remove } from "/v2/state/actions.js";
import { navigate, openProject } from "/v2/state/workspace.js";
import { toast, fail } from "/v2/state/toasts.js";

// A project's own settings: what it is about and the folder its agents may read, which model
// each kind of task uses here, the GPU machines it may rent, and what its agents may use. A
// project that is part of others can use only what they allow too; where that narrows a
// setting, the page says so. Changes wait in a draft (kept while you switch tabs) until saved.

const MEDIA = [["text", "Chat"], ["image", "Images"], ["speech", "Speech"], ["transcribe", "Transcription"], ["video", "Video"]];
const POLICIES = [["swap", "Swap", "Unload idle models of other kinds to make room"], ["queue", "Queue", "Wait until there is room"],
  ["remote", "Remote", "Send it to the route's remote runner"]];

const drafts = new Map();                 // project id -> its unsaved draft
const tools = signal(null);               // every tool agents have: [{name, description, acts}]
let toolsLoad = null;
const loadTools = () => (toolsLoad ||= api("/api/agents").then(r => (tools.value = r.tools || []), e => { toolsLoad = null; throw e; }));

// the project as the studio keeps it, without what the page adds for showing
const own = p => {
  const { effective, ...rest } = p;
  return JSON.parse(JSON.stringify(rest));
};
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export function ProjectSettings({ tab }) {
  const p = byId.value.get(tab.project);
  const [draft, setDraftState] = useState(() => drafts.get(tab.project) || (p && own(p)));
  const [busy, setBusy] = useState(false);
  const [version, setVersion] = useState(0);     // fields that keep their own text start over on undo and save
  useEffect(() => { loadTools().catch(fail); if (!modelInfo.value) loadModels(); }, []);
  useEffect(() => { if (!draft && p) setDraftState(own(p)); }, [p]);
  if (!p || !draft) return html`<div class="loading" aria-busy="true" />`;

  const saved = own(p);
  const dirty = !same(draft, saved);
  const set = changes => {
    const d = { ...draft, ...changes };
    if (same(d, saved)) drafts.delete(tab.project); else drafts.set(tab.project, d);
    setDraftState(d);
  };
  const undo = () => { drafts.delete(tab.project); setDraftState(saved); setVersion(v => v + 1); };
  const save = async () => {
    if (!draft.name.trim()) return fail(new Error("A project needs a name."));
    setBusy(true);
    try {
      const runners = Object.fromEntries(Object.entries(draft.runners || {}).map(([n, r]) => [n, Object.fromEntries(Object.entries(r)
        .map(([k, v]) => [k, NUMBERS.includes(k) && v !== "" && !isNaN(+v) ? +v : v]))]));
      const d = { ...draft, runners, name: draft.name.trim(), folder: draft.folder?.trim() || null, description: draft.description || "" };
      await saveProject(p.id, d);
      drafts.delete(tab.project);
      setDraftState(own(byId.value.get(p.id) || d));
      setVersion(v => v + 1);
      toast("Saved");
    } catch (e) {
      fail(e);
    }
    setBusy(false);
  };
  const eff = p.effective;

  return html`<div class="page page-narrow project-settings">
    <header class="page-head">
      <div>
        <p class="eyebrow">Project settings</p>
        <h1 class="page-title">${p.name}</h1>
        <p class="page-lead">What its agents work on, where its tasks run, and what they may use.</p>
      </div>
      <div class="page-actions"><${Button} variant="ghost" icon="home" onClick=${() => navigate(tab, { screen: "home" })}>Project home<//></div>
    </header>

    <section class="card">
      <h2 class="card-title">About</h2>
      <${Field} label="Name"><input class="input" value=${draft.name} maxlength="60" onInput=${e => set({ name: e.target.value })} /><//>
      <${Field} label="What it is about" hint="Agents get this with every task.">
        <textarea class="input" rows="3" value=${draft.description || ""} onInput=${e => set({ description: e.target.value })} /><//>
      <${Field} label="A folder on this PC its agents may read"
        hint=${eff && !draft.folder && eff.folder ? `Empty: it uses ${eff.folder}, from the project it is part of.` : "Agents may read and list files there, and nothing outside it. Empty for none."}>
        <input class="input mono" value=${draft.folder || ""} placeholder="such as D:/Photos/low-light" onInput=${e => set({ folder: e.target.value })} /><//>
      ${p.id !== "default" && html`<div class="field"><span class="field-label">Part of</span>
        <p class="settings-line">${(p.parents || []).length
          ? (p.parents.map((id, i) => html`${i ? ", " : ""}<button type="button" class="link" onClick=${() => openProject(id)}>${projectName(id)}</button>`))
          : html`<span class="muted">No other project</span>`}
          <button type="button" class="link small" onClick=${() => editParents(p.id)}>Change…</button></p></div>`}
    </section>

    <section class="card">
      <h2 class="card-title">Where tasks run</h2>
      <p class="muted small card-lead">Auto in this project uses the model set here. When it runs on this PC and the GPU is full, the policy decides; Remote sends it to a runner below.</p>
      <div class="routes">${MEDIA.map(([m, label]) => html`<${Route} key=${m} m=${m} label=${label} draft=${draft} set=${set} />`)}</div>
    </section>

    <${Runners} draft=${draft} set=${set} />

    <${Allowed} key=${"a" + version} draft=${draft} set=${set} eff=${eff} />

    <${AgentNotes} project=${p.id} />

    ${p.id !== "default" && html`<section class="card">
      <h2 class="card-title">Requests from subprojects</h2>
      <${Choices} name="Who answers requests" value=${draft.requests || "user"} onChange=${v => set({ requests: v })} options=${[
        { value: "user", label: "Me", text: "They wait in your overview." },
        { value: "agents", label: "This project's agents", text: "Within what this project allows; you can still answer." },
      ]} />
    </section>`}

    ${p.id !== "default" && html`<section class="card danger-zone">
      <h2 class="card-title">Delete this project</h2>
      <p class="muted small card-lead">It goes to the trash with its chats, agents and networks, and its subprojects that are part of nothing else. Restore it from there for 30 days.</p>
      <${Button} variant="danger" icon="trash" onClick=${() => remove(tab, { kind: "project", id: p.id, title: p.name })}>Delete ${p.name}<//>
    </section>`}

    <div class=${cx("savebar", dirty && "on")} role="region" aria-label="Unsaved changes" hidden=${!dirty}>
      <span class="live live-wait" /><span>Unsaved changes to ${p.name}</span><span class="grow" />
      <${Button} variant="ghost" onClick=${undo} disabled=${busy}>Undo changes<//>
      <${Button} variant="primary" icon="check" onClick=${save} disabled=${busy}>${busy ? "Saving…" : "Save"}<//>
    </div>
  </div>`;
}

/** What agents noted about this project for later tasks: read, corrected or cleared here. Saved on its own. */
function AgentNotes({ project }) {
  const [text, setText] = useState(null);
  const [kept, setKept] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => { api("/api/agents/notes", { project }).then(r => { setText(r.text || ""); setKept(r.text || ""); }, fail); }, [project]);
  const save = async () => {
    setBusy(true);
    try { await api("/api/agents/notes/save", { project, text }); setKept(text); toast(text.trim() ? "Notes saved" : "Notes cleared"); }
    catch (e) { fail(e); }
    setBusy(false);
  };
  return html`<section class="card">
    <h2 class="card-title">Agent notes</h2>
    <p class="muted small card-lead">Agents keep notes here for their later tasks in this project, and read them first. Correct what is wrong, or clear it to start over.</p>
    ${text === null ? html`<p class="muted small">Reading…</p>` : html`
      <textarea class="input mono notes-text" rows=${Math.min(16, Math.max(4, text.split("\n").length + 1))} value=${text} spellcheck="false"
        placeholder="Nothing noted yet." onInput=${e => setText(e.target.value)} />
      ${text !== kept && html`<div class="row-btns"><span class="grow" />
        <${Button} variant="ghost" onClick=${() => setText(kept)} disabled=${busy}>Undo changes<//>
        <${Button} variant="primary" icon="check" onClick=${save} disabled=${busy}>${busy ? "Saving…" : "Save notes"}<//></div>`}`}
  </section>`;
}

function modelsFor(m, remoteOnly) {
  return (modelInfo.value?.models || []).filter(x => (x.media || []).includes(m) && (!remoteOnly || x.provider !== "local"));
}

function Route({ m, label, draft, set }) {
  const r = (draft.routes || {})[m] || {};
  const change = (k, v) => {
    const routes = { ...(draft.routes || {}) };
    const n = { model: r.model ?? null, policy: r.policy ?? null, fallback: r.fallback ?? null, [k]: v || null };
    if (!n.model && !n.policy && !n.fallback) delete routes[m]; else routes[m] = n;
    set({ routes });
  };
  const ms = modelsFor(m, false);
  const runners = Object.keys(draft.runners || {}).map(n => "vast:" + n);
  const remote = [...runners, ...modelsFor(m, true).map(x => x.key)];
  const missing = (v, list) => v && !list.includes(v);
  return html`<div class="route">
    <span class="route-name">${label}</span>
    <label><span class="route-l">Model</span>
      <select class="input select-sm" value=${r.model || ""} onChange=${e => change("model", e.target.value)} aria-label=${label + " model"}>
        <option value="">Auto, as the studio picks</option>
        ${missing(r.model, ms.map(x => x.key)) && html`<option value=${r.model}>${r.model} (not found)</option>`}
        ${ms.map(x => html`<option value=${x.key}>${x.name} · ${placeOf(x)}</option>`)}
      </select></label>
    <label><span class="route-l">When the GPU is full</span>
      <select class="input select-sm" value=${r.policy || ""} onChange=${e => change("policy", e.target.value)} aria-label=${label + ": when the GPU is full"}>
        <option value="">The studio's choice</option>
        ${POLICIES.map(([k, n, d]) => html`<option value=${k} title=${d}>${n}</option>`)}
      </select></label>
    <label><span class="route-l">Remote runner</span>
      <select class="input select-sm" value=${r.fallback || ""} onChange=${e => change("fallback", e.target.value)} aria-label=${label + " remote runner"}>
        <option value="">None</option>
        ${missing(r.fallback, remote) && html`<option value=${r.fallback}>${r.fallback} (not found)</option>`}
        ${runners.map(k => html`<option value=${k}>${k.slice(5)} · Vast.ai runner</option>`)}
        ${modelsFor(m, true).map(x => html`<option value=${x.key}>${x.name} · ${placeOf(x)}</option>`)}
      </select></label>
  </div>`;
}

const NUMBERS = ["minGpuRam", "maxPrice", "hours"];
const RUNNER_FIELDS = [["gpu", "GPU", "any, or 4090"], ["minGpuRam", "GB at least", "24", true], ["maxPrice", "Max $/hour", "0.60", true],
  ["hours", "Hours rented", "2", true], ["repo", "Hugging Face repository", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"], ["option", "File or folder", "folder"]];

function Runners({ draft, set }) {
  const runners = Object.entries(draft.runners || {});
  const put = (list) => set({ runners: Object.fromEntries(list) });
  const add = () => {
    let n = "runner";
    for (let i = 2; draft.runners?.[n]; i++) n = "runner-" + i;
    put([...runners, [n, { hours: 2, option: "folder" }]]);
  };
  return html`<section class="card">
    <h2 class="card-title">GPU machines to rent<${Button} size="sm" icon="plus" onClick=${add}>Add runner<//></h2>
    <p class="muted small card-lead">A Vast.ai machine rented when a route first sends it a task: the cheapest verified offer within your price, with the model fetched from Hugging Face. It stays rented for the hours set, so later tasks reuse it.</p>
    ${runners.length === 0 ? html`<p class="muted small">None. Add one, then pick it as a route's remote runner above.</p>`
      : runners.map(([name, r], at) => html`<div class="runner" key=${at}>
        <div class="runner-head"><span class="mono muted">vast:</span>
          <input class="input select-sm mono" value=${name} aria-label="Runner name"
            onChange=${e => { const v = e.target.value.trim(); if (v && !draft.runners[v]) put(runners.map((x, i) => (i === at ? [v, x[1]] : x))); }} />
          <span class="grow" />
          <${Button} variant="ghost" size="sm" icon="trash" label=${"Remove runner " + name} onClick=${() => put(runners.filter((_, i) => i !== at))} /></div>
        <div class="runner-grid">${RUNNER_FIELDS.map(([k, label, ph, num]) => html`<label class=${cx("field", (k === "repo") && "wide")} key=${k}>
          <span class="field-label">${label}</span>
          <input class=${cx("input", num && "mono")} inputmode=${num ? "decimal" : undefined} placeholder=${ph} value=${r[k] ?? ""}
            onInput=${e => { const v = e.target.value.trim(); const n = { ...r }; if (v === "") delete n[k]; else n[k] = v; put(runners.map((x, i) => (i === at ? [name, n] : x))); }} /></label>`)}</div>
      </div>`)}
  </section>`;
}

function Allowed({ draft, set, eff }) {
  const all = tools.value;
  const every = draft.tools == null;
  const allowedByParents = eff?.tools;            // null: the parents allow every tool
  const toggle = (name, on) => set({ tools: on ? [...draft.tools, name] : draft.tools.filter(t => t !== name) });
  const [q, setQ] = useState("");
  const [models, setModels] = useState(() => (draft.models || []).join("\n"));
  const shown = (all || []).filter(t => !q || (t.name + " " + (t.description || "")).toLowerCase().includes(q.toLowerCase()));
  return html`<section class="card">
    <h2 class="card-title">What agents may use</h2>
    <label class="check"><input type="checkbox" checked=${every}
      onChange=${e => set({ tools: e.target.checked ? null : (all || []).map(t => t.name) })} />
      <span><b>Every tool</b> <span class="muted">· new tools are allowed as they appear</span></span></label>
    ${allowedByParents && html`<p class="flag small">The projects it is part of allow ${allowedByParents.length} tools; others stay off here even when ticked.</p>`}
    ${!every && html`<div class="search tool-search"><${Icon} name="search" size=${14} />
        <input class="input select-sm" placeholder="Find a tool" value=${q} onInput=${e => setQ(e.target.value)} /></div>
      <div class="toolpills">${all ? shown.map(t => html`<label key=${t.name} class=${cx("pill", allowedByParents && !allowedByParents.includes(t.name) && "pill-off")} title=${t.description || ""}>
        <input type="checkbox" checked=${draft.tools.includes(t.name)} onChange=${e => toggle(t.name, e.target.checked)} />
        <span class="mono">${t.name}</span>${t.acts && html`<span class="pill-acts">acts</span>`}</label>`)
        : html`<span class="muted small">Reading the tools…</span>`}</div>
      <p class="muted small">${plural(draft.tools.length, "tool")} allowed. Tools marked <i>acts</i> change something outside the studio.</p>`}
    <${Field} label="Models allowed" hint=${html`One per line: <code>provider/id</code>, or <code>local/</code> for every model of a provider. Empty allows all.${eff?.models ? ` The projects it is part of allow: ${eff.models.join(", ")}.` : ""}`}>
      <textarea class="input mono" rows="3" placeholder=${"local/\nanthropic/claude-sonnet-5-5"} value=${models}
        onInput=${e => { setModels(e.target.value); const ms = e.target.value.split(/\n|,/).map(s => s.trim()).filter(Boolean); set({ models: ms.length ? ms : null }); }} /><//>
  </section>`;
}

const plural = (n, one) => `${n} ${n === 1 ? one : one + "s"}`;
