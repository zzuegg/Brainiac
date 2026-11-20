import { html, useEffect, useState, cx } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button, IconAction } from "/v2/ui/Button.js";
import { Layer } from "/v2/ui/Layer.js";
import { openLayer } from "/v2/ui/overlay.js";
import { plural, when } from "/v2/lib/format.js";
import { fail, toast } from "/v2/state/toasts.js";

// How agents work, for every project: the roles they take, the rules every call is checked
// against, the outside services their tools use, and the reminders waiting to go off.

const TABS = [["roles", "Roles"], ["rules", "Rules"], ["services", "Services"], ["reminders", "Reminders"]];
const FAMILIES = { general: "General", coordination: "Coordination", build: "Build", ml: "Models", content: "Content", personal: "Personal" };
const MODES = [["", "As the role it extends"], ["plan", "Plans first"], ["ask", "Asks first"], ["auto", "Auto"]];
const EFFECT = { allow: "Allow", ask: "Ask me", deny: "Deny" };

export function AgentSettings({ tab }) {
  const [at, setAt] = useState("roles");
  return html`<div class="page agent-settings">
    <header class="page-head"><div><p class="eyebrow">Studio</p><h1 class="page-title">Agent settings</h1>
      <p class="page-lead">For every project: who agents can be, what they may do without asking, the outside services their tools reach, and reminders.</p></div></header>
    <div class="seg page-seg" role="tablist" aria-label="Agent settings">${TABS.map(([k, n]) => html`<button type="button" role="tab" aria-selected=${at === k} aria-pressed=${at === k} onClick=${() => setAt(k)}>${n}</button>`)}</div>
    ${at === "roles" ? html`<${Roles} />` : at === "rules" ? html`<${Rules} />` : at === "services" ? html`<${Services} />` : html`<${Reminders} tab=${tab} />`}
  </div>`;
}

// ------------------------------------------------------------------ roles

function Roles() {
  const [d, setD] = useState(null);
  const [sure, setSure] = useState(null);
  const read = () => api("/api/roles").then(setD, fail);
  useEffect(() => { read(); }, []);
  if (!d) return html`<p class="muted">Reading the roles…</p>`;
  const name = id => d.roles.find(r => r.id === id)?.name || id;
  const fams = {};
  for (const r of d.roles) (fams[r.family] ||= []).push(r);
  const edit = role => openLayer(RoleEdit, { role, data: d, saved: setD });
  const del = r => {
    if (sure !== r.id) { setSure(r.id); setTimeout(() => setSure(x => (x === r.id ? null : x)), 4000); return; }
    api("/api/roles/delete", { id: r.id }).then(j => { setD(j); toast(`${r.name} deleted.`); }, fail);
  };
  return html`
    <p class="muted section-lead">A role is who an agent is: its instructions, its tools, how much it may do alone, its budget, and which roles it may start as helpers. Built-in roles stay as they are; a copy of one can be changed.</p>
    <div class="row-btns"><${Button} variant="primary" icon="plus" onClick=${() => edit({ id: "", name: "", family: "general", isNew: true })}>New role<//></div>
    ${Object.entries(fams).map(([f, list]) => html`<section key=${f}><h2 class="group-title">${FAMILIES[f] || f}</h2>
      <div class="role-grid">${list.map(r => html`<article key=${r.id} class="card role-card">
        <h3 class="card-title">${r.name}<span class="mono muted small">${r.id}</span>
          <span class=${cx("badge push", !r.builtin && "s-done")}>${r.builtin ? "built in" : "yours"}</span></h3>
        <p class="muted small">${r.description}</p>
        <div class="role-chips">
          ${r.mode && html`<span class="badge">${MODES.find(m => m[0] === r.mode)?.[1] || r.mode}</span>`}
          <span class="badge">${r.tools ? plural(r.tools.length, "tool") : r.extends ? "tools of " + name(r.extends) : "every tool"}</span>
          ${r.extends && html`<span class="badge">extends ${name(r.extends)}</span>`}
          ${(r.canSpawn || []).includes("*") ? html`<span class="badge s-running">starts any role</span>`
            : (r.canSpawn || []).length > 0 && html`<span class="badge s-running" title=${r.canSpawn.map(name).join(", ")}>starts ${plural(r.canSpawn.length, "role")}</span>`}
          ${r.budget?.steps && html`<span class="badge">${r.budget.steps} steps</span>`}
        </div>
        <div class="role-acts">${r.builtin
          ? html`<${IconAction} icon="copy" label="Make a copy to change" onClick=${() => edit({ ...r, id: r.id + "-mine", name: r.name + " (mine)", builtin: false, isNew: true, extends: r.extends || "" })} />`
          : html`<${IconAction} icon="edit" label="Edit" onClick=${() => edit(r)} /><${IconAction} icon=${sure === r.id ? "check" : "trash"} label=${sure === r.id ? "Click again to delete it" : "Delete"} onClick=${() => del(r)} />`}</div>
      </article>`)}</div></section>`)}`;
}

function RoleEdit({ role, data, saved, close, top }) {
  const [d, setD] = useState({ budget: {}, policy: {}, canSpawn: [], ...role });
  const [find, setFind] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const set = (k, v) => setD({ ...d, [k]: v });
  const own = d.tools != null;
  const q = find.trim().toLowerCase();
  const tools = data.tools.filter(t => !q || (t.name + " " + t.description).toLowerCase().includes(q));
  const toggle = (k, v, on) => set(k, on ? [...(d[k] || []).filter(x => x !== v), v] : (d[k] || []).filter(x => x !== v));
  const save = async () => {
    setBusy(true); setError(null);
    const { isNew, builtin, ...r } = d;
    try { const j = await api("/api/roles/save", { role: r }); saved(j); toast(`${d.name || d.id} saved.`); close(); }
    catch (x) { setError(x.message); setBusy(false); }
  };
  const footer = html`${error && html`<p class="msg-error small">${error}</p>`}<span class="grow" />
    <${Button} variant="ghost" onClick=${close}>Cancel<//><${Button} variant="primary" disabled=${busy || !d.id || !d.name} onClick=${save}>${busy ? "Saving…" : "Save role"}<//>`;
  return html`<${Layer} kind="drawer" wide title=${role.isNew ? "New role" : d.name} subtitle="What is left empty comes from the role it extends." close=${close} top=${top} footer=${footer}>
    <div class="form-grid">
      <label class="field"><span class="field-label">Name</span><input class="input" autofocus value=${d.name} onInput=${e => set("name", e.target.value)} placeholder="My writer" /></label>
      <label class="field"><span class="field-label">Id</span><input class="input mono" value=${d.id} disabled=${!role.isNew} onInput=${e => set("id", e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, "-"))} placeholder="my-writer" /></label>
      <label class="field"><span class="field-label">Family</span><select class="input" value=${d.family} onChange=${e => set("family", e.target.value)}>${Object.entries(FAMILIES).map(([k, v]) => html`<option value=${k}>${v}</option>`)}</select></label>
      <label class="field"><span class="field-label">Extends</span><select class="input" value=${d.extends || ""} onChange=${e => set("extends", e.target.value)}>
        <option value="">Nothing</option>${data.roles.filter(r => r.id !== d.id).map(r => html`<option value=${r.id}>${r.name}</option>`)}</select></label>
    </div>
    <label class="field"><span class="field-label">What it is for</span><input class="input" value=${d.description || ""} onInput=${e => set("description", e.target.value)} /></label>
    <label class="field"><span class="field-label">Instructions</span><textarea class="input" rows="6" value=${d.instructions || ""} onInput=${e => set("instructions", e.target.value)}
      placeholder="Who the agent is and how it works. Added after the instructions of the role it extends." /></label>
    <div class="form-grid">
      <label class="field"><span class="field-label">Mode</span><select class="input" value=${d.mode || ""} onChange=${e => set("mode", e.target.value)}>${MODES.map(([v, n]) => html`<option value=${v}>${n}</option>`)}</select></label>
      <label class="field"><span class="field-label">Model</span><input class="input mono" value=${d.model || ""} onInput=${e => set("model", e.target.value)} placeholder="as it extends, auto, or a model key" /></label>
    </div>
    <h3 class="group-title">Budget</h3>
    <div class="form-grid">${[["steps", "Model turns"], ["tokens", "Tokens, with its helpers"], ["media", "Media files"], ["turnTokens", "Tokens per turn"], ["context", "Context tokens"]].map(([k, n]) => html`
      <label class="field" key=${k}><span class="field-label">${n}</span><input class="input mono" type="number" min="1" value=${d.budget?.[k] ?? ""} placeholder="as it extends"
        onInput=${e => { const b = { ...(d.budget || {}) }; e.target.value ? (b[k] = +e.target.value) : delete b[k]; set("budget", b); }} /></label>`)}</div>
    <h3 class="group-title">What it may do alone</h3>
    <div class="tier-rows">${data.tiers.map(t => html`<div class="tier-row" key=${t.key}><span><b>${t.what[0].toUpperCase() + t.what.slice(1)}</b></span>
      <div class="seg" role="group" aria-label=${t.what}>${["", "allow", "ask", "deny"].filter(v => !(t.key === "outward" && v === "allow")).map(v => html`<button type="button"
        aria-pressed=${(d.policy?.[t.key] || "") === v} title=${v ? "" : `Asks first: ${t.ask}; Auto: ${t.auto}`}
        onClick=${() => { const p = { ...(d.policy || {}) }; v ? (p[t.key] = v) : delete p[t.key]; set("policy", p); }}>${v ? EFFECT[v] : "By mode"}</button>`)}</div></div>`)}</div>
    <h3 class="group-title">Tools</h3>
    <label class="pedit-row"><input type="checkbox" checked=${own} onChange=${e => set("tools", e.target.checked ? [] : null)} />
      <span><b>Choose its tools</b><span class="muted small">Otherwise it has the tools of the role it extends${d.extends ? "" : ", which is every tool"}.</span></span></label>
    ${own && html`<input class="input" type="search" placeholder="Find a tool" value=${find} onInput=${e => setFind(e.target.value)} aria-label="Find a tool" />
      <ul class="picks tool-picks">${tools.map(t => html`<li key=${t.name}><label title=${t.description}>
        <input type="checkbox" checked=${d.tools.includes(t.name)} onChange=${e => toggle("tools", t.name, e.target.checked)} />
        <span class="who"><b class="mono">${t.name}</b><span class="muted small">${t.acts || t.description}</span></span><span class="badge">${t.risk}</span></label></li>`)}</ul>
      <p class="field-hint">${plural(d.tools.length, "tool")} chosen.</p>`}
    <h3 class="group-title">May start these roles as helpers</h3>
    <div class="spawn-picks">
      <label class="check-inline"><input type="checkbox" checked=${(d.canSpawn || []).includes("*")} onChange=${e => toggle("canSpawn", "*", e.target.checked)} /><b>Any role</b></label>
      ${!(d.canSpawn || []).includes("*") && data.roles.map(r => html`<label class="check-inline" key=${r.id}><input type="checkbox" checked=${(d.canSpawn || []).includes(r.id)} onChange=${e => toggle("canSpawn", r.id, e.target.checked)} />${r.name}</label>`)}
    </div>
  <//>`;
}

// ------------------------------------------------------------------ rules

const blankRule = () => ({ id: "rule-" + Math.random().toString(36).slice(2, 6), effect: "ask", tools: [], risks: [], paths: [], access: "any", project: "", role: "", note: "", enabled: true });
const listOf = s => s.split(",").map(x => x.trim()).filter(Boolean);

function Rules() {
  const [d, setD] = useState(null);
  const [rules, setRules] = useState(null);
  const [dirty, setDirty] = useState(false);
  const [resetting, setResetting] = useState(false);
  const [roles, setRoles] = useState([]);
  const take = j => { setD(j); setRules(j.rules.map(r => ({ ...r }))); setDirty(false); };
  useEffect(() => { api("/api/rules").then(take, fail); api("/api/roles").then(j => setRoles(j.roles), () => {}); }, []);
  if (!rules) return html`<p class="muted">Reading the rules…</p>`;
  const put = (i, ch) => { setRules(rules.map((r, j) => (j === i ? { ...r, ...ch } : r))); setDirty(true); };
  const move = (i, j) => { const l = [...rules]; const [r] = l.splice(i, 1); l.splice(j, 0, r); setRules(l); setDirty(true); };
  const save = () => api("/api/rules/save", { rules }).then(j => { take(j); toast(`Saved ${plural(j.rules.length, "rule")}.`); }, fail);
  const reset = () => {
    if (!resetting) { setResetting(true); setTimeout(() => setResetting(false), 4000); return; }
    setResetting(false); api("/api/rules/reset", {}).then(j => { take(j); toast("The shipped rules are back."); }, fail);
  };
  return html`
    <p class="muted section-lead">Every call an agent makes is checked against these rules from the top; the first that matches decides. Calls no rule matches are judged by their risk, the role and the mode. Two things always ask, whatever the rules say: anything that leaves this PC under your name, and writing after an agent read text from the web, mail or another agent.</p>
    <datalist id="rule-tools">${d.tools.map(t => html`<option value=${t} />`)}</datalist>
    <div class="row-btns"><${Button} icon="plus" onClick=${() => { setRules([blankRule(), ...rules]); setDirty(true); }}>Add a rule<//>
      <span class="grow" />
      <${Button} variant="ghost" onClick=${reset}>${resetting ? "Click again to reset" : "Back to the shipped rules"}<//>
      <${Button} variant="primary" icon="check" disabled=${!dirty} onClick=${save}>Save rules<//></div>
    <ol class="rules">${rules.map((r, i) => html`<li key=${r.id + i} class=${cx("card rule", !r.enabled && "off", "rule-" + r.effect)}>
      <div class="rule-head">
        <span class="mono muted small">${i + 1}</span>
        <div class="seg" role="group" aria-label="What the rule does">${Object.entries(EFFECT).map(([k, n]) => html`<button type="button" aria-pressed=${r.effect === k} onClick=${() => put(i, { effect: k })}>${n}</button>`)}</div>
        <input class="input mono rule-id" value=${r.id} aria-label="Rule name" onInput=${e => put(i, { id: e.target.value })} />
        <label class="check-inline small"><input type="checkbox" checked=${r.enabled} onChange=${e => put(i, { enabled: e.target.checked })} /> On</label>
        <span class="grow" />
        <${IconAction} icon="up" label="Move up" disabled=${!i} onClick=${() => move(i, i - 1)} />
        <${IconAction} icon="chevronDown" label="Move down" disabled=${i === rules.length - 1} onClick=${() => move(i, i + 1)} />
        <${IconAction} icon="trash" label="Delete the rule" onClick=${() => { setRules(rules.filter((_, j) => j !== i)); setDirty(true); }} />
      </div>
      <input class="input rule-why" value=${r.note} placeholder="Why, in a few words" aria-label="Why" onInput=${e => put(i, { note: e.target.value })} />
      <div class="form-grid rule-grid">
        <label class="field"><span class="field-label">Tools <span class="muted small">comma separated; empty for all</span></span>
          <input class="input mono" list="rule-tools" value=${(r.tools || []).join(", ")} onInput=${e => put(i, { tools: listOf(e.target.value) })} /></label>
        <label class="field"><span class="field-label">Risks <span class="muted small">empty for all</span></span>
          <input class="input mono" value=${(r.risks || []).join(", ")} placeholder=${d.tiers.join(", ")} onInput=${e => put(i, { risks: listOf(e.target.value) })} /></label>
        <label class="field span-2"><span class="field-label">Paths <span class="muted small">@project, @outside, or patterns such as ~/Documents/**; empty for any call</span></span>
          <input class="input mono" value=${(r.paths || []).join(", ")} onInput=${e => put(i, { paths: listOf(e.target.value) })} /></label>
        <label class="field"><span class="field-label">Reading or writing</span><select class="input" value=${r.access} onChange=${e => put(i, { access: e.target.value })}>
          <option value="any">Either</option><option value="read">Reading</option><option value="write">Writing</option></select></label>
        <label class="field"><span class="field-label">Role</span><select class="input" value=${r.role || ""} onChange=${e => put(i, { role: e.target.value })}>
          <option value="">Every role</option>${roles.map(x => html`<option value=${x.id}>${x.name}</option>`)}</select></label>
        <label class="field"><span class="field-label">Project <span class="muted small">empty for all</span></span>
          <input class="input mono" value=${r.project || ""} onInput=${e => put(i, { project: e.target.value })} /></label>
      </div>
    </li>`)}</ol>`;
}

// ------------------------------------------------------------------ services

const SERVICES = [
  ["search", "Web search", "For web_search. Brave Search needs a key; SearXNG needs its address; Ollama uses an Ollama key.", [["provider", "Provider", ["brave", "searxng", "ollama"]], ["url", "SearXNG address", "https://search.example.org"]], "API key"],
  ["embeddings", "Meaning search", "index_files and search_knowledge find passages by meaning as well as words, with an embedding model downloaded here (such as BAAI/bge-small-en-v1.5) or an OpenAI-compatible embeddings API.", [["source", "Use", ["", "off"]], ["url", "API address", "empty for a model on this PC"], ["model", "API model", "nomic-embed-text"], ["folder", "Model folder", "found among downloads if empty"]], "API key"],
  ["sandbox", "Code sandbox", "run_command runs in a Docker or Podman container on a copy of the project, never on this PC directly.", [["kind", "Engine", ["", "docker", "podman"]], ["image", "Image", "debian:stable-slim"], ["cpus", "CPUs", "2"], ["memory", "Memory", "4g"], ["network", "Network", ["none", "on"]], ["timeout", "Time limit, seconds", "600"]], null],
  ["github", "GitHub", "Reading issues and, after you approve, opening issues and pull requests and commenting. A fine-grained token limited to the repositories agents may touch is best.", [["repo", "Repository", "owner/name; the project's origin if empty"], ["api", "API address", "https://api.github.com/"]], "Token"],
  ["mastodon", "Mastodon", "For publish_post. Make an access token under Preferences, Development.", [["instance", "Server", "mastodon.social"]], "Access token"],
  ["bluesky", "Bluesky", "For publish_post. Use an app password, not your account password.", [["handle", "Handle", "you.bsky.social"], ["service", "Server", "https://bsky.social"]], "App password"],
  ["discord", "Discord", "Posts go to a channel through its webhook.", [], "Webhook address"],
  ["webhook", "Webhook", "publish_post can send {text, media} as JSON to any address.", [["url", "Address", "https://…"]], null],
  ["mail", "Mail", "read_mail reads over IMAP without marking anything; send_mail sends over SMTP after you approve.", [["imapHost", "IMAP server", "imap.example.org"], ["imapPort", "IMAP port", "993"], ["smtpHost", "SMTP server", "smtp.example.org"], ["smtpPort", "SMTP port", "587"], ["user", "User", "you@example.org"], ["from", "From", "You <you@example.org>"]], "Password"],
  ["calendar", "Calendar", "calendar_events reads this iCalendar address, along with the project's calendar.ics.", [["url", "Calendar address", "https://…/basic.ics"]], null],
];
const isSet = v => !!v && (v.hasSecret || Object.entries(v).some(([k, x]) => k !== "hasSecret" && x));

function Services() {
  const [s, setS] = useState(null);
  useEffect(() => { api("/api/services").then(setS, fail); }, []);
  if (!s) return html`<p class="muted">Reading the services…</p>`;
  return html`
    <p class="muted section-lead">The outside services agent tools use. Passwords and tokens are kept with the studio's other keys and never shown again.</p>
    <p class=${cx("flag pedit-flag", s.ffmpeg ? "flag-ok" : "")}><${Icon} name=${s.ffmpeg ? "check" : "warn"} size=${14} /><span>FFmpeg, for audio and video, is ${s.ffmpeg ? "found." : "not found: install it or set BRAINIAC_FFMPEG."}</span></p>
    <div class="svc-grid">${SERVICES.map(([id, title, blurb, fields, secret]) => html`<${Service} key=${id} id=${id} title=${title} blurb=${blurb} fields=${fields} secret=${secret} v=${s[id] || {}} saved=${setS} />`)}</div>`;
}

function Service({ id, title, blurb, fields, secret, v, saved }) {
  const [vals, setVals] = useState(() => Object.fromEntries(fields.map(([k]) => [k, v[k] || ""])));
  const [pass, setPass] = useState("");
  const [busy, setBusy] = useState(false);
  const save = async e => {
    e.preventDefault(); setBusy(true);
    try { saved(await api("/api/services/save", { section: id, values: vals, secret: pass || null })); setPass(""); toast(`${title} saved.`); } catch (x) { fail(x); }
    setBusy(false);
  };
  return html`<form class="card svc" onSubmit=${save} autocomplete="off">
    <h3 class="card-title">${title}${isSet(v) ? html`<span class="badge s-done push">set up</span>` : html`<span class="badge push">not set up</span>`}</h3>
    <p class="muted small">${blurb}</p>
    <div class="form-grid">${fields.map(([k, n, hint]) => html`<label class="field" key=${k}><span class="field-label">${n}</span>${Array.isArray(hint)
      ? html`<select class="input" value=${vals[k]} onChange=${e => setVals({ ...vals, [k]: e.target.value })}>${hint.map(o => html`<option value=${o}>${o || "Find one"}</option>`)}</select>`
      : html`<input class="input mono" value=${vals[k]} placeholder=${hint} onInput=${e => setVals({ ...vals, [k]: e.target.value })} />`}</label>`)}
      ${secret && html`<label class="field"><span class="field-label">${secret}${v.hasSecret && html` <span class="muted small"><${Icon} name="lock" size=${11} /> kept</span>`}</span>
        <input class="input mono" type="password" value=${pass} placeholder=${v.hasSecret ? "type to replace" : ""} autocomplete="new-password" onInput=${e => setPass(e.target.value)} /></label>`}</div>
    <div class="row-btns"><span class="grow" /><${Button} size="sm" type="submit" disabled=${busy}>Save<//></div>
  </form>`;
}

// ------------------------------------------------------------------ reminders

function Reminders({ tab }) {
  const [list, setList] = useState(null);
  const [roles, setRoles] = useState([]);
  const [f, setF] = useState({ at: "", text: "", goal: "", role: "", repeat: "" });
  const [busy, setBusy] = useState(false);
  const read = () => api("/api/reminders").then(j => setList(j.reminders || []), fail);
  useEffect(() => { read(); api("/api/roles").then(j => setRoles(j.roles), () => {}); }, []);
  const add = async e => {
    e.preventDefault(); setBusy(true);
    try { await api("/api/reminders/add", f, { project: tab.project }); setF({ at: "", text: "", goal: "", role: "", repeat: "" }); toast("Reminder set."); read(); } catch (x) { fail(x); }
    setBusy(false);
  };
  const waiting = (list || []).filter(r => r.state === "waiting").sort((a, b) => a.at - b.at);
  const past = (list || []).filter(r => r.state !== "waiting").sort((a, b) => b.at - a.at).slice(0, 20);
  const row = r => html`<li key=${r.id} class=${cx(r.state !== "waiting" && "dim")}>
    <${Icon} name=${r.goal ? "agent" : "info"} size=${14} />
    <div class="who"><b>${r.text}</b>${r.goal && html`<span class="muted small">then an agent${r.role ? " as " + (roles.find(x => x.id === r.role)?.name || r.role) : ""}: ${r.goal}</span>`}</div>
    <span class="mono small">${new Date(r.at).toLocaleString([], { dateStyle: "medium", timeStyle: "short" })}</span>
    <span class="muted small">${[r.repeat, r.project, r.by !== "you" && "set by an agent"].filter(Boolean).join(" · ")}</span>
    ${r.state === "waiting" ? html`<${IconAction} icon="close" label="Cancel it" onClick=${() => api("/api/reminders/dismiss", { id: r.id }).then(j => setList(j.reminders || []), fail)} />` : html`<span class="badge">${r.state}</span>`}
  </li>`;
  return html`
    <p class="muted section-lead">A reminder goes off at its time in the project it belongs to. With a goal, an agent starts then and works on it. Agents can set reminders too.</p>
    <form class="card reminder-form" onSubmit=${add}>
      <h3 class="card-title">New reminder <span class="muted small">in this project</span></h3>
      <div class="form-grid">
        <label class="field span-2"><span class="field-label">Remind me of</span><input class="input" value=${f.text} onInput=${e => setF({ ...f, text: e.target.value })} placeholder="Check the training run" required /></label>
        <label class="field"><span class="field-label">When</span><input class="input" type="datetime-local" value=${f.at} onInput=${e => setF({ ...f, at: e.target.value })} required /></label>
        <label class="field"><span class="field-label">Again</span><select class="input" value=${f.repeat} onChange=${e => setF({ ...f, repeat: e.target.value })}>
          <option value="">Once</option><option value="daily">Every day</option><option value="weekly">Every week</option></select></label>
        <label class="field span-2"><span class="field-label">Then an agent should <span class="muted small">optional</span></span><input class="input" value=${f.goal} onInput=${e => setF({ ...f, goal: e.target.value })} placeholder="Look at the loss and tell me whether to stop it" /></label>
        ${f.goal && html`<label class="field"><span class="field-label">As</span><select class="input" value=${f.role} onChange=${e => setF({ ...f, role: e.target.value })}>
          <option value="">General assistant</option>${roles.map(r => html`<option value=${r.id}>${r.name}</option>`)}</select></label>`}
      </div>
      <div class="row-btns"><span class="grow" /><${Button} variant="primary" type="submit" icon="plus" disabled=${busy || !f.text.trim() || !f.at}>Set reminder<//></div>
    </form>
    ${list && html`<section class="card"><h3 class="card-title">Waiting <span class="muted small">${waiting.length}</span></h3>
      ${waiting.length ? html`<ul class="reminders">${waiting.map(row)}</ul>` : html`<p class="muted small">Nothing waiting.</p>`}</section>`}
    ${past.length > 0 && html`<section class="card"><h3 class="card-title">Gone off</h3><ul class="reminders">${past.map(row)}</ul></section>`}`;
}
