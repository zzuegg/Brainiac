import { html, useEffect, useMemo, useState, cx } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button, IconAction } from "/v2/ui/Button.js";
import { Layer } from "/v2/ui/Layer.js";
import { openLayer } from "/v2/ui/overlay.js";
import { MEDIA_NAMES } from "/v2/state/models.js";
import { projectName } from "/v2/state/projects.js";
import { toast } from "/v2/state/toasts.js";

// What the studio can do: the capabilities its modules bring, and the tools agents call, grouped
// by what a call may do. Any tool that only reads, computes or makes something can be tried by
// hand here, with the arguments an agent would give it, to see what the agent would see.

// what a call may do, mildest first, as the rules and the modes name it
const TIERS = [
  ["read", "Reads", "Looks at models, files and runs; changes nothing."],
  ["compute", "Uses this PC", "Runs on this PC's GPU or CPU."],
  ["spend", "May cost money", "Calls a paid provider."],
  ["create", "Creates files", "Makes a new file; nothing is overwritten."],
  ["write", "Changes files", "Changes or replaces files; an agent's changes can be undone."],
  ["code", "Runs commands", "Runs code in the sandbox: only an agent, after you allow it."],
  ["outward", "Leaves this PC", "Sends something out under your name: only an agent, after you allow it."],
];
const tierOf = k => TIERS.find(t => t[0] === k) || [k, k, ""];
const byHand = t => t.risk !== "code" && t.risk !== "outward";
const firstLine = s => (s || "").split(/(?<=\.)\s/)[0];

export function Tools({ tab }) {
  const [tools, setTools] = useState(null);
  const [caps, setCaps] = useState(null);
  const [error, setError] = useState(null);
  const [q, setQ] = useState("");
  useEffect(() => {
    api("/api/tools").then(r => setTools(r.tools || []), e => setError(e.message));
    api("/api/capabilities").then(r => setCaps(r.capabilities || []), () => setCaps([]));
  }, []);
  const shown = useMemo(() => {
    const words = q.toLowerCase().split(/\s+/).filter(Boolean);
    return (tools || []).filter(t => words.every(w => (t.name + " " + t.description + " " + (t.medium || "")).toLowerCase().includes(w)));
  }, [tools, q]);
  return html`<div class="page tools-page">
    <header class="page-head"><div><p class="eyebrow">Studio</p><h1 class="page-title">Tools</h1>
      <p class="page-lead">What agents can call, grouped by what a call may do. Your rules and each agent's mode decide which calls ask first. Try one by hand to see what an agent would get back.</p></div></header>
    ${error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${error}</p>`}
    <section class="card">
      <h2 class="card-title">Tools<span class="count">${tools ? tools.length : ""}</span><span class="push" />
        <input class="input select-sm tools-filter" type="search" value=${q} placeholder="Find a tool" aria-label="Find a tool" onInput=${e => setQ(e.target.value)} /></h2>
      ${!tools ? html`<p class="muted small">Reading the tools…</p>`
        : !shown.length ? html`<p class="muted small">No tool matches “${q}”.</p>`
        : TIERS.map(([k, name, what]) => {
            const list = shown.filter(t => t.risk === k);
            return list.length > 0 && html`<div class="tool-tier" key=${k}>
              <h3 class="group-title">${name} <span class="muted small">· ${what}</span></h3>
              <ul class="tool-rows">${list.map(t => html`<li key=${t.name}>
                <button type="button" class="tool-row" onClick=${() => openLayer(TryTool, { tool: t, project: tab?.project })}>
                  <span class="mono">${t.name}</span>
                  <span class="muted small tool-what">${firstLine(t.description)}</span>
                  <span class="tool-flags">${t.medium && html`<span class="badge">${MEDIA_NAMES[t.medium] || t.medium}</span>`}
                    ${t.untrusted && html`<span class="badge s-waiting" title="What it returns comes from outside: the web, mail or another agent">outside text</span>`}</span>
                  <${Icon} name=${byHand(t) ? "play" : "lock"} size=${14} />
                </button></li>`)}</ul></div>`;
          })}
    </section>
    ${caps?.length > 0 && html`<section class="card">
      <h2 class="card-title">Capabilities<span class="count">${caps.filter(c => c.status === "available").length}</span></h2>
      <p class="muted small card-lead">What the studio's modules can do, found when it starts. New ones show up here and as tools without changes to the page.</p>
      <ul class="cap-grid">${caps.slice().sort((a, b) => (a.status === "available" ? 0 : 1) - (b.status === "available" ? 0 : 1)).map(c => html`<li key=${c.key} class=${cx("cap", c.status !== "available" && "dim")}>
        <div class="cap-head"><b>${c.title}</b>${c.status !== "available" && html`<span class="badge">planned</span>`}</div>
        <p class="small">${c.description}</p>
        <span class="muted small mono">${c.key}${c.source ? " · " + c.source : ""}</span>
        ${c.note && html`<span class="muted small">${c.note}</span>`}</li>`)}</ul>
    </section>`}
  </div>`;
}

/** A tool's arguments as a form, built from its parameters, and the answer it gives. */
function TryTool({ tool, project, close, top }) {
  const props = tool.parameters?.properties || {}, required = new Set(tool.parameters?.required || []);
  const [args, setArgs] = useState({});
  const [busy, setBusy] = useState(false);
  const [out, setOut] = useState(null);
  const [error, setError] = useState(null);
  const tier = tierOf(tool.risk);
  const put = (k, v) => setArgs(a => { const n = { ...a }; if (v === "" || v == null) delete n[k]; else n[k] = v; return n; });
  const missing = [...required].filter(k => args[k] == null || args[k] === "");
  const run = async () => {
    const send = {};
    for (const [k, v] of Object.entries(args)) {
      const p = props[k] || {};
      if (p.type === "number" || p.type === "integer") send[k] = +v;
      else if (p.type === "array" || p.type === "object") {
        try { send[k] = JSON.parse(v); } catch { setError(`“${k}” is not JSON, such as ${p.type === "array" ? "[\"a\", \"b\"]" : "{\"a\": 1}"}.`); return; }
      } else send[k] = v;
    }
    setBusy(true); setError(null); setOut(null);
    try { setOut((await api("/api/tools/call", { name: tool.name, arguments: send }, { project })).result); }
    catch (e) { setError(e.message); }
    setBusy(false);
  };
  const pretty = useMemo(() => { if (out == null) return ""; try { return JSON.stringify(JSON.parse(out), null, 2); } catch { return out; } }, [out]);
  const footer = byHand(tool)
    ? html`${missing.length > 0 && html`<span class="muted small">Needs ${missing.join(", ")}</span>`}<span class="grow" />
        <${Button} variant="ghost" onClick=${close}>Close<//>
        <${Button} variant="primary" icon="play" onClick=${run} disabled=${busy || missing.length > 0}>${busy ? "Running…" : "Run it"}<//>`
    : html`<span class="muted small">${tier[2]}</span><span class="grow" /><${Button} onClick=${close}>Close<//>`;
  return html`<${Layer} kind="drawer" wide title=${tool.name} subtitle=${tier[1] + (project ? " · in " + projectName(project) : "")} close=${close} top=${top} footer=${footer}>
    <p class="tool-desc">${tool.description}</p>
    ${tool.returns && html`<p class="muted small">Returns ${tool.returns}</p>`}
    ${Object.keys(props).length === 0 ? html`<p class="muted small">It takes nothing.</p>`
      : html`<div class="tool-args">${Object.entries(props).map(([k, p]) => html`<label class="field" key=${k}>
          <span class="field-label"><span class="mono">${k}</span>${required.has(k) ? "" : html` <span class="muted small">optional</span>`}</span>
          ${p.enum ? html`<select class="input" value=${args[k] ?? ""} disabled=${!byHand(tool)} onChange=${e => put(k, e.target.value)}>
                <option value="">${required.has(k) ? "Choose…" : "Not given"}</option>${p.enum.map(v => html`<option value=${v}>${v}</option>`)}</select>`
            : p.type === "boolean" ? html`<select class="input" value=${args[k] == null ? "" : String(args[k])} disabled=${!byHand(tool)} onChange=${e => put(k, e.target.value === "" ? null : e.target.value === "true")}>
                <option value="">Not given</option><option value="true">yes</option><option value="false">no</option></select>`
            : p.type === "array" || p.type === "object" || (p.type === "string" && /text|prompt|content|instructions|script/i.test(k))
              ? html`<textarea class="input mono" rows="3" value=${args[k] ?? ""} disabled=${!byHand(tool)} placeholder=${p.type === "array" ? "[ … ]" : p.type === "object" ? "{ … }" : ""} onInput=${e => put(k, e.target.value)} />`
            : html`<input class=${cx("input", p.type !== "string" && "mono")} value=${args[k] ?? ""} disabled=${!byHand(tool)} inputmode=${p.type === "number" || p.type === "integer" ? "decimal" : undefined}
                placeholder=${p.default != null ? String(p.default) : ""} onInput=${e => put(k, e.target.value)} />`}
          ${p.description && html`<span class="field-hint">${p.description}</span>`}</label>`)}</div>`}
    ${error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${error}</p>`}
    ${out != null && html`<div class="tool-out"><div class="tool-out-head"><span class="field-label">What it answered</span><span class="grow" />
        <${IconAction} icon="copy" label="Copy the answer" onClick=${() => navigator.clipboard?.writeText(out).then(() => toast("Copied."))} /></div>
      <pre class="mono">${pretty || "(nothing)"}</pre></div>`}
  <//>`;
}
