import { html, useEffect, useState, signal, cx } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button, IconAction } from "/v2/ui/Button.js";
import { Layer } from "/v2/ui/Layer.js";
import { openLayer } from "/v2/ui/overlay.js";
import { plural } from "/v2/lib/format.js";
import { modelInfo, MEDIA_NAMES } from "/v2/state/models.js";
import { fmtCount } from "/v2/state/hub.js";
import { fail, toast } from "/v2/state/toasts.js";

// Providers reached with an API key: the keys (several are tried in order), what each one has
// left where the provider says, how the provider is used, and what every call cost in tokens.

export const usage = signal(null);
export const balances = signal(null);
export const readUsage = () => api("/api/usage").then(u => (usage.value = u), () => {});
export const readBalances = () => api("/api/balances").then(b => (balances.value = b), () => (balances.value = {}));

/** Providers that take a key or have settings of their own. */
export const keyed = p => !!p.settings;
const keys = body => api("/api/keys", body).then(r => { modelInfo.value = r; return r; });
const money = b => (b.amount === undefined ? "no limit" : `${b.amount.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} ${b.currency || ""}`.trim());
export const usageLine = t => (!t?.calls ? "" : [plural(t.calls, "call"), t.in + t.out ? `${fmtCount(t.in)} in · ${fmtCount(t.out)} out` : "", t.failed ? `${t.failed} failed` : ""].filter(Boolean).join(" · "));

/** What is set differently from the provider's defaults, in a few words. */
export function providerFlags(p) {
  const st = p.settings || {};
  return [st.auto === false && "not used by Auto", st.models && `${st.models.length} of ${p.models} shown`,
    st.url && "own address", st.options?.billTo && `bills ${st.options.billTo}`].filter(Boolean);
}

/** The keys of one provider: the first is tried first, the next takes over when one is refused. */
export function ProviderKeys({ p }) {
  const [sure, setSure] = useState(null);
  const [adding, setAdding] = useState(false);
  const [editing, setEditing] = useState(null);   // a saved key's name, while its new value is typed
  const bal = balances.value?.[p.id];
  useEffect(() => { if (p.balance && p.keys.length && !balances.value) readBalances(); }, []);
  const saved = p.keys.filter(k => k.source === "saved").length;
  return html`<div class="pkeys">
    ${p.keys.length > 0 && html`<ul class="key-rows">${p.keys.map((k, n) => {
      const b = bal?.find(x => x.label === k.label);
      const use = usageLine(usage.value?.keys?.[p.id + "/" + k.label]);
      return html`<li key=${k.label}>
        <span class=${cx("key-dot", k.error ? "bad" : "ok")} aria-hidden="true" />
        <${Icon} name="lock" size=${13} /><b>${k.label}</b><span class="mono muted small">${k.hint}</span>
        ${n === 0 && p.keys.length > 1 && html`<span class="badge">tried first</span>`}
        ${p.balance && (b ? b.error ? html`<span class="badge" title=${b.error}>balance unknown</span>`
          : html`<span class=${cx("badge", b.balance.amount !== undefined && b.balance.amount < 1 ? "s-waiting" : "s-done")} title=${b.balance.detail || ""}>${money(b.balance)} left</span>`
          : !balances.value && html`<span class="badge muted">balance…</span>`)}
        ${use && html`<span class="mono muted small key-use">${use}</span>`}
        <span class="grow" />
        ${k.source === "saved" ? html`
          <${IconAction} icon="edit" label="Replace this key with a new one under the same name" onClick=${() => setEditing(editing === k.label ? null : k.label)} />
          ${n > 0 && html`<${IconAction} icon="up" label="Try this key first" onClick=${() => keys({ provider: p.id, label: k.label, first: true }).catch(fail)} />`}
          <${IconAction} icon=${sure === k.label ? "check" : "trash"} label=${sure === k.label ? "Click again to remove the key" : "Remove this key"}
            onClick=${() => (sure === k.label ? (setSure(null), keys({ provider: p.id, label: k.label, remove: true }).then(() => toast("Key removed.")).catch(fail)) : setSure(k.label))} />`
          : html`<span class="badge" title="Set in the environment the studio started from; remove it there">from ${k.source || "the environment"}</span>`}
        ${k.error && html`<p class="msg-error small key-err">${k.error}</p>`}
        ${editing === k.label && html`<div class="key-err"><${KeyForm} p=${p} label=${k.label} done=${() => setEditing(null)} cancel=${() => setEditing(null)} /></div>`}
      </li>`;
    })}</ul>`}
    ${p.kind === "local" && html`<p class="muted small key-note">${p.id === "ollama"
      ? "Models on this PC need no key. A key is for Ollama Cloud (set the address https://ollama.com with the gear icon above) and for agents' web search through Ollama."
      : "Models on this PC need no key. A key is for another address, set with the gear icon above."}</p>`}
    ${adding || (!p.keys.length && p.kind !== "local") ? html`<${KeyForm} p=${p} first=${!saved} done=${() => setAdding(false)} cancel=${p.keys.length || p.kind === "local" ? () => setAdding(false) : null} />`
      : html`<button type="button" class="link small link-icon" onClick=${() => setAdding(true)}><${Icon} name="plus" size=${12} /> ${p.keys.length ? "Add another key" : "Add a key"}</button>
        ${p.keys.length > 1 && html`<span class="muted small"> · tried in order; the next takes over when one is refused or out of credit</span>`}`}
  </div>`;
}

function KeyForm({ p, first, done, cancel, label: fixed }) {
  const [label, setLabel] = useState(fixed || (first ? "default" : ""));
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const save = async e => {
    e.preventDefault();
    setBusy(true);
    try { await keys({ provider: p.id, label: label.trim() || "default", key: key.trim() }); setKey(""); toast(fixed ? `Key “${fixed}” replaced.` : `${p.name} key saved. Its models are listed now.`); done?.(); }
    catch (x) { fail(x); }
    setBusy(false);
  };
  return html`<form class="key-form" onSubmit=${save}>
    ${!fixed && html`<input class="input" value=${label} onInput=${e => setLabel(e.target.value)} placeholder="Name, such as work" aria-label="Key name" maxlength="40" required />`}
    <input class="input mono" type="password" value=${key} onInput=${e => setKey(e.target.value)} placeholder=${fixed ? "New key for “" + fixed + "”" : "API key"} aria-label=${p.name + " API key"} autocomplete="off" required autofocus=${!first} />
    <${Button} type="submit" icon=${fixed ? "check" : "plus"} disabled=${busy || !key.trim()}>${busy ? "Checking…" : fixed ? "Replace" : "Add key"}<//>
    ${cancel && html`<${Button} variant="ghost" onClick=${cancel}>Cancel<//>`}
  </form>`;
}

/** Providers that take a key and have none yet, each one click away from a key form. */
export function MoreProviders({ list }) {
  const [open, setOpen] = useState(null);
  if (!list.length) return null;
  return html`<section class="card">
    <h2 class="card-title">Connect a provider <span class="muted small">${plural(list.length, "more takes", "more take")} an API key</span></h2>
    <ul class="more-provs">${list.map(p => html`<li key=${p.id} class=${cx(open === p.id && "open")}>
      <span class="plogo" aria-hidden="true">${p.name[0]}</span>
      <div class="who"><b>${p.name}</b><span class="muted small">${p.note}</span></div>
      <span class="mono muted small env" title="The studio also reads the key from this environment variable">${(p.env || []).join(" or ")}</span>
      <${Button} size="sm" icon=${open === p.id ? "close" : "plus"} aria-expanded=${open === p.id} onClick=${() => setOpen(open === p.id ? null : p.id)}>${open === p.id ? "Close" : "Add key"}<//>
      ${open === p.id && html`<div class="more-form"><${KeyForm} p=${p} first done=${() => setOpen(null)} /></div>`}
    </li>`)}</ul>
    <p class="muted small">Keys stay on this PC, readable only by your user account; the page only ever sees their last four characters.</p>
  </section>`;
}

/** How one provider is used: in Auto or not, its address, its options, and which models it offers. */
export function ProviderEdit({ id, close, top }) {
  const info = modelInfo.value;
  const p = info.providers.find(x => x.id === id);
  const st = p.settings || {};
  const all = info.models.filter(m => m.provider === p.id);
  const [auto, setAuto] = useState(st.auto !== false);
  const [url, setUrl] = useState(st.url || "");
  const [options, setOptions] = useState({ ...(st.options || {}) });
  const [pick, setPick] = useState(!!st.models);
  const [picked, setPicked] = useState(() => new Set(st.models || []));
  const [find, setFind] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const ids = all.map(m => m.id);
  const gone = [...picked].filter(x => !ids.includes(x));
  const q = find.trim().toLowerCase();
  const shown = [...all.map(m => ({ id: m.id, name: m.name, detail: m.detail, media: m.media })), ...gone.map(x => ({ id: x, name: x, detail: "no longer listed", media: [] }))]
    .filter(m => !q || (m.name + " " + m.id).toLowerCase().includes(q));
  const toggle = (x, on) => { const s = new Set(picked); on ? s.add(x) : s.delete(x); setPicked(s); };
  const choose = on => {
    setPick(on);
    if (on && !picked.size) { // start from what Auto uses there, so the list is not empty
      const a = Object.values(info.auto || {}).find(k => k.startsWith(p.id + "/"));
      if (a) setPicked(new Set([a.slice(p.id.length + 1)]));
    }
  };
  const save = async () => {
    if (pick && !picked.size) { setError("Pick at least one model, or choose All."); return; }
    setBusy(true);
    try {
      modelInfo.value = await api("/api/providers", { provider: p.id, auto, models: pick ? [...picked] : null, url: url.trim(), options });
      toast(`${p.name} saved.`); close();
    } catch (x) { setError(x.message); setBusy(false); }
  };
  const OPTION = { billTo: ["Organization to bill", "my-org", "Hugging Face charges this organization's credits instead of yours. Empty pays from your own account."] };
  const footer = html`${error && html`<p class="msg-error small">${error}</p>`}<span class="grow" />
    <${Button} variant="ghost" onClick=${close}>Cancel<//><${Button} variant="primary" disabled=${busy} onClick=${save}>${busy ? "Saving…" : "Save"}<//>`;
  return html`<${Layer} kind="drawer" title=${p.name} subtitle=${p.note} close=${close} top=${top} footer=${footer}>
    <label class="pedit-row"><input type="checkbox" checked=${auto} onChange=${e => setAuto(e.target.checked)} />
      <span><b>Use in Auto</b><span class="muted small">Auto may pick this provider's models. Off, they can still be chosen by name.</span></span></label>
    <label class="field"><span class="field-label">Address</span>
      <input class="input mono" value=${url} onInput=${e => setUrl(e.target.value)} placeholder=${st.home || ""} spellcheck="false" autocomplete="off" />
      <span class="field-hint">Empty uses ${st.home || "the provider's own"}. Set a gateway, a proxy or another region${p.id === "qwen" ? ", such as https://dashscope.aliyuncs.com/compatible-mode/v1/ for mainland China" : ""}.</span></label>
    ${Object.keys(st.options || {}).map(o => html`<label class="field" key=${o}><span class="field-label">${OPTION[o]?.[0] || o}</span>
      <input class="input" value=${options[o] || ""} onInput=${e => setOptions({ ...options, [o]: e.target.value })} placeholder=${OPTION[o]?.[1] || ""} spellcheck="false" autocomplete="off" />
      <span class="field-hint">${OPTION[o]?.[2] || ""}</span></label>`)}
    ${p.id === "huggingface" && html`<p class="flag pedit-flag"><${Icon} name="info" size=${14} /> Hugging Face's plan decides how calls are paid, not which models you may call: Free has no included credits, PRO includes some each month. Gated models also need their license accepted on the model page.</p>`}
    <div class="field"><span class="field-label">Models</span>
      <div class="seg" role="group" aria-label="Which models to offer">
        <button type="button" aria-pressed=${!pick} onClick=${() => choose(false)}>All ${all.length}</button>
        <button type="button" aria-pressed=${pick} onClick=${() => choose(true)}>Only the ones I pick</button></div>
      <span class="muted small">${pick ? `${picked.size} picked. Only these appear in the model pickers and in Auto.` : "Every model the provider lists appears in the pickers, new ones too."}</span>
      ${pick && all.length > 8 && html`<div class="pedit-tools"><input class="input" type="search" placeholder="Find a model" value=${find} onInput=${e => setFind(e.target.value)} aria-label="Find a model" />
        <${Button} size="sm" onClick=${() => setPicked(new Set([...picked, ...shown.map(m => m.id)]))}>Pick shown<//>
        <${Button} size="sm" variant="ghost" onClick=${() => { const s = new Set(picked); shown.forEach(m => s.delete(m.id)); setPicked(s); }}>Clear shown<//></div>`}
      ${pick && html`<ul class="picks">${shown.map(m => html`<li key=${m.id}><label>
          <input type="checkbox" checked=${picked.has(m.id)} onChange=${e => toggle(m.id, e.target.checked)} />
          <span class="who"><b>${m.name}</b>${m.name !== m.id && html`<span class="mono muted small">${m.id}</span>`}</span>
          <span class="model-can">${(m.media || []).map(x => html`<span class=${"cap cap-" + x}>${MEDIA_NAMES[x] || x}</span>`)}</span>
          <span class="mono muted small">${m.detail || ""}</span></label></li>`)}
        ${!shown.length && html`<li class="muted small pad">${p.ready ? (q ? "No model matches." : "The provider listed no models. Check its key or address.") : "Add a key to list this provider's models."}</li>`}</ul>`}
    </div>
  <//>`;
}
export const editProvider = id => openLayer(ProviderEdit, { id });

/** Every call this studio made, per day, provider and model, as the providers count tokens. */
export function Usage() {
  const u = usage.value;
  const [hover, setHover] = useState(null);
  useEffect(() => { readUsage(); }, []);
  if (!u) return null;
  const provs = Object.entries(u.providers || {}).sort((a, b) => b[1].in + b[1].out - (a[1].in + a[1].out) || b[1].calls - a[1].calls);
  const name = id => (id === "local" ? "This PC" : modelInfo.value?.providers.find(p => p.id === id)?.name || id);
  if (!provs.length) return html`<section class="card"><h2 class="card-title">Usage</h2>
    <p class="muted small">No calls yet. Chats, pictures and speech show up here with their tokens, per provider, key and model.</p></section>`;
  const days = u.daily || [];
  const tokens = days.map(d => Object.values(d.providers).reduce((a, t) => a + t.in + t.out, 0));
  const calls = days.map(d => Object.values(d.providers).reduce((a, t) => a + t.calls, 0));
  const max = Math.max(1, ...tokens);
  const tot = days.reduce((a, d) => (Object.values(d.providers).forEach(t => { a.calls += t.calls; a.in += t.in; a.out += t.out; a.failed += t.failed; }), a), { calls: 0, in: 0, out: 0, failed: 0 });
  const day = d => new Date(d + "T12:00").toLocaleDateString(undefined, { month: "short", day: "numeric" });
  const tip = i => `${day(days[i].day)}: ${tokens[i] ? fmtCount(tokens[i]) + " tokens, " : ""}${plural(calls[i], "call")}`;
  const h = hover ?? tokens.length - 1;
  return html`<section class="card usage">
    <h2 class="card-title">Usage <span class="muted small">every call this studio made, kept on this PC</span></h2>
    <div class="tiles">
      <div class="tile"><span class="tile-value">${fmtCount(tot.calls)}</span><span class="tile-label">calls, 30 days</span></div>
      <div class="tile"><span class="tile-value">${fmtCount(tot.in)}</span><span class="tile-label">tokens in</span></div>
      <div class="tile"><span class="tile-value">${fmtCount(tot.out)}</span><span class="tile-label">tokens out</span></div>
      ${tot.failed > 0 && html`<div class="tile"><span class="tile-value">${fmtCount(tot.failed)}</span><span class="tile-label">failed</span></div>`}
    </div>
    <figure class="daychart">
      <figcaption class="small"><span class="muted">Tokens per day</span><span class="mono">${tip(h)}</span></figcaption>
      <div class="day-bars" role="group" aria-label="Tokens per day, last 30 days" onPointerLeave=${() => setHover(null)}>
        ${days.map((d, i) => html`<button type="button" key=${d.day} class=${cx(i === hover && "on")} aria-label=${tip(i)}
          onPointerEnter=${() => setHover(i)} onFocus=${() => setHover(i)} onBlur=${() => setHover(null)}>
          <i style=${`height:${tokens[i] ? Math.max(2, (100 * tokens[i]) / max) : 0}%`} /></button>`)}</div>
      <div class="day-axis muted small"><span>${days[0] && day(days[0].day)}</span><span>today</span></div>
    </figure>
    <div class="table-box"><table class="usage-table">
      <thead><tr><th>Provider and model</th><th class="r">Calls</th><th class="r">Failed</th><th class="r">Tokens in</th><th class="r">Tokens out</th><th class="r">Pictures or characters</th><th class="r">Average</th></tr></thead>
      <tbody>${provs.map(([p, t]) => {
        const ms = Object.entries(u.models || {}).filter(([k]) => k.startsWith(p + "/")).sort((a, b) => b[1].calls - a[1].calls);
        const row = (label, t, cls) => html`<tr class=${cls}><td>${label}</td><td class="r">${fmtCount(t.calls)}</td><td class=${cx("r", t.failed ? "bad" : "muted")}>${t.failed}</td>
          <td class="r">${fmtCount(t.in)}</td><td class="r">${fmtCount(t.out)}</td><td class="r">${t.units ? fmtCount(t.units) : "–"}</td><td class="r">${t.calls ? (t.seconds / t.calls).toFixed(1) + " s" : "–"}</td></tr>`;
        return [row(html`<b>${name(p)}</b>`, t, "u-total"), ...ms.map(([k, mt]) => row(html`<span class="mono">${k.slice(p.length + 1)}</span>`, mt, "u-model"))];
      })}</tbody></table></div>
  </section>`;
}
