import { html, useEffect, useState, signal, cx } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button, IconAction } from "/v2/ui/Button.js";
import { Layer } from "/v2/ui/Layer.js";
import { openLayer } from "/v2/ui/overlay.js";
import { plural } from "/v2/lib/format.js";
import { loadModels } from "/v2/state/models.js";
import { fmtCount } from "/v2/state/hub.js";
import { GpuCard } from "/v2/screens/Gpu.js";
import { fail, toast } from "/v2/state/toasts.js";

// The machines models run and train on: this PC, other machines of yours running the studio
// as a worker, and GPU machines rented on Vast.ai, which join as workers once they are set up.

const hosts = signal(null), system = signal(null), runs = signal([]), vast = signal(null), rentals = signal({ rentals: [] });
const usd = n => (n == null ? "–" : "$" + (n < 1 ? n.toFixed(3) : n.toFixed(2)));
const fmtHours = h => (h >= 1 ? `${h.toFixed(h >= 10 ? 0 : 1)} h` : `${Math.max(1, Math.round(h * 60))} min`);
const STAGE = { renting: "renting", starting: "starting up", copying: "copying the studio", building: "building the worker",
  ready: "ready as a machine", destroying: "destroying", destroyed: "destroyed", failed: "failed" };
const readHosts = () => api("/api/hosts").then(h => (hosts.value = h.hosts || []), () => {});
const readVast = () => api("/api/vast").then(v => (vast.value = v), e => (vast.value = { error: e.message }));
function takeRentals(r) {
  const ready = n => (n.rentals || []).filter(x => x.stage === "ready").length;
  if (ready(r) !== ready(rentals.value)) { readHosts(); loadModels(true); } // a rental that became ready is a new machine
  rentals.value = r;
}
const readRentals = () => api("/api/vast/rentals").then(takeRentals, () => {});

export function Machines() {
  useEffect(() => {
    let alive = true, t = 0;
    const tick = async () => {
      await Promise.all([readHosts(), api("/api/train/runs").then(r => (runs.value = r.runs || []), () => {}), readRentals()]);
      if (!alive) return;
      const settingUp = rentals.value.rentals.some(r => !r.ended && r.stage !== "ready");
      t = setTimeout(tick, settingUp ? 4000 : 8000);
    };
    tick(); readVast();
    api("/api/system").then(s => (system.value = s), () => {});
    return () => { alive = false; clearTimeout(t); };
  }, []);
  const here = runs.value.find(r => r.host === "This PC" && (r.state === "running" || r.state === "loading"));
  const queued = runs.value.filter(r => r.host === "This PC" && r.state === "queued").length;
  return html`<div class="page machines-page">
    <header class="page-head"><div><p class="eyebrow">Studio</p><h1 class="page-title">Machines</h1>
      <p class="page-lead">Where models run and train: this PC, your other machines running the studio as a worker, and GPUs you rent by the hour. A machine's models show up on the Models page and answer in chats.</p></div>
      <div class="page-actions"><${Button} variant="primary" icon="plus" onClick=${() => openLayer(AddMachine)}>Add a machine…<//></div></header>
    <section class="card machine">
      <h2 class="card-title"><span class=${cx("live-dot", here ? "busy" : "ok")} aria-hidden="true" />This PC
        <span class=${cx("badge", here ? "s-running" : "s-done")}>${here ? "training" : "idle"}</span>
        <span class="muted small push">${queued ? `${queued} queued` : "one training run at a time"}</span></h2>
      <${Devices} list=${system.value?.devices} speed=${here} />
      ${here && html`<p class="muted small">${here.name} · epoch ${here.epoch + 1} of ${here.epochs}</p>`}
    </section>
    <${GpuCard} />
    ${(hosts.value || []).map(h => html`<${Host} key=${h.id} h=${h} />`)}
    ${hosts.value && !hosts.value.length && html`<section class="card hint-card">
      <h2 class="card-title">Another machine of yours</h2>
      <p class="muted">Run <code>brainiac-studio --worker</code> on it. It prints its addresses and a token; add it here and its models answer in chats, and it can train.</p>
      <${Button} icon="plus" onClick=${() => openLayer(AddMachine)}>Add a machine…<//></section>`}
    <${Vast} />
  </div>`;
}

function Devices({ list, speed }) {
  if (!list?.length) return null;
  return html`<ul class="devices">${list.map(d => html`<li key=${d.name}>
    <${Icon} name=${d.kind === "gpu" ? "build" : "dot"} size=${14} /><span>${d.name}</span>
    <span class="mono muted small">${d.kind === "gpu" ? "GPU" : d.name.toLowerCase() === "cpu" ? "" : "CPU"}</span>
    ${speed?.samplesPerSecond > 0 && (speed.device === d.name || (!speed.device && d.kind === "gpu")) && html`<span class="mono small push">${fmtCount(speed.samplesPerSecond)} samples/s</span>`}
  </li>`)}</ul>`;
}

function Host({ h }) {
  const [sure, setSure] = useState(false);
  const [busy, setBusy] = useState(false);
  const lent = (h.lend || []).includes("huggingface");
  const state = !h.online ? ["s-failed", "offline", "bad"] : h.active ? ["s-running", "training", "busy"] : ["s-done", "idle", "ok"];
  const tr = h.training;
  const lend = async () => {
    setBusy(true);
    try { hosts.value = (await api("/api/hosts/secret", { host: h.id, name: "huggingface", lend: !lent })).hosts || hosts.value; toast(lent ? "Token taken back." : `${h.name} holds your Hugging Face token now.`); }
    catch (x) { fail(x); }
    setBusy(false);
  };
  const forget = async () => {
    if (!sure) { setSure(true); setTimeout(() => setSure(false), 4000); return; }
    try { await api("/api/hosts", { id: h.id, remove: true }); hosts.value = hosts.value.filter(x => x.id !== h.id); loadModels(true); toast(`${h.name} forgotten. Its worker keeps running there.`); } catch (x) { fail(x); }
  };
  return html`<section class=${cx("card machine", !h.online && "off")}>
    <h2 class="card-title"><span class=${cx("live-dot", state[2])} aria-hidden="true" />${h.name}
      <span class=${"badge " + state[0]}>${state[1]}</span>
      <span class="mono muted small push" title=${"Token " + (h.token || "")}>${h.url.replace(/^https?:\/\//, "").replace(/\/$/, "")}</span>
      <${IconAction} icon=${sure ? "check" : "trash"} label=${sure ? "Click again to forget it" : "Forget this machine"} onClick=${forget} /></h2>
    <${Devices} list=${h.devices} speed=${tr} />
    <div class="machine-foot">
      <span class=${cx("small", h.online ? "muted" : "bad")}>${!h.online ? h.error || "does not answer" : tr ? `${tr.name} · epoch ${tr.epoch + 1} of ${tr.epochs}` : h.queued ? `${h.queued} queued` : "Ready."}</span>
      ${h.lendable && html`<${Button} size="sm" icon="lock" aria-pressed=${lent} disabled=${busy} class=${cx(lent && "btn-on")}
        title=${lent ? "This machine holds your Hugging Face token in its memory, for gated models. It was sent sealed, is not written to its disk or log, and is gone when its worker stops. Click to take it back."
          : "Lend this machine your Hugging Face token, so it can download gated models. It is sent sealed and held in the worker's memory only. Whoever controls the machine could still read its memory, so lend it only to machines you trust."}
        onClick=${lend}>${lent ? "Holds your HF token" : "Lend HF token"}<//>`}
    </div>
  </section>`;
}

function AddMachine({ close, top }) {
  const [url, setUrl] = useState("");
  const [token, setToken] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const add = async e => {
    e?.preventDefault();
    setBusy(true); setError(null);
    try {
      const h = await api("/api/hosts", { url: url.trim(), token: token.trim(), name: name.trim() });
      await readHosts(); loadModels(true); toast(`${h.name} added.`); close();
    } catch (x) { setError(x.message.replace(/^\/api\/hosts: /, "")); setBusy(false); }
  };
  const footer = html`${error && html`<p class="msg-error small">${error}</p>`}<span class="grow" />
    <${Button} variant="ghost" onClick=${close}>Cancel<//><${Button} variant="primary" disabled=${busy || !url.trim() || !token.trim()} onClick=${add}>${busy ? "Connecting…" : "Connect"}<//>`;
  return html`<${Layer} title="Add a machine" subtitle="Another machine of yours running the studio as a worker." close=${close} top=${top} footer=${footer}>
    <form onSubmit=${add} autocomplete="off">
      <p class="flag pedit-flag"><${Icon} name="info" size=${14} /><span>On the other machine run <code>brainiac-studio --worker</code>. It prints its addresses and a token.</span></p>
      <label class="field"><span class="field-label">Address</span>
        <input class="input mono" autofocus value=${url} onInput=${e => setUrl(e.target.value)} placeholder="192.168.1.20:7870" spellcheck="false" /></label>
      <label class="field"><span class="field-label">Token</span>
        <input class="input mono" type="password" value=${token} onInput=${e => setToken(e.target.value)} spellcheck="false" /></label>
      <label class="field"><span class="field-label">Name</span>
        <input class="input" value=${name} onInput=${e => setName(e.target.value)} placeholder="Optional, such as Workshop PC" />
        <span class="field-hint">Empty uses the name the machine gives itself.</span></label>
      <button type="submit" hidden />
    </form>
  <//>`;
}

/** Vast.ai: the key, the balance, the machines rented from here and from elsewhere, and finding more. */
function Vast() {
  const s = vast.value;
  const [key, setKey] = useState("");
  const [busy, setBusy] = useState(false);
  const [sure, setSure] = useState(null);
  const r = rentals.value;
  if (!s) return null;
  const saveKey = async e => {
    e.preventDefault(); setBusy(true);
    try { const v = await api("/api/vast/key", { key: key.trim() }); vast.value = v; setKey(""); if (!v.error) toast("Vast.ai key saved."); } catch (x) { fail(x); }
    setBusy(false);
  };
  const arm = (k, then) => { if (sure !== k) { setSure(k); setTimeout(() => setSure(x => (x === k ? null : x)), 4000); return; } setSure(null); then(); };
  const destroy = id => arm(id, () => api("/api/vast/destroy", { id }).then(v => { takeRentals(v); readVast(); toast("Destroyed. Billing ends."); }, fail));
  const mine = new Set(r.rentals.filter(x => x.instance && !x.ended).map(x => x.instance));
  const others = (s.instances || []).filter(i => !mine.has(i.id));
  return html`<section class="card machine vast">
    <h2 class="card-title"><span class=${cx("live-dot", !s.key ? "" : s.error ? "bad" : "ok")} aria-hidden="true" />Vast.ai
      <span class="badge">rented by the hour</span>
      ${s.key && html`<span class="mono muted small push link-icon" title=${s.key.source === "environment" ? "Key from " + s.key.label : "Key saved on this PC"}><${Icon} name="lock" size=${12} /> ${s.key.hint}</span>
        ${s.balance != null && html`<span class="mono" title="Balance on Vast.ai">${usd(s.balance)}</span>`}
        ${s.key.source === "saved" && html`<${IconAction} icon=${sure === "key" ? "check" : "trash"} label=${sure === "key" ? "Click again to remove the key" : "Remove the Vast.ai key"}
          onClick=${() => arm("key", () => api("/api/vast/key", { remove: true }).then(v => (vast.value = v), fail))} />`}`}</h2>
    ${!s.key ? html`<p class="muted small">Rent GPU machines by the hour; each one sets itself up as a worker and joins here. Paste an API key from <a href="https://cloud.vast.ai/manage-keys/" target="_blank" rel="noopener">cloud.vast.ai</a>; it stays on this PC and goes only to Vast.ai.</p>
      <form class="key-form" onSubmit=${saveKey} autocomplete="off">
        <input class="input mono" type="password" value=${key} onInput=${e => setKey(e.target.value)} placeholder="Vast.ai API key" aria-label="Vast.ai API key" spellcheck="false" />
        <${Button} type="submit" disabled=${busy || !key.trim()}>${busy ? "Checking…" : "Save"}<//></form>`
    : html`
      ${s.error && html`<p class="msg-error small"><${Icon} name="warn" size=${14} />${s.error}</p>`}
      ${r.rentals.length > 0 && html`<ul class="rentals">${r.rentals.map(x => {
        const live = !x.ended, tone = x.stage === "ready" ? "ok" : x.stage === "failed" ? "bad" : live ? "busy" : "";
        return html`<li key=${x.key} class=${cx(!live && "ended")}>
          <span class=${cx("live-dot", tone)} aria-hidden="true" /><b>${x.gpu}</b>
          <span class=${cx("badge", x.stage === "ready" ? "s-done" : x.stage === "failed" ? "s-failed" : live ? "s-running" : "")}>${STAGE[x.stage] || x.stage}</span>
          ${live && x.stage !== "destroying" && html`<span class="muted small">${usd(x.pricePerHour)}/h · destroyed in ${fmtHours(Math.max(0, (x.until - Date.now()) / 3600000))}${x.instance ? ` · #${x.instance}` : ""}</span>`}
          <span class="mono push" title=${`Cost so far at ${usd(x.pricePerHour)} an hour`}>${usd(x.cost)}</span>
          ${live && x.stage !== "destroying" && (x.instance
            ? html`<${IconAction} icon=${sure === x.instance ? "check" : "trash"} label=${sure === x.instance ? "Click again to destroy it; billing ends" : "Destroy it now"} onClick=${() => destroy(x.instance)} />`
            : html`<${IconAction} icon="stop" label="Stop renting it" onClick=${() => api("/api/vast/cancel", { key: x.key }).then(takeRentals, fail)} />`)}
          ${x.leftover && html`<p class="muted small rent-note">Left running when the studio last stopped without closing, so it was destroyed when the studio started.</p>`}
          ${x.last && html`<p class="mono muted small rent-note rent-log" title=${x.last}>${x.last}</p>`}
          ${x.error && html`<p class="msg-error small rent-note">${x.error}</p>`}
        </li>`;
      })}</ul>`}
      ${others.length > 0 && html`<p class="muted small">Also rented on this account:</p><ul class="rentals">${others.map(i => html`<li key=${i.id}>
        <span class="live-dot" aria-hidden="true" /><b>${i.gpu}${i.gpus > 1 ? " × " + i.gpus : ""}</b><span class="mono muted small">${i.status || ""}</span>
        <span class="mono push">${usd(i.pricePerHour)}/h</span>
        <${IconAction} icon=${sure === i.id ? "check" : "trash"} label=${sure === i.id ? "Click again to destroy it; billing ends" : `Destroy instance ${i.id}`} onClick=${() => destroy(i.id)} /></li>`)}</ul>`}
      <div class="machine-foot"><span class="muted small">${(s.instances || []).length ? plural(s.instances.length, "machine") + " rented" : "Nothing rented."}</span>
        <${Button} size="sm" icon="search" onClick=${() => openLayer(FindGpus)}>Find GPUs…<//></div>`}
  </section>`;
}

const filters = signal({ gpu: "", minGpuRam: 0, gpus: 1, maxPrice: 0 });
function FindGpus({ close, top }) {
  const [f, setF] = useState(filters.value);
  const [offers, setOffers] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [pick, setPick] = useState(null);
  const [hours, setHours] = useState(2);
  const [renting, setRenting] = useState(false);
  const max = rentals.value.maxHours || 48, unavailable = rentals.value.unavailable;
  const search = async e => {
    e?.preventDefault(); setBusy(true); setError(null); filters.value = f;
    try { setOffers((await api("/api/vast/offers", f)).offers || []); } catch (x) { setError(x.message); }
    setBusy(false);
  };
  useEffect(() => { search(); }, []);
  const rent = async x => {
    if (!(hours > 0) || hours > max) { setError(`Choose up to ${max} hours.`); return; }
    setRenting(true); setError(null);
    try {
      takeRentals(await api("/api/vast/rent", { offer: x.id, pricePerHour: x.pricePerHour, hours }));
      readVast(); toast(`Renting ${x.gpu}. It sets itself up in 10 to 20 minutes.`); close();
    } catch (e) { setError(e.message); setRenting(false); }
  };
  const num = (k, v) => setF({ ...f, [k]: +v || 0 });
  return html`<${Layer} title="GPUs on Vast.ai" subtitle="Verified machines you can rent now, cheapest first." wide close=${close} top=${top}
      footer=${html`<span class="muted small">Each rental asks first, with its price.</span><span class="grow" /><${Button} onClick=${close}>Done<//>`}>
    <form class="vast-filters" onSubmit=${search} autocomplete="off">
      <label class="field"><span class="field-label">GPU</span><input class="input" value=${f.gpu} onInput=${e => setF({ ...f, gpu: e.target.value })} placeholder="any, or 4090, A100…" /></label>
      <label class="field"><span class="field-label">Memory per GPU</span><input class="input mono" type="number" min="0" value=${f.minGpuRam || ""} onInput=${e => num("minGpuRam", e.target.value)} placeholder="any GB" /></label>
      <label class="field"><span class="field-label">GPUs</span><input class="input mono" type="number" min="1" max="16" value=${f.gpus || 1} onInput=${e => num("gpus", e.target.value)} /></label>
      <label class="field"><span class="field-label">Most per hour</span><input class="input mono" type="number" min="0" step="0.05" value=${f.maxPrice || ""} onInput=${e => num("maxPrice", e.target.value)} placeholder="any $" /></label>
      <${Button} variant="primary" icon="search" type="submit" disabled=${busy}>${busy ? "Looking…" : "Search"}<//>
    </form>
    ${unavailable && html`<p class="flag pedit-flag"><${Icon} name="warn" size=${14} /><span>${unavailable}</span></p>`}
    ${error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${error}</p>`}
    ${offers && !offers.length && html`<p class="muted">No offers match. Loosen the filters.</p>`}
    ${offers?.length > 0 && html`<div class="table-box"><table class="usage-table offers">
      <thead><tr><th>GPU</th><th class="r">Memory</th><th class="r">Per hour</th><th class="r">Reliable</th><th>Where</th><th class="r">CPU · RAM</th><th class="r">Download</th><th /></tr></thead>
      <tbody>${offers.map(x => html`<tr key=${x.id} class=${cx(pick === x.id && "on")}>
          <td><b>${x.gpu}</b>${x.gpus > 1 && html` <span class="muted">× ${x.gpus}</span>`}</td><td class="r">${x.gpuRamGb ?? "?"} GB</td><td class="r">${usd(x.pricePerHour)}</td>
          <td class="r">${x.reliability != null ? (x.reliability * 100).toFixed(1) + "%" : "–"}</td><td>${x.location || "–"}</td>
          <td class="r">${x.cpuCores != null ? Math.round(x.cpuCores) : "?"} · ${x.ramGb ?? "?"} GB</td><td class="r">${x.downMbps != null ? Math.round(x.downMbps) + " Mb/s" : "–"}</td>
          <td class="r"><${Button} size="sm" disabled=${!!unavailable} onClick=${() => setPick(pick === x.id ? null : x.id)}>Rent…<//></td></tr>
        ${pick === x.id && html`<tr class="rent-confirm"><td colspan="8"><div class="rent-box">
          <b>Rent ${x.gpus > 1 ? x.gpus + " × " : ""}${x.gpu} for ${usd(x.pricePerHour)} an hour?</b>
          <span class="muted small">Billing starts when Vast.ai rents it to you and runs until it is destroyed; storage costs a few cents an hour on top. Setting up takes 10 to 20 minutes and is billed too. It logs in with your SSH key from ~/.ssh, which must be added at cloud.vast.ai.</span>
          <div class="rent-row"><label>Destroy it after <input class="input mono" type="number" min="0.25" max=${max} step="0.25" value=${hours} onInput=${e => setHours(+e.target.value)} /> hours</label>
            <span class="muted">at most <b class="mono">${usd(x.pricePerHour * (hours || 0))}</b> for the machine</span><span class="grow" />
            <${Button} variant="ghost" onClick=${() => setPick(null)}>Cancel<//>
            <${Button} variant="primary" disabled=${renting} onClick=${() => rent(x)}>${renting ? "Renting…" : `Rent for ${usd(x.pricePerHour)}/h`}<//></div>
        </div></td></tr>`}`)}</tbody></table></div>`}
  <//>`;
}
