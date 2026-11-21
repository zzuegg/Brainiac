import { html, useEffect, useState } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Button } from "/v2/ui/Button.js";
import { fail } from "/v2/state/toasts.js";

// This PC's GPU memory: which model each kind of task keeps loaded and how big it is, unloading
// one by hand, how long an unused one stays, and what happens when the next one does not fit.

const KINDS = { chat: "Chat", image: "Pictures", speech: "Speech", transcribe: "Transcription", video: "Video" };
const COLOR = { chat: 1, image: 2, speech: 3, transcribe: 4, video: 5 };
const POLICIES = [
  ["swap", "Swap", "Unload idle models of other kinds to make room"],
  ["queue", "Queue", "Wait until there is room"],
  ["remote", "Remote", "Send it to the route's remote runner"],
  ["ask", "Ask", "Ask me each time"],
  ["spill", "Spill", "Load anyway; what does not fit goes to slower system memory"],
];
const IDLE = [[1, "1 min"], [5, "5 min"], [15, "15 min"], [60, "1 hour"], [0, "Until the room is needed"]];
const gb = b => (b == null ? "?" : (b / 1073741824).toFixed(1) + " GB");

export function GpuCard() {
  const [g, setG] = useState(null);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(null);
  const read = () => api("/api/gpu").then(d => { setG(d); setError(null); }, e => setError(e.message));
  useEffect(() => { read(); const t = setInterval(read, 3000); return () => clearInterval(t); }, []);
  const change = body => api("/api/gpu", body).then(setG, fail);
  const unload = kind => api("/api/gpu/unload", { kind }).then(setG, fail);

  if (!g) return html`<section class="card"><h2 class="card-title">GPU memory</h2><p class="muted">${error || "Asking this PC…"}</p></section>`;
  const loaded = g.slots.filter(s => s.model);
  const ours = loaded.reduce((a, s) => a + (s.bytes || 0), 0);
  const other = g.budget ? Math.max(0, g.inUse - ours) : 0;
  return html`<section class="card gpu">
    <h2 class="card-title">GPU memory
      ${loaded.some(s => !s.busy) && html`<${Button} size="sm" onClick=${() => unload("all")}>Unload all idle<//>`}</h2>
    <p class="muted small">${g.device || "No device"}${g.budget ? ` · ${gb(g.inUse)} of ${gb(g.budget)} in use · ${gb(Math.max(0, g.budget - g.inUse))} free`
      : " · this device does not count its memory"}</p>
    ${g.budget > 0 && html`<div class="gpu-bar" role="img" aria-label=${`${gb(g.inUse)} of ${gb(g.budget)} in use`}>
      ${loaded.filter(s => s.bytes).map(s => html`<i style=${`--c:var(--series-${COLOR[s.kind]});width:${(100 * s.bytes) / g.budget}%`}
        title=${`${KINDS[s.kind]} · ${s.model} · ${gb(s.bytes)}`} />`)}
      ${other > g.budget * 0.005 && html`<i class="gpu-other" style=${`width:${(100 * other) / g.budget}%`} title=${`Other programs · ${gb(other)}`} />`}
    </div>`}
    ${g.waiting.length > 0 && html`<p class="gpu-wait">Waiting for room: ${g.waiting.map(w => `${KINDS[w.kind] || w.kind} · ${w.model} (${gb(w.need)})`).join(", ")}. Unload a model to let it start.</p>`}
    <ul class="gpu-slots">${g.slots.map(s => html`<li key=${s.kind} class=${s.model ? "on" : ""}>
      <span class="gpu-dot" style=${`--c:var(--series-${COLOR[s.kind]})`} />
      <b>${KINDS[s.kind] || s.kind}</b>
      <span class="gpu-model">${s.model ? html`<span class="mono">${s.model}</span>${s.bytes > 0 && html` <span class="muted">${gb(s.bytes)}</span>`}` : html`<span class="muted">nothing loaded</span>`}</span>
      ${s.busy && html`<span class="badge s-running">working</span>`}
      ${s.pinned && html`<span class="badge">kept loaded</span>`}
      <span class="grow" />
      <button type="button" class="link" aria-expanded=${open === s.kind} onClick=${() => setOpen(open === s.kind ? null : s.kind)}>When full: ${POLICIES.find(p => p[0] === g.policies[s.kind])?.[1]}</button>
      ${s.model && html`<${Button} size="sm" disabled=${s.busy} onClick=${() => unload(s.kind)}>Unload<//>`}
      ${open === s.kind && html`<div class="gpu-policy">
        <div class="seg" role="group" aria-label=${KINDS[s.kind] + ": when the GPU is full"}>${POLICIES.map(([k, n, d]) => html`<button type="button" title=${d}
          aria-pressed=${g.policies[s.kind] === k} onClick=${() => change({ policies: { [s.kind]: k } })}>${n}</button>`)}</div>
        <span class="muted small">${POLICIES.find(p => p[0] === g.policies[s.kind])?.[2]}.</span>
        <label class="small"><input type="checkbox" checked=${s.pinned}
          onChange=${e => change({ pinned: [...g.pinned.filter(k => k !== s.kind), ...(e.target.checked ? [s.kind] : [])] })} /> Keep loaded, never unload it to make room</label>
      </div>`}
    </li>`)}</ul>
    <div class="gpu-idle"><span>Unload a model nobody used for</span>
      <div class="seg" role="group" aria-label="Unload after">${IDLE.map(([m, n]) => html`<button type="button" aria-pressed=${g.idleMinutes === m}
        onClick=${() => change({ idleMinutes: m })}>${n}</button>`)}
        ${!IDLE.some(([m]) => m === g.idleMinutes) && html`<button type="button" aria-pressed="true">${g.idleMinutes} min</button>`}</div></div>
  </section>`;
}
