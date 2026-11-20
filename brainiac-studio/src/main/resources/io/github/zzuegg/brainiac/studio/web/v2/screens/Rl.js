import { html, useEffect, useRef, useState, cx } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button, IconAction } from "/v2/ui/Button.js";
import { fmtCount } from "/v2/state/hub.js";
import { fail } from "/v2/state/toasts.js";

// Reinforcement learning: an agent learns an environment by acting in it, on this PC's CPU. A run
// shows how its return grew, and when it ends one of its episodes can be watched step by step.
// Runs are kept until the studio closes.

const STATES = { running: ["learning", "s-running"], done: ["done", "s-done"], stopped: ["stopped", ""], failed: ["failed", "s-failed"] };
const num = v => (v == null ? "–" : Math.abs(v) >= 100 ? Math.round(v).toLocaleString() : (+v).toFixed(1));
const fits = (a, e) => !!a && !!e && !(a.discreteOnly && e.continuous) && (!a.table || e.grid);

export function Rl() {
  const [d, setD] = useState(null);
  const [sel, setSel] = useState(null);
  const [run, setRun] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    let alive = true, t = 0;
    const tick = async () => {
      try {
        const j = await api("/api/rl");
        if (!alive) return;
        setD(j); setError(null);
        setSel(s => (s && j.runs.some(r => r.id === s) ? s : j.runs[0]?.id || null));
      } catch (e) { setError(e.message); }
      if (alive) t = setTimeout(tick, (d?.runs || []).some(r => r.state === "running") ? 1200 : 5000);
    };
    tick();
    return () => { alive = false; clearTimeout(t); };
  }, []);
  // the run shown is read again while it learns; once ended it does not change any more
  const listed = d?.runs.find(r => r.id === sel);
  useEffect(() => {
    if (!sel) { setRun(null); return; }
    if (run?.id === sel && run.state !== "running" && listed?.state === run.state) return;
    api("/api/rl/get", { id: sel }).then(setRun, () => {});
  }, [sel, listed?.done, listed?.state]);
  const started = id => { setSel(id); api("/api/rl").then(setD, () => {}); };
  return html`<div class="page rl-page">
    <header class="page-head"><div><p class="eyebrow">Studio</p><h1 class="page-title">Learning from rewards</h1>
      <p class="page-lead">An agent learns an environment by acting in it and being rewarded, on this PC's CPU. Runs are kept until the studio closes.</p></div></header>
    ${error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${error}</p>`}
    ${!d ? html`<p class="muted">Looking for environments…</p>` : html`<div class="rl-cols">
      <${NewRun} d=${d} started=${started} />
      <div class="rl-main">
        <section class="card"><h2 class="card-title">Runs</h2>
          ${!d.runs.length ? html`<p class="muted small">No runs yet. CartPole with PPO learns to balance in about 50,000 steps.</p>`
            : html`<div class="table-box"><table class="usage-table rl-runs">
              <thead><tr><th>Run</th><th>State</th><th class="r">Steps</th><th class="r">Episodes</th><th class="r" title="The mean return of the last 20 episodes while learning">Return</th><th class="r" title="The mean return of 10 fresh episodes without exploring, after the run ended">Score</th></tr></thead>
              <tbody>${d.runs.map(r => html`<tr key=${r.id} class=${cx("rl-row", r.id === sel && "on")} tabindex="0" onClick=${() => setSel(r.id)} onKeyDown=${e => e.key === "Enter" && setSel(r.id)}>
                <td><b>${r.envName}</b> <span class="muted">${r.algorithmName}</span></td>
                <td><span class=${"badge " + (STATES[r.state]?.[1] || "")}>${STATES[r.state]?.[0] || r.state}</span></td>
                <td class="r">${fmtCount(r.done)} <span class="muted">/ ${fmtCount(r.steps)}</span></td><td class="r">${r.episodes.toLocaleString()}</td>
                <td class="r">${num(r.meanReturn)}</td><td class="r">${num(r.score)}</td></tr>`)}</tbody></table></div>`}
        </section>
        ${run && html`<${RunDetail} key=${run.id} r=${run} gone=${() => { setSel(null); setRun(null); api("/api/rl").then(setD, () => {}); }} />`}
      </div></div>`}
  </div>`;
}

function NewRun({ d, started }) {
  const [env, setEnv] = useState("cartpole");
  const [algo, setAlgo] = useState("ppo");
  const [steps, setSteps] = useState(50000);
  const [seed, setSeed] = useState(0);
  const [hidden, setHidden] = useState("64, 64");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const e = d.environments.find(x => x.id === env), a = d.algorithms.find(x => x.id === algo);
  const pickEnv = id => { setEnv(id); const ne = d.environments.find(x => x.id === id); if (!fits(a, ne)) setAlgo(d.algorithms.find(x => fits(x, ne))?.id || "ppo"); };
  const start = async ev => {
    ev.preventDefault();
    const h = String(hidden).split(/[\s,x×]+/).filter(Boolean).map(Number), n = Math.round(+steps);
    if (!(n >= 100 && n <= d.maxSteps)) { setError(`Steps are between 100 and ${d.maxSteps.toLocaleString()}.`); return; }
    if (!a.table && (!h.length || h.length > 2 || h.some(x => !Number.isInteger(x) || x < 4 || x > 512))) { setError("Give one or two hidden layers of 4 to 512 units, such as 64, 64."); return; }
    setBusy(true); setError(null);
    try { const r = await api("/api/rl/start", { env, algorithm: algo, steps: n, seed: Math.round(+seed) || 0, hidden: a.table ? undefined : h }); started(r.id); }
    catch (x) { setError(x.message); }
    setBusy(false);
  };
  return html`<form class="card rl-new" onSubmit=${start} autocomplete="off" noValidate>
    <h2 class="card-title">New run</h2>
    <label class="field"><span class="field-label">Environment</span>
      <select class="input" value=${env} onChange=${x => pickEnv(x.target.value)}>${d.environments.map(x => html`<option value=${x.id}>${x.name}</option>`)}</select>
      <span class="field-hint">${e && `${e.what} It sees ${e.observations} numbers and has ${e.continuous ? e.actions + " continuous action" + (e.actions === 1 ? "" : "s") : e.actions + " actions"}.`}</span></label>
    <label class="field"><span class="field-label">Learner</span>
      <select class="input" value=${algo} onChange=${x => setAlgo(x.target.value)}>${d.algorithms.map(x => html`<option value=${x.id} disabled=${!fits(x, e)}>${x.name}</option>`)}</select>
      <span class="field-hint">${a?.what}</span></label>
    <div class="form-grid">
      <label class="field"><span class="field-label">Steps</span><input class="input mono" type="number" min="100" max=${d.maxSteps} step="100" value=${steps} onInput=${x => setSteps(x.target.value)} /></label>
      <label class="field"><span class="field-label">Seed</span><input class="input mono" type="number" step="1" value=${seed} onInput=${x => setSeed(x.target.value)} /></label>
    </div>
    ${!a?.table && html`<label class="field"><span class="field-label">Hidden layers</span><input class="input mono" value=${hidden} placeholder="64, 64" spellcheck="false" onInput=${x => setHidden(x.target.value)} /></label>`}
    ${error && html`<p class="msg-error small">${error}</p>`}
    <${Button} variant="primary" icon="play" type="submit" disabled=${busy}>${busy ? "Starting…" : "Start learning"}<//>
  </form>`;
}

/** The return per episode over the steps: one line on its own scale, with its range labelled. */
function Curve({ curve }) {
  const [at, setAt] = useState(null);
  if (!curve || curve.length < 2) return html`<p class="muted small">The curve shows after a few episodes.</p>`;
  const W = 640, H = 180, L = 52, B = 22, T = 8, xs = curve.map(p => p[0]), ys = curve.map(p => p[1]);
  const x1 = Math.max(1, ...xs), lo = Math.min(...ys), hi = Math.max(...ys), span = Math.max(1e-9, hi - lo);
  const X = v => L + (v / x1) * (W - L - 8), Y = v => T + (1 - (v - lo) / span) * (H - T - B);
  const near = at == null ? null : curve.reduce((b, p) => (Math.abs(p[0] - at) < Math.abs(b[0] - at) ? p : b), curve[0]);
  const move = e => { const r = e.currentTarget.getBoundingClientRect(); setAt(Math.max(0, ((e.clientX - r.left) / r.width * W - L) / (W - L - 8) * x1)); };
  return html`<figure class="rl-curve">
    <figcaption class="small"><span class="muted">Return per episode</span><span class="mono">${near ? `${fmtCount(near[0])} steps: ${num(near[1])}` : `${num(ys[0])} → ${num(ys.at(-1))}`}</span></figcaption>
    <svg viewBox=${`0 0 ${W} ${H}`} role="img" aria-label=${`Return from ${num(ys[0])} to ${num(ys.at(-1))} over ${fmtCount(x1)} steps`} onPointerMove=${move} onPointerLeave=${() => setAt(null)}>
      <line x1=${L} y1=${Y(lo)} x2=${W - 8} y2=${Y(lo)} class="grid" /><line x1=${L} y1=${Y(hi)} x2=${W - 8} y2=${Y(hi)} class="grid" />
      <text x=${L - 6} y=${Y(hi) + 4} text-anchor="end">${num(hi)}</text><text x=${L - 6} y=${Y(lo) + 4} text-anchor="end">${num(lo)}</text>
      <text x=${L} y=${H - 5}>0</text><text x=${W - 8} y=${H - 5} text-anchor="end">${fmtCount(x1)} steps</text>
      <polyline class="line" points=${curve.map(p => X(p[0]).toFixed(1) + "," + Y(p[1]).toFixed(1)).join(" ")} />
      ${near && html`<line x1=${X(near[0])} y1=${T} x2=${X(near[0])} y2=${H - B} class="cross" /><circle cx=${X(near[0])} cy=${Y(near[1])} r="4" class="dot" />`}
    </svg></figure>`;
}

function RunDetail({ r, gone }) {
  const frames = r.frames || [];
  const [i, setI] = useState(0);
  const [playing, setPlaying] = useState(false);
  const timer = useRef(0);
  useEffect(() => {
    clearInterval(timer.current);
    if (playing) timer.current = setInterval(() => setI(n => { if (n >= frames.length - 1) { setPlaying(false); return n; } return n + 1; }), 90);
    return () => clearInterval(timer.current);
  }, [playing, frames.length]);
  const st = STATES[r.state] || [r.state, ""];
  const at = Math.min(i, Math.max(0, frames.length - 1));
  return html`<section class="card rl-detail">
    <h2 class="card-title">${r.envName} · ${r.algorithmName}<span class=${"badge " + st[1]}>${st[0]}</span>
      <span class="push" />${r.state === "running"
        ? html`<${Button} size="sm" icon="stop" onClick=${() => api("/api/rl/stop", { id: r.id }).catch(fail)}>Stop<//>`
        : html`<${IconAction} icon="trash" label="Forget this run" onClick=${() => api("/api/rl/delete", { id: r.id }).then(gone, fail)} />`}</h2>
    ${r.error && html`<p class="msg-error small">${r.error}</p>`}
    <dl class="serve-rows rl-kv">
      <dt>Steps</dt><dd class="mono">${r.done.toLocaleString()} of ${r.steps.toLocaleString()}</dd>
      <dt>Episodes</dt><dd class="mono">${r.episodes.toLocaleString()}</dd>
      <dt>Return</dt><dd class="mono">${r.meanReturn != null ? html`${num(r.meanReturn)} <span class="muted">last 20 · best ${num(r.bestReturn)}</span>` : "–"}</dd>
      ${r.score != null && html`<dt>Score</dt><dd class="mono">${num(r.score)} <span class="muted">10 fresh episodes, no exploring</span></dd>`}
      ${r.hidden?.length > 0 && !["qlearning", "sarsa"].includes(r.algorithm) && html`<dt>Networks</dt><dd class="mono">${r.hidden.join(" × ")} hidden</dd>`}
      <dt>Seed</dt><dd class="mono">${r.seed}</dd></dl>
    <${Curve} curve=${r.curve} />
    ${frames.length > 0 ? html`<div class="rl-watch">
        <div class="row-btns"><b class="small">Watch it play</b><span class="grow" />
          <${IconAction} icon=${playing ? "pause" : "play"} label=${playing ? "Pause" : "Play"} onClick=${() => { if (!playing && at >= frames.length - 1) setI(0); setPlaying(!playing); }} /></div>
        <pre class="rl-frame" aria-live="off">${frames[at]}</pre>
        <div class="row-btns"><input type="range" min="0" max=${frames.length - 1} value=${at} aria-label="Step of the episode" class="grow" onInput=${e => { setPlaying(false); setI(+e.target.value); }} />
          <span class="mono muted small">${at + 1} / ${frames.length}</span></div></div>`
      : r.state === "running" && html`<p class="muted small">When the run ends, one of its episodes can be watched here.</p>`}
  </section>`;
}
