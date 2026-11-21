import { html, useEffect, useRef, useState, cx } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Button } from "/v2/ui/Button.js";
import { Icon } from "/v2/ui/Icon.js";
import { navigate } from "/v2/state/workspace.js";
import { fail, toast } from "/v2/state/toasts.js";
import { ItemHeader, routeItem, Gone } from "/v2/screens/ItemHeader.js";
import { newTraining } from "/v2/wizards/Training.js";
import { watchRun, readRun, runStore, stopRun, isActive, STATE_NAMES, runColor, fmtNum, eta, entryLabel, entry,
  loadSetup, setup } from "/v2/state/training.js";

// One training run: how far it is, its loss as it learns (beside other runs of the same network
// when asked), every epoch's numbers, the settings it ran with, and the model it saved, which can
// be tried on a sample at once.

export function Training({ tab }) {
  const item = routeItem(tab);
  const id = tab.route.value.id;
  const s = runStore(id);
  useEffect(() => watchRun(id), [id]);
  useEffect(() => { loadSetup().catch(() => {}); }, []);
  // the item list notices a run ending first; read the run once more then
  useEffect(() => { if (item && s.run.value && item.state !== s.run.value.state) readAgain(id); }, [item && item.state]);
  const [compare, setCompare] = useState([]);

  if (item === false || s.missing.value) return html`<${Gone} tab=${tab} what="training run" />`;
  const r = s.run.value;
  if (!item || !r) return html`<div class="loading" aria-busy="true" />`;

  const network = tab.items.value.find(i => i.kind === "graph" && i.id === r.spec.model);
  const siblings = tab.items.value.filter(i => i.kind === "training" && i.id !== r.id && i.model === r.spec.model);
  const left = eta(r);
  const sub = [STATE_NAMES[r.state] || r.state, `epoch ${r.epoch} of ${r.epochs}`, left && `done in ${left}`, r.host !== "This PC" && "on " + r.host]
    .filter(Boolean).join(" · ");
  const header = html`<${ItemHeader} tab=${tab} item=${item} sub=${sub}>
    <div class="row-btns">
      ${network && html`<${Button} icon="graph" onClick=${() => navigate(tab, { screen: "graph", id: network.id })}>Open network<//>`}
      <${Button} icon="redo" onClick=${() => newTraining(tab, { spec: r.spec })} title="Starts a new run with these settings, to change first">Train again…<//>
      ${isActive(r) && html`<${Button} variant="danger" icon="stop" onClick=${() => stopRun(r.id).then(() => toast("Stopping; what it learned so far is kept")).catch(fail)}
          title="Stops after the current step and keeps what it learned">Stop<//>`}
    </div>
  <//>`;
  return html`<div class="run-screen">
    <header class="run-head">${header}</header>
    <div class="run-body">
      ${r.state === "failed" && html`<p class="flag flag-bad" role="alert"><${Icon} name="warn" />${r.error || "The run failed."}</p>`}
      ${r.state === "interrupted" && html`<p class="flag flag-warn"><${Icon} name="info" />The studio stopped while this run trained. What it learned up to its last saved epoch is kept${r.output ? " in " + r.output : ""}.</p>`}
      <${Progress} r=${r} />
      <section class="card">
        <div class="card-title">Loss<span class="muted">per epoch, lower is better</span><span class="grow" />
          ${siblings.length > 0 && html`<span class="compare-label">Compare with</span>
            ${siblings.slice(0, 6).map(o => html`<button type="button" class=${cx("compare-chip", compare.includes(o.id) && "on")} aria-pressed=${compare.includes(o.id)}
              style=${"--c:" + runColor(o.id)} onClick=${() => setCompare(compare.includes(o.id) ? compare.filter(x => x !== o.id) : [...compare, o.id].slice(-3))}>
              <i />${o.title}</button>`)}`}
        </div>
        <${LossChart} runs=${[r, ...compare.map(c => readRun(c).run.value).filter(Boolean)]} />
      </section>
      <${Epochs} r=${r} />
      <div class="run-cols">
        <${Settings} r=${r} />
        <${Result} r=${r} tab=${tab} />
      </div>
    </div>
  </div>`;
}

function readAgain(id) {
  const s = runStore(id);
  api("/api/train/run", { id }).then(r => (s.run.value = r)).catch(() => {});
}

// ---------------------------------------------------------------- progress and headline numbers

const lastOf = r => r.history?.[r.history.length - 1];
function metricOf(h) {
  if (!h) return null;
  const vm = h.validationMetrics || {}, tm = h.metrics || {};
  const k = Object.keys(vm)[0] || Object.keys(tm)[0];
  return k ? { name: (k in vm ? "checked " : "") + k.replace(/_/g, " "), v: vm[k] ?? tm[k], k } : null;
}

function Progress({ r }) {
  const pct = r.steps ? Math.min(100, (r.step / r.steps) * 100) : r.state === "finished" ? 100 : 0;
  const h = lastOf(r), m = metricOf(h);
  const tiles = [
    ["Loss", fmtNum(h ? h.loss : r.loss), "on the data it learns from"],
    h?.validationLoss != null && ["Checked loss", fmtNum(h.validationLoss), "on data held back"],
    m && [m.name[0].toUpperCase() + m.name.slice(1), m.k === "accuracy" ? (m.v * 100).toFixed(1) + " %" : fmtNum(m.v), ""],
    r.samplesPerSecond && ["Speed", `${Math.round(r.samplesPerSecond).toLocaleString()} / s`, r.device || ""],
    r.bestEpoch > 0 && ["Best epoch", String(r.bestEpoch), r.stoppedEarly ? "stopped early, kept the best" : "its weights are kept"],
  ].filter(Boolean);
  return html`<section class="card run-progress">
    <div class="prog" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${Math.round(pct)} aria-label="Progress">
      <i class=${cx(isActive(r) && "prog-live")} style=${`width:${pct.toFixed(1)}%`} /></div>
    <div class="tiles">${tiles.map(([label, value, note]) => html`<div class="tile"><span class="tile-label">${label}</span>
      <span class="tile-value">${value}</span>${note && html`<span class="tile-note">${note}</span>`}</div>`)}</div>
  </section>`;
}

// ---------------------------------------------------------------- the chart

/** Loss against epochs: a line per run from its step losses, dots for the checked loss. */
function LossChart({ runs }) {
  const box = useRef(null);
  const [w, setW] = useState(640);
  const [log, setLog] = useState(false);
  const [hover, setHover] = useState(null);
  useEffect(() => {
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.floor(e.contentRect.width))));
    ro.observe(box.current);
    return () => ro.disconnect();
  }, []);
  const series = runs.map(r => {
    const per = r.steps && r.epochs ? r.steps / r.epochs : 1;
    return {
      r, color: runColor(r.id),
      line: (r.curve || []).map(([step, loss]) => [step / per, loss]),
      checked: (r.history || []).filter(h => h.validationLoss != null).map(h => [h.epoch, h.validationLoss]),
    };
  }).filter(s => s.line.length > 1 || s.checked.length);
  const H = 240, L = 48, R = 14, T = 12, B = 28;
  const f = v => (log ? Math.log10(Math.max(v, 1e-8)) : v);
  let maxX = 1, lo = Infinity, hi = -Infinity;
  for (const s of series) for (const [x, y] of [...s.line, ...s.checked]) {
    maxX = Math.max(maxX, x);
    if (isFinite(y)) { lo = Math.min(lo, f(y)); hi = Math.max(hi, f(y)); }
  }
  for (const s of series) maxX = Math.max(maxX, s.r.epochs || 1);
  if (!isFinite(lo)) { lo = 0; hi = 1; }
  if (!log) lo = Math.min(0, lo);
  if (hi - lo < 1e-9) hi = lo + 1;
  const X = x => L + (x / maxX) * (w - L - R), Y = v => T + (1 - (f(v) - lo) / (hi - lo)) * (H - T - B);
  const ticksY = log ? [...new Set(Array.from({ length: 5 }, (_, i) => lo + (i * (hi - lo)) / 4))].map(t => 10 ** t)
    : Array.from({ length: 5 }, (_, i) => lo + (i * (hi - lo)) / 4);
  const ticksX = niceEpochs(maxX);
  const path = pts => pts.filter(([, y]) => isFinite(y)).map(([x, y], i) => `${i ? "L" : "M"}${X(x).toFixed(1)},${Y(y).toFixed(1)}`).join("");
  const move = e => {
    const rect = box.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left - L) / (w - L - R)) * maxX;
    if (x < 0 || x > maxX) return setHover(null);
    setHover({ x, at: series.map(s => ({ s, p: nearest(s.line, x), c: nearest(s.checked, x, 0.5) })) });
  };
  return html`<div class="chart">
    <div class="chart-tools">
      ${series.length > 1 && html`<div class="chart-legend">${series.map(s => html`<span style=${"--c:" + s.color}><i />${s.r.name}</span>`)}</div>`}
      <span class="chart-legend legend-key"><span><i class="k-line" />while learning</span><span><i class="k-dot" />checked</span></span>
      <span class="grow" />
      <div class="seg" role="group" aria-label="Scale">
        <button type="button" aria-pressed=${!log} onClick=${() => setLog(false)}>Linear</button>
        <button type="button" aria-pressed=${log} onClick=${() => setLog(true)}>Log</button></div>
    </div>
    <div class="chart-box" ref=${box} onMouseMove=${move} onMouseLeave=${() => setHover(null)}>
      ${!series.length ? html`<p class="chart-empty">The loss shows here once the first steps are done.</p>` : html`
      <svg width=${w} height=${H} role="img" aria-label="Loss against epochs">
        ${ticksY.map(t => html`<g><line x1=${L} x2=${w - R} y1=${Y(t)} y2=${Y(t)} class="grid" />
          <text x=${L - 8} y=${Y(t) + 4} text-anchor="end" class="tick">${fmtNum(t)}</text></g>`)}
        ${ticksX.map(t => html`<text x=${X(t)} y=${H - 8} text-anchor="middle" class="tick">${t}</text>`)}
        <line x1=${L} x2=${w - R} y1=${H - B} y2=${H - B} class="axis" />
        ${series.map(s => html`<g style=${"--c:" + s.color}>
          <path d=${path(s.line)} class="ln" />
          ${s.checked.map(([x, y]) => html`<circle cx=${X(x)} cy=${Y(y)} r="4" class="pt" />`)}
        </g>`)}
        ${hover && html`<line x1=${X(hover.x)} x2=${X(hover.x)} y1=${T} y2=${H - B} class="cross" />
          ${hover.at.filter(a => a.p).map(a => html`<circle cx=${X(a.p[0])} cy=${Y(a.p[1])} r="4" class="hv" style=${"--c:" + a.s.color} />`)}`}
      </svg>`}
      ${hover && series.length > 0 && html`<div class="chart-tip" style=${`left:${Math.min(X(hover.x) + 12, w - 190)}px`}>
        <b>epoch ${hover.x.toFixed(1)}</b>
        ${hover.at.map(a => html`<div><i style=${"--c:" + a.s.color} />${series.length > 1 ? a.s.r.name + ": " : ""}${a.p ? fmtNum(a.p[1]) : "–"}${a.c ? html` <span class="muted">· checked ${fmtNum(a.c[1])}</span>` : ""}</div>`)}
      </div>`}
    </div>
  </div>`;
}
function nearest(pts, x, within = Infinity) {
  let best = null, d = within;
  for (const p of pts) { const e = Math.abs(p[0] - x); if (e <= d) { d = e; best = p; } }
  return best;
}
function niceEpochs(max) {
  const step = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].find(s => max / s <= 8) || Math.ceil(max / 8);
  const out = [];
  for (let t = 0; t <= max + 1e-9; t += step) out.push(t);
  return out;
}

// ---------------------------------------------------------------- epochs, settings, result

function Epochs({ r }) {
  const [all, setAll] = useState(false);
  const rows = r.history || [];
  if (!rows.length) return null;
  const metricKeys = [...new Set(rows.flatMap(h => Object.keys(h.validationMetrics || h.metrics || {})))];
  const shown = all ? rows : rows.slice(-8);
  return html`<section class="card">
    <div class="card-title">Epochs<span class="grow" />
      ${rows.length > 8 && html`<button type="button" class="link" onClick=${() => setAll(!all)}>${all ? "Show the last 8" : `Show all ${rows.length}`}</button>`}</div>
    <div class="table-wrap"><table class="table num-table">
      <thead><tr><th>Epoch</th><th>Loss</th><th>Checked loss</th>${metricKeys.map(k => html`<th>${k.replace(/_/g, " ")}</th>`)}</tr></thead>
      <tbody>${[...shown].reverse().map(h => html`<tr class=${cx(h.epoch === r.bestEpoch && "best")}>
        <td>${h.epoch}${h.epoch === r.bestEpoch ? html` <span class="tag">best</span>` : ""}</td>
        <td>${fmtNum(h.loss)}</td><td>${fmtNum(h.validationLoss)}</td>
        ${metricKeys.map(k => { const v = (h.validationMetrics || {})[k] ?? (h.metrics || {})[k]; return html`<td>${k === "accuracy" && v != null ? (v * 100).toFixed(1) + " %" : fmtNum(v)}</td>`; })}
      </tr>`)}</tbody>
    </table></div>
  </section>`;
}

function Settings({ r }) {
  const s = r.spec || {};
  const opt = entry("optimizer", s.optimizer?.entry);
  const args = (e, c) => e && e.params.length ? " (" + e.params.map((p, i) => `${p.name} ${c.args?.[i]}`).join(", ") + ")" : "";
  const rows = [
    ["Model", s.model],
    ["Data", r.data],
    ["Optimizer", setup.value ? entryLabel(opt) + args(opt, s.optimizer) : s.optimizer?.entry],
    ["Loss", setup.value ? entryLabel(entry("loss", s.loss?.entry)) : s.loss?.entry],
    s.schedule && ["Schedule", setup.value ? entryLabel(entry("schedule", s.schedule.entry)) + args(entry("schedule", s.schedule.entry), s.schedule) : s.schedule.entry],
    ["Epochs", `${s.epochs}, batches of ${s.batchSize}`],
    ["Held back", `${Math.round((s.validation || 0) * 100)} % of the data`],
    +s.patience > 0 && ["Patience", `${s.patience} epochs`],
    ["Seed", s.seed],
    ["Ran on", [r.host, r.device].filter(Boolean).join(", ")],
  ].filter(Boolean);
  return html`<section class="card"><div class="card-title">Settings</div>
    <dl class="kv">${rows.map(([k, v]) => html`<dt>${k}</dt><dd>${v ?? "–"}</dd>`)}</dl></section>`;
}

/** The model the run saved, and a quick try on one sample of its data. */
function Result({ r, tab }) {
  const [tried, setTried] = useState(null);
  const [busy, setBusy] = useState(false);
  useEffect(() => setTried(null), [r.id]);
  if (!r.output) return html`<section class="card"><div class="card-title">Result</div>
    <p class="muted">${isActive(r) ? "The trained model is saved here when the run ends, or when you stop it." : "This run saved no model."}</p></section>`;
  const tryOne = async () => {
    setBusy(true);
    try {
      const samples = tried?.samples || 100;
      setTried(await api("/api/train/predict", { model: r.output, data: r.spec.data, index: Math.floor(Math.random() * samples) }, { project: tab.project }));
    } catch (e) { fail(e); }
    setBusy(false);
  };
  const classes = tried && tried.expectedBest != null;
  const right = classes && tried.best === tried.expectedBest;
  const name = (label, i) => label ?? `class ${i}`;
  const vals = v => (v || []).slice(0, 6).map(fmtNum).join(", ") + ((v || []).length > 6 ? ", …" : "");
  return html`<section class="card"><div class="card-title">Result</div>
    <p>Saved as the model <b class="mono">${r.output}</b>${r.bestEpoch ? `, with the weights of epoch ${r.bestEpoch}` : ""}.</p>
    <div class="row-btns"><${Button} icon="sparkle" disabled=${busy} onClick=${tryOne}>${tried ? "Try another sample" : "Try it on a sample"}<//>
      <${Button} icon="redo" onClick=${() => newTraining(tab, { spec: { ...r.spec, model: r.output } })} title="A new run that starts from these weights">Keep training…<//></div>
    ${tried && html`<div class=${cx("try", classes && (right ? "try-ok" : "try-bad"))}>
      ${classes
        ? html`<span>Sample ${tried.index}: it says <b>${name(tried.label, tried.best)}</b> (${Math.round((tried.output?.[tried.best] ?? 0) * 100)} % sure), the answer is <b>${name(tried.expectedLabel, tried.expectedBest)}</b>. ${right ? "Right." : "Wrong."}</span>`
        : html`<span>Sample ${tried.index}: it answers <b class="mono">${vals(tried.output)}</b>${tried.expected ? html`, the answer is <b class="mono">${vals(tried.expected)}</b>` : ""}.</span>`}
      ${tried.input?.length && tried.input.length <= 8 ? html`<span class="muted mono">input ${vals(tried.input)}</span>` : ""}
    </div>`}
  </section>`;
}
