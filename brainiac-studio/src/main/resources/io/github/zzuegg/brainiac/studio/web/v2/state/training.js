import { signal } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";

// Training: what a run can be made of (the optimizers, losses, schedules and metrics the
// installed modules offer, the data sets and the machines), and the runs being watched.
// A run belongs to the project it was started in, like a chat or a network.

/** {options, data, models, hosts} once loaded; read again each time a new run is set up. */
export const setup = signal(null);
let loading = null;
export function loadSetup(again = false) {
  if (loading && !again) return loading;
  loading = Promise.all([api("/api/train/options"), api("/api/train/data"), api("/api/train/models"), api("/api/hosts")])
    .then(([options, data, models, hosts]) => (setup.value = { options, data, models: models.models || [], hosts: hosts.hosts || [] }))
    .catch(e => { loading = null; throw e; });
  return loading;
}

const GROUPS = { optimizer: "optimizers", loss: "losses", schedule: "schedules", metric: "metrics" };
export const entries = kind => setup.value?.options?.[GROUPS[kind]] || [];
export const entry = (kind, key) => entries(kind).find(e => e.key === key);

/** "AdamW", "Optimizer.sgd", "cross entropy": the name a person reads for an entry. */
export function entryLabel(e) {
  if (!e) return "?";
  const plain = ["Optimizer", "Loss", "Metric", "LearningRateSchedule"].includes(e.group);
  const name = e.name === "new" ? e.group : e.name;
  const text = plain ? name : e.group + " " + name;
  return /^[A-Z_]+$/.test(name) ? text.toLowerCase().replace(/_/g, " ") : text;
}
const sig = e => (e?.params?.length ? e.params.map(p => p.name).join(", ") : "");
export const entryTitle = e => entryLabel(e) + (sig(e) ? ` (${sig(e)})` : "");

/** A sensible first value for a parameter, from its name. */
export function guess(p, epochs = 20) {
  const n = p.name.toLowerCase();
  if (p.array) return [];
  if (p.kind === "choice") return p.choices?.[0] ?? "";
  if (p.kind === "boolean") return false;
  if (/learningrate|^lr$|rate$/.test(n)) return 0.001;
  if (/weightdecay|decay/.test(n)) return 0.01;
  if (/momentum|beta1/.test(n)) return 0.9;
  if (/beta2/.test(n)) return 0.999;
  if (/beta3/.test(n)) return 0.9999;
  if (/epsilon|eps/.test(n)) return 1e-8;
  if (/totalepochs|epochs/.test(n)) return epochs;
  if (/minrate/.test(n)) return 0;
  if (/factor|gamma/.test(n)) return 0.5;
  if (/every|warmup/.test(n)) return 10;
  if (/delta|smooth|alpha/.test(n)) return 1;
  if (p.kind === "int" || p.kind === "long") return 1;
  if (p.kind === "float" || p.kind === "double") return 0.1;
  return "";
}
export const choiceFor = key => { const e = entry(kindOfKey(key), key); return e ? { entry: e.key, args: e.params.map(p => guess(p)) } : null; };
const kindOfKey = key => Object.keys(GROUPS).find(k => entries(k).some(e => e.key === key));
export function pickKey(kind, ...wanted) {
  const keys = entries(kind).map(e => e.key);
  for (const w of wanted) { const k = keys.find(k => k.endsWith(w)); if (k) return k; }
  return keys[0] || null;
}
const isRate = p => /learningrate|^lr$|rate$/i.test(p.name);
/** The position of the learning rate among an optimizer's parameters, or -1. */
export const rateAt = choice => (entry("optimizer", choice?.entry)?.params || []).findIndex(isRate);

/** The data set a run's data points at, with its shape and whether it has classes. */
export function dataInfo(d) {
  const s = setup.value?.data;
  if (!s || !d) return null;
  if (d.kind === "builtin") return s.builtins.find(b => b.name === d.name) || null;
  return (s.files || []).find(f => f.name === (d.file || d.images || d.folder)) || null;
}
export const hasClasses = d => { const i = dataInfo(d); return i ? i.classes === true || Array.isArray(i.classes) || i.kind === "idx" || i.kind === "csv" : true; };

/** The settings of a new run on a model, with every default picked. */
export function freshSpec(model, data = { kind: "builtin", name: "spirals" }) {
  const classes = hasClasses(data);
  return {
    model, data,
    optimizer: choiceFor(pickKey("optimizer", "Optimizer#adamW(float,float)", "Optimizer#adam(float)")),
    loss: choiceFor(pickKey("loss", classes ? "#CROSS_ENTROPY" : "#MSE")),
    schedule: null,
    metrics: [pickKey("metric", classes ? "#ACCURACY" : "#MEAN_ABSOLUTE_ERROR")].filter(Boolean),
    epochs: 20, batchSize: 32, validation: 0.1, patience: 0, seed: 42, clipNorm: 0, device: "best", host: "local",
  };
}

/** Swaps the loss and metrics for ones that suit the data, when the data changes kind. */
export function fitToData(spec, data) {
  const was = hasClasses(spec.data), now = hasClasses(data);
  if (was === now) return { ...spec, data };
  return { ...spec, data, loss: choiceFor(pickKey("loss", now ? "#CROSS_ENTROPY" : "#MSE")),
    metrics: [pickKey("metric", now ? "#ACCURACY" : "#MEAN_ABSOLUTE_ERROR")].filter(Boolean) };
}

/** What the studio is sent: numbers as numbers, nothing it does not need. */
export function wireSpec(s) {
  const out = { model: s.model, data: s.data, optimizer: s.optimizer, loss: s.loss, metrics: s.metrics,
    epochs: +s.epochs, batchSize: +s.batchSize, validation: +s.validation, patience: +s.patience, seed: +s.seed, device: s.device || "best" };
  if (s.schedule) out.schedule = s.schedule;
  if (+s.clipNorm > 0) out.clipNorm = +s.clipNorm;
  if (s.name) out.name = s.name;
  if (s.host && s.host !== "local") out.host = s.host;
  return out;
}

/** The rates to try besides the first, from "3e-4, 3e-3". */
export const sweepRates = text => String(text || "").split(/[\s,;]+/).filter(Boolean).map(Number).filter(v => isFinite(v) && v > 0);

export const isActive = r => ["queued", "loading", "running"].includes(r?.state);
export const STATE_NAMES = { queued: "queued", loading: "loading", running: "training", finished: "finished", stopped: "stopped", failed: "failed", interrupted: "interrupted" };

/** Fixed colour per run, so a run keeps its colour whatever else is shown with it. */
export const runColor = id => `var(--series-${(((parseInt(id, 10) || 1) - 1) % 8) + 1})`;

export function fmtNum(v) {
  if (v == null || !isFinite(v)) return "–";
  const a = Math.abs(v);
  return a >= 1000 ? Math.round(v).toLocaleString() : a >= 1 ? (+v).toFixed(3) : a === 0 ? "0" : (+v).toPrecision(3);
}
export function eta(r) {
  if (r.state !== "running" || !r.samplesPerSecond || !r.steps) return null;
  const left = (r.steps - r.step) * r.batchSize / r.samplesPerSecond;
  return left < 60 ? Math.ceil(left) + " s" : left < 3600 ? Math.ceil(left / 60) + " min" : (left / 3600).toFixed(1) + " h";
}

// ------------------------------------------------------------------ watched runs
// A run's full record (with its loss curve) is read while a screen shows it: every second while
// it trains, then once more when it ends.

const runs = new Map();
export function runStore(id) {
  let s = runs.get(id);
  if (!s) runs.set(id, (s = { id, run: signal(null), missing: signal(false), timer: 0, watchers: 0 }));
  return s;
}
async function poll(s) {
  clearTimeout(s.timer);
  try {
    const r = await api("/api/train/run", { id: s.id });
    s.run.value = r;
    s.missing.value = false;
    if (s.watchers > 0 && isActive(r)) s.timer = setTimeout(() => poll(s), 1000);
  } catch (e) {
    if (e.status === 404) s.missing.value = true;
    else if (s.watchers > 0) s.timer = setTimeout(() => poll(s), 5000);
  }
}
/** Follows a run while something shows it; returns the function that stops following. */
export function watchRun(id) {
  const s = runStore(id);
  s.watchers++;
  poll(s);
  return () => { s.watchers--; if (!s.watchers) clearTimeout(s.timer); };
}
/** Reads a run once (for comparing), without following it. */
export const readRun = id => { const s = runStore(id); if (!s.run.value) poll(s); return s; };

export async function stopRun(id) {
  await api("/api/train/stop", { id });
  const s = runStore(id);
  if (s.watchers) poll(s);
}

/**
 * Starts one run, or one per learning rate of a sweep, in a project. A network is built from its
 * design first, so the run starts from fresh weights of what the canvas shows.
 * @returns the ids of the runs started
 */
export async function startRuns(project, spec, sweep) {
  const base = wireSpec(spec);
  const at = rateAt(spec.optimizer);
  const rates = at >= 0 ? sweepRates(sweep) : [];
  const specs = [base, ...rates.map(rate => {
    const args = [...base.optimizer.args];
    args[at] = rate;
    return { ...base, optimizer: { ...base.optimizer, args }, name: `${base.name || base.model} · lr ${rate}` };
  })];
  if (rates.length && !base.name) specs[0].name = `${base.model} · lr ${base.optimizer.args[at]}`;
  const ids = [];
  for (const s of specs) ids.push((await api("/api/train/start", { spec: s }, { project })).id);
  return ids;
}
