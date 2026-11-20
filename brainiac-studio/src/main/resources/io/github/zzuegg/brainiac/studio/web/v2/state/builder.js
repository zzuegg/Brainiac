import { signal } from "/v2/lib/html.js";
import { api, local } from "/v2/lib/api.js";
import { fail } from "/v2/state/toasts.js";

// The builder's state. Each network open in a tab has one editor, kept while the page lives, so
// its undo history survives switching tabs. Every change is checked with the core's own shape
// rules; one that would break a block that worked is refused and the reason shown. Changes save
// themselves a moment later. A change made elsewhere (the docked assistant, the classic page) is
// read in and can be undone like one made on the canvas.

/** Every layer the installed modules offer: {entries, skipped}. */
export const catalog = signal(null);
let catalogLoad = null;
export function loadCatalog() {
  catalogLoad ||= api("/api/builder/layers").then(c => (catalog.value = c), e => { catalogLoad = null; throw e; });
  return catalogLoad;
}

const OP_LABEL = { input: "Input", add: "Add", multiply: "Multiply", concat: "Concat" };
export const WIRING = [["input", "Input", "a sample's shape"], ["add", "Add", "same shapes"], ["multiply", "Multiply", "same shapes"],
  ["concat", "Concat", "along the first dim"]];

export const entryOf = n => n.op === "layer" ? catalog.value?.entries.find(e => e.key === n.entry) || null : null;
export const entryLabel = e => e.group + (["new", "of"].includes(e.name) ? "" : "." + e.name);
export const nodeLabel = n => n.op === "layer" ? (entryOf(n) ? entryLabel(entryOf(n)) : String(n.entry).split("#").pop()) : OP_LABEL[n.op] || n.op;
export const multiIn = n => ["add", "multiply", "concat"].includes(n.op) || (n.op === "layer" && !!entryOf(n)?.multiInput);
/** A block still being wired: something to finish, not an error to refuse. */
export const incomplete = err => /not connected|needs at least two inputs/.test(err || "");
export const shapeKind = s => !s ? "vector" : s.length >= 3 ? "image" : s.length === 2 ? "seq" : "vector";
export const fmtShape = s => s ? s.join(" × ") : "?";
export const fmtCount = n => n >= 1e9 ? (n / 1e9).toFixed(2) + " B" : n >= 1e6 ? (n / 1e6).toFixed(2) + " M" : n >= 1e3 ? (n / 1e3).toFixed(1) + " k" : String(n);
export function fmtVal(v) {
  if (v == null || v === "") return "–";
  if (Array.isArray(v)) return v.join(" × ");
  if (typeof v === "boolean") return v ? "yes" : "no";
  return String(v);
}

/** A first value for a parameter, so a new block works at once. */
export function guess(p) {
  const n = p.name.toLowerCase();
  if (p.kind === "random") return null;
  if (p.kind === "choice") return p.choices.includes("RELU") ? "RELU" : p.choices.includes("HE") ? "HE" : p.choices[0];
  if (p.kind === "boolean") return false;
  if (p.kind === "string") return "";
  let v = 1;
  if (p.kind === "float" || p.kind === "double") v = /rate|drop|prob|eps|alpha|slope|momentum/.test(n) ? 0.1 : 1;
  else if (/kernel|window/.test(n)) v = 3;
  else if (/head/.test(n)) v = 4;
  else if (/units|filters|features|dim|channels|hidden|width|out|size|vocab|capacity|experts|slots/.test(n)) v = 64;
  else if (/group|stride|dilation|multiplier/.test(n)) v = 1;
  else if (/factor|scale/.test(n)) v = 2;
  return p.array ? (/kernel|window/.test(n) ? [v, v] : [v]) : v;
}

/** What a field's text means for a parameter. */
export function parseVal(p, raw) {
  if (p.kind === "boolean") return !!raw;
  if (p.kind === "choice" && !p.array) return raw;
  const one = s => p.kind === "string" || p.kind === "choice" ? s : s === "" ? null : Number(s);
  if (p.array) return String(raw).split(/[\s,×x]+/).filter(Boolean).map(one);
  return one(String(raw).trim());
}

// The graph as the checker reads it: options as a list, only those switched on.
const optionList = o => Object.entries(o || {}).map(([key, args]) => ({ key, args }));
export const wireGraph = g => ({
  output: g.output || undefined,
  nodes: g.nodes.map(n => ({ id: n.id, op: n.op, shape: n.shape, entry: n.entry, inputs: n.inputs || [], args: n.args, options: optionList(n.options) })),
});
// as kept on disk: positions too, options by key (the classic page reads both)
const stored = g => ({ ...(g.output ? { output: g.output } : {}), nodes: g.nodes });
const fromStored = g => ({
  output: g.output || undefined,
  nodes: (g.nodes || []).map((n, i) => ({
    x: 40 + i * 250, y: 120, ...n, inputs: n.inputs || [],
    options: Array.isArray(n.options) ? Object.fromEntries(n.options.map(o => [o.key, o.args])) : n.options || {},
  })),
});
const clone = v => JSON.parse(JSON.stringify(v));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const editors = new Map();
/** The editor of one network in one project. */
export function editorFor(project, name) {
  const key = project + "\n" + name;
  let ed = editors.get(key);
  if (!ed) {
    ed = {
      project, name,
      graph: signal(null),      // {output?, nodes: [{id, op, entry?, args?, options: {key: args}, inputs, shape?, x, y}]}
      check: signal(null),      // {ok, nodes: {id: {shape} | {error}}, input, output, outputShape, problem?}
      sel: signal(null),
      refusal: signal(null),    // {id, text}: the last change refused, and why
      notice: signal(null),     // {text, ids}: a change made elsewhere, just read in
      built: signal(null),
      building: signal(false),
      saving: signal(false),
      past: signal([]),
      future: signal([]),
      view: signal(local.get("builderView:" + key, null)),
      known: 0, timer: null, seq: 0,
    };
    editors.set(key, ed);
  }
  return ed;
}

export function setView(ed, v) {
  ed.view.value = v;
  local.set("builderView:" + ed.project + "\n" + ed.name, v);
}

async function runCheck(ed, g) {
  const seq = ++ed.seq;
  const r = await api("/api/builder/check", { graph: wireGraph(g) }, { project: ed.project });
  return seq === ed.seq ? r : null;
}

export async function recheck(ed) {
  try {
    const r = await runCheck(ed, ed.graph.value);
    if (r) ed.check.value = r;
  } catch (e) {
    console.warn("cannot check the network", e);
  }
}

/**
 * Reads the network from the studio. With `by`, a change made there is kept on the undo list
 * and pointed out: "by the assistant", "elsewhere".
 */
export async function reload(ed, { by } = {}) {
  if (ed.timer) await flush(ed);
  const r = await api("/api/builder/graphs/get", { name: ed.name }, { project: ed.project });
  const next = fromStored(r.graph);
  const was = ed.graph.value;
  ed.known = r.graph.updated || 0;
  if (was && same(stored(was), stored(next))) return false;
  if (was && by) {
    ed.past.value = [...ed.past.value.slice(-79), was];
    ed.future.value = [];
    const before = new Map(was.nodes.map(n => [n.id, JSON.stringify(n)]));
    const ids = next.nodes.filter(n => before.get(n.id) !== JSON.stringify(n)).map(n => n.id);
    ed.notice.value = { text: "Changed " + by, ids, at: Date.now() };
  }
  ed.graph.value = next;
  if (ed.sel.value && !next.nodes.some(n => n.id === ed.sel.value)) ed.sel.value = null;
  ed.built.value = null;
  await recheck(ed);
  return true;
}

/** Opens the network the first time it is shown. */
export function open(ed) {
  return ed.graph.value ? Promise.resolve(false) : reload(ed);
}

function commit(ed, next, before) {
  ed.past.value = [...ed.past.value.slice(-79), before];
  ed.future.value = [];
  ed.graph.value = next;
  ed.built.value = null;
  ed.notice.value = null;
  save(ed);
}

/**
 * Applies a change and keeps it only if no block that worked stops working, and the block it
 * aims at works (or only waits for a wire). Returns whether it was kept.
 */
export async function tryChange(ed, change, focus) {
  const before = ed.graph.value, was = ed.check.value;
  const next = clone(before);
  change(next);
  let r;
  try {
    r = await runCheck(ed, next);
  } catch (e) {
    ed.refusal.value = { id: focus, text: e.message };
    return false;
  }
  if (!r) return false;
  const why = verdict(r, was, before, next, focus);
  if (why) {
    ed.refusal.value = { id: focus, text: why };
    return false;
  }
  ed.check.value = r;
  ed.refusal.value = null;
  commit(ed, next, before);
  return true;
}

// why a checked change is refused, or null: a block that worked breaks, or the one aimed at does
function verdict(r, was, before, next, focus) {
  const broken = Object.entries(r.nodes).filter(([id, v]) => v.error && (!was || was.nodes?.[id]?.shape || id === focus))
    .filter(([id, v]) => !(id === focus && incomplete(v.error)));
  if (!broken.length) return null;
  const [id, v] = broken.find(([i]) => i === focus) || broken[0];
  const n = before.nodes.find(x => x.id === id) || next.nodes.find(x => x.id === id);
  return `${n ? nodeLabel(n) : id}: ${v.error}`;
}

const wired = (g, src, dst, slot) => {
  const t = g.nodes.find(n => n.id === dst);
  const ins = [...(t.inputs || [])];
  if (multiIn(t)) { if (slot < ins.length) ins[slot] = src; else ins.push(src); } else ins[0] = src;
  t.inputs = [...new Set(ins)];
};

/**
 * While a wire is pulled from a block: for every block it could go to, null if it fits there,
 * else why not. The same check a dropped wire gets, asked ahead.
 */
export async function wireTargets(ed, src) {
  const before = ed.graph.value, was = ed.check.value;
  const out = {};
  const targets = before.nodes.filter(n => n.id !== src && n.op !== "input").slice(0, 60);
  await Promise.all(targets.map(async t => {
    const next = clone(before);
    wired(next, src, t.id, 99);
    try {
      const r = await api("/api/builder/check", { graph: wireGraph(next) }, { project: ed.project });
      out[t.id] = verdict(r, was, before, next, t.id);
    } catch (e) {
      out[t.id] = e.message;
    }
  }));
  for (const n of before.nodes) if (n.op === "input" && n.id !== src) out[n.id] = "an Input block takes no wire";
  return out;
}

/** A change that cannot break a block worth refusing it for: moving, removing, unwiring. */
export async function change(ed, fn, { check = true } = {}) {
  const before = ed.graph.value;
  const next = clone(before);
  fn(next);
  if (same(before, next)) return;
  ed.refusal.value = null;
  commit(ed, next, before);
  if (check) await recheck(ed);
}

function newId(g, base) {
  const b = String(base).toLowerCase().replace(/[^a-z0-9]+/g, "").slice(0, 12) || "block";
  for (let i = 1; ; i++) if (!g.nodes.some(n => n.id === b + i)) return b + i;
}

/** A palette entry as a new block. */
export function specOf(key) {
  if (key.startsWith("op:")) {
    const op = key.slice(3);
    return { label: op, node: op === "input" ? { op, shape: [16] } : { op } };
  }
  const e = catalog.value.entries.find(x => x.key === key);
  return { label: e.group, node: { op: "layer", entry: e.key, args: e.params.map(guess), options: {} } };
}

/** Adds a block: at a point, or after the selected block, wired from it when that works. */
export async function addBlock(ed, spec, at) {
  const before = ed.graph.value;
  const next = clone(before);
  const from = next.nodes.find(n => n.id === ed.sel.value);
  const node = { id: newId(next, spec.label), inputs: [], options: {}, x: 0, y: 0, ...spec.node };
  if (at) Object.assign(node, at);
  else if (from) Object.assign(node, { x: from.x + 250, y: from.y });
  else Object.assign(node, { x: 40 + 250 * Math.min(6, next.nodes.length), y: 120 });
  while (next.nodes.some(n => Math.abs(n.x - node.x) < 60 && Math.abs(n.y - node.y) < 60)) node.y += 90;
  if (from && !at && node.op !== "input") node.inputs = [from.id];
  next.nodes.push(node);
  let r = null;
  try {
    r = await runCheck(ed, next);
    const err = r?.nodes?.[node.id]?.error;
    if (node.inputs.length && err && !incomplete(err)) {
      node.inputs = [];               // the wire does not fit; the block comes in loose
      r = await runCheck(ed, next);
    }
  } catch (e) {
    fail(e);
    return;
  }
  if (r) ed.check.value = r;
  ed.refusal.value = null;
  commit(ed, next, before);
  ed.sel.value = node.id;
}

export function connect(ed, src, dst, slot) {
  const target = ed.graph.value.nodes.find(n => n.id === dst);
  if (src === dst || !target || target.op === "input") return;
  return tryChange(ed, g => wired(g, src, dst, slot), dst);
}

/** Unplugs every wire going into a block. */
export const unplug = (ed, id) => change(ed, g => { g.nodes.find(x => x.id === id).inputs = []; });

/** Marks a block as the network's output, or unmarks it. */
export const toggleOutput = (ed, id) => tryChange(ed, g => { g.output = g.output === id ? undefined : id; }, id);

export const disconnect = (ed, id, slot) => change(ed, g => {
  const n = g.nodes.find(x => x.id === id);
  n.inputs = n.inputs.filter((_, i) => i !== slot);
});

export function removeBlock(ed, id) {
  if (ed.sel.value === id) ed.sel.value = null;
  return change(ed, g => {
    g.nodes = g.nodes.filter(n => n.id !== id);
    g.nodes.forEach(n => { n.inputs = (n.inputs || []).filter(s => s !== id); });
    if (g.output === id) delete g.output;
  });
}

export const moveBlock = (ed, id, x, y) => change(ed, g => {
  const n = g.nodes.find(v => v.id === id);
  n.x = x;
  n.y = y;
}, { check: false });

/** Lays the blocks out left to right by depth. */
export const arrange = ed => change(ed, g => {
  const depth = {}, byId = Object.fromEntries(g.nodes.map(n => [n.id, n]));
  const d = (id, seen = new Set()) => {
    if (depth[id] != null) return depth[id];
    if (seen.has(id)) return 0;
    seen.add(id);
    const n = byId[id];
    return (depth[id] = n && n.inputs?.length ? 1 + Math.max(...n.inputs.map(s => d(s, seen))) : 0);
  };
  const cols = {};
  g.nodes.forEach(n => (cols[d(n.id)] ||= []).push(n));
  Object.entries(cols).forEach(([k, list]) => list.forEach((n, i) => { n.x = 40 + k * 250; n.y = 80 + i * 210; }));
}, { check: false });

function step(ed, from, to) {
  if (!from.value.length) return false;
  const back = from.value.at(-1);
  from.value = from.value.slice(0, -1);
  to.value = [...to.value, ed.graph.value];
  ed.graph.value = back;
  if (ed.sel.value && !back.nodes.some(n => n.id === ed.sel.value)) ed.sel.value = null;
  ed.refusal.value = null;
  ed.notice.value = null;
  ed.built.value = null;
  save(ed);
  recheck(ed);
  return true;
}
export const undo = ed => step(ed, ed.past, ed.future);
export const redo = ed => step(ed, ed.future, ed.past);

function save(ed) {
  clearTimeout(ed.timer);
  ed.saving.value = true;
  ed.timer = setTimeout(() => flush(ed), 500);
}

/** Saves now what waits to be saved. */
export async function flush(ed) {
  clearTimeout(ed.timer);
  ed.timer = null;
  try {
    const r = await api("/api/builder/graphs", { name: ed.name, graph: stored(ed.graph.value) }, { project: ed.project });
    if (r.updated) ed.known = r.updated;
  } catch (e) {
    fail(e);
  }
  ed.saving.value = !!ed.timer;
}

/** Builds the model and saves it beside the network. */
export async function build(ed) {
  ed.building.value = true;
  try {
    if (ed.timer) await flush(ed);
    const g = ed.graph.value;
    ed.built.value = await api("/api/builder/build", { name: ed.name, graph: { ...wireGraph(g), nodes: g.nodes.map(n => ({ ...n, options: optionList(n.options) })) } },
      { project: ed.project });
    const r = await api("/api/builder/graphs/get", { name: ed.name }, { project: ed.project });
    ed.known = r.graph.updated || ed.known;
  } catch (e) {
    ed.built.value = null;
    ed.refusal.value = { id: null, text: e.message };
  }
  ed.building.value = false;
}

// ------------------------------------------------------------------ the docked assistant
// One agent run per network, remembered in this browser. It is followed while it works even
// when its panel is closed, so what it changes reaches the canvas, marked as its doing.

const LIVE = ["running", "waiting"];
const EDITS = ["edit_network", "create_network"];
const editCount = run => (run.steps || []).filter(s => s.kind === "result" && EDITS.includes(s.tool) && s.ok !== false).length;

export function assistantOf(ed) {
  if (!ed.agent) {
    const key = "builderAgent:" + ed.project + "\n" + ed.name;
    ed.agent = { key, id: signal(local.get(key, null)), run: signal(null), seen: -1, timer: null };
    if (ed.agent.id.value) follow(ed);
  }
  return ed.agent;
}

export const assistantLive = ed => !!ed.agent?.run.value && LIVE.includes(ed.agent.run.value.state);

/** Reads the run, and again shortly while it works. */
export async function follow(ed) {
  const a = ed.agent, id = a.id.value;
  clearTimeout(a.timer);
  if (!id) return;
  try {
    const r = await api("/api/agents/get", { id }, { project: ed.project });
    if (a.id.value !== id) return;
    const n = editCount(r);
    if (a.seen >= 0 && n > a.seen) reload(ed, { by: "by the assistant" }).catch(fail);
    a.seen = n;
    a.run.value = r;
    if (LIVE.includes(r.state)) a.timer = setTimeout(() => follow(ed), document.hidden ? 3000 : 700);
  } catch (e) {
    if (e.status === 404) forgetAssistant(ed);
    else fail(e);
  }
}

/** Starts the conversation over; the old run stays in the project's agents. */
export function forgetAssistant(ed) {
  const a = ed.agent;
  clearTimeout(a.timer);
  local.set(a.key, null);
  a.id.value = null;
  a.run.value = null;
  a.seen = -1;
}

/** Tells the assistant something: starts its run, or replies to it. */
export async function tellAssistant(ed, start, reply) {
  const a = assistantOf(ed);
  if (!a.id.value || !a.run.value) {
    const r = await api("/api/agents/start", start, { project: ed.project });
    local.set(a.key, r.id);
    a.seen = 0;
    a.id.value = r.id;
  } else {
    await api("/api/agents/reply", { id: a.id.value, text: reply }, { project: ed.project });
  }
  await follow(ed);
}

const TOOLS = ["list_networks", "read_network", "edit_network", "create_network", "list_layers", "check_shapes"];
const SAYS = "\n\nThe user says:\n\n";
const SELECTED = /\n\n\(The user has the block [^)]*selected\.\)$/;

/** What the person wrote, without the context the page added to it. */
export const saidToAssistant = t => (t.includes(SAYS) ? t.slice(t.indexOf(SAYS) + SAYS.length) : t).replace(SELECTED, "");

const goal = (name, text) => `Builder assistant for the network "${name}"\n\nThe user has it open on the canvas and sees each `
  + `change you make as you make it; they can undo any of them. Read it first with read_network("${name}"). Find layer keys with `
  + `list_layers. Change it with edit_network, a few steps at a time, and keep every block that works working. If a change is `
  + `refused, read why and try another way. When you are done, say in a few words what you changed and why.${SAYS}${text}`;

/** Asks the assistant for a change, naming the selected block. */
export function askAssistant(ed, text, model = "auto") {
  const sel = ed.sel.value ? `\n\n(The user has the block ${ed.sel.value} selected.)` : "";
  return tellAssistant(ed, { goal: goal(ed.name, text) + sel, role: "model-builder", model, mode: "auto", maxSteps: 40, tools: TOOLS }, text + sel);
}
