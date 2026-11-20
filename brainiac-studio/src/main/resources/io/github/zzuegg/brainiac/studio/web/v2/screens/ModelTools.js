import { html, useEffect, useState, signal, cx } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button, IconAction } from "/v2/ui/Button.js";
import { Layer } from "/v2/ui/Layer.js";
import { openLayer, openMenu } from "/v2/ui/overlay.js";
import { modelInfo, loadModels } from "/v2/state/models.js";
import { fmtBytes } from "/v2/state/hub.js";
import { fail, toast } from "/v2/state/toasts.js";

// What can be done to a chat model on this PC: serve it to other programs, make a smaller copy,
// teach it from examples, merge several into one, and measure it. Each runs one at a time and
// is watched while it works; finished copies and merges become models in the list.

const S = { serve: signal(null), quantize: signal(null), finetune: signal(null), merge: signal(null), evaluate: signal(null) };
const BUSY = {
  serve: s => false,
  quantize: s => ["loading", "writing"].includes(s?.job?.state),
  finetune: s => ["loading", "training", "saving"].includes(s?.job?.state),
  merge: s => s?.job?.state === "merging",
  evaluate: s => !!s?.running,
};
const timers = {};
export function readTool(name) {
  const was = BUSY[name](S[name].value);
  return api("/api/" + name).then(s => {
    S[name].value = s;
    clearTimeout(timers[name]);
    if (BUSY[name](s)) timers[name] = setTimeout(() => readTool(name), 1300);
    else if (was && name !== "evaluate") loadModels(true); // a finished copy, merge or fine-tune is a model now
  }, () => {});
}
async function post(name, body) {
  const s = await api("/api/" + name, body);
  S[name].value = s;
  clearTimeout(timers[name]);
  if (BUSY[name](s)) timers[name] = setTimeout(() => readTool(name), 900);
  return s;
}

/** This PC's chat models; `for` narrows them to what a tool can take. */
const localChat = () => (modelInfo.value?.models || []).filter(m => m.provider === "local" && (m.media || []).includes("text"));
const FOR = {
  serve: m => true,
  quantize: m => !(m.media || []).includes("vision") && !/looks at pictures/.test(m.detail || ""),
  finetune: m => FOR.quantize(m),
  merge: m => FOR.quantize(m) && !/\.gguf$/i.test(m.id),
  evaluate: m => FOR.quantize(m),
};
const modelsFor = tool => localChat().filter(FOR[tool]);
const secs = s => (s >= 90 ? `${Math.floor(s / 60)} min ${Math.round(s % 60)} s` : `${s >= 10 ? Math.round(s) : (+s).toFixed(1)} s`);

const TOOLS = [
  { id: "serve", icon: "plug", name: "Serve", text: "Let editors and scripts use a model through the OpenAI API", layer: () => Serve },
  { id: "quantize", icon: "shrink", name: "Make smaller", text: "A quantized GGUF copy: less memory, often faster", layer: () => Quantize },
  { id: "finetune", icon: "sparkle", name: "Fine-tune", text: "Teach a model from your examples with LoRA", layer: () => FineTune },
  { id: "merge", icon: "merge", name: "Merge", text: "Combine models of one architecture into a new one", layer: () => Merge },
  { id: "evaluate", icon: "gauge", name: "Measure", text: "Score it on questions, or time its speed", layer: () => Evaluate },
];
export const openTool = (id, model) => openLayer(TOOLS.find(t => t.id === id).layer(), { model });

/** What a tool is doing now, in a few words, or nothing. */
function doing(id) {
  const s = S[id].value, j = s?.job;
  if (id === "serve") return s?.running ? ["on", `serving ${s.model}`] : null;
  if (id === "evaluate") { const r = s?.runs?.find(r => r.state === "loading" || r.state === "running"); return r ? ["busy", `measuring ${r.model}`] : null; }
  if (!j) return null;
  if (BUSY[id](s)) return ["busy", id === "finetune" ? `teaching ${j.model} · step ${j.step} of ${j.steps}` : id === "merge" ? `merging into ${j.name}` : `writing ${j.size} of ${j.model}`];
  if (j.state === "failed") return ["bad", "the last one failed"];
  return null;
}

/** The tools as one card on the Models page, each with what it is doing. */
export function ModelTools() {
  useEffect(() => { TOOLS.forEach(t => readTool(t.id)); }, []);
  const none = !localChat().length;
  return html`<section class="card">
    <h2 class="card-title">Work on this PC's models ${none && html`<span class="muted small">once a chat model is on this PC</span>`}</h2>
    <div class="tool-tiles">${TOOLS.map(t => {
      const d = doing(t.id);
      return html`<button type="button" key=${t.id} class=${cx("tool-tile", d && "tool-" + d[0])} disabled=${none && !d} onClick=${() => openTool(t.id)}>
        <${Icon} name=${t.icon} size=${18} /><b>${t.name}</b><span class="muted small">${d ? d[1] : t.text}</span></button>`;
    })}</div>
  </section>`;
}

/** The ⋯ menu of one of this PC's chat models. */
export function toolMenu(m, el) {
  openMenu(el, TOOLS.filter(t => FOR[t.id](m)).map(t => ({ label: t.name + "…", icon: t.icon, run: () => openTool(t.id, m.id) })), m.name);
}

function ModelSelect({ tool, value, onChange, label = "Model" }) {
  const list = modelsFor(tool);
  return html`<label class="field"><span class="field-label">${label}</span>
    <select class="input" value=${value} onChange=${e => onChange(e.target.value)}>${list.map(m => html`<option value=${m.id}>${m.name}</option>`)}</select></label>`;
}
const useModel = (tool, given) => {
  const list = modelsFor(tool);
  return useState(list.some(m => m.id === given) ? given : list[0]?.id || "");
};
const Empty = ({ what }) => html`<p class="muted">No chat model on this PC that can be ${what} yet. Download one, or add a folder that has one, on the Models page.</p>`;
const Status = ({ tone, icon, children }) => html`<p class=${cx("flag pedit-flag", tone && "flag-" + tone)}><${Icon} name=${icon} size=${14} /><span>${children}</span></p>`;
const copy = text => navigator.clipboard?.writeText(text).then(() => toast("Copied."), () => toast("Copy it by hand; the browser would not."));

function Serve({ model, close, top }) {
  const s = S.serve.value;
  const [pick, setPick] = useModel("serve", model);
  const [port, setPort] = useState(8080);
  const [network, setNetwork] = useState(false);
  const [busy, setBusy] = useState(false);
  useEffect(() => { readTool("serve"); }, []);
  const run = async body => { setBusy(true); try { await post("serve", body); } catch (x) { fail(x); } setBusy(false); };
  const running = s?.running;
  const curl = running && `curl ${s.url}/chat/completions -H "Content-Type: application/json"${s.key ? ` -H "Authorization: Bearer ${s.key}"` : ""} -d '{"messages":[{"role":"user","content":"Hello"}]}'`;
  const footer = html`<span class="grow" /><${Button} variant="ghost" onClick=${close}>Close<//>
    ${running ? html`<${Button} icon="stop" disabled=${busy} onClick=${() => run({ stop: true })}>${busy ? "Stopping…" : "Stop serving"}<//>`
      : modelsFor("serve").length > 0 && html`<${Button} variant="primary" icon="play" disabled=${busy || !pick} onClick=${() => run({ model: pick, port: Math.floor(+port) || 8080, network })}>${busy ? "Loading the model…" : "Start serving"}<//>`}`;
  return html`<${Layer} kind="drawer" title="Serve a model" subtitle="Editors, scripts and tools that speak the OpenAI API can use a chat model on this PC." close=${close} top=${top} footer=${footer}>
    ${running ? html`<dl class="serve-rows">
        <dt>Model</dt><dd><b>${s.model}</b> <span class="badge">${s.device}</span></dd>
        <dt>Base URL</dt><dd><code>${s.url}</code><${IconAction} icon="copy" label="Copy the URL" onClick=${() => copy(s.url)} /></dd>
        <dt>API key</dt><dd>${s.key ? html`<code>${s.key}</code><${IconAction} icon="copy" label="Copy the key" onClick=${() => copy(s.key)} />` : html`<span class="muted">none needed: only programs on this PC can reach it</span>`}</dd>
        <dt>Try it</dt><dd><code class="serve-curl">${curl}</code><${IconAction} icon="copy" label="Copy the command" onClick=${() => copy(curl)} /></dd></dl>
      ${s.network && html`<${Status} icon="lock">Open to other computers on your network. Every request needs the key, which changes each time you start serving.<//>`}`
    : !modelsFor("serve").length ? html`<${Empty} what="served" />` : html`
      <${ModelSelect} tool="serve" value=${pick} onChange=${setPick} />
      <label class="field"><span class="field-label">Port</span><input class="input mono narrow" type="number" min="1024" max="65535" value=${port} onInput=${e => setPort(e.target.value)} /></label>
      <label class="pedit-row"><input type="checkbox" checked=${network} onChange=${e => setNetwork(e.target.checked)} />
        <span><b>Let other computers on my network use it</b><span class="muted small">They need a key the studio makes up for this run.</span></span></label>
      <p class="field-hint">The model is loaded for the server alone, so serving the model you also chat with holds it in memory twice.</p>`}
  <//>`;
}

function Quantize({ model, close, top }) {
  const s = S.quantize.value, j = s?.job, sizes = s?.sizes || [];
  const [pick, setPick] = useModel("quantize", model);
  const [size, setSize] = useState("Q4_K_M");
  const [busy, setBusy] = useState(false);
  useEffect(() => { readTool("quantize"); }, []);
  const working = BUSY.quantize(s);
  const chosen = sizes.find(x => x.id === size) || sizes[0];
  const start = async () => { setBusy(true); try { await post("quantize", { model: pick, size: chosen.id }); } catch (x) { fail(x); } setBusy(false); };
  const footer = html`<span class="grow" /><${Button} variant="ghost" onClick=${close}>Close<//>
    ${modelsFor("quantize").length > 0 && html`<${Button} variant="primary" icon="shrink" disabled=${busy || working || !pick} onClick=${start}>${working ? "Working…" : "Make the copy"}<//>`}`;
  return html`<${Layer} kind="drawer" title="Make a smaller copy" subtitle="Each weight in fewer bits: less disk and memory, often faster, at a small cost in quality. The original stays as it is." close=${close} top=${top} footer=${footer}>
    ${!modelsFor("quantize").length ? html`<${Empty} what="copied" />` : html`
      <${ModelSelect} tool="quantize" value=${pick} onChange=${setPick} />
      <div class="field"><span class="field-label">Size</span>
        <div class="size-picks" role="radiogroup" aria-label="Size of the copy">${sizes.map(x => html`<label key=${x.id} class=${cx("size-pick", chosen?.id === x.id && "on")}>
          <input type="radio" name="qsize" checked=${chosen?.id === x.id} onChange=${() => setSize(x.id)} />
          <b class="mono">${x.id}</b><span class="mono muted small">${x.bits} bits a weight</span><span class="muted small">${x.note}</span></label>`)}</div>
        <span class="field-hint">The copy is a GGUF file next to the original; llama.cpp, Ollama and LM Studio run it too. While it works, the whole model is held in memory at full size.</span></div>`}
    ${j && (j.state === "done" ? html`<${Status} tone="ok" icon="check"><b>${j.name}</b> is ready${j.bytes ? ", " + fmtBytes(j.bytes) : ""}, and in the model list. <span class="mono muted small">${j.file}</span><//>`
      : j.state === "failed" ? html`<${Status} tone="bad" icon="warn">The ${j.size} copy of ${j.model} was not made: ${j.error || "it failed"}<//>`
      : html`<${Status} icon="sparkle"><b>${j.state === "loading" ? "Reading" : `Writing the ${j.size} copy of`} ${j.model}</b> for ${secs(j.seconds)}. It runs on the CPU and can take minutes; you can close this.<//>`)}
  <//>`;
}

/** The loss over the steps so far, on its own scale: one line, newest on the right. */
function LossCurve({ points }) {
  if (points.length < 2) return null;
  const w = 400, h = 72, top = Math.max(...points.map(p => p[1])), low = Math.min(0, ...points.map(p => p[1]));
  const a = points[0][0], span = Math.max(1, points.at(-1)[0] - a);
  const x = s => 2 + ((s - a) / span) * (w - 4), y = v => h - 3 - ((v - low) / Math.max(1e-9, top - low)) * (h - 6);
  const d = points.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join(" ");
  return html`<figure class="loss"><svg viewBox=${`0 0 ${w} ${h}`} preserveAspectRatio="none" role="img" aria-label=${`Loss from ${points[0][1].toFixed(3)} to ${points.at(-1)[1].toFixed(3)}`}>
      <path d=${`${d} L${w - 2},${h} L2,${h} Z`} class="loss-area" /><path d=${d} class="loss-line" vector-effect="non-scaling-stroke" /></svg>
    <figcaption class="mono small muted">loss ${(+points[0][1]).toFixed(3)} at the start, ${(+points.at(-1)[1]).toFixed(3)} now</figcaption></figure>`;
}

/** Reads a JSON-lines file chosen on this PC. */
function FileField({ label, file, setFile, what }) {
  const pickFile = async e => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (f.size > 24e6) { fail(new Error("That file is over 24 MB: use a smaller one.")); return; }
    const text = await f.text();
    setFile({ name: f.name, text, lines: text.split(/\r?\n/).filter(l => l.trim()).length });
  };
  return html`<div class="field"><span class="field-label">${label}</span>
    <label class="btn file-btn"><${Icon} name="folder" size=${14} /><span>${file ? `${file.name} · ${file.lines} ${what}` : "Choose a .jsonl file…"}</span>
      <input type="file" accept=".jsonl,.json,.txt,application/json" hidden onChange=${pickFile} /></label></div>`;
}

function FineTune({ model, close, top }) {
  const s = S.finetune.value, j = s?.job;
  const [pick, setPick] = useModel("finetune", model);
  const [name, setName] = useState("");
  const [file, setFile] = useState(null);
  const [n, setN] = useState({ rank: 8, epochs: 3, learningRate: 0.0002, length: 512, rows: 1 });
  const [busy, setBusy] = useState(false);
  useEffect(() => { readTool("finetune"); }, []);
  const working = BUSY.finetune(s);
  const start = async () => {
    if (!file) { fail(new Error("Choose a file of examples first.")); return; }
    setBusy(true);
    try { await post("finetune", { model: pick, name: name.trim(), data: file.text, rank: +n.rank, alpha: 2 * n.rank, epochs: +n.epochs, learningRate: +n.learningRate, length: +n.length, rows: +n.rows }); } catch (x) { fail(x); }
    setBusy(false);
  };
  const num = (k, label, hint, attrs) => html`<label class="num-field" title=${hint}><span class="small">${label}</span>
    <input class="input mono" type="number" ...${attrs} value=${n[k]} onInput=${e => setN({ ...n, [k]: e.target.value })} /></label>`;
  const footer = html`<span class="grow" /><${Button} variant="ghost" onClick=${close}>Close<//>
    ${working ? html`<${Button} icon="stop" disabled=${j.state !== "training"} onClick=${() => post("finetune", { stop: true }).catch(fail)}>Stop and keep it<//>`
      : modelsFor("finetune").length > 0 && html`<${Button} variant="primary" icon="sparkle" disabled=${busy || !pick || !file} onClick=${start}>${busy ? "Starting…" : "Start teaching"}<//>`}`;
  return html`<${Layer} kind="drawer" title="Fine-tune a model" subtitle="LoRA trains small adapters while the model's own weights stay as they are. You get the adapter and a copy with it merged in." close=${close} top=${top} footer=${footer}>
    ${!modelsFor("finetune").length ? html`<${Empty} what="taught" />` : html`
      <${ModelSelect} tool="finetune" value=${pick} onChange=${setPick} />
      <label class="field"><span class="field-label">Name of what it learns</span><input class="input" value=${name} maxlength="40" placeholder="such as support" onInput=${e => setName(e.target.value)} /></label>
      <${FileField} label="Examples" file=${file} setFile=${setFile} what="examples" />
      <p class="field-hint">JSON lines: <code>{"messages": [...]}</code> conversations, or <code>{"prompt": "...", "completion": "..."}</code>. Only the replies are learned.</p>
      <details class="more-settings"><summary class="small">Settings</summary><div class="num-row">
        ${num("rank", "Rank", "The adapters' size. 8 or 16 suits most tasks; more learns more and takes more memory.", { min: 1, max: 256 })}
        ${num("epochs", "Epochs", "How often every example is seen.", { min: 1, max: 100 })}
        ${num("learningRate", "Learning rate", "The step size. 0.0002 is the usual start.", { min: 0.000001, max: 1, step: "any" })}
        ${num("length", "Longest example", "In tokens. Longer examples are cut; memory grows with it.", { min: 16, max: 8192, step: 16 })}
        ${num("rows", "Rows a step", "More rows a step is steadier and takes more memory.", { min: 1, max: 64 })}</div></details>`}
    ${j && html`<div class="tool-status">
      ${working && html`<div class="prog" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${j.steps ? Math.round((100 * j.step) / j.steps) : 0}><i style=${`width:${j.steps ? (100 * j.step) / j.steps : 0}%`} /></div>
        <p class="mono small muted">${j.state === "loading" ? `reading the model and ${j.examples || "the"} examples` : j.state === "saving" ? "saving" : `step ${j.step} of ${j.steps}`} · ${secs(j.seconds)}</p>`}
      <${LossCurve} points=${j.losses || []} />
      ${j.state === "failed" ? html`<${Status} tone="bad" icon="warn">Teaching ${j.model} failed: ${j.error || "it failed"}<//>`
        : (j.state === "done" || j.state === "stopped") && html`<${Status} tone=${j.adapter ? "ok" : ""} icon=${j.adapter ? "check" : "stop"}>
          ${j.state === "stopped" ? `Stopped after ${j.step} of ${j.steps} steps. ` : ""}${j.mergedName ? html`<b>${j.mergedName}</b> is in the model list: ${j.model} with what it learned from ${j.examples} examples. ` : j.adapter ? "" : "Nothing was learned yet, so nothing was saved. "}${j.note || ""}
          ${j.adapter && html`<span class="mono muted small"> Adapter: ${j.adapter}</span>`}<//>`}</div>`}
  <//>`;
}

function Merge({ model, close, top }) {
  const s = S.merge.value, j = s?.job, ways = s?.ways || [];
  const list = modelsFor("merge");
  const [wayId, setWay] = useState("linear");
  const [base, setBase] = useState("");
  const [picked, setPicked] = useState(() => new Set(model ? [model] : []));
  const [weights, setWeights] = useState({});
  const [balance, setBalance] = useState(0.5);
  const [density, setDensity] = useState(0.5);
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  useEffect(() => { readTool("merge"); }, []);
  const way = ways.find(w => w.id === wayId) || ways[0];
  const chosen = list.filter(m => picked.has(m.id) && !(way?.base && m.id === base));
  const working = BUSY.merge(s);
  const start = async () => {
    if (way?.base && !base) { fail(new Error(`${way.label} needs the base model the others were tuned from.`)); return; }
    const given = chosen.map(m => String(weights[m.id] ?? "").trim());
    setBusy(true);
    try { await post("merge", { way: way.id, base, models: chosen.map(m => m.id), weights: given.some(Boolean) ? given.map(Number) : [], balance: +balance, density: +density, name: name.trim() }); } catch (x) { fail(x); }
    setBusy(false);
  };
  const toggle = (id, on) => { const p = new Set(picked); on ? p.add(id) : p.delete(id); setPicked(p); };
  const footer = html`<span class="grow" /><${Button} variant="ghost" onClick=${close}>Close<//>
    ${list.length > 1 && html`<${Button} variant="primary" icon="merge" disabled=${busy || working || chosen.length < 2} onClick=${start}>${working ? "Merging…" : chosen.length < 2 ? "Tick two or more" : `Merge ${chosen.length} models`}<//>`}`;
  return html`<${Layer} kind="drawer" title="Merge models" subtitle="Models that share an architecture, such as two fine-tunes of one base, combined weight by weight into a new model folder. The originals stay as they are." close=${close} top=${top} footer=${footer}>
    ${list.length < 2 ? html`<p class="muted">Merging needs two or more chat models on this PC that are model folders (a single GGUF file cannot be merged). Download fine-tunes of one base model, or teach one with Fine-tune.</p>` : html`
      <div class="field"><span class="field-label">How</span>
        <div class="seg" role="group" aria-label="How to merge">${ways.map(w => html`<button type="button" aria-pressed=${way?.id === w.id} onClick=${() => setWay(w.id)}>${w.label}</button>`)}</div>
        <span class="field-hint">${way && way.note[0].toUpperCase() + way.note.slice(1)}.</span></div>
      ${way?.base && html`<label class="field"><span class="field-label">Base model</span>
        <select class="input" value=${base} onChange=${e => setBase(e.target.value)}><option value="">Choose the model the others were tuned from…</option>
          ${list.map(m => html`<option value=${m.id}>${m.name}</option>`)}</select></label>`}
      <div class="field"><span class="field-label">Models to merge</span>
        <ul class="picks">${list.map(m => {
          const isBase = way?.base && m.id === base, on = picked.has(m.id) && !isBase;
          return html`<li key=${m.id}><label class=${cx(isBase && "dim")}>
            <input type="checkbox" checked=${on} disabled=${isBase} onChange=${e => toggle(m.id, e.target.checked)} />
            <span class="who"><b>${m.name}</b>${isBase && html`<span class="muted small">the base</span>`}</span>
            ${on && way?.id !== "slerp" ? html`<input class="input mono weight" type="number" step="any" aria-label=${"Weight of " + m.name} placeholder=${way?.id === "linear" ? "equal" : "1"}
              value=${weights[m.id] ?? ""} onClick=${e => e.preventDefault()} onInput=${e => setWeights({ ...weights, [m.id]: e.target.value })} />` : html`<span />`}</label></li>`;
        })}</ul>
        ${way?.id !== "slerp" && html`<span class="field-hint">Weights are optional; empty weighs them ${way?.id === "linear" ? "equally" : "as 1"}.</span>`}</div>
      <div class="num-row">
        ${way?.id === "slerp" && html`<label class="num-field" title="0 is the first model you ticked, 1 the second."><span class="small">Balance</span><input class="input mono" type="number" min="0" max="1" step="0.05" value=${balance} onInput=${e => setBalance(e.target.value)} /></label>`}
        ${(way?.id === "ties" || way?.id === "dare") && html`<label class="num-field" title="The share of each model's changes from the base that is kept."><span class="small">Share kept</span><input class="input mono" type="number" min="0.05" max="1" step="0.05" value=${density} onInput=${e => setDensity(e.target.value)} /></label>`}
      </div>
      <label class="field"><span class="field-label">Name of the new model</span><input class="input" value=${name} maxlength="60" placeholder="Empty makes one up" onInput=${e => setName(e.target.value)} /></label>`}
    ${j && (j.state === "done" ? html`<${Status} tone="ok" icon="check"><b>${j.name}</b> is in the model list: ${j.models.join(" and ")}, merged. <span class="mono muted small">${j.folder}</span><//>`
      : j.state === "failed" ? html`<${Status} tone="bad" icon="warn">${j.name} was not made: ${j.error || "it failed"}<//>`
      : html`<${Status} icon="merge"><b>Merging ${j.models.join(" and ")} into ${j.name}</b> for ${secs(j.seconds)}. It runs on the CPU, a tensor at a time; you can close this.<//>`)}
  <//>`;
}

function Evaluate({ model, close, top }) {
  const s = S.evaluate.value, runs = s?.runs || [];
  const [pick, setPick] = useModel("evaluate", model);
  const [file, setFile] = useState(null);
  const [limit, setLimit] = useState(100);
  const [shots, setShots] = useState(0);
  const [busy, setBusy] = useState(false);
  useEffect(() => { readTool("evaluate"); }, []);
  const working = BUSY.evaluate(s);
  const run = async body => { setBusy(true); try { await post("evaluate", body); } catch (x) { fail(x); } setBusy(false); };
  const live = r => r.state === "loading" || r.state === "running";
  const rate = v => (v >= 100 ? Math.round(v) : (+v).toFixed(1));
  const result = r => r.state === "failed" ? html`<span class="bad">${r.error || "it failed"}</span>`
    : r.kind === "speed" ? (live(r) ? "timing…" : `reads ${rate(r.readPerSecond)} · writes ${rate(r.writePerSecond)} tokens/s`)
    : !r.done ? (live(r) ? (r.state === "loading" ? "reading the model…" : `0 of ${r.total}`) : "nothing asked")
    : html`<b>${(100 * r.accuracy).toFixed(1)}%</b> ± ${(100 * r.stderr).toFixed(1)} · ${r.correct} of ${r.done} right${live(r) || r.state === "stopped" ? ` · ${r.done} of ${r.total} asked` : ""}${r.shots ? ` · ${r.shots} shown solved` : ""}`;
  const footer = html`<span class="grow" /><${Button} variant="ghost" onClick=${close}>Close<//>
    ${modelsFor("evaluate").length > 0 && html`<${Button} icon="play" disabled=${busy || working || !pick} onClick=${() => run({ model: pick, speed: true })}>Time its speed<//>
      <${Button} variant="primary" icon="gauge" disabled=${busy || working || !pick || !file}
        onClick=${() => run({ model: pick, task: file.name.replace(/\.(jsonl?|txt)$/i, ""), data: file.text, limit: Math.max(0, Math.round(+limit) || 0), shots: Math.max(0, Math.round(+shots) || 0) })}>${working ? "Measuring…" : "Score it"}<//>`}`;
  return html`<${Layer} kind="drawer" title="Measure a model" subtitle="Score a chat model on a file of questions, or time how fast it reads and writes. Results stay here, so a model and its smaller copies can be compared." close=${close} top=${top} footer=${footer}>
    ${!modelsFor("evaluate").length ? html`<${Empty} what="measured" />` : html`
      <${ModelSelect} tool="evaluate" value=${pick} onChange=${setPick} />
      <${FileField} label="Questions" file=${file} setFile=${setFile} what="questions" />
      <p class="field-hint">JSON lines as the common benchmarks have them: MMLU, ARC, HellaSwag and GSM8K files work as they are, and so does <code>{"question": "...", "choices": [...], "answer": 0}</code> or <code>{"question": "...", "answer": "..."}</code>.</p>
      <div class="num-row">
        <label class="num-field" title="How many of the file's questions are asked. 0 asks all."><span class="small">Ask at most</span><input class="input mono" type="number" min="0" value=${limit} onInput=${e => setLimit(e.target.value)} /></label>
        <label class="num-field" title="This many of the file's first questions are shown solved before each question, and not asked."><span class="small">Shown solved</span><input class="input mono" type="number" min="0" max="8" value=${shots} onInput=${e => setShots(e.target.value)} /></label>
      </div>`}
    ${runs.length > 0 && html`<div class="table-box"><table class="usage-table eval-table">
      <thead><tr><th>Model</th><th>Questions</th><th>Result</th><th class="r">Time</th><th /></tr></thead>
      <tbody>${runs.map(r => html`<tr key=${r.id} class=${cx(live(r) && "running")}>
        <td class="mono">${r.model}</td><td>${r.kind === "speed" ? "Speed" : r.task}</td><td>${result(r)}${r.state === "stopped" && html` <span class="muted">stopped</span>`}</td>
        <td class="r">${secs(r.seconds)}</td>
        <td class="r">${live(r) ? r.kind !== "speed" && html`<${IconAction} icon="stop" label="Stop" onClick=${() => post("evaluate", { stop: true }).catch(fail)} />`
          : html`<${IconAction} icon="trash" label="Forget this result" onClick=${() => post("evaluate", { remove: r.id }).catch(fail)} />`}</td></tr>`)}</tbody></table></div>`}
  <//>`;
}
