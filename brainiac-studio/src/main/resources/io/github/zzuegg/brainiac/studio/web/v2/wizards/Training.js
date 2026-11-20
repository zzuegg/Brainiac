import { html, useEffect, useState } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { openLayer } from "/v2/ui/overlay.js";
import { Wizard, Field, Summary } from "/v2/ui/Wizard.js";
import { navigate } from "/v2/state/workspace.js";
import { toast } from "/v2/state/toasts.js";
import { editorFor, reload, build } from "/v2/state/builder.js";
import { setup, loadSetup, entries, entry, entryLabel, entryTitle, choiceFor, freshSpec, fitToData, dataInfo, rateAt,
  sweepRates, wireSpec, startRuns } from "/v2/state/training.js";

// "New training run": what to train and on what, how it learns, and where. Every setting has a
// default picked, so a run on a network can start with three clicks; the rest sits under
// "More settings". Also opened from the builder's Train button, and from "Train again" on a run
// with that run's settings.

/**
 * Opens the wizard.
 * @param tab the project's tab
 * @param {{network?: string, spec?: object}} [from] a network to train, or a run's settings to start from
 */
export function newTraining(tab, from = {}) {
  return openLayer(NewTraining, { tab, from });
}

const fmtShape = s => (s || []).join(" × ");

function NewTraining({ tab, from, close, top }) {
  const [ready, setReady] = useState(!!setup.value);
  const [error, setError] = useState(null);
  useEffect(() => { loadSetup(true).then(() => setReady(true), e => setError(e.message)); }, []);
  if (error) return html`<${Wizard} title="New training run" close=${close} top=${top} initial=${{}}
    steps=${[{ title: "What to train", body: () => html`<p class="wizard-problem">${error}</p>` }]} summary=${() => null} finish=${async () => null} />`;
  if (!ready) return html`<div class="loading" aria-busy="true" />`;

  const networks = tab.items.value.filter(i => i.kind === "graph");
  const results = tab.items.value.filter(i => i.kind === "training" && i.output);
  const what = from.spec ? targetOf(from.spec.model, networks) : from.network ? "graph:" + from.network : networks[0] ? "graph:" + networks[0].id : "";
  const initial = { what, spec: from.spec ? { ...freshSpec(null), ...from.spec, name: "" } : freshSpec(null), sweep: "" };

  return html`<${Wizard} title="New training run" draftKey=${from.spec || from.network ? null : "training:" + tab.project} close=${close} top=${top}
    initial=${initial} finishLabel="Start training"
    steps=${[
      {
        title: "What to train",
        missing: d => !d.what ? (networks.length ? "Pick a network to train." : "This project has no network yet. Make one in Networks first.") : null,
        body: (d, set) => html`<${WhatStep} d=${d} set=${set} networks=${networks} results=${results} project=${tab.project} />`,
      },
      {
        title: "How it learns",
        missing: d => badNumbers(d.spec),
        body: (d, set) => html`<${HowStep} d=${d} set=${set} />`,
      },
      {
        title: "Where",
        body: (d, set) => html`<${WhereStep} d=${d} set=${set} />`,
      },
    ]}
    summary=${(d, goTo) => {
      const rates = rateAt(d.spec.optimizer) >= 0 ? sweepRates(d.sweep) : [];
      const target = d.what.startsWith("graph:") ? `the network “${d.what.slice(6)}”, built fresh from its design` : `the result “${d.what.slice(6)}”, carrying on from what it learned`;
      const host = d.spec.host && d.spec.host !== "local" ? setup.value.hosts.find(h => h.id === d.spec.host)?.name : "this PC";
      return html`<${Summary} goTo=${goTo} lead=${`Trains ${target}${rates.length ? `, ${rates.length + 1} times with different learning rates` : ""}. The trained model is saved as a new model; the network stays as it is.`}
        rows=${[
          { label: "Data", value: dataLabel(d.spec.data), step: 0 },
          { label: "Optimizer", value: optimizerLine(d.spec.optimizer, rates), step: 1 },
          { label: "Loss", value: entryLabel(entry("loss", d.spec.loss?.entry)), step: 1 },
          { label: "Length", value: `${d.spec.epochs} epochs, batches of ${d.spec.batchSize}${+d.spec.patience > 0 ? `, stops early after ${d.spec.patience} epochs without gain` : ""}`, step: 1 },
          { label: "Runs on", value: host, step: 2 },
        ]} />`;
    }}
    finish=${async d => {
      let model = d.what.slice(6);
      if (d.what.startsWith("graph:")) {
        const ed = editorFor(tab.project, model);
        if (!ed.graph.value) await reload(ed);
        await build(ed);
        if (!ed.built.value) throw new Error(`The network “${model}” cannot be built: ${ed.refusal.value?.text || "it does not check out yet"}. Open it to fix it.`);
      }
      const spec = { ...d.spec, model };
      const check = await api("/api/train/check", { spec: wireSpec(spec) }, { project: tab.project });
      if (!check.ok) throw new Error(check.problem || "The model and the data do not fit together.");
      const ids = await startRuns(tab.project, spec, d.sweep);
      await tab.refresh();
      navigate(tab, { screen: "training", id: ids[0] });
      toast(ids.length > 1 ? `${ids.length} runs started` : "Training started");
      return ids[0];
    }} />`;
}

// a run's model is either a network of the project (built under its own name) or a saved result
const targetOf = (model, networks) => networks.some(n => n.id === model) ? "graph:" + model : "model:" + model;

function dataLabel(d) {
  const i = dataInfo(d);
  if (!d) return "";
  if (d.kind === "builtin") return `${d.name}${i ? ` · ${fmtShape(i.input)} in, ${i.targets} out` : ""}`;
  return d.file || d.images || d.folder || "";
}
function optimizerLine(choice, rates) {
  const e = entry("optimizer", choice?.entry);
  const at = rateAt(choice);
  const lr = at >= 0 ? `, learning rate ${choice.args[at]}${rates.length ? " (also " + rates.join(", ") + ")" : ""}` : "";
  return entryLabel(e) + lr;
}
function badNumbers(s) {
  if (!(Number.isInteger(+s.epochs) && +s.epochs >= 1)) return "Epochs is a whole number, at least 1.";
  if (!(Number.isInteger(+s.batchSize) && +s.batchSize >= 1)) return "Batch is a whole number, at least 1.";
  if (!(+s.validation >= 0 && +s.validation <= 0.5)) return "The validation share is between 0 and 0.5.";
  if (!(Number.isInteger(+s.patience) && +s.patience >= 0)) return "Patience is a whole number; 0 turns it off.";
  return null;
}

// ---------------------------------------------------------------- step 1: what and on what

function WhatStep({ d, set, networks, results, project }) {
  const s = setup.value;
  const builtins = s.data.builtins || [], files = s.data.files || [];
  const dataKey = dk(d.spec.data);
  const pickData = key => {
    const [kind, name] = [key.slice(0, key.indexOf(":")), key.slice(key.indexOf(":") + 1)];
    const f = files.find(x => x.name === name);
    const data = kind === "builtin" ? { kind: "builtin", name } : { kind: f?.kind || "file", file: name };
    set({ spec: fitToData(d.spec, data) });
  };
  const info = dataInfo(d.spec.data);
  return html`
    <${Field} label="Network" hint=${d.what.startsWith("model:") ? "Carries on from the weights that run saved." : "It is built fresh from its design when the run starts."}>
      <select class="input" value=${d.what} onChange=${e => set({ what: e.target.value })}>
        ${!networks.length && html`<option value="">No network in this project yet</option>`}
        <optgroup label="Networks in this project">${networks.map(n => html`<option value=${"graph:" + n.id}>${n.title}</option>`)}</optgroup>
        ${results.length && html`<optgroup label="Keep training a result">${results.map(r => html`<option value=${"model:" + r.output}>${r.output} (from ${r.title})</option>`)}</optgroup>`}
        ${d.what.startsWith("model:") && !results.some(r => "model:" + r.output === d.what) && html`<option value=${d.what}>${d.what.slice(6)}</option>`}
      </select><//>
    <${Field} label="Data" hint=${info?.what || (files.length ? "" : `Put CSV or image folders in ${s.data.folder} to train on your own data.`)}>
      <select class="input" value=${dataKey} onChange=${e => pickData(e.target.value)}>
        <optgroup label="Built in">${builtins.map(b => html`<option value=${"builtin:" + b.name}>${b.name} · ${fmtShape(b.input)} in, ${b.targets} out</option>`)}</optgroup>
        ${files.length && html`<optgroup label=${"In " + s.data.folder}>${files.map(f => html`<option value=${"file:" + f.name}>${f.name}</option>`)}</optgroup>`}
      </select><//>
    <${Fit} d=${d} project=${project} />`;
}
const dk = d => !d ? "" : d.kind === "builtin" ? "builtin:" + d.name : "file:" + (d.file || d.images || d.folder);

/** Whether the network's input and output fit the data, said before the run starts. */
function Fit({ d, project }) {
  const [shape, setShape] = useState(null);
  useEffect(() => {
    let live = true;
    setShape(null);
    const name = d.what.slice(6);
    if (!name) return;
    const known = setup.value.models.find(m => m.name === name);
    if (d.what.startsWith("model:")) { setShape(known ? { input: known.input, output: known.output } : null); return; }
    const ed = editorFor(project, name);
    reload(ed)
      .then(() => ed.check.value)
      .then(c => live && c && setShape(c.ok ? { input: c.nodes?.[c.input]?.shape, output: c.outputShape } : { problem: "The network does not check out yet, so it cannot be trained. Open it to finish it." }))
      .catch(() => live && setShape(null));
    return () => { live = false; };
  }, [d.what]);
  const info = dataInfo(d.spec.data);
  if (!shape || !info) return null;
  if (shape.problem) return html`<p class="fit fit-bad" role="alert">${shape.problem}</p>`;
  const inOk = !shape.input || fmtShape(shape.input) === fmtShape(info.input);
  const outN = (shape.output || []).reduce((a, b) => a * b, 1);
  const outOk = !shape.output || outN === info.targets;
  return html`<p class=${"fit " + (inOk && outOk ? "fit-ok" : "fit-bad")}>
    ${inOk && outOk ? `Fits: the network takes ${fmtShape(info.input)} and answers ${info.targets} value${info.targets === 1 ? "" : "s"}, as this data needs.`
      : `${!inOk ? `The network takes ${fmtShape(shape.input)}, the data gives ${fmtShape(info.input)}. ` : ""}${!outOk ? `It answers ${outN} values, the data has ${info.targets}.` : ""} Pick other data, or change the network.`}
  </p>`;
}

// ---------------------------------------------------------------- step 2: how it learns

function HowStep({ d, set }) {
  const s = d.spec;
  const put = changes => set({ spec: { ...s, ...changes } });
  const at = rateAt(s.optimizer);
  const [more, setMore] = useState(false);
  return html`
    <${Field} label="Optimizer" hint="AdamW is a good first choice. Every optimizer the installed modules offer is listed.">
      <${EntrySelect} kind="optimizer" value=${s.optimizer} onChange=${v => put({ optimizer: v })} /><//>
    <div class="form-row">
      ${at >= 0 && html`<${Field} label="Learning rate"><input class="input mono" value=${s.optimizer.args[at]}
          onInput=${e => put({ optimizer: setArg(s.optimizer, at, num(e.target.value)) })} /><//>`}
      <${Field} label="Epochs" hint="Passes over the data"><input class="input mono" value=${s.epochs} onInput=${e => put({ epochs: e.target.value })} /><//>
      <${Field} label="Batch"><input class="input mono" value=${s.batchSize} onInput=${e => put({ batchSize: e.target.value })} /><//>
    </div>
    <button type="button" class="link more-toggle" aria-expanded=${more} onClick=${() => setMore(!more)}>${more ? "Fewer settings" : "More settings"}</button>
    ${more && html`<div class="more-settings">
      <${Args} choice=${s.optimizer} skip=${at} onChange=${v => put({ optimizer: v })} />
      <${Field} label="Loss"><${EntrySelect} kind="loss" value=${s.loss} onChange=${v => put({ loss: v })} /><//>
      <${Args} choice=${s.loss} onChange=${v => put({ loss: v })} />
      <${Field} label="Learning rate schedule"><${EntrySelect} kind="schedule" value=${s.schedule} none="Constant" onChange=${v => put({ schedule: v })} /><//>
      <${Args} choice=${s.schedule} onChange=${v => put({ schedule: v })} />
      <div class="form-row">
        <${Field} label="Validation share" hint="Held back to check on, 0 to 0.5"><input class="input mono" value=${s.validation} onInput=${e => put({ validation: e.target.value })} /><//>
        <${Field} label="Patience" hint="Stop after this many epochs without gain; 0 is off"><input class="input mono" value=${s.patience} onInput=${e => put({ patience: e.target.value })} /><//>
        <${Field} label="Seed"><input class="input mono" value=${s.seed} onInput=${e => put({ seed: e.target.value })} /><//>
        <${Field} label="Clip gradients" hint="0 is off"><input class="input mono" value=${s.clipNorm} onInput=${e => put({ clipNorm: e.target.value })} /><//>
      </div>
      ${entries("metric").length > 0 && html`<div class="field"><span class="field-label">Measure</span><div class="check-list">
        ${entries("metric").map(e => html`<label class="check"><input type="checkbox" checked=${s.metrics.includes(e.key)}
          onChange=${ev => put({ metrics: ev.target.checked ? [...s.metrics, e.key] : s.metrics.filter(k => k !== e.key) })} />${entryLabel(e)}</label>`)}
      </div></div>`}
    </div>`}`;
}
const num = v => (v.trim() === "" || isNaN(+v) ? v : +v);
const setArg = (choice, i, v) => { const args = [...choice.args]; args[i] = v; return { ...choice, args }; };

/** A select over every optimizer, loss or schedule, the plain ones first. */
function EntrySelect({ kind, value, onChange, none }) {
  const all = entries(kind);
  const plain = all.filter(e => ["Optimizer", "Loss", "LearningRateSchedule", "Metric"].includes(e.group));
  const rest = all.filter(e => !plain.includes(e));
  const groups = [...new Set(rest.map(e => e.module || "more"))];
  return html`<select class="input" value=${value?.entry || ""} onChange=${e => onChange(e.target.value ? choiceFor(e.target.value) : null)}>
    ${none && html`<option value="">${none}</option>`}
    <optgroup label="Common">${plain.map(e => html`<option value=${e.key}>${entryTitle(e)}</option>`)}</optgroup>
    ${groups.map(g => html`<optgroup label=${g.replace(/^io\.github\.zzuegg\.brainiac\./, "")}>
      ${rest.filter(e => (e.module || "more") === g).map(e => html`<option value=${e.key}>${entryTitle(e)}${e.experimental ? " · experimental" : ""}</option>`)}</optgroup>`)}
  </select>`;
}

/** The parameters of a chosen entry, each as a field of its kind. */
function Args({ choice, onChange, skip = -1 }) {
  const e = choice && entry(kindFor(choice.entry), choice.entry);
  if (!e || !e.params.length || (e.params.length === 1 && skip === 0)) return null;
  return html`<div class="form-row args">${e.params.map((p, i) => i === skip ? null : html`<${Field} label=${p.name}>
    ${p.kind === "choice" ? html`<select class="input" value=${choice.args[i]} onChange=${ev => onChange(setArg(choice, i, ev.target.value))}>
        ${p.choices.map(c => html`<option value=${c}>${c.toLowerCase().replace(/_/g, " ")}</option>`)}</select>`
      : p.kind === "boolean" ? html`<label class="check"><input type="checkbox" checked=${!!choice.args[i]} onChange=${ev => onChange(setArg(choice, i, ev.target.checked))} />yes</label>`
      : p.array ? html`<input class="input mono" value=${(choice.args[i] || []).join(", ")} placeholder="comma separated"
          onInput=${ev => onChange(setArg(choice, i, ev.target.value.split(/[\s,]+/).filter(Boolean).map(Number)))} />`
      : html`<input class="input mono" value=${choice.args[i]} onInput=${ev => onChange(setArg(choice, i, num(ev.target.value)))} />`}<//>`)}
  </div>`;
}
const kindFor = key => ["optimizer", "loss", "schedule", "metric"].find(k => entries(k).some(e => e.key === key));

// ---------------------------------------------------------------- step 3: where

function WhereStep({ d, set }) {
  const s = d.spec, hosts = setup.value.hosts;
  const put = changes => set({ spec: { ...s, ...changes } });
  const host = hosts.find(h => h.id === s.host);
  const devices = !host ? setup.value.options.devices
    : [{ key: "best", name: "Best available" }, ...(host.devices || []).map(x => ({ key: x.kind, name: x.name }))];
  const at = rateAt(s.optimizer);
  return html`
    <${Field} label="Train on" hint=${hosts.length ? "" : "Other machines running brainiac-studio --worker show up here once added on the Machines page."}>
      <select class="input" value=${s.host || "local"} onChange=${e => put({ host: e.target.value, device: "best" })}>
        <option value="local">This PC</option>
        ${hosts.map(h => html`<option value=${h.id} disabled=${!h.online}>${h.name}${h.online ? "" : " (offline)"}</option>`)}
      </select><//>
    <${Field} label="Device"><select class="input" value=${s.device} onChange=${e => put({ device: e.target.value })}>
      ${devices.map(x => html`<option value=${x.key}>${x.name}</option>`)}</select><//>
    ${at >= 0 && html`<${Field} label="Also try learning rates" hint="One more run for each, to compare. Leave empty for one run.">
      <input class="input mono" value=${d.sweep} placeholder="for example 3e-4, 3e-3" onInput=${e => set({ sweep: e.target.value })} /><//>`}`;
}
