import { html, useEffect, useLayoutEffect, useRef, useState } from "/v2/lib/html.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button, IconAction } from "/v2/ui/Button.js";
import { Working } from "/v2/ui/Media.js";
import { modelInfo, placeOf } from "/v2/state/models.js";
import { MEDIA, modelsFor, rewriteTasks } from "/v2/state/media.js";
import { toast } from "/v2/state/toasts.js";

// Rewriting: a text goes in and another comes out, on a summarizing, translating or T5 model on
// this PC. Each turn keeps what went in, so an answer can be taken further (a summary translated,
// say) with "Rewrite this".

const LANGS = new Intl.DisplayNames(undefined, { type: "language" });
const lang = c => { try { return LANGS.of(c) || c; } catch { return c; } };

/** A task's name as people say it: "Summary", "English to German". */
export function taskName(t) {
  if (!t) return "";
  const tr = /^translation_([a-z]{2,3})_to_([a-z]{2,3})$/i.exec(t);
  if (tr) return `${lang(tr[1])} to ${lang(tr[2])}`;
  if (/^summari[sz]ation/i.test(t)) return "Summary";
  return t.replace(/[_-]+/g, " ").replace(/^./, c => c.toUpperCase());
}

/** One text and what was written from it. */
export function RewriteTurn({ t, busy, reuse }) {
  const { ask, made } = t;
  const o = ask.settings || {};
  const [open, setOpen] = useState(false);
  const long = (ask.text || "").length > 280;
  const facts = [o.task && taskName(o.task), o.maxTokens && "at most " + o.maxTokens + " tokens"].filter(Boolean);
  const copy = () => navigator.clipboard?.writeText(made.text).then(() => toast("Copied."));
  return html`<li class="msg media-turn rewrite-turn">
    <div class="media-ask">
      <p class=${"rewrite-source" + (long && !open ? " clipped" : "")}>${ask.text}</p>
      <div class="rewrite-ask-foot">
        ${facts.length > 0 && html`<span class="muted small">${facts.join(" · ")}</span>`}
        ${long && html`<button type="button" class="link small" aria-expanded=${open} onClick=${() => setOpen(!open)}>${open ? "Less" : "All of it"}</button>`}
      </div></div>
    ${made?.streaming && html`<${Working} doing=${made.model ? "writing with " + made.model.name : "loading the model"} />`}
    ${made?.text && html`<div class="heard"><p>${made.text}</p></div>`}
    ${made?.error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${made.error}</p>`}
    ${made?.stopped && html`<p class="muted small">Stopped.</p>`}
    ${made && !made.streaming && made.text && html`<div class="msg-acts media-acts">
      <span class="muted small">${[made.model?.name, made.task && !o.task && taskName(made.task), made.seconds >= 0.1 && made.seconds.toFixed(1) + " s"].filter(Boolean).join(" · ")}</span>
      <span class="grow" />
      <${IconAction} icon="copy" label="Copy the text" onClick=${copy} />
      ${!busy && html`<${IconAction} icon="redo" label="Rewrite this: put it in the box, to summarize or translate it further" onClick=${() => reuse(made.text)} />`}
    </div>`}
  </li>`;
}

/** The text to rewrite, the model, and its task. */
export function RewriteComposer({ settings: o, put, prompt, setPrompt, busy, go, stop }) {
  const ref = useRef(null);
  const [more, setMore] = useState(false);
  const [tasks, setTasks] = useState([]);
  const models = modelsFor("rewrite");
  const auto = modelInfo.value?.auto?.rewrite || models[0]?.key;
  const key = o.model === "auto" ? auto : o.model;
  useEffect(() => {
    let on = true;
    setTasks([]);
    if (key) rewriteTasks(key).then(r => on && setTasks(r));
    return () => { on = false; };
  }, [key]);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, innerHeight / 3) + "px";
  }, [prompt]);
  const task = tasks.find(t => t.name === o.task);
  return html`<div class="composer">
    <div class="composer-box">
      <textarea ref=${ref} class="composer-input media-prompt" rows="1" value=${prompt} aria-label="Text to rewrite"
        placeholder=${models.length ? "Paste or type the text to summarize or translate" : "No summarizing or translating model on this PC yet"}
        onInput=${e => setPrompt(e.target.value)}
        onKeyDown=${e => {
          if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && !e.isComposing) { e.preventDefault(); if (!busy && prompt.trim()) go(); }
          else if (e.key === "Escape" && busy) { e.preventDefault(); stop(); }
        }} />
      <div class="composer-bar media-bar">
        <select class="input select-sm" value=${o.model} disabled=${busy} aria-label="Model" onChange=${e => put({ model: e.target.value, task: "" })}>
          <option value="auto">Auto${auto ? " (" + (models.find(m => m.key === auto)?.name || auto) + ")" : ""}</option>
          ${models.map(m => html`<option value=${m.key}>${m.name} · ${placeOf(m)}</option>`)}
        </select>
        ${tasks.length > 0 && html`<select class="input select-sm" value=${o.task || ""} disabled=${busy} aria-label="Task" onChange=${e => put({ task: e.target.value })}>
          <option value="">${tasks.length === 1 ? taskName(tasks[0].name) : "As the model does"}</option>
          ${tasks.length > 1 && tasks.map(t => html`<option value=${t.name}>${taskName(t.name)}</option>`)}</select>`}
        <button type="button" class="link" aria-expanded=${more} onClick=${() => setMore(!more)}>${more ? "Less" : "More"}</button>
        <span class="grow" />
        <span class="muted small kbd-hint">Ctrl+Enter</span>
        ${busy ? html`<${Button} icon="stop" onClick=${stop}>Stop<//>`
          : html`<${Button} variant="primary" icon=${MEDIA.rewrite.icon} onClick=${go} disabled=${!prompt.trim() || !models.length}>${MEDIA.rewrite.verb}<//>`}
      </div>
      ${more && html`<div class="composer-bar media-more">
        <label>Longest answer <input class="input mono select-sm" value=${o.maxTokens} placeholder=${task?.maxTokens || "model's"} inputmode="numeric"
          onInput=${e => put({ maxTokens: e.target.value.replace(/[^0-9]/g, "") })} /> tokens</label>
        ${task?.prefix && html`<span class="muted small">Sent as “${task.prefix.trim()}” and the text.</span>`}
      </div>`}
    </div>
  </div>`;
}
