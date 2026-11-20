import { html, useEffect, useLayoutEffect, useRef, useState, cx } from "/v2/lib/html.js";
import { Reading, Empty } from "/v2/ui/Reading.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button } from "/v2/ui/Button.js";
import { ItemHeader } from "/v2/screens/ItemHeader.js";
import { MediaGrid, Working } from "/v2/ui/Media.js";
import { plural } from "/v2/lib/format.js";
import { configure } from "/v2/state/chats.js";
import { modelInfo, placeOf } from "/v2/state/models.js";
import { MEDIA, modelsFor, defaults, generate, stopMaking } from "/v2/state/media.js";

// An image, speech or video chat: each turn is a prompt with the settings it was made with, and
// what came out. Any picture can be taken back into the composer (Fine tune: its prompt and seed)
// to be drawn again with a change; "Again" makes new ones from the same turn.

const ASPECTS = ["1:1", "4:3", "3:4", "16:9", "9:16"];

export function MediaChat({ tab, item, s }) {
  const chat = s.chat.value, busy = s.busy.value, M = MEDIA[chat.medium];
  const settings = { ...defaults(chat.medium), ...(chat.settings || {}) };
  const scroll = useRef(null);
  const [prompt, setPrompt] = useState(chat.prompt || "");
  useEffect(() => setPrompt(chat.prompt || ""), [chat.id]);
  useLayoutEffect(() => { const el = scroll.current; if (el) el.scrollTop = el.scrollHeight; }, [chat.messages.length, busy]);
  const put = changes => configure(s, { settings: { ...settings, ...changes } });
  // a picture back in the composer: its prompt and seed, the rest as it was made
  const tune = (it, turn) => {
    setPrompt(turn?.text ?? it.prompt ?? prompt);
    configure(s, { settings: { ...settings, ...(turn?.settings || {}), seed: it.seed ?? "" } });
    document.querySelector(".media-prompt")?.focus();
  };
  const turns = pairs(chat.messages);
  const header = html`<${ItemHeader} tab=${tab} item=${item}
    sub=${`${M.label} · ${plural(turns.length, "prompt")}`} />`;
  const footer = html`<${MediaComposer} M=${M} medium=${chat.medium} settings=${settings} put=${put} prompt=${prompt} setPrompt=${setPrompt}
    busy=${busy} go=${() => generate(s, prompt)} stop=${() => stopMaking(s)} />`;
  return html`<${Reading} header=${header} footer=${footer} scrollRef=${scroll} class="chat media-chat">
    ${!turns.length ? html`<${Empty} title=${M.label} icon=${html`<${Icon} name=${M.icon} size=${28} />`}>${M.lead}<//>`
      : html`<ol class="transcript">${turns.map((t, i) => html`<${Turn} key=${i} t=${t} M=${M} busy=${busy}
          tune=${it => tune(it, t.ask)} again=${() => { setPrompt(t.ask.text); configure(s, { settings: { ...settings, ...t.ask.settings, seed: "" } }); setTimeout(() => generate(s, t.ask.text)); }} />`)}</ol>`}
  <//>`;
}

function pairs(messages) {
  const out = [];
  for (const m of messages) {
    if (m.role === "user") out.push({ ask: m, made: null });
    else if (out.length) out.at(-1).made = m;
  }
  return out;
}

function Turn({ t, M, busy, tune, again }) {
  const { ask, made } = t;
  const o = ask.settings || {};
  const facts = [o.aspect && M !== MEDIA.speech && o.aspect, o.count > 1 && o.count + " " + M.noun + "s", o.steps && o.steps + " steps",
    o.guidance && "guidance " + o.guidance, o.seed !== "" && o.seed != null && "seed " + o.seed, o.voice && "voice " + o.voice].filter(Boolean);
  const items = (made?.media || []).map(m => ({ ...m, prompt: ask.text, model: made.model?.key, aspect: o.aspect }));
  return html`<li class="msg media-turn">
    <div class="media-ask"><span class="media-prompt-text">${ask.text}</span>
      ${ask.from === "agent" && html`<span class="badge">from an agent</span>`}
      ${facts.length > 0 && html`<span class="muted small">${facts.join(" · ")}</span>`}</div>
    ${made?.streaming && html`<${Working} progress=${made.progress ? { ...made.progress, tool: M === MEDIA.image ? "draw_image" : M === MEDIA.video ? "make_video" : "make_speech" } : null}
        doing=${made.model ? M.doing.toLowerCase() + " with " + made.model.name : "starting"} />`}
    <${MediaGrid} items=${items} onFineTune=${M === MEDIA.image ? tune : null} />
    ${made?.error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${made.error}</p>`}
    ${made?.stopped && html`<p class="muted small">Stopped.</p>`}
    ${made && !made.streaming && html`<div class="msg-acts media-acts">
      <span class="muted small">${[made.model?.name, made.seconds >= 0.1 && made.seconds.toFixed(1) + " s"].filter(Boolean).join(" · ")}</span>
      <span class="grow" />
      ${!busy && html`<button type="button" class="link" onClick=${again} title="The same prompt and settings, new seeds">${M.verb} again</button>`}
    </div>`}
  </li>`;
}

function MediaComposer({ M, medium, settings: o, put, prompt, setPrompt, busy, go, stop }) {
  const ref = useRef(null);
  const [more, setMore] = useState(false);
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, innerHeight / 3) + "px";
  }, [prompt]);
  const models = modelsFor(medium);
  const auto = modelInfo.value?.auto?.[M.modelMedia];
  return html`<div class="composer">
    <div class="composer-box">
      <textarea ref=${ref} class="composer-input media-prompt" rows="1" value=${prompt} aria-label="Prompt"
        placeholder=${medium === "speech" ? "What should be said?" : medium === "video" ? "What happens in the clip?" : "What should the picture show?"}
        onInput=${e => setPrompt(e.target.value)}
        onKeyDown=${e => {
          if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!busy && prompt.trim()) go(); }
          else if (e.key === "Escape" && busy) { e.preventDefault(); stop(); }
        }} />
      <div class="composer-bar media-bar">
        <select class="input select-sm" value=${o.model} disabled=${busy} aria-label="Model" onChange=${e => put({ model: e.target.value })}>
          <option value="auto">Auto${auto ? " (" + (models.find(m => m.key === auto)?.name || auto) + ")" : ""}</option>
          ${models.map(m => html`<option value=${m.key}>${m.name} · ${placeOf(m)}</option>`)}
          ${o.model !== "auto" && !models.some(m => m.key === o.model) && html`<option value=${o.model}>${o.model}</option>`}
        </select>
        ${medium !== "speech" && html`<div class="seg" role="group" aria-label="Shape">${ASPECTS.map(a => html`<button type="button" aria-pressed=${o.aspect === a} disabled=${busy} onClick=${() => put({ aspect: a })}>${a}</button>`)}</div>`}
        ${medium === "image" && html`<div class="seg" role="group" aria-label="How many">${[1, 2, 3, 4].map(n => html`<button type="button" aria-pressed=${+o.count === n} disabled=${busy} onClick=${() => put({ count: n })}>${n}</button>`)}</div>`}
        ${medium === "speech" && html`<input class="input select-sm" value=${o.voice} placeholder="Voice" aria-label="Voice" disabled=${busy} onInput=${e => put({ voice: e.target.value })} />`}
        ${medium !== "speech" && html`<label class="seed-field" title="The same seed with the same prompt and settings gives the same result; empty picks a new one">Seed
          <input class="input mono select-sm" value=${o.seed} placeholder="new" disabled=${busy} onInput=${e => put({ seed: e.target.value.replace(/[^0-9]/g, "") })} />
          ${o.seed !== "" && html`<button type="button" class="mini" aria-label="New seed each time" title="New seed each time" onClick=${() => put({ seed: "" })}><${Icon} name="close" size=${12} /></button>`}</label>`}
        ${medium !== "speech" && html`<button type="button" class="link" aria-expanded=${more} onClick=${() => setMore(!more)}>${more ? "Less" : "More"}</button>`}
        <span class="grow" />
        ${busy ? html`<${Button} icon="stop" onClick=${stop}>Stop<//>`
          : html`<${Button} variant="primary" icon=${M.icon} onClick=${go} disabled=${!prompt.trim()}>${M.verb}<//>`}
      </div>
      ${more && medium !== "speech" && html`<div class="composer-bar media-more">
        <label>Steps <input class="input mono select-sm" value=${o.steps} placeholder="model's" onInput=${e => put({ steps: e.target.value })} /></label>
        <label>Guidance <input class="input mono select-sm" value=${o.guidance} placeholder="model's" onInput=${e => put({ guidance: e.target.value })} /></label>
        ${medium === "video" && html`<label>Seconds <input class="input mono select-sm" value=${o.seconds} onInput=${e => put({ seconds: e.target.value })} /></label>`}
        <label class="grow">Leave out <input class="input select-sm" value=${o.negativePrompt} placeholder="blurry, text, watermark" onInput=${e => put({ negativePrompt: e.target.value })} /></label>
      </div>`}
    </div>
  </div>`;
}
