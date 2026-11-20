import { html, useEffect, useRef, useState } from "/v2/lib/html.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button, IconAction } from "/v2/ui/Button.js";
import { MediaGrid, Working } from "/v2/ui/Media.js";
import { toWav, clock } from "/v2/lib/audio.js";
import { modelInfo, placeOf } from "/v2/state/models.js";
import { MEDIA, modelsFor, transcribeOptions, writeDown, stopMaking } from "/v2/state/media.js";
import { toast } from "/v2/state/toasts.js";

// A transcription: each turn is a recording (made here or chosen) and what was said in it, with
// the times. The recording is kept, so it can be heard again next to its text.

const LANGS = new Intl.DisplayNames(undefined, { type: "language" });
const langName = c => { try { return LANGS.of(c) || c; } catch { return c; } };

/** One recording and its transcript. */
export function TranscriptTurn({ t }) {
  const { ask, made } = t;
  const o = ask.settings || {};
  const [times, setTimes] = useState(false);
  const facts = [ask.audio?.seconds && clock(ask.audio.seconds), o.language && o.language !== "auto" && langName(o.language), o.translate && "into English",
    made?.language && (!o.language || o.language === "auto") && "heard " + langName(made.language)].filter(Boolean);
  const segs = made?.segments || [];
  const text = made?.text || segs.map(x => x.text.trim()).join(" ");
  const save = () => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([times ? segs.map(x => `[${clock(x.start)}] ${x.text.trim()}`).join("\n") : text], { type: "text/plain" }));
    a.download = (ask.audio?.label || "transcript").replace(/\.[^.]+$/, "") + ".txt";
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };
  return html`<li class="msg media-turn transcript-turn">
    <div class="media-ask"><span class="media-prompt-text"><${Icon} name="mic" size=${14} /> ${ask.audio?.label || "Recording"}</span>
      ${facts.length > 0 && html`<span class="muted small">${facts.join(" · ")}</span>`}</div>
    ${ask.audio?.file && html`<${MediaGrid} items=${[{ file: ask.audio.file, kind: "audio" }]} />`}
    ${made?.streaming && !segs.length && html`<${Working} doing=${made.model ? "listening with " + made.model.name : "starting"} />`}
    ${(segs.length > 0 || text) && html`<div class="heard">${times && segs.length
      ? html`<ol class="segments">${segs.map((x, i) => html`<li key=${i}><span class="mono muted small">${clock(x.start)}</span><span>${x.text.trim()}</span></li>`)}</ol>`
      : html`<p>${text}</p>`}${made?.streaming && html`<span class="caret" aria-hidden="true" />`}</div>`}
    ${made?.error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${made.error}</p>`}
    ${made?.stopped && html`<p class="muted small">Stopped.</p>`}
    ${made && !made.streaming && text && html`<div class="msg-acts media-acts">
      <span class="muted small">${[made.model?.name, made.seconds >= 0.1 && made.seconds.toFixed(1) + " s"].filter(Boolean).join(" · ")}</span>
      <span class="grow" />
      ${segs.length > 0 && html`<button type="button" class="link small" aria-pressed=${times} onClick=${() => setTimes(!times)}>${times ? "As text" : "With times"}</button>`}
      <${IconAction} icon="copy" label="Copy the text" onClick=${() => navigator.clipboard?.writeText(times ? segs.map(x => `[${clock(x.start)}] ${x.text.trim()}`).join("\n") : text).then(() => toast("Copied."))} />
      <${IconAction} icon="download" label="Save as a text file" onClick=${save} />
    </div>`}
  </li>`;
}

/** Record or choose a recording, set the language, and write it down. */
export function TranscribeComposer({ s, settings: o, put, busy }) {
  const [clip, setClip] = useState(null);
  const [rec, setRec] = useState(null);
  const [secs, setSecs] = useState(0);
  const [error, setError] = useState(null);
  const [opts, setOpts] = useState({ languages: [], translate: false });
  const [more, setMore] = useState(false);
  const file = useRef(null);
  const models = modelsFor("transcribe");
  const auto = modelInfo.value?.auto?.transcribe;
  useEffect(() => { let on = true; transcribeOptions(o.model || "auto").then(r => on && setOpts(r)); return () => { on = false; }; }, [o.model]);
  const live = useRef(null);
  live.current = rec;
  useEffect(() => () => { if (live.current?.state === "recording") live.current.stop(); }, []);
  useEffect(() => {
    if (!rec) return;
    const at = Date.now(), t = setInterval(() => setSecs((Date.now() - at) / 1000), 250);
    return () => clearInterval(t);
  }, [rec]);
  const read = async (blob, label) => {
    setError(null);
    try { setClip({ ...(await toWav(blob, 16000)), label }); }
    catch (e) { setError("This could not be read as sound: " + e.message); }
  };
  const record = async () => {
    if (rec) { rec.stop(); return; }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (e) { setError("The microphone is not available: " + e.message); return; }
    const chunks = [], r = new MediaRecorder(stream);
    r.ondataavailable = e => e.data.size && chunks.push(e.data);
    r.onstop = () => { stream.getTracks().forEach(t => t.stop()); setRec(null); read(new Blob(chunks, { type: r.mimeType || "audio/webm" }), "Recording " + new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })); };
    r.start(500);
    setSecs(0); setRec(r); setClip(null);
  };
  const go = () => { const c = clip; setClip(null); writeDown(s, c); };
  const langs = opts.languages || [];
  return html`<div class="composer">
    <div class="composer-box">
      <div class="heard-pick">
        ${rec ? html`<span class="live live-run" aria-hidden="true" /><span>Recording… <span class="mono">${clock(secs)}</span></span>`
          : clip ? html`<${Icon} name="mic" size=${16} /><span><b>${clip.label}</b> <span class="mono muted small">${clock(clip.seconds)}</span></span>
              <${IconAction} icon="close" label="Leave it out" onClick=${() => setClip(null)} />`
          : html`<span class="muted">Record something, or choose a recording or a video.</span>`}
        <span class="grow" />
        <${Button} icon="folder" disabled=${busy || !!rec} onClick=${() => file.current.click()}>Choose…<//>
        <input ref=${file} type="file" accept="audio/*,video/*" hidden onChange=${e => { const f = e.target.files[0]; if (f) read(f, f.name); e.target.value = ""; }} />
        <${Button} icon=${rec ? "stop" : "mic"} disabled=${busy} onClick=${record}>${rec ? "Stop" : "Record"}<//>
      </div>
      ${error && html`<p class="msg-error small"><${Icon} name="warn" size=${14} />${error}</p>`}
      <div class="composer-bar media-bar">
        <select class="input select-sm" value=${o.model} disabled=${busy} aria-label="Model" onChange=${e => put({ model: e.target.value })}>
          <option value="auto">Auto${auto ? " (" + (models.find(m => m.key === auto)?.name || auto) + ")" : ""}</option>
          ${models.map(m => html`<option value=${m.key}>${m.name} · ${placeOf(m)}</option>`)}
        </select>
        ${langs.length > 0 && html`<select class="input select-sm" value=${o.language || "auto"} disabled=${busy} aria-label="Language spoken" onChange=${e => put({ language: e.target.value })}>
          <option value="auto">Any language</option>
          ${langs.map(l => html`<option value=${l}>${langName(l)}</option>`)}</select>`}
        ${opts.translate && html`<label class="small check-inline"><input type="checkbox" checked=${!!o.translate} disabled=${busy} onChange=${e => put({ translate: e.target.checked })} /> Into English</label>`}
        <button type="button" class="link" aria-expanded=${more} onClick=${() => setMore(!more)}>${more ? "Less" : "More"}</button>
        <span class="grow" />
        ${busy ? html`<${Button} icon="stop" onClick=${() => stopMaking(s)}>Stop<//>`
          : html`<${Button} variant="primary" icon="mic" onClick=${go} disabled=${!clip || !!rec}>${MEDIA.transcribe.verb}<//>`}
      </div>
      ${more && html`<div class="composer-bar media-more">
        <label class="grow">Words to expect <input class="input select-sm" value=${o.hint} placeholder="names, terms, how it is spelled" onInput=${e => put({ hint: e.target.value })} /></label>
      </div>`}
    </div>
  </div>`;
}
