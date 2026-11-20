import { html, useEffect, useRef, useState, cx } from "/v2/lib/html.js";
import { Layer } from "/v2/ui/Layer.js";
import { Button } from "/v2/ui/Button.js";
import { Icon } from "/v2/ui/Icon.js";
import { outputUrl } from "/v2/ui/Media.js";
import { toWav, clock } from "/v2/lib/audio.js";
import { modelInfo, modelByKey } from "/v2/state/models.js";
import { speechOptions, voiceName, preview, addVoice } from "/v2/state/media.js";

// The voices a speech model offers, each one a click away from being heard, and a new voice
// made from a short recording when the model can copy one.

/** A voice's round face: its first letter on a colour that stays the same for that voice. */
export function VoiceFace({ id, name, size = 22 }) {
  let h = 0;
  for (const c of id || "") h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return html`<span class="voice-face" style=${`--c:var(--series-${(h % 8) + 1});width:${size}px;height:${size}px`} aria-hidden="true">${(name || id || "?")[0].toUpperCase()}</span>`;
}

let playing = null;

export function VoicePicker({ project, model, value, onPick, close, top }) {
  const key = model === "auto" ? modelInfo.value?.auto?.speech : model;
  const m = modelByKey(key);
  const opt = speechOptions(model);
  const [hearing, setHearing] = useState(null);
  const [error, setError] = useState(null);
  const [filter, setFilter] = useState("");
  const hear = async (v, name) => {
    setHearing(v); setError(null);
    try {
      const file = await preview(project, model, v, name);
      const url = await outputUrl(file);
      playing?.pause();
      playing = new Audio(url);
      await playing.play();
    } catch (e) { setError(e.message); }
    setHearing(null);
  };
  const shown = opt.voices.filter(v => {
    const n = voiceName(v, opt.names);
    return !filter || (n.name + " " + n.about + " " + v).toLowerCase().includes(filter.toLowerCase());
  });
  const footer = html`<span class="muted small">${m ? m.name : "No speech model"}${m?.provider && m.provider !== "local" ? " · hearing a voice is billed to your key" : ""}</span>
    <span class="grow" /><${Button} onClick=${close}>Close<//>`;
  return html`<${Layer} title="Voices" subtitle=${m ? `${opt.voices.length} voices of ${m.name}` : ""} close=${close} top=${top} footer=${footer}>
    ${opt.voices.length > 8 && html`<input class="input" type="search" placeholder="Find a voice: a name, female, British…" value=${filter}
      onInput=${e => setFilter(e.target.value)} aria-label="Find a voice" autofocus />`}
    ${error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${error}</p>`}
    ${!opt.voices.length && html`<p class="muted">${opt.clone ? "No voices yet. This model speaks in a voice made from a recording: add one below." : "This model names no voices."}</p>`}
    <ul class="voice-list">${shown.map(v => {
      const n = voiceName(v, opt.names);
      return html`<li key=${v} class=${cx(v === value && "on")}>
        <button type="button" class="voice-pick" aria-pressed=${v === value} onClick=${() => { onPick(v); close(); }}>
          <${VoiceFace} id=${v} name=${n.name} size=${30} />
          <span><b>${n.name}</b>${n.about && html`<span class="muted small">${n.about}</span>`}</span>
          ${v === value && html`<${Icon} name="check" />`}
        </button>
        <${Button} variant="ghost" icon=${hearing === v ? "dot" : "play"} label=${"Hear " + n.name} disabled=${!!hearing} onClick=${() => hear(v, n.name)} />
      </li>`;
    })}</ul>
    ${opt.clone && key && html`<${NewVoice} project=${project} model=${key} onMade=${v => { onPick(v); close(); }} />`}
  <//>`;
}

/** A new voice from 3 to 12 seconds of clear speech: a file or the microphone, and what is said. */
function NewVoice({ project, model, onMade }) {
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [sample, setSample] = useState(null);
  const [rec, setRec] = useState(null);
  const [secs, setSecs] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const file = useRef(null);
  useEffect(() => () => rec?.stop(), []);
  useEffect(() => {
    if (!rec) return;
    const at = Date.now(), t = setInterval(() => setSecs((Date.now() - at) / 1000), 250);
    return () => clearInterval(t);
  }, [rec]);
  const read = async (blob, label) => {
    setError(null);
    try {
      const w = await toWav(blob, 24000);
      setSample({ ...w, label });
      if (w.seconds < 2) setError("That is too short: 3 to 12 seconds of speech works best.");
    } catch (e) { setError("This could not be read as sound: " + e.message); }
  };
  const record = async () => {
    if (rec) { rec.stop(); return; }
    let stream;
    try { stream = await navigator.mediaDevices.getUserMedia({ audio: true }); }
    catch (e) { setError("The microphone is not available: " + e.message); return; }
    const chunks = [], r = new MediaRecorder(stream);
    r.ondataavailable = e => e.data.size && chunks.push(e.data);
    r.onstop = () => { stream.getTracks().forEach(t => t.stop()); setRec(null); read(new Blob(chunks, { type: r.mimeType || "audio/webm" }), "Recording"); };
    r.start(500);
    setSecs(0); setRec(r);
  };
  const save = async () => {
    setBusy(true); setError(null);
    try { onMade(await addVoice(project, model, name.trim(), sample.data, text.trim())); }
    catch (e) { setError(e.message); }
    setBusy(false);
  };
  return html`<section class="new-voice">
    <h3 class="card-title">New voice from a recording</h3>
    <div class="new-voice-row">
      <input class="input" placeholder="Its name, such as Me" maxlength="40" value=${name} onInput=${e => setName(e.target.value)} aria-label="Name of the new voice" />
      <${Button} icon="folder" disabled=${busy || !!rec} onClick=${() => file.current.click()}>Choose a file<//>
      <input ref=${file} type="file" accept="audio/*,video/*" hidden onChange=${e => { const f = e.target.files[0]; if (f) read(f, f.name); e.target.value = ""; }} />
      <${Button} icon=${rec ? "stop" : "mic"} disabled=${busy} onClick=${record}>${rec ? `Stop ${clock(secs)}` : "Record"}<//>
    </div>
    <p class="muted small">${sample ? `${sample.label} · ${sample.seconds.toFixed(1)} s${sample.seconds > 12 ? ", only the first 12 s are used" : ""}`
      : "3 to 12 seconds of clear speech, one speaker, little noise."}</p>
    <textarea class="input" rows="2" placeholder="What is said in it. Left empty, it is written down for you." value=${text} onInput=${e => setText(e.target.value)} aria-label="What is said in the recording" />
    ${error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${error}</p>`}
    <div class="row-btns"><span class="grow" /><${Button} variant="primary" disabled=${busy || !sample || !name.trim() || !!rec} onClick=${save}>${busy ? "Adding…" : "Add the voice"}<//></div>
  </section>`;
}
