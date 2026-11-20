import { html, useEffect, useState, cx } from "/v2/lib/html.js";
import { Icon } from "/v2/ui/Icon.js";
import { openLayer } from "/v2/ui/overlay.js";
import { Layer } from "/v2/ui/Layer.js";
import { Button } from "/v2/ui/Button.js";

// Pictures, recordings and clips the studio made, shown where they were asked for: in a chat,
// on an agent's page, in an image, speech or video chat. They are read from the outputs folder
// with the session token, once, and kept as blob URLs while the page is open.

const token = document.querySelector('meta[name="studio-token"]')?.content ?? "";
const urls = new Map();

/** A blob URL for a file in the studio's outputs folder. */
export function outputUrl(path) {
  let u = urls.get(path);
  if (!u) {
    u = fetch("/api/outputs/file", { method: "POST", headers: { "X-Studio-Token": token, "Content-Type": "application/json" }, body: JSON.stringify({ path }) })
      .then(r => (r.ok ? r.blob() : Promise.reject(new Error(r.status === 404 ? "The file is not in the outputs folder any more." : "The studio answered " + r.status))))
      .then(b => URL.createObjectURL(b));
    u.catch(() => urls.delete(path));
    urls.set(path, u);
  }
  return u;
}

function useOutput(path) {
  const [state, setState] = useState({ url: null, error: null });
  useEffect(() => {
    let live = true;
    setState({ url: null, error: null });
    if (path) outputUrl(path).then(url => live && setState({ url, error: null }), e => live && setState({ url: null, error: e.message }));
    return () => { live = false; };
  }, [path]);
  return state;
}

export const kindOfFile = f => /\.(png|jpe?g|webp|gif)$/i.test(f) ? "image" : /\.(wav|mp3|ogg|flac)$/i.test(f) ? "audio" : /\.(mp4|webm)$/i.test(f) ? "video"
  : /\.apng$/i.test(f) ? "image" : null;
export const fileName = f => String(f).split(/[\\/]/).pop();

const MEDIA_TOOLS = { draw_image: "image", make_speech: "audio", make_video: "video" };

/**
 * What an agent's steps made: for each tool call that answered with files, the files and what
 * was asked for (the prompt, model and settings, so it can be made again with changes).
 */
export function mediaOf(steps = []) {
  const out = [];
  let call = null;
  for (const s of steps) {
    if (s.kind === "call") call = s;
    if (s.kind !== "result" || s.ok === false) continue;
    let r;
    try { r = JSON.parse(s.text); } catch { continue; }
    const files = Array.isArray(r?.files) ? r.files.filter(f => kindOfFile(f)) : [];
    if (!files.length) continue;
    const args = call && call.tool === s.tool ? call.arguments || {} : {};
    files.forEach((file, i) => out.push({
      file, kind: kindOfFile(file), tool: s.tool, model: r.model, seed: r.seeds?.[i] ?? null,
      prompt: args.prompt || args.text || "", aspect: args.aspect || null, voice: args.voice || null, speed: args.speed || null, instructions: args.instructions || null,
    }));
  }
  return out;
}

/** A grid of what was made; a picture opens large on click. */
export function MediaGrid({ items, onFineTune, compact }) {
  if (!items?.length) return null;
  return html`<div class=${cx("media-grid", compact && "media-grid-compact")}>
    ${items.map((it, i) => html`<${MediaTile} key=${it.file + i} it=${it} open=${() => openLayer(Viewer, { items, at: i, onFineTune })} onFineTune=${onFineTune} />`)}
  </div>`;
}

function MediaTile({ it: given, open, onFineTune }) {
  const it = { ...given, kind: kindOfFile(given.file) || given.kind };
  const { url, error } = useOutput(it.file);
  const title = `${fileName(it.file)}${it.seed != null ? " · seed " + it.seed : ""}`;
  if (error) return html`<div class="media-tile media-missing" title=${it.file}><${Icon} name="warn" size=${14} />${error}</div>`;
  if (it.kind === "audio") return html`<div class="media-tile media-audio" title=${title}>
    ${url ? html`<audio controls preload="metadata" src=${url} />` : html`<span class="media-wait" />`}
    <span class="media-cap">${it.prompt ? `“${short(it.prompt, 80)}”` : fileName(it.file)}${it.voice ? html` <span class="muted">· ${it.voice}</span>` : ""}
      ${onFineTune && html`<button type="button" class="link media-tune" onClick=${() => onFineTune(it)}>Fine tune…</button>`}</span></div>`;
  return html`<button type="button" class=${cx("media-tile", "media-" + it.kind)} title=${title} onClick=${open} aria-label=${"Open " + title}>
    ${!url ? html`<span class="media-wait" />`
      : it.kind === "video" ? html`<video src=${url} muted loop autoplay playsinline />`
      : html`<img src=${url} alt=${it.prompt || fileName(it.file)} />`}
    ${it.seed != null && html`<span class="media-seed">seed ${it.seed}</span>`}
  </button>`;
}
const short = (t, n) => (t.length > n ? t.slice(0, n - 1) + "…" : t);

/** One picture or clip large, with the others a key press away, and what to do with it. */
function Viewer({ items, at: first, onFineTune, close, top }) {
  const [at, setAt] = useState(first);
  const it = { ...items[at], kind: kindOfFile(items[at].file) || items[at].kind };
  const { url } = useOutput(it.file);
  useEffect(() => {
    const key = e => {
      if (e.key === "ArrowRight") setAt(a => Math.min(items.length - 1, a + 1));
      else if (e.key === "ArrowLeft") setAt(a => Math.max(0, a - 1));
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);
  const footer = html`
    ${items.length > 1 && html`<${Button} icon="chevronLeft" label="Previous (←)" disabled=${at === 0} onClick=${() => setAt(at - 1)} />
      <span class="muted small">${at + 1} of ${items.length}</span>
      <${Button} icon="chevronRight" label="Next (→)" disabled=${at === items.length - 1} onClick=${() => setAt(at + 1)} />`}
    <span class="grow" />
    ${url && html`<a class="btn" href=${url} download=${fileName(it.file)}><${Icon} name="download" /><span>Save</span></a>`}
    ${onFineTune && html`<${Button} variant="primary" icon="sparkle" onClick=${() => { close(); onFineTune(it); }}>Fine tune…<//>`}`;
  return html`<${Layer} title=${it.prompt ? short(it.prompt, 90) : fileName(it.file)} close=${close} top=${top} wide footer=${footer}
      subtitle=${[it.model, it.seed != null && "seed " + it.seed, fileName(it.file)].filter(Boolean).join(" · ")}>
    <div class="viewer">${!url ? html`<span class="media-wait" />`
      : it.kind === "video" ? html`<video src=${url} controls autoplay loop />` : html`<img src=${url} alt=${it.prompt || ""} />`}</div>
  <//>`;
}

const VERBS = { draw_image: "Drawing", make_video: "Filming", make_speech: "Speaking" };

/**
 * What a working agent is doing now. A drawing or clip shows its step and a rough preview; any
 * other long call shows what it called and for how long, so a quiet minute never looks stuck.
 */
export function Working({ progress, doing, thinking }) {
  const [, tick] = useState(0);
  const [since] = useState(() => ({ doing, at: Date.now() }));
  if (since.doing !== doing) { since.doing = doing; since.at = Date.now(); }
  useEffect(() => { const t = setInterval(() => tick(n => n + 1), 1000); return () => clearInterval(t); }, []);
  const secs = Math.floor((Date.now() - since.at) / 1000);
  if (progress?.total) {
    const pct = Math.round((100 * progress.step) / progress.total);
    return html`<div class="working working-steps" role="status">
      ${progress.preview && html`<img class="working-preview" src=${progress.preview} alt="Rough preview" />`}
      <div class="working-main">
        <span><span class="live live-run" />${VERBS[progress.tool] || "Working"} · step ${progress.step} of ${progress.total}</span>
        <div class="prog" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${pct}><i style=${`width:${pct}%`} /></div>
      </div>
    </div>`;
  }
  const what = thinking ? "Thinking" : doing ? doing[0].toUpperCase() + doing.slice(1) : "Working";
  return html`<p class="working" role="status"><span class="live live-run" />${what}…${secs >= 5 ? html` <span class="muted small">${secs < 120 ? secs + " s" : Math.floor(secs / 60) + " min"}</span>` : ""}</p>`;
}
