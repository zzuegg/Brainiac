import { html, useEffect, useState, cx } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button, IconAction } from "/v2/ui/Button.js";
import { Layer } from "/v2/ui/Layer.js";
import { openLayer } from "/v2/ui/overlay.js";
import { plural } from "/v2/lib/format.js";
import { modelInfo, modelsError, loadModels } from "/v2/state/models.js";
import { hub, readHub, active, download, cancel, forget, fmtBytes, fmtWait, fmtCount } from "/v2/state/hub.js";
import { fail, toast } from "/v2/state/toasts.js";

// Every model the studio can use, wherever it runs: this PC, your other machines, and the
// providers you have a key for. New ones come from Hugging Face or from a folder on this PC.

export const MEDIA_NAMES = { text: "Chat", vision: "Sees pictures", image: "Pictures", speech: "Speech", transcribe: "Transcription", video: "Video", embed: "Embeddings" };
const placeName = p => (p.id === "local" ? "This PC" : p.name || p.id);
const placeNote = p => (p.kind === "local" ? "free · private" : p.kind === "host" ? "your machine" : p.kind === "rental" ? "rented" : "your key");

export function Models() {
  const info = modelInfo.value;
  const [find, setFind] = useState("");
  const [folded, setFolded] = useState(() => new Set());
  useEffect(() => { loadModels(); readHub(); }, []);
  const q = find.trim().toLowerCase();
  const auto = new Set(Object.values(info?.auto || {}));
  const groups = (info?.providers || []).map(p => ({ p, list: (info.models || []).filter(m => m.provider === p.id && (!q || (m.name + " " + m.key).toLowerCase().includes(q))) }))
    .filter(g => g.list.length || (!q && (g.p.kind === "local" || g.p.error)));
  const downloads = hub.value.downloads;

  return html`<div class="page models-page">
    <header class="page-head"><div><p class="eyebrow">Studio</p><h1 class="page-title">Models</h1>
      <p class="page-lead">Everything the studio can use, wherever it runs. Auto tries this PC and your machines first, then providers you have a key for.</p></div>
      <div class="page-actions">
        <${Button} icon="folder" onClick=${() => openLayer(AddFolder)}>Add from this PC…<//>
        <${Button} variant="primary" icon="download" onClick=${() => openLayer(HubSearch)}>Download…<//>
      </div></header>
    ${downloads.length > 0 && html`<section class="card"><h2 class="card-title">Downloads</h2><${Downloads} list=${downloads} /></section>`}
    ${modelsError.value && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${modelsError.value}</p>`}
    ${!info ? html`<p class="muted">Looking for models…</p>` : html`
      <div class="models-bar">
        <input class="input" type="search" placeholder="Find a model" value=${find} onInput=${e => setFind(e.target.value)} aria-label="Find a model" />
        <${IconAction} icon="redo" label="Look again: new files and provider lists" onClick=${() => loadModels(true)} />
      </div>
      ${groups.map(({ p, list }) => {
        const open = q || !folded.has(p.id);
        const hidden = list.filter(m => m.shown === false).length;
        return html`<section class="card model-group" key=${p.id}>
          <h2 class="card-title">
            <button type="button" class="group-toggle" aria-expanded=${!!open} onClick=${() => { const f = new Set(folded); f.has(p.id) ? f.delete(p.id) : f.add(p.id); setFolded(f); }}>
              <${Icon} name=${open ? "chevronDown" : "chevronRight"} size=${14} />${placeName(p)}</button>
            <span class="muted small">${plural(list.length, "model")}${hidden ? ` · ${hidden} hidden` : ""}</span>
            <span class="badge">${placeNote(p)}</span>
          </h2>
          ${p.error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${p.error}</p>`}
          ${open && p.id === "local" && html`<${Folders} p=${p} />`}
          ${open && (list.length ? html`<ul class="model-rows">${list.map(m => html`<li key=${m.key} class=${cx(m.shown === false && "dim")}>
              <div class="model-name"><b>${m.name}</b><span class="mono muted small">${m.key}</span></div>
              <div class="model-can">${(m.media || []).map(x => html`<span class=${"cap cap-" + x}>${MEDIA_NAMES[x] || x}</span>`)}</div>
              <div class="model-use">${m.shown === false ? html`<span class="badge" title="Not offered in the model pickers or by Auto">hidden</span>`
                : auto.has(m.key) ? html`<span class="badge s-done">Auto picks it</span>` : ""}</div>
              <div class="mono muted small model-detail">${m.detail}</div>
            </li>`)}</ul>`
            : html`<p class="muted small">${p.kind === "local" ? "No models here yet: download one, or add a folder that has some." : "No models."}</p>`)}
        </section>`;
      })}
      ${q && !groups.length && html`<p class="muted">No model matches “${find}”.</p>`}`}
  </div>`;
}

/** The folders this PC looks in; ones you added can be taken out again. */
function Folders({ p }) {
  const added = new Set(p.added || []);
  const remove = f => api("/api/local", { path: f, remove: true }).then(() => { loadModels(true); toast("Taken out. The files stay where they are."); }, fail);
  if (!(p.folders || []).length) return null;
  return html`<details class="folders"><summary class="muted small">Looks in ${plural(p.folders.length, "folder")}</summary>
    <ul>${p.folders.map(f => html`<li key=${f}><span class="mono small">${f}</span>
      ${added.has(f) && html`<${IconAction} icon="close" label="Stop looking here (the files stay)" onClick=${() => remove(f)} />`}</li>`)}</ul></details>`;
}

function Downloads({ list }) {
  const [sure, setSure] = useState(null);
  return html`<ul class="downloads">${list.map(d => {
    const pct = d.total ? Math.min(100, (100 * d.done) / d.total) : 0;
    const status = d.state === "done" ? "on this PC" : d.state === "checking" ? "checking…" : d.state === "queued" ? "starting…"
      : d.state === "downloading" ? `${fmtBytes(d.done)} of ${fmtBytes(d.total)}${d.speed ? ` · ${fmtBytes(d.speed)}/s · ${fmtWait((d.total - d.done) / d.speed)}` : ""}`
      : d.state === "stopped" ? `stopped at ${fmtBytes(d.done)} of ${fmtBytes(d.total)}` : "failed";
    return html`<li key=${d.id} class=${"dl dl-" + d.state}>
      <div class="dl-name"><b>${d.name}</b><span class="muted small">${d.repo}</span></div>
      <div class="prog" role="progressbar" aria-valuemin="0" aria-valuemax="100" aria-valuenow=${Math.round(pct)}><i style=${`width:${pct}%`} /></div>
      <span class="mono small muted">${status}</span>
      <span class="dl-acts">${active(d) ? html`<${IconAction} icon="stop" label="Stop" onClick=${() => cancel(d.id)} />`
        : d.state === "done" ? html`<${IconAction} icon=${sure === d.id ? "check" : "trash"} label=${sure === d.id ? "Click again to delete the files" : "Delete it from this PC"}
            onClick=${() => (sure === d.id ? (setSure(null), forget(d.name)) : setSure(d.id))} />`
        : html`<${IconAction} icon="redo" label=${d.state === "stopped" ? "Go on" : "Try again"} onClick=${() => download(d.repo, d.option).catch(fail)} />`}</span>
      ${d.error && html`<p class="msg-error small">${d.error}</p>`}
    </li>`;
  })}</ul>`;
}

/** Search Hugging Face for models the studio runs, and fetch one. */
function HubSearch({ close, top }) {
  const [kind, setKind] = useState("text");
  const [query, setQuery] = useState("");
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [repo, setRepo] = useState(null);
  const [files, setFiles] = useState(null);
  useEffect(() => { readHub(); }, []);
  const search = async e => {
    e?.preventDefault();
    setBusy(true); setError(null);
    try { const r = await api("/api/hub/search", { query, kind }); setRes(r); } catch (x) { setError(x.message); setRes(null); }
    setBusy(false);
  };
  const pick = async id => {
    setRepo(id); setFiles(null);
    try { const l = await api("/api/hub/files", { repo: id }); setFiles(l); } catch (x) { setFiles({ error: x.message }); }
  };
  const runs = hub.value.runs;
  const footer = html`<span class="muted small">Files go to <span class="mono">${hub.value.folder || "downloads"}</span>.</span><span class="grow" /><${Button} onClick=${close}>Done<//>`;
  return html`<${Layer} title="Download from Hugging Face" wide close=${close} top=${top} footer=${footer}
      subtitle=${runs ? `Only models the studio runs: ${runs.families.length} kinds of model folder and ${runs.architectures.length} of GGUF file.` : ""}>
    <${Token} />
    <form class="hub-search" onSubmit=${search}>
      <div class="seg" role="group" aria-label="What kind of model">
        ${[["text", "Chat"], ["speech", "Speech"]].map(([k, n]) => html`<button type="button" aria-pressed=${kind === k} onClick=${() => { setKind(k); setRes(null); }}>${n}</button>`)}</div>
      <input class="input" type="search" autofocus value=${query} onInput=${e => setQuery(e.target.value)} aria-label="Model name"
        placeholder=${kind === "speech" ? "whisper base, kokoro, mms-tts-deu…" : "qwen2.5 instruct, llama 3.2 1b, gemma…"} />
      <${Button} variant="primary" icon="search" type="submit" disabled=${busy}>${busy ? "Searching…" : "Search"}<//>
    </form>
    ${error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${error}</p>`}
    <div class="hub-cols">
      <ul class="hub-list" aria-label="Found">${!res ? html`<li class="muted small pad">Search by name. Pick a result to see its files.</li>`
        : !res.models?.length ? html`<li class="muted small pad">Nothing found that the studio runs${res.hidden ? ` (${res.hidden} others found)` : ""}.</li>`
        : res.models.map(r => html`<li key=${r.id}><button type="button" class=${cx("hub-item", repo === r.id && "on")} onClick=${() => pick(r.id)}>
            <b>${r.id}</b>
            <span class="hub-meta">${r.family && html`<span class="badge s-done">${r.family}</span>`}${r.gguf && html`<span class="badge">GGUF</span>`}
              ${r.parameters > 0 && html`<span class="badge">${r.parameters >= 1e9 ? (r.parameters / 1e9).toFixed(r.parameters < 1e10 ? 1 : 0) + "B" : Math.round(r.parameters / 1e6) + "M"}</span>`}
              ${r.sees && html`<span class="badge">sees pictures</span>`}${r.gated && html`<span class="badge s-waiting" title="Accept its license on huggingface.co and save your token above">gated</span>`}
              <span class="muted small"><${Icon} name="download" size=${12} /> ${fmtCount(r.downloads)}</span></span></button></li>`)}
        ${res?.hidden > 0 && res.models?.length > 0 && html`<li class="muted small pad">${res.hidden} more that the studio cannot run yet.</li>`}</ul>
      <div class="hub-files">${!repo ? html`<p class="muted small pad">Its files and which ones run here show here.</p>`
        : html`<div class="hub-files-head"><span class="mono small">${repo}</span><a class="small" href=${"https://huggingface.co/" + repo} target="_blank" rel="noopener">Model page</a></div>
          ${!files ? html`<p class="muted small pad">Reading its files…</p>` : files.error ? html`<p class="msg-error">${files.error}</p>`
            : html`<ul class="hub-opts">${(files.options || []).slice().sort((a, b) => b.supported - a.supported || a.size - b.size).map(o => {
                const d = hub.value.downloads.find(x => x.repo === repo && (o.kind === "folder" ? x.kind === "folder" : x.name === o.name));
                return html`<li key=${o.id} class=${cx(!o.supported && "dim")}>
                  <span class="hub-opt-name">${o.kind === "folder" ? "Whole model folder" : o.name}${o.quant && html` <span class="badge">${o.quant}</span>`}${o.sees && html` <span class="badge">sees pictures</span>`}</span>
                  <span class="mono small muted">${o.size ? fmtBytes(o.size + (o.projectorSize || 0)) : ""}</span>
                  ${o.have || d?.state === "done" ? html`<span class="badge s-done">on this PC</span>` : d && active(d) ? html`<span class="badge s-running">downloading</span>`
                    : o.supported ? html`<${Button} size="sm" icon="download" onClick=${() => download(repo, o.id).then(() => toast("Downloading " + (o.kind === "folder" ? repo : o.name)), fail)}>Download<//>`
                    : html`<span class="badge s-failed" title=${o.reason || ""}>won’t run</span>`}
                  ${o.reason && !o.supported && html`<span class="muted small hub-why">${o.reason}</span>`}
                </li>`;
              })}</ul>`}`}</div>
    </div>
    ${hub.value.downloads.length > 0 && html`<h3 class="card-title hub-dl-title">Downloads</h3><${Downloads} list=${hub.value.downloads} />`}
  <//>`;
}

/** The token the downloader sends to Hugging Face, for gated models; shown by its last characters. */
function Token() {
  const t = hub.value.token;
  const [edit, setEdit] = useState(false);
  const [value, setValue] = useState("");
  const save = async e => {
    e.preventDefault();
    try { await api("/api/hub/token", { token: value.trim() }); setEdit(false); setValue(""); readHub(); toast("Token saved."); } catch (x) { fail(x); }
  };
  if (t && !edit) return html`<p class="hub-token"><${Icon} name="lock" size=${14} /> Hugging Face token <span class="mono">${t.hint}</span> is set.
    <button type="button" class="link small" onClick=${() => setEdit(true)}>Change</button></p>`;
  return html`<form class="hub-token" onSubmit=${save}><${Icon} name="lock" size=${14} />
    <span class="small">${t ? "A new token replaces the one set." : html`Gated models such as Llama or Gemma need a token: <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noopener">make a read token</a> and accept the model's license on its page.`}</span>
    <input class="input mono" type="password" placeholder="hf_…" value=${value} onInput=${e => setValue(e.target.value)} aria-label="Hugging Face token" />
    <${Button} type="submit" disabled=${!value.trim()}>Save<//>${t && html`<${Button} variant="ghost" onClick=${() => setEdit(false)}>Cancel<//>`}</form>`;
}

/** Points the studio at models already on this PC: a GGUF file, a model folder, or a folder of them. */
function AddFolder({ close, top }) {
  const [path, setPath] = useState("");
  const [list, setList] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const browse = async p => {
    try { const l = await api("/api/local/browse", { path: p || "" }); setList(l); setError(null); if (p) setPath(l.path); } catch (x) { setError(x.message); }
  };
  useEffect(() => { browse(""); }, []);
  const add = async e => {
    e?.preventDefault();
    setBusy(true); setError(null);
    try {
      const r = await api("/api/local", { path: path.trim() });
      await loadModels(true);
      toast(r.models ? `Added ${plural(r.models, "model")}: they are under This PC.` : "Added. Models put there later show up too.");
      close();
    } catch (x) { setError(x.message); }
    setBusy(false);
  };
  const footer = html`<span class="grow" /><${Button} onClick=${close}>Cancel<//><${Button} variant="primary" disabled=${busy || !path.trim()} onClick=${add}>${busy ? "Adding…" : "Add"}<//>`;
  const inside = list ? list.entries.filter(e => e.kind !== "folder").length : 0;
  return html`<${Layer} title="Add models from this PC" close=${close} top=${top} footer=${footer}
      subtitle="A .gguf file, a Hugging Face model folder, or a folder that holds them. The files stay where they are.">
    <form onSubmit=${add}><input class="input mono" value=${path} onInput=${e => setPath(e.target.value)} placeholder="A file or folder, such as D:/Models" spellcheck="false" aria-label="File or folder" autofocus /></form>
    ${error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${error}</p>`}
    <div class="browse">
      <div class="browse-head">${list?.path ? html`<${IconAction} icon="chevronLeft" label="Up" onClick=${() => browse(list.parent ?? "")} /><span class="mono small">${list.path}</span>
        <span class="grow" /><span class="muted small">${list.kind === "model" ? "This folder is a model." : `${inside || "No"} ${inside === 1 ? "model" : "models"} right here`}</span>`
        : html`<span class="muted small">Start from</span>`}</div>
      <ul class="browse-list">${!list ? html`<li class="muted small">Looking…</li>` : !list.entries.length ? html`<li class="muted small">No folders or models here.</li>`
        : list.entries.map(e => html`<li key=${e.path}><button type="button" class=${cx("browse-item", e.path === path && "on")} title=${e.path}
            onClick=${() => (e.kind === "folder" ? browse(e.path) : setPath(e.path))}>
            <${Icon} name=${e.kind === "folder" ? "folder" : "build"} size=${14} /><span>${e.name}</span>
            ${e.kind === "gguf" && html`<span class="badge">GGUF</span>`}${e.kind === "model" && html`<span class="badge">model folder</span>`}</button></li>`)}</ul>
    </div>
  <//>`;
}
