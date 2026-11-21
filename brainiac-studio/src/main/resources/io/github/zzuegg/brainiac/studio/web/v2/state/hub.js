import { signal } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { loadModels } from "/v2/state/models.js";

// Models fetched from Hugging Face onto this PC: what is downloading, the token the downloader
// sends, and where the files go. Read while a download runs, and when the Models page opens.
// A download can also go onto another machine running the studio as a worker: its downloads
// are listed with the rest, marked with the machine (`host`, `hostName`).

export const hub = signal({ downloads: [], token: null, folder: "", runs: null });
/** The machines a model can be downloaded onto: the workers that answer. */
export const hosts = signal([]);
/** Where a download goes: "" for this PC, or a worker's id. */
export const target = signal("");
const remote = new Map();   // host id -> its downloads
export const active = d => d.state === "queued" || d.state === "downloading" || d.state === "checking";

let timer = 0, finished = new Set();
export async function readHub() {
  try {
    const r = await api("/api/hub/downloads");
    // a finished download is a new model: the lists learn of it
    const done = (r.downloads || []).filter(d => d.state === "done").map(d => d.id);
    if (done.some(id => !finished.has(id)) && finished.size + done.length > done.length) loadModels(true);
    finished = new Set(done);
    // the workers' downloads: the one chosen, and any with a download going
    const watch = new Set([target.value, ...[...remote].filter(([, l]) => l.some(active)).map(([id]) => id)].filter(Boolean));
    await Promise.all([...watch].map(async id => {
      try {
        const was = new Set((remote.get(id) || []).filter(d => d.state === "done").map(d => d.id));
        const list = (await api("/api/hosts/downloads", { host: id })).downloads || [];
        if (list.some(d => d.state === "done" && !was.has(d.id)) && remote.has(id)) loadModels(true);   // the worker now serves it
        remote.set(id, list);
      } catch { /* the worker went away; its list stays as it was */ }
    }));
    const name = id => hosts.value.find(h => h.id === id)?.name || "another machine";
    const theirs = [...remote].flatMap(([id, l]) => l.map(d => ({ ...d, host: id, hostName: name(id) })));
    hub.value = { downloads: [...(r.downloads || []), ...theirs], token: r.token || null, folder: r.folder || "", runs: r.runs || null };
  } catch { /* the page says the studio is not answering */ }
  clearTimeout(timer);
  if (hub.value.downloads.some(active)) timer = setTimeout(readHub, 1200);
}

/** Reads the workers a model could go onto. */
export async function readHosts() {
  try { hosts.value = ((await api("/api/hosts")).hosts || []).filter(h => h.online); } catch { hosts.value = []; }
  if (target.value && !hosts.value.some(h => h.id === target.value)) target.value = "";
}

/** Downloads a file or folder of a repo onto this PC, or onto a worker when `host` is given. */
export async function download(repo, option, host = "") {
  if (host) await api("/api/hosts/download", { host, repo, option });
  else await api("/api/hub/download", { repo, option });
  if (host && !remote.has(host)) remote.set(host, []);
  readHub();
}
export const cancel = (id, host = "") => (host ? api("/api/hosts/cancel", { host, id }) : api("/api/hub/cancel", { id })).finally(readHub);
export const forget = name => api("/api/hub/delete", { name }).finally(() => { readHub(); loadModels(true); });

export const fmtBytes = n => (n >= 1e9 ? (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + " GB" : n >= 1e6 ? Math.round(n / 1e6) + " MB" : n >= 1e3 ? Math.round(n / 1e3) + " kB" : (n || 0) + " B");
export const fmtWait = s => (!isFinite(s) || s <= 0 ? "" : s < 90 ? Math.round(s) + " s left" : s < 5400 ? Math.round(s / 60) + " min left" : (s / 3600).toFixed(1) + " h left");
export const fmtCount = n => (n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? Math.round(n / 1e3) + "k" : String(n || 0));
