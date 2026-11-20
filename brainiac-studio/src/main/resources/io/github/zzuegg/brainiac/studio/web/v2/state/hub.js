import { signal } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { loadModels } from "/v2/state/models.js";

// Models fetched from Hugging Face onto this PC: what is downloading, the token the downloader
// sends, and where the files go. Read while a download runs, and when the Models page opens.

export const hub = signal({ downloads: [], token: null, folder: "", runs: null });
export const active = d => d.state === "queued" || d.state === "downloading" || d.state === "checking";

let timer = 0, finished = new Set();
export async function readHub() {
  try {
    const r = await api("/api/hub/downloads");
    // a finished download is a new model: the lists learn of it
    const done = (r.downloads || []).filter(d => d.state === "done").map(d => d.id);
    if (done.some(id => !finished.has(id)) && finished.size + done.length > done.length) loadModels(true);
    finished = new Set(done);
    hub.value = { downloads: r.downloads || [], token: r.token || null, folder: r.folder || "", runs: r.runs || null };
  } catch { /* the page says the studio is not answering */ }
  clearTimeout(timer);
  if (hub.value.downloads.some(active)) timer = setTimeout(readHub, 1200);
}

export async function download(repo, option) {
  await api("/api/hub/download", { repo, option });
  readHub();
}
export const cancel = id => api("/api/hub/cancel", { id }).finally(readHub);
export const forget = name => api("/api/hub/delete", { name }).finally(() => { readHub(); loadModels(true); });

export const fmtBytes = n => (n >= 1e9 ? (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + " GB" : n >= 1e6 ? Math.round(n / 1e6) + " MB" : n >= 1e3 ? Math.round(n / 1e3) + " kB" : (n || 0) + " B");
export const fmtWait = s => (!isFinite(s) || s <= 0 ? "" : s < 90 ? Math.round(s) + " s left" : s < 5400 ? Math.round(s / 60) + " min left" : (s / 3600).toFixed(1) + " h left");
export const fmtCount = n => (n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : n >= 1e3 ? Math.round(n / 1e3) + "k" : String(n || 0));
