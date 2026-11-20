// Calls to the studio's API. Every call carries the session token, and the project it belongs
// to, so each tab works in its own project without a server-wide "active" one.

const token = document.querySelector('meta[name="studio-token"]')?.content ?? "";

/** An API call that failed; `message` is the studio's own explanation. */
export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

/**
 * Calls the API: a GET without a body, a POST with one.
 * @param {string} path such as "/api/items"
 * @param {object} [body] sent as JSON
 * @param {{project?: string, signal?: AbortSignal}} [opts]
 */
export async function api(path, body, { project, signal } = {}) {
  const headers = { "X-Studio-Token": token };
  if (project) headers["X-Studio-Project"] = project;
  const init = { headers, signal };
  if (body !== undefined) {
    init.method = "POST";
    headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(body);
  }
  let r;
  try {
    r = await fetch(path, init);
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError(0, "The studio is not answering. Is it still running?");
  }
  const text = await r.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    data = { error: text };
  }
  if (!r.ok) throw new ApiError(r.status, data.error || `The studio answered ${r.status}`);
  return data;
}

/** What the studio remembers for the page (shared by every browser on this PC). */
export const pageState = (() => {
  try {
    return JSON.parse(document.getElementById("studio-state")?.textContent || "{}");
  } catch {
    return {};
  }
})();

const pending = new Map();
/** Remembers a value in the studio, a moment after the last change. */
export function remember(name, value) {
  clearTimeout(pending.get(name));
  pending.set(name, setTimeout(() => {
    pending.delete(name);
    api("/api/state", { [name]: value }).catch(e => console.warn("cannot remember", name, e));
  }, 600));
}

/** The browser's own storage, which may be missing or full; never throws. */
export const local = {
  get(name, otherwise) {
    try {
      const v = localStorage.getItem("studio.v2." + name);
      return v == null ? otherwise : JSON.parse(v);
    } catch {
      return otherwise;
    }
  },
  set(name, value) {
    try {
      localStorage.setItem("studio.v2." + name, JSON.stringify(value));
    } catch { /* private window or full: the studio's copy still holds */ }
  },
};

/**
 * Calls an API that answers with server-sent events, handing each event to `onEvent(name, data)`.
 * Aborting the signal closes the stream, which stops the work in the studio.
 */
export async function stream(path, body, { project, signal, onEvent }) {
  const headers = { "X-Studio-Token": token, "Content-Type": "application/json" };
  if (project) headers["X-Studio-Project"] = project;
  let r;
  try {
    r = await fetch(path, { method: "POST", headers, body: JSON.stringify(body), signal });
  } catch (e) {
    if (e.name === "AbortError") throw e;
    throw new ApiError(0, "The studio is not answering. Is it still running?");
  }
  if (!r.ok) {
    const data = await r.json().catch(() => ({}));
    throw new ApiError(r.status, data.error || `The studio answered ${r.status}`);
  }
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let cut;
    while ((cut = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      let event = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (data) onEvent(event, JSON.parse(data));
    }
  }
}
