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
