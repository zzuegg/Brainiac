import { signal, computed } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { toast, fail } from "/v2/state/toasts.js";

// What agents in subprojects asked of the projects theirs is part of: questions, a tool or
// model they need, or a result they hand up. Each project says whether its own agents may
// answer; the user always can. Shared by the overview, the project homes and the tab badges.

export const requests = signal([]);

export async function loadRequests() {
  try {
    const r = await api("/api/requests");
    requests.value = r.requests || [];
  } catch {
    // the status bar says when the studio is not answering
  }
}

export const openRequests = computed(() => requests.value.filter(q => q.state === "open"));

/** Open requests sent to a project. */
export const openTo = project => openRequests.value.filter(q => (q.to || []).includes(project));

/** Open requests that wait for the user, because no project they went to lets its agents answer. */
export const forUser = computed(() => openRequests.value.filter(q => q.answeredBy === "user"));

/**
 * Answers a request as the user.
 * @param {object} q the request
 * @param {{answer?: string, approve?: boolean}} body
 */
export async function answerRequest(q, body) {
  try {
    await api("/api/requests/answer", { id: q.id, ...body });
    toast(q.kind === "tool" || q.kind === "model"
      ? `${body.approve ? "Gave" : "Refused"} ${q.subject} to ${q.projectName}`
      : `Answered ${q.projectName}`);
    await loadRequests();
    return true;
  } catch (e) {
    fail(e);
    await loadRequests();
    return false;
  }
}

export async function withdrawRequest(q) {
  try {
    await api("/api/requests/withdraw", { id: q.id });
    toast("Request withdrawn; the agent hears so");
    await loadRequests();
  } catch (e) {
    fail(e);
  }
}

loadRequests();
setInterval(() => { if (!document.hidden) loadRequests(); }, 5000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) loadRequests(); });
