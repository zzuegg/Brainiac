import { signal, computed } from "/v2/lib/html.js";
import { api, local } from "/v2/lib/api.js";

// The models the studio found, on this PC and at the providers with keys. Loaded once and
// shared; the chat's model picker and the agent wizard read it.

export const modelInfo = signal(null);
export const modelsError = signal(null);
let loading = null;

export function loadModels(refresh = false) {
  if (loading && !refresh) return loading;
  loading = api("/api/models" + (refresh ? "?refresh=1" : ""))
    .then(r => { modelInfo.value = r; modelsError.value = null; return r; })
    .catch(e => { modelsError.value = e.message; loading = null; });
  return loading;
}

/** Models that can chat, grouped by where they run. */
export const chatModels = computed(() => (modelInfo.value?.models || []).filter(m => (m.media || []).includes("text")));

export const modelByKey = key => (modelInfo.value?.models || []).find(m => m.key === key) || null;

/** The model Auto sends text to now, if any. */
export const autoModel = computed(() => modelByKey(modelInfo.value?.auto?.text) || null);

/** "this PC", "OpenAI", … */
export function placeOf(m) {
  if (!m) return "";
  if (m.provider === "local") return "this PC";
  const p = (modelInfo.value?.providers || []).find(x => x.id === m.provider);
  return p?.name || m.provider;
}

/** What a new chat starts with: the model last picked. */
export const lastChatModel = signal(local.get("chatModel", "auto"));
export function rememberChatModel(key) {
  lastChatModel.value = key;
  local.set("chatModel", key);
}
