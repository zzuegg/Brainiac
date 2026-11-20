import { signal, batch } from "/v2/lib/html.js";
import { api, stream } from "/v2/lib/api.js";
import { lastChatModel } from "/v2/state/models.js";
import { fail } from "/v2/state/toasts.js";

// Open conversations, each a session that outlives its screen: an answer keeps streaming while
// you look at another project, and the screen shows it again where it is when you come back.
// A conversation is saved whole to the studio after each change, as the classic page does,
// so both pages show the same thing.

const sessions = new Map();

/** The session of a conversation, loading it the first time. */
export function session(id, project) {
  let s = sessions.get(id);
  if (!s) {
    s = { id, project, chat: signal(null), busy: signal(false), missing: signal(false), ctrl: null, run: null };
    sessions.set(id, s);
    reload(s);
  }
  return s;
}

/** Reads the conversation again, unless an answer is coming in. */
export async function reload(s) {
  if (s.busy.value) return;
  try {
    const r = await api("/api/chats/get", { id: s.id }, { project: s.project });
    if (!s.busy.value) batch(() => { s.chat.value = r.chat; s.missing.value = false; });
  } catch (e) {
    if (e.status === 404) s.missing.value = true;
    else fail(e);
  }
}

const set = (s, changes) => { s.chat.value = { ...s.chat.value, ...changes }; };
const setLast = (s, changes) => {
  const ms = s.chat.value.messages;
  set(s, { messages: [...ms.slice(0, -1), { ...ms.at(-1), ...changes }] });
};

let saving = Promise.resolve();
/** Saves the conversation as it is now. */
export function save(s) {
  const c = s.chat.value;
  const body = { ...c, messages: c.messages.map(({ streaming, ...m }) => m) };
  saving = saving.then(() => api("/api/chats", body, { project: s.project })).catch(fail);
  return saving;
}

/** Changes a setting of the conversation (model, system, temperature, tools, …) and saves it. */
export function configure(s, changes) {
  const c = { ...s.chat.value };
  for (const [k, v] of Object.entries(changes)) {
    if (v == null || v === "") delete c[k];
    else c[k] = v;
  }
  s.chat.value = c;
  save(s);
}

// the turns sent to the model: answers that failed are left out, and two turns in a row from
// the same side are joined, as models expect them to alternate
function turns(messages) {
  const out = [];
  for (const m of messages) {
    if (m.role === "assistant" && (m.error || !m.text)) continue;
    const last = out.at(-1);
    if (last && last.role === m.role) {
      last.text += "\n\n" + m.text;
      if (m.images?.length) last.images = [...(last.images || []), ...m.images];
    } else out.push(m.images?.length ? { role: m.role, text: m.text, images: m.images } : { role: m.role, text: m.text });
  }
  return out;
}

const titleOf = text => {
  const t = text.replace(/\s+/g, " ").trim();
  return t.length > 60 ? t.slice(0, 57).replace(/\s+\S*$/, "") + "…" : t;
};

/**
 * Sends a message and has the model answer.
 * @param {object} s the session
 * @param {string} text
 * @param {{images?: object[], replaceFrom?: number}} [opts] replaceFrom: an edited message takes
 *   the place of the one at that index, and what came after it goes
 */
export function send(s, text, { images = [], replaceFrom } = {}) {
  if (s.busy.value || (!text.trim() && !images.length)) return;
  const c = s.chat.value;
  const kept = replaceFrom != null ? c.messages.slice(0, replaceFrom) : c.messages;
  const msg = { role: "user", text: text.trim(), at: Date.now(), ...(images.length ? { images } : {}) };
  const changes = { messages: [...kept, msg], updated: Date.now() };
  if (!c.messages.length && (!c.title || c.title === "New chat") && text.trim()) changes.title = titleOf(text);
  if (!c.model) changes.model = lastChatModel.value;
  set(s, changes);
  return answer(s);
}

/** Answers the last message again, in place of the answer it had. */
export function regenerate(s) {
  if (s.busy.value) return;
  const ms = s.chat.value.messages;
  let end = ms.length;
  while (end > 0 && ms[end - 1].role === "assistant") end--;
  if (!end) return;
  set(s, { messages: ms.slice(0, end) });
  return answer(s);
}

export function stop(s) {
  if (s.ctrl) s.ctrl.abort();
  if (s.run) api("/api/agents/stop", { id: s.run }, { project: s.project }).catch(() => {});
}

async function answer(s) {
  const c = s.chat.value;
  const history = turns(c.messages);
  const a = { role: "assistant", text: "", thinking: "", model: null, streaming: true, at: Date.now() };
  batch(() => {
    set(s, { messages: [...c.messages, a] });
    s.busy.value = true;
  });
  save(s);
  try {
    if (c.tools) await agentAnswer(s, c, history);
    else await streamAnswer(s, c, history);
  } catch (e) {
    if (e.name === "AbortError") setLast(s, { stopped: true });
    else setLast(s, { error: e.message });
  }
  batch(() => {
    setLast(s, { streaming: false, thinking: s.chat.value.messages.at(-1).thinking || undefined });
    set(s, { updated: Date.now() });
    s.busy.value = false;
  });
  s.ctrl = null;
  s.run = null;
  await save(s);
}

async function streamAnswer(s, c, history) {
  s.ctrl = new AbortController();
  let text = "", thinking = "", frame = 0;
  // pieces come faster than the screen draws; they are shown once a frame
  const flush = () => { frame = 0; setLast(s, { text, thinking }); };
  await stream("/api/chat", {
    model: c.model || "auto",
    messages: history,
    system: (c.system || "").trim() || undefined,
    maxTokens: c.maxTokens > 0 ? c.maxTokens : undefined,
    temperature: c.temperature ?? undefined,
    topP: c.topP ?? undefined,
  }, {
    project: s.project,
    signal: s.ctrl.signal,
    onEvent(event, d) {
      if (event === "route") setLast(s, { model: d });
      else if (event === "text") text += d.text;
      else if (event === "thinking") thinking += d.text;
      else if (event === "done") setLast(s, { stats: d });
      else if (event === "error") setLast(s, { error: d.message });
      if ((event === "text" || event === "thinking") && !frame) frame = requestAnimationFrame(flush);
    },
  }).finally(() => {
    if (frame) cancelAnimationFrame(frame);
    flush();
  });
}

// With tools on, an agent answers: the same tools, rules and approvals as an agent of its own.
// A conversation keeps one agent for good: the next message carries on with it, with the chat's
// model and tool mode as they are now. It shows in the chat, not among the agents.
async function agentAnswer(s, c, history) {
  const last = history.at(-1);
  const mode = c.tools, model = c.model || "auto";
  let id = c.agentRun, from = 0;
  let old = id ? await api("/api/agents/get", { id }, { project: s.project }).catch(() => null) : null;
  if (old && ["running", "waiting"].includes(old.state)) {
    // left working or waiting (another window, a reload): this message takes over from it
    await api("/api/agents/stop", { id }, { project: s.project }).catch(() => {});
    for (let i = 0; i < 20 && old && ["running", "waiting"].includes(old.state); i++) {
      await new Promise(res => setTimeout(res, 250));
      old = await api("/api/agents/get", { id }, { project: s.project }).catch(() => null);
    }
  }
  if (old && !["running", "waiting"].includes(old.state)) {
    from = (old.steps || []).length;
    await api("/api/agents/reply", { id, text: last.text, mode, model }, { project: s.project });
  } else {
    const earlier = history.slice(0, -1).filter(m => m.text);
    const goal = ((c.system || "").trim() ? `Instructions for this chat: ${c.system.trim()}\n\n` : "")
      + (earlier.length ? "Earlier in this chat:\n\n" + earlier.map(m => (m.role === "user" ? "User: " : "You: ") + m.text).join("\n\n")
        + "\n\nNow the user says:\n\n" : "")
      + last.text;
    const r = await api("/api/agents/start", { goal, model, mode, maxSteps: 30, chat: c.id }, { project: s.project });
    id = r.id;
    set(s, { agentRun: id });
  }
  s.run = id;
  setLast(s, { agent: { id, state: "running", steps: [] } });
  let run;
  for (;;) {
    await new Promise(res => setTimeout(res, 600));
    run = await api("/api/agents/get", { id }, { project: s.project });
    setLast(s, {
      model: run.by ? { key: run.by, name: run.by } : null,
      thinking: run.state === "running" ? run.thinking || "" : "",
      agent: { id, state: run.state, pending: run.pending || null, progress: run.progress || null, doing: run.doing || null,
        steps: (run.steps || []).slice(from).filter(x => x.kind !== "goal" && x.kind !== "answer") },
    });
    if (!["running", "waiting"].includes(run.state)) break;
  }
  setLast(s, {
    text: run.answer || "",
    stats: { inputTokens: run.inputTokens || 0, outputTokens: run.outputTokens || 0 },
    ...(run.state === "failed" ? { error: run.error || "the agent failed" } : {}),
    ...(run.state === "stopped" ? { stopped: true } : {}),
  });
}
