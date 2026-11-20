import { batch } from "/v2/lib/html.js";
import { api, stream } from "/v2/lib/api.js";
import { modelInfo, modelByKey, loadModels } from "/v2/state/models.js";
import { openProject, refreshAll } from "/v2/state/workspace.js";
import { save } from "/v2/state/chats.js";
import { fail, toast } from "/v2/state/toasts.js";

// Image, speech and video chats: a conversation whose turns are a prompt with its settings and
// what the model made. They keep the same file as a chat (with `medium` set), so they sit with
// the chats, and rename, move and delete the same way. A picture an agent drew anywhere opens
// in one with Fine tune: same prompt, model and seed, ready to change.

export const MEDIA = {
  image: { label: "Image chat", noun: "picture", verb: "Draw", doing: "Drawing", icon: "image", modelMedia: "image",
    lead: "Describe a picture. Each one keeps its seed, so a good one can be drawn again with small changes." },
  speech: { label: "Speech chat", noun: "recording", verb: "Speak", doing: "Speaking", icon: "speech", modelMedia: "speech",
    lead: "Type what should be said. Pick a voice and a model below." },
  video: { label: "Video chat", noun: "clip", verb: "Film", doing: "Filming", icon: "video", modelMedia: "video",
    lead: "Describe a short clip. It takes a while on this PC; the steps show as it goes." },
};

/** The models that make a medium. */
export const modelsFor = medium => (modelInfo.value?.models || []).filter(m => (m.media || []).includes(MEDIA[medium].modelMedia));

export const defaults = medium => ({
  model: "auto", aspect: medium === "video" ? "16:9" : "1:1", count: 1, steps: "", guidance: "", seed: "", negativePrompt: "",
  voice: "", blend: "", mix: 0.7, format: "", seconds: 2, speed: 1, instructions: "",
});

/** What a speech model offers: its voices, and whether it takes a speed and instructions. */
export function speechOptions(key) {
  const k = key === "auto" ? modelInfo.value?.auto?.speech : key;
  return { voices: [], names: {}, speed: false, instructions: false, mix: false, clone: false, tags: [], formats: [], ...(modelByKey(k)?.speech || {}) };
}

/** A voice's name and what it is like, from "Heart (American female)" or its id. */
export function voiceName(v, names = {}) {
  if (!v) return { name: "", about: "" };
  const full = names[v], hit = full && /^(.*?)\s*\((.*)\)$/.exec(full);
  return hit ? { name: hit[1], about: hit[2] } : { name: full || v[0].toUpperCase() + v.slice(1), about: "" };
}

/** What is sent as the voice: the voice, or a blend such as af_heart*0.70+am_adam*0.30. */
export const spokenVoice = o => (o.blend && o.voice && o.blend !== o.voice
  ? `${o.voice}*${(+o.mix || 0.7).toFixed(2)}+${o.blend}*${(1 - (+o.mix || 0.7)).toFixed(2)}` : o.voice || "");

/** Speaks a short sentence in a voice, to hear it before choosing it; resolves to the file. */
export async function preview(project, model, voice, name) {
  const r = await api("/api/speech", { model: model || "auto", text: `Hello, I'm ${name}. This is how I sound.`, voice, format: "wav" }, { project });
  return r.items?.[0]?.file;
}

/** Makes a voice from 3 to 12 seconds of speech; resolves to its id. */
export async function addVoice(project, model, name, wav, transcript) {
  const r = await api("/api/speech/voice", { model, name, audio: wav, transcript }, { project });
  await loadModels(true);
  return r.voice;
}

const chatId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
const titleOf = t => { t = t.replace(/\s+/g, " ").trim(); return t.length > 60 ? t.slice(0, 57).replace(/\s+\S*$/, "") + "…" : t; };

/** Turns an empty chat into an image, speech or video chat. */
export function makeMedium(s, medium) {
  const c = s.chat.value;
  s.chat.value = { ...c, medium, settings: { ...defaults(medium), ...(c.settings || {}) }, title: c.title === "New chat" ? MEDIA[medium].label : c.title };
  loadModels();
  return save(s);
}

/**
 * Opens what an agent made in a new image chat of its project, with its prompt, model and seed
 * set, and the picture itself as the first turn, so it can be drawn again with changes.
 */
export async function fineTune(project, it) {
  try {
    const id = chatId();
    const medium = it.kind === "audio" ? "speech" : it.kind === "video" ? "video" : "image";
    const settings = { ...defaults(medium), model: it.model || "auto", seed: it.seed ?? "", ...(it.aspect ? { aspect: it.aspect } : {}),
      ...(it.voice ? { voice: it.voice } : {}), ...(it.speed ? { speed: it.speed } : {}), ...(it.instructions ? { instructions: it.instructions } : {}) };
    const at = Date.now();
    await api("/api/chats", {
      id, title: "Fine tune: " + titleOf(it.prompt || MEDIA[medium].noun), medium, settings, prompt: it.prompt || "",
      messages: [
        { role: "user", text: it.prompt || "", settings, at, from: "agent" },
        { role: "assistant", media: [{ file: it.file, seed: it.seed, kind: it.kind }], model: it.model ? { key: it.model, name: modelByKey(it.model)?.name || it.model } : null, at },
      ],
      status: "open", created: at, updated: at,
    }, { project });
    await refreshAll();
    openProject(project, { route: { screen: "chat", id } });
    toast(`Opened in a ${MEDIA[medium].label.toLowerCase()}: change the ${medium === "speech" ? "text, voice" : "prompt"} or settings and ${MEDIA[medium].verb.toLowerCase()} again`);
  } catch (e) { fail(e); }
}

// ------------------------------------------------------------------ making

const setLast = (s, changes) => {
  const ms = s.chat.value.messages;
  s.chat.value = { ...s.chat.value, messages: [...ms.slice(0, -1), { ...ms.at(-1), ...changes }] };
};
const numberOr = (v, d) => (v === "" || v == null || isNaN(+v) ? d : +v);

/** Makes what the prompt asks for with the chat's settings, as a new turn. */
export async function generate(s, prompt) {
  const c = s.chat.value, medium = c.medium;
  if (s.busy.value || !prompt.trim()) return;
  const settings = { ...defaults(medium), ...(c.settings || {}) };
  const at = Date.now();
  batch(() => {
    s.chat.value = {
      ...c, prompt: "", updated: at, title: !c.messages.length && Object.values(MEDIA).some(m => m.label === c.title) ? titleOf(prompt) : c.title,
      messages: [...c.messages, { role: "user", text: prompt.trim(), settings, at }, { role: "assistant", media: [], streaming: true, at }],
    };
    s.busy.value = true;
  });
  s.ctrl = new AbortController();
  const started = performance.now();
  try {
    if (medium === "image") await image(s, prompt.trim(), settings);
    else if (medium === "video") await video(s, prompt.trim(), settings);
    else await speech(s, prompt.trim(), settings);
    setLast(s, { seconds: (performance.now() - started) / 1000 });
  } catch (e) {
    if (e.name === "AbortError" || e.message === "stopped") setLast(s, { stopped: true });
    else setLast(s, { error: e.message });
  }
  batch(() => {
    setLast(s, { streaming: false, progress: undefined });
    s.busy.value = false;
  });
  s.ctrl = null;
  await save(s);
}

export function stopMaking(s) {
  s.ctrl?.abort();
}

const add = (s, item) => setLast(s, { media: [...(s.chat.value.messages.at(-1).media || []), item] });

// on this PC (or a worker) step by step with previews; elsewhere one call to the provider
async function image(s, prompt, o) {
  const body = { model: o.model || "auto", prompt, negativePrompt: o.negativePrompt || "", aspect: o.aspect, count: +o.count || 1,
    ...(o.steps !== "" ? { steps: numberOr(o.steps, 25) } : {}), ...(o.guidance !== "" ? { guidance: numberOr(o.guidance, 7) } : {}),
    ...(o.seed !== "" && o.seed != null ? { seed: numberOr(o.seed, undefined) } : {}) };
  let failed = null;
  try {
    await stream("/api/images/draw", body, {
      project: s.project, signal: s.ctrl.signal,
      onEvent(event, d) {
        if (event === "route") setLast(s, { model: { key: d.key, name: d.name } });
        else if (event === "step") setLast(s, { progress: { ...(s.chat.value.messages.at(-1).progress || {}), step: d.step, total: d.total } });
        else if (event === "preview") setLast(s, { progress: { ...(s.chat.value.messages.at(-1).progress || {}), preview: "data:image/png;base64," + d.data } });
        else if (event === "image") add(s, { file: d.file, seed: d.seed, kind: "image" });
        else if (event === "error") failed = d.message;
      },
    });
  } catch (e) {
    if (!/does not run on this PC/.test(e.message)) throw e;
    const r = await api("/api/images", { model: body.model, prompt, aspect: body.aspect, count: body.count }, { project: s.project, signal: s.ctrl.signal });
    setLast(s, { model: r.model, media: (r.items || []).map(it => ({ file: it.file, kind: "image" })) });
    return;
  }
  if (failed) throw new Error(failed);
}

async function video(s, prompt, o) {
  const body = { model: o.model || "auto", prompt, negativePrompt: o.negativePrompt || "", aspect: o.aspect, seconds: numberOr(o.seconds, 2),
    ...(o.steps !== "" ? { steps: numberOr(o.steps, 30) } : {}), ...(o.guidance !== "" ? { guidance: numberOr(o.guidance, 5) } : {}),
    ...(o.seed !== "" && o.seed != null ? { seed: numberOr(o.seed, undefined) } : {}) };
  let failed = null;
  await stream("/api/video", body, {
    project: s.project, signal: s.ctrl.signal,
    onEvent(event, d) {
      if (event === "route") setLast(s, { model: { key: d.key, name: d.name } });
      else if (event === "step") setLast(s, { progress: { step: d.step, total: d.total } });
      else if (event === "video") add(s, { file: d.file, seed: d.seed, kind: "video" });
      else if (event === "error") failed = d.message;
    },
  });
  if (failed) throw new Error(failed);
}

async function speech(s, text, o) {
  const opt = speechOptions(o.model || "auto");
  const format = opt.formats?.includes(o.format) ? o.format : opt.formats?.[0] || "wav";
  const r = await api("/api/speech", { model: o.model || "auto", text, voice: spokenVoice(o), format,
    speed: numberOr(o.speed, 1), instructions: o.instructions || "" }, { project: s.project, signal: s.ctrl.signal });
  setLast(s, { model: r.model, media: (r.items || []).map(it => ({ file: it.file, kind: "audio" })) });
}
