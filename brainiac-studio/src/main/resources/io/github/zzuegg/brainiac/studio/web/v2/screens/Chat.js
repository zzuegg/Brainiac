import { html, useEffect, useLayoutEffect, useRef, useState, cx } from "/v2/lib/html.js";
import { Reading, Empty } from "/v2/ui/Reading.js";
import { Text } from "/v2/ui/Text.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button, IconAction } from "/v2/ui/Button.js";
import { Pending, stepLine } from "/v2/ui/Pending.js";
import { ModelPicker } from "/v2/ui/ModelPicker.js";
import { ItemHeader, routeItem, Gone } from "/v2/screens/ItemHeader.js";
import { openLayer } from "/v2/ui/overlay.js";
import { ChatSettings } from "/v2/wizards/ChatSettings.js";
import { plural } from "/v2/lib/format.js";
import { session, reload, send, regenerate, stop, configure, setTools } from "/v2/state/chats.js";
import { modelByKey, autoModel, loadModels, chatModels, modelInfo, rememberChatModel } from "/v2/state/models.js";
import { MediaGrid, mediaOf, Working } from "/v2/ui/Media.js";
import { fineTune, MEDIA, makeMedium } from "/v2/state/media.js";
import { MediaChat } from "/v2/screens/MediaChat.js";
import { toast, fail } from "/v2/state/toasts.js";
import { navigate } from "/v2/state/workspace.js";

// A conversation in the reading column: the transcript, and a composer that stays at the
// bottom with the model, tools and pictures. Answers stream in; one can be stopped, answered
// again, or a message edited and sent anew.

const TOOLS = [
  { value: "", label: "No tools", text: "The model only answers." },
  { value: "ask", label: "Tools, ask first", text: "An agent answers and asks before it changes anything." },
  { value: "auto", label: "Tools, automatic", text: "An agent answers and acts within the project's rules." },
];

export function Chat({ tab }) {
  const item = routeItem(tab);
  const id = tab.route.value.id;
  const s = session(id, tab.project);
  const chat = s.chat.value;
  const busy = s.busy.value;
  const scroll = useRef(null);
  const follow = useRef(true);
  const [away, setAway] = useState(false);
  const [editing, setEditing] = useState(null);
  useEffect(() => { loadModels(); }, []);

  // the list's copy changed (another page, a rename): read the conversation again
  const updated = item ? item.updated : 0;
  useEffect(() => { if (chat && updated > (chat.updated || 0)) reload(s); }, [updated]);

  // follow the answer as it comes in, unless the reader scrolled up
  useLayoutEffect(() => {
    const el = scroll.current;
    if (el && follow.current) el.scrollTop = el.scrollHeight;
  });
  useLayoutEffect(() => {
    const el = scroll.current;
    if (!el || !chat) return;
    const at = tab.scroll.get("chat:" + id);
    follow.current = at == null;
    el.scrollTop = at ?? el.scrollHeight;
  }, [id, !!chat]);
  const onScroll = e => {
    const el = e.currentTarget;
    const bottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    follow.current = bottom;
    setAway(!bottom);
    tab.scroll.set("chat:" + id, bottom ? null : el.scrollTop);
  };
  const toBottom = () => {
    follow.current = true;
    setAway(false);
    scroll.current?.scrollTo({ top: scroll.current.scrollHeight, behavior: "smooth" });
  };

  if (item === false || s.missing.value) return html`<${Gone} tab=${tab} what="chat" />`;
  if (!item || !chat) return html`<div class="loading" aria-busy="true" />`;
  if (chat.medium && MEDIA[chat.medium]) return html`<${MediaChat} tab=${tab} item=${item} s=${s} />`;
  const messages = chat.messages || [];
  const model = chat.model || "auto";
  const m = model === "auto" ? autoModel.value : modelByKey(model);

  const header = html`<${ItemHeader} tab=${tab} item=${item} sub=${html`${plural(messages.filter(x => x.role !== "note").length, "message")}
      ${chat.tools && html` · <span class="badge s-running">tools ${chat.tools === "auto" ? "automatic" : "ask first"}</span>`}`}>
    <${Button} variant="ghost" icon="settings" label="Chat settings" onClick=${() => openLayer(ChatSettings, { s })} />
  <//>`;

  const footer = html`<${Composer} key=${id} s=${s} busy=${busy} model=${model} editing=${editing}
    onModel=${k => { configure(s, { model: k }); rememberChatModel(k); }}
    onTools=${v => setTools(s, v)}
    cancelEdit=${() => setEditing(null)}
    onSend=${(text, images) => {
      follow.current = true;
      send(s, text, { images, replaceFrom: editing?.at });
      setEditing(null);
    }} />`;

  const last = messages.length - 1;
  return html`<${Reading} header=${header} footer=${footer} scrollRef=${scroll} onScroll=${onScroll} class="chat">
    ${messages.length === 0
      ? html`<${Welcome} tab=${tab} model=${m} auto=${model === "auto"} s=${s} />`
      : html`<ol class="transcript">${messages.map((msg, i) => msg.role === "note"
          ? html`<li key=${i} class="msg-note" role="separator"><span>${msg.tools ? `Tools on from here, ${msg.tools === "auto" ? "automatic" : "ask first"}` : "Tools off from here"}</span></li>`
          : msg.role === "user"
          ? html`<${UserMsg} key=${i} msg=${msg} editing=${editing?.at === i}
              edit=${busy ? null : () => setEditing({ at: i, text: msg.text, images: msg.images || [] })} />`
          : html`<${AnswerMsg} key=${i} msg=${msg} s=${s} last=${i === last} busy=${busy} />`)}</ol>`}
    ${away && html`<button type="button" class="jump" onClick=${toBottom}><${Icon} name="chevronDown" size=${14} />Latest</button>`}
  <//>`;
}

function Welcome({ tab, model, auto, s }) {
  const none = modelInfo.value && !chatModels.value.length;
  return html`<${Empty} title=${none ? "No chat model yet" : "What should we work on?"} icon=${html`<${Icon} name="chat" size=${28} />`}>
    ${none
      ? html`Download one, add a folder on this PC, or add a key for a provider on the <button type="button" class="link" onClick=${() => navigate(tab, { screen: "models" })}>Models</button> page.`
      : model ? `${auto ? "Auto sends this to" : "This goes to"} ${model.name}. Pick another model below the message box.` : ""}
    <div class="medium-picks"><span class="muted small">Or make this</span>
      ${Object.entries(MEDIA).map(([k, M]) => html`<button type="button" class="btn btn-sm" onClick=${() => makeMedium(s, k)}><${Icon} name=${M.icon} /><span>${M.label.toLowerCase()}</span></button>`)}</div>
  <//>`;
}

const copy = text => navigator.clipboard?.writeText(text).then(() => toast("Copied"), fail);

function UserMsg({ msg, edit, editing }) {
  return html`<li class=${cx("msg msg-user", editing && "msg-editing")}>
    <div class="msg-text"><${Text} text=${msg.text || ""} /></div>
    ${msg.images?.length > 0 && html`<div class="msg-pics">${msg.images.map((p, i) =>
      html`<img key=${i} src=${"data:" + p.type + ";base64," + p.data} alt=${"Picture " + (i + 1)} />`)}</div>`}
    <div class="msg-acts">
      <${IconAction} icon="copy" label="Copy" onClick=${() => copy(msg.text || "")} />
      ${edit && html`<${IconAction} icon="edit" label="Edit and send again" onClick=${edit} />`}
    </div>
  </li>`;
}

function AnswerMsg({ msg, s, last, busy }) {
  const by = msg.model ? modelByKey(msg.model.key)?.name || msg.model.name || msg.model.key : null;
  const st = msg.stats;
  const g = msg.agent;
  const steps = (g?.steps || []).filter(x => ["call", "result", "decision", "plan", "error", "say", "spawn", "request"].includes(x.kind));
  const live = msg.streaming;
  return html`<li class=${cx("msg msg-assistant", live && "msg-live")}>
    <div class="msg-who">${by || "Assistant"}
      ${st && !live && (st.outputTokens > 0) && html`<span class="msg-stats">${plural(st.outputTokens, "token")}${st.seconds ? ` · ${(st.outputTokens / st.seconds).toFixed(1)}/s` : ""}</span>`}
    </div>
    ${msg.thinking && html`<details class="thinking" open=${live && !msg.text}><summary>${live && !msg.text ? "Thinking…" : "Reasoning"}</summary><${Text} text=${msg.thinking} /></details>`}
    ${steps.length > 0 && html`<details class="activity" open=${live}>
      <summary>${plural(steps.length, "step")}${live ? " so far" : ""}</summary>
      <ol class="activity-list">${steps.map((x, i) => {
        const l = stepLine(x);
        return html`<li key=${i}><${Icon} name=${l.icon} size=${13} /><span>${l.text}</span>${l.detail && html`<span class="muted mono small">${l.detail}</span>`}</li>`;
      })}</ol></details>`}
    <${MediaGrid} items=${mediaOf(g?.steps)} onFineTune=${it => fineTune(s.project, it)} />
    ${g?.state === "running" && live && html`<${Working} progress=${g.progress} doing=${g.doing} thinking=${!!msg.thinking && !msg.text} />`}
    ${g?.state === "waiting" && g.pending && html`<${Pending} id=${g.id} project=${s.project} pending=${g.pending} />`}
    ${(msg.text || !live) && html`<div class="msg-text"><${Text} text=${msg.text || ""} />${live && html`<span class="caret" />`}</div>`}
    ${live && !msg.text && !msg.thinking && !g && html`<span class="typing" aria-label="Waiting for the answer"><i /><i /><i /></span>`}
    ${msg.error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${msg.error}</p>`}
    ${msg.stopped && html`<p class="muted small">Stopped.</p>`}
    ${!live && html`<div class="msg-acts">
      <${IconAction} icon="copy" label="Copy" onClick=${() => copy(msg.text || "")} />
      ${last && !busy && html`<${IconAction} icon="undo" label="Answer again" onClick=${() => regenerate(s)} />`}
    </div>`}
  </li>`;
}

const MAX_PICTURE = 8 << 20;

function Composer({ s, busy, model, editing, onSend, onModel, onTools, cancelEdit }) {
  const ref = useRef(null);
  const file = useRef(null);
  const [text, setText] = useState("");
  const [pics, setPics] = useState([]);
  const tools = s.chat.value.tools || "";

  useEffect(() => {
    if (!editing) return;
    setText(editing.text);
    setPics(editing.images);
    ref.current?.focus();
  }, [editing]);
  // the box grows with the text, up to a third of the window
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, innerHeight / 3) + "px";
  }, [text]);
  useEffect(() => { ref.current?.focus(); }, []);

  const addFiles = async files => {
    for (const f of files) {
      if (!f.type.startsWith("image/")) continue;
      if (f.size > MAX_PICTURE) { fail(`${f.name} is larger than 8 MB.`); continue; }
      const data = await new Promise(res => {
        const r = new FileReader();
        r.onload = () => res(String(r.result).split(",")[1]);
        r.readAsDataURL(f);
      });
      setPics(p => [...p, { type: f.type, data, name: f.name }].slice(0, 8));
    }
  };
  const go = () => {
    if (busy || (!text.trim() && !pics.length)) return;
    onSend(text, pics.map(({ type, data }) => ({ type, data })));
    setText("");
    setPics([]);
  };
  return html`<div class="composer" onDragOver=${e => e.preventDefault()} onDrop=${e => { e.preventDefault(); addFiles(e.dataTransfer.files); }}>
    ${editing && html`<div class="composer-note"><${Icon} name="edit" size=${14} />Editing a message: sending it replaces the answers after it.
      <button type="button" class="link" onClick=${() => { cancelEdit(); setText(""); setPics([]); }}>Cancel</button></div>`}
    <div class="composer-box">
      ${pics.length > 0 && html`<div class="composer-pics">${pics.map((p, i) => html`<span class="pic" key=${i}>
        <img src=${"data:" + p.type + ";base64," + p.data} alt=${p.name || "Picture"} />
        <button type="button" class="pic-x" aria-label=${"Remove " + (p.name || "picture")} onClick=${() => setPics(ps => ps.filter((_, j) => j !== i))}><${Icon} name="close" size=${11} /></button>
      </span>`)}</div>`}
      <textarea ref=${ref} class="composer-input" rows="1" value=${text} aria-label="Message"
        placeholder=${busy ? "The answer is coming in…" : "Message"}
        onInput=${e => setText(e.target.value)}
        onPaste=${e => { const f = [...(e.clipboardData?.files || [])]; if (f.length) { e.preventDefault(); addFiles(f); } }}
        onKeyDown=${e => {
          if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); go(); }
          else if (e.key === "Escape" && busy) { e.preventDefault(); stop(s); }
          else if (e.key === "Escape" && editing) { e.preventDefault(); cancelEdit(); setText(""); setPics([]); }
        }} />
      <div class="composer-bar">
        <${ModelPicker} value=${model} onChange=${onModel} disabled=${busy} />
        <select class="input select-sm" value=${tools} disabled=${busy} aria-label="Tools" title=${TOOLS.find(t => t.value === tools)?.text}
          onChange=${e => onTools(e.target.value)}>
          ${TOOLS.map(t => html`<option value=${t.value}>${t.label}</option>`)}
        </select>
        <${Button} variant="ghost" size="sm" icon="image" label="Add a picture (or paste or drop one)" onClick=${() => file.current.click()} disabled=${busy} />
        <input type="file" accept="image/*" multiple hidden ref=${file} onChange=${e => { addFiles([...e.target.files]); e.target.value = ""; }} />
        <span class="grow" />
        <span class="muted small composer-hint">${busy ? "Esc stops" : "Enter sends · Shift+Enter new line"}</span>
        ${busy
          ? html`<${Button} icon="stop" onClick=${() => stop(s)}>Stop<//>`
          : html`<${Button} variant="primary" onClick=${go} disabled=${!text.trim() && !pics.length}>${editing ? "Send again" : "Send"}<//>`}
      </div>
    </div>
  </div>`;
}
