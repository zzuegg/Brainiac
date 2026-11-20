import { html, useEffect, useRef, useState } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Reading, Empty } from "/v2/ui/Reading.js";
import { Text } from "/v2/ui/Text.js";
import { Icon } from "/v2/ui/Icon.js";
import { ItemHeader, routeItem, Gone } from "/v2/screens/ItemHeader.js";
import { plural } from "/v2/lib/format.js";

// A conversation in the reading column. This step shows it; talking moves here with the Chat
// step, until then the composer points to the classic page.

export function Chat({ tab }) {
  const item = routeItem(tab);
  const [chat, setChat] = useState(null);
  const scroll = useRef(null);
  const id = tab.route.value.id;
  const updated = item ? item.updated : 0;

  useEffect(() => {
    let live = true;
    api("/api/chats", undefined, { project: tab.project })
      .then(r => live && setChat((r.chats || []).find(c => c.id === id) || false))
      .catch(() => live && setChat(false));
    return () => { live = false; };
  }, [id, updated]);

  // each conversation comes back where it was left
  useEffect(() => {
    const el = scroll.current;
    if (el && chat) el.scrollTop = tab.scroll.get("chat:" + id) ?? el.scrollHeight;
  }, [chat && chat.id]);

  if (item === false) return html`<${Gone} tab=${tab} what="chat" />`;
  if (!item) return html`<div class="loading" aria-busy="true" />`;
  const messages = (chat && chat.messages) || [];
  const header = html`<${ItemHeader} tab=${tab} item=${item} sub=${plural(messages.length, "message")} />`;
  const footer = html`<div class="composer composer-off">
    <div class="composer-box"><span class="muted">Chatting in this view arrives with the next step.</span></div>
    <a class="btn" href="/" target="_blank" rel="noopener"><${Icon} name="classic" /><span>Continue in classic Studio</span></a>
  </div>`;
  return html`<${Reading} header=${header} footer=${footer} scrollRef=${scroll}
      onScroll=${e => tab.scroll.set("chat:" + id, e.currentTarget.scrollTop)}>
    ${messages.length === 0
      ? html`<${Empty} title="No messages yet" icon=${html`<${Icon} name="chat" size=${28} />`}>Start the conversation from the composer below.<//>`
      : html`<ol class="transcript">${messages.map((m, i) => html`<li key=${i} class=${"msg msg-" + m.role}>
          <div class="msg-who">${m.role === "user" ? "You" : m.model?.name || "Assistant"}</div>
          ${m.thinking && html`<details class="thinking"><summary>Reasoning</summary><${Text} text=${m.thinking} /></details>`}
          <div class="msg-text"><${Text} text=${m.text || ""} /></div>
          ${m.images?.length > 0 && html`<p class="muted small">${plural(m.images.length, "image")} attached</p>`}
        </li>`)}</ol>`}
  <//>`;
}
