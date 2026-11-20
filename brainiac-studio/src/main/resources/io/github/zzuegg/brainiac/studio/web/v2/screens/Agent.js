import { html, useEffect, useLayoutEffect, useRef, useState } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Reading } from "/v2/ui/Reading.js";
import { Button } from "/v2/ui/Button.js";
import { Thread, Reply, LIVE } from "/v2/ui/AgentThread.js";
import { ItemHeader, routeItem, Gone } from "/v2/screens/ItemHeader.js";
import { fail } from "/v2/state/toasts.js";
import { when, plural } from "/v2/lib/format.js";
import { modelByKey } from "/v2/state/models.js";
import { navigate } from "/v2/state/workspace.js";

// An agent's run, read like a conversation: the goal and each reply as your messages, what it
// did in between as activity, its answers, and what it waits on, answered in place. When it
// has finished, the composer lets it carry on with another message.

export function Agent({ tab }) {
  const item = routeItem(tab);
  const id = tab.route.value.id;
  const [run, setRun] = useState(null);
  const scroll = useRef(null);
  const follow = useRef(true);
  const sig = item ? item.state + item.updated : "";

  const load = () => api("/api/agents/get", { id }, { project: tab.project })
    .then(setRun).catch(e => (e.status === 404 ? setRun(false) : fail(e)));
  useEffect(() => { setRun(null); follow.current = true; }, [id]);
  useEffect(() => { load(); }, [id, sig]);
  // while it works, it is read every second
  const live = run && LIVE.includes(run.state);
  useEffect(() => {
    if (!live) return;
    const t = setInterval(() => !document.hidden && load(), 1000);
    return () => clearInterval(t);
  }, [id, live]);
  useLayoutEffect(() => {
    const el = scroll.current;
    if (el && follow.current) el.scrollTop = el.scrollHeight;
  });

  if (item === false || run === false) return html`<${Gone} tab=${tab} what="agent" />`;
  if (!item || !run) return html`<div class="loading" aria-busy="true" />`;
  const after = () => { load(); tab.refresh(); };
  const stop = () => api("/api/agents/stop", { id }, { project: tab.project }).then(after, fail);
  const model = run.by || run.model;
  const header = html`<${ItemHeader} tab=${tab} item=${item} sub=${html`<span class=${"badge s-" + run.state}>${run.pending ? "waits for you" : run.state}</span>
      ${" · " + (run.role || "agent")} · ${modelByKey(model)?.name || (model === "auto" ? "Auto" : model)} · ${run.mode === "auto" ? "automatic" : run.mode === "plan" ? "plan first" : "asks first"}`}>
    ${run.chat && html`<${Button} icon="chat" onClick=${() => navigate(tab, { screen: "chat", id: run.chat })} title="This agent writes the answers of a chat">Open its chat<//>`}
    ${live && html`<${Button} icon="stop" onClick=${stop}>Stop<//>`}
  <//>`;
  return html`<${Reading} header=${header} scrollRef=${scroll} class="chat agent"
      onScroll=${e => { const el = e.currentTarget; follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60; }}
      footer=${html`<${Reply} run=${run} project=${tab.project} done=${after} />`}>
    <${Thread} run=${run} project=${tab.project} done=${after} />
    ${!live && (run.inputTokens || run.outputTokens) ? html`<p class="muted small">${plural((run.inputTokens || 0) + (run.outputTokens || 0), "token")} used · ended ${when(run.ended)}</p>` : null}
  <//>`;
}

