import { html, useEffect, useLayoutEffect, useRef, useState } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Reading } from "/v2/ui/Reading.js";
import { Button, IconAction } from "/v2/ui/Button.js";
import { Layer } from "/v2/ui/Layer.js";
import { openLayer } from "/v2/ui/overlay.js";
import { Thread, Reply, LIVE } from "/v2/ui/AgentThread.js";
import { ItemHeader, routeItem, Gone } from "/v2/screens/ItemHeader.js";
import { fail, toast } from "/v2/state/toasts.js";
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
  // the files it changed, which can be put back once it has stopped
  const [changes, setChanges] = useState([]);
  useEffect(() => {
    if (!run || live) return;
    api("/api/agents/changes", { id }, { project: tab.project }).then(r => setChanges(r.changes || []), () => setChanges([]));
  }, [id, live, !!run]);
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
    ${!live && changes.length > 0 && html`<${IconAction} icon="undo" label=${`Undo its changes to ${plural(new Set(changes.map(c => c.path)).size, "file")}…`}
      onClick=${() => openLayer(UndoChanges, { id, project: tab.project, changes, done: () => { setChanges([]); after(); } })} />`}
    ${live && html`<${Button} icon="stop" onClick=${stop}>Stop<//>`}
  <//>`;
  return html`<${Reading} header=${header} scrollRef=${scroll} class="chat agent"
      onScroll=${e => { const el = e.currentTarget; follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60; }}
      footer=${html`<${Reply} run=${run} project=${tab.project} done=${after} />`}>
    <${Thread} run=${run} project=${tab.project} done=${after} />
    ${!live && (run.inputTokens || run.outputTokens) ? html`<p class="muted small">${plural((run.inputTokens || 0) + (run.outputTokens || 0), "token")} used · ended ${when(run.ended)}</p>` : null}
  <//>`;
}

/** The files an agent (and its sub-agents) changed, and putting them back as they were. */
function UndoChanges({ id, project, changes, done, close, top }) {
  const [busy, setBusy] = useState(false);
  // one line per file: what it was before the first change decides what undoing does
  const files = [];
  for (const c of changes) if (!files.some(f => f.path === c.path)) files.push(c);
  const go = async () => {
    setBusy(true);
    try {
      const r = await api("/api/agents/undo", { id }, { project });
      toast(r.restored?.length ? `Put back ${plural(r.restored.length, "file")}` : "It changed no files");
      done(); close();
    } catch (e) { fail(e); setBusy(false); }
  };
  return html`<${Layer} title="Undo its changes" subtitle=${plural(files.length, "file") + " as they were before it started"} close=${close} top=${top}
      footer=${html`<span class="grow" /><${Button} variant="ghost" onClick=${close}>Keep them<//><${Button} variant="primary" icon="undo" onClick=${go} disabled=${busy}>${busy ? "Putting back…" : "Put them back"}<//>`}>
    <ul class="undo-files">${files.map(f => html`<li key=${f.path}>
      <span class="mono small">${f.path}</span>
      <span class=${"badge " + (f.existed ? "" : "s-failed")}>${f.existed ? "put back" : "removed, it was new"}</span></li>`)}</ul>
    <p class="muted small">Changes made after it stopped, by you or another agent, are lost for these files.</p>
  <//>`;
}
