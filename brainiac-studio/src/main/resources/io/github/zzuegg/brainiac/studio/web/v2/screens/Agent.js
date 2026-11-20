import { html, useEffect, useLayoutEffect, useRef, useState, cx } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Reading } from "/v2/ui/Reading.js";
import { Text } from "/v2/ui/Text.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button } from "/v2/ui/Button.js";
import { Pending, stepLine } from "/v2/ui/Pending.js";
import { ItemHeader, routeItem, Gone } from "/v2/screens/ItemHeader.js";
import { fail } from "/v2/state/toasts.js";
import { when, plural } from "/v2/lib/format.js";
import { modelByKey } from "/v2/state/models.js";

// An agent's run, read like a conversation: the goal and each reply as your messages, what it
// did in between as activity, its answers, and what it waits on, answered in place. When it
// has finished, the composer lets it carry on with another message.

const LIVE = ["running", "waiting"];

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
    ${live && html`<${Button} icon="stop" onClick=${stop}>Stop<//>`}
  <//>`;
  return html`<${Reading} header=${header} scrollRef=${scroll} class="chat agent"
      onScroll=${e => { const el = e.currentTarget; follow.current = el.scrollHeight - el.scrollTop - el.clientHeight < 60; }}
      footer=${html`<${Reply} run=${run} project=${tab.project} done=${after} />`}>
    ${run.plan?.length > 0 && html`<${Plan} plan=${run.plan} />`}
    <ol class="transcript">${turnsOf(run).map((t, i, all) => t.kind === "you"
      ? html`<li key=${i} class="msg msg-user"><div class="msg-text"><${Text} text=${t.text} /></div></li>`
      : t.kind === "answer"
        ? html`<li key=${i} class="msg msg-assistant"><div class="msg-who">${roleName(run.role)}</div><div class="msg-text"><${Text} text=${t.text} /></div></li>`
        : html`<li key=${i} class="msg msg-assistant"><${Activity} steps=${t.steps} open=${live && i === all.length - 1} live=${live && i === all.length - 1} /></li>`)}
    </ol>
    ${run.pending && html`<${Pending} id=${id} project=${tab.project} pending=${run.pending} done=${after} />`}
    ${live && !run.pending && html`<p class="working"><span class="live live-run" />${run.thinking ? "Thinking…" : "Working…"}</p>`}
    ${run.state === "failed" && run.error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${run.error}</p>`}
    ${run.state === "stopped" && html`<p class="muted small">Stopped${run.error ? ": " + run.error : ""}.</p>`}
    ${!live && (run.inputTokens || run.outputTokens) ? html`<p class="muted small">${plural((run.inputTokens || 0) + (run.outputTokens || 0), "token")} used · ended ${when(run.ended)}</p>` : null}
  <//>`;
}

const roleName = r => !r || r === "assistant" ? "Agent" : r[0].toUpperCase() + r.slice(1).replace(/-/g, " ");

// the steps in turns: what you said, what it did, what it answered
function turnsOf(run) {
  const out = [];
  for (const s of run.steps || []) {
    if (s.kind === "goal") out.push({ kind: "you", text: s.text });
    else if (s.kind === "answer") out.push({ kind: "answer", text: s.text });
    else if (s.kind === "think") continue;
    else {
      const last = out.at(-1);
      if (last?.kind === "steps") last.steps.push(s);
      else out.push({ kind: "steps", steps: [s] });
    }
  }
  if (!out.some(t => t.kind === "you")) out.unshift({ kind: "you", text: run.goal });
  if (run.answer && out.at(-1)?.kind !== "answer" && !LIVE.includes(run.state)) out.push({ kind: "answer", text: run.answer });
  return out;
}

function Activity({ steps, open, live }) {
  const shown = steps;
  return html`<details class="activity" open=${open}>
    <summary>${plural(shown.length, "step")}${live ? " so far" : ""}</summary>
    <ol class="activity-list">${shown.map((x, i) => {
      const l = stepLine(x);
      return html`<li key=${i}><${Icon} name=${l.icon} size=${13} /><span>${l.text}</span>${l.detail && html`<span class="muted mono small" title=${l.detail}>${l.detail}</span>`}</li>`;
    })}</ol>
  </details>`;
}

function Plan({ plan }) {
  const done = plan.filter(p => p.state === "done").length;
  return html`<section class="card plan-card" aria-label="Plan">
    <h2 class="card-title">Plan<span class="count">${done}/${plan.length}</span></h2>
    <ol class="plan-steps">${plan.map((p, i) => html`<li key=${i} class=${"plan-" + p.state}>
      <${Icon} name=${p.state === "done" ? "check" : p.state === "doing" ? "dot" : "chevronRight"} size=${13} />${p.step}</li>`)}</ol>
  </section>`;
}

// carry on after it ended; while it works, the composer says so
function Reply({ run, project, done }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const live = LIVE.includes(run.state);
  const send = async () => {
    if (!text.trim() || live) return;
    setBusy(true);
    try {
      await api("/api/agents/reply", { id: run.id, text: text.trim() }, { project });
      setText("");
      done();
    } catch (e) {
      fail(e);
    }
    setBusy(false);
  };
  return html`<div class="composer">
    <div class="composer-box">
      <textarea class="composer-input" rows="1" value=${text} disabled=${busy || live} aria-label="Reply to the agent"
        placeholder=${run.pending ? "It waits for your answer above" : live ? "It is working; you can reply once it has answered" : "Reply to carry on"}
        onInput=${e => setText(e.target.value)}
        onKeyDown=${e => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } }} />
      <div class="composer-bar"><span class="grow" />
        ${!live && html`<span class="muted small composer-hint">Enter sends</span>`}
        <${Button} variant="primary" onClick=${send} disabled=${busy || live || !text.trim()}>Send<//>
      </div>
    </div>
  </div>`;
}
