import { html, useState } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Text } from "/v2/ui/Text.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button } from "/v2/ui/Button.js";
import { Pending, stepLine } from "/v2/ui/Pending.js";
import { fail } from "/v2/state/toasts.js";
import { plural } from "/v2/lib/format.js";

// An agent's run read as a conversation, the same on the agent's own page and where an agent is
// docked beside other work (the builder's assistant): what you said, what it did, what it
// answered, and what it waits on, answered in place.

export const LIVE = ["running", "waiting"];

/**
 * The run's transcript. `said` turns the goal into what the person wrote, for a run whose goal
 * also carries context the page added.
 */
export function Thread({ run, project, done, said = t => t }) {
  const live = LIVE.includes(run.state);
  return html`
    ${run.plan?.length > 0 && html`<${Plan} plan=${run.plan} />`}
    <ol class="transcript">${turnsOf(run).map((t, i, all) => t.kind === "you"
      ? html`<li key=${i} class="msg msg-user"><div class="msg-text"><${Text} text=${said(t.text)} /></div></li>`
      : t.kind === "answer"
        ? html`<li key=${i} class="msg msg-assistant"><div class="msg-who">${roleName(run.role)}</div><div class="msg-text"><${Text} text=${t.text} /></div></li>`
        : html`<li key=${i} class="msg msg-assistant"><${Activity} steps=${t.steps} open=${live && i === all.length - 1} live=${live && i === all.length - 1} /></li>`)}
    </ol>
    ${run.pending && html`<${Pending} id=${run.id} project=${project} pending=${run.pending} done=${done} />`}
    ${live && !run.pending && html`<p class="working"><span class="live live-run" />${run.thinking ? "Thinking…" : "Working…"}</p>`}
    ${run.state === "failed" && run.error && html`<p class="msg-error"><${Icon} name="warn" size=${14} />${run.error}</p>`}
    ${run.state === "stopped" && html`<p class="muted small">Stopped${run.error ? ": " + run.error : ""}.</p>`}`;
}

export const roleName = r => !r || r === "assistant" ? "Agent" : r[0].toUpperCase() + r.slice(1).replace(/-/g, " ");

// the steps in turns: what you said, what it did, what it answered
export function turnsOf(run) {
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
export function Reply({ run, project, done, placeholder = "Reply to carry on" }) {
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
        placeholder=${run.pending ? "It waits for your answer above" : live ? "It is working; you can reply once it has answered" : placeholder}
        onInput=${e => setText(e.target.value)}
        onKeyDown=${e => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } }} />
      <div class="composer-bar"><span class="grow" />
        ${!live && html`<span class="muted small composer-hint">Enter sends</span>`}
        <${Button} variant="primary" onClick=${send} disabled=${busy || live || !text.trim()}>Send<//>
      </div>
    </div>
  </div>`;
}
