import { html, useState } from "/v2/lib/html.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button } from "/v2/ui/Button.js";
import { openMenu } from "/v2/ui/overlay.js";
import { when } from "/v2/lib/format.js";
import { answerRequest, withdrawRequest } from "/v2/state/requests.js";
import { openProject } from "/v2/state/workspace.js";

// One request from an agent in a subproject, with the way to answer it right there: write an
// answer to a question, give or refuse a tool or model, or take a result in. The same card
// shows in the overview and on the home of the project it was sent to.

const ASKS = { question: "asks", tool: "needs a tool", model: "needs a model", result: "hands up a result" };

export function RequestCard({ q }) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const send = async body => {
    setBusy(true);
    const ok = await answerRequest(q, body);
    setBusy(false);
    if (ok) setText("");
  };
  const note = text.trim() || undefined;
  const openAgent = () => openProject(q.project, { route: { screen: "agent", id: q.run } });
  return html`<article class=${"req req-" + q.kind} aria-label=${"Request from " + q.projectName}>
    <header class="req-head">
      <span class="item-ic k-agent"><${Icon} name=${q.kind === "result" ? "check" : q.kind === "question" ? "chat" : "info"} /></span>
      <span class="req-who">
        <button type="button" class="link" onClick=${openAgent}>${q.runTitle || "Agent " + q.run}</button>
        <span class="muted"> in </span><button type="button" class="link" onClick=${() => openProject(q.project)}>${q.projectName}</button>
        <span class="muted"> ${ASKS[q.kind] || "asks"}${q.toNames?.length > 1 ? " (" + q.toNames.join(" or ") + ")" : ""}</span>
      </span>
      <span class="item-when">${when(q.created)}</span>
      <${Button} variant="ghost" size="sm" icon="more" label="More" onClick=${e => openMenu(e.currentTarget, [
        { label: "Open the agent", icon: "open", run: openAgent },
        { label: "Withdraw without answer", icon: "close", danger: true, run: () => withdrawRequest(q) },
      ], "Request")} />
    </header>
    ${q.subject && html`<p class="req-subject"><span class="muted">${q.kind === "tool" ? "Tool" : "Model"}</span> <code class="mono">${q.subject}</code></p>`}
    <p class="req-text">${q.text}</p>
    <form class="req-answer" onSubmit=${e => { e.preventDefault(); if (q.kind === "question" && note) send({ answer: note }); }}>
      <textarea class="input" rows=${q.kind === "question" ? 2 : 1} value=${text} disabled=${busy}
        placeholder=${q.kind === "question" ? "Your answer" : "A note with it (optional)"} aria-label="Answer"
        onInput=${e => setText(e.target.value)}
        onKeyDown=${e => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey) && q.kind === "question" && note) { e.preventDefault(); send({ answer: note }); } }} />
      <div class="row-btns">
        ${q.kind === "question" && html`<${Button} variant="primary" type="submit" disabled=${busy || !note}>Answer<//>`}
        ${(q.kind === "tool" || q.kind === "model") && html`
          <${Button} variant="primary" icon="check" disabled=${busy} onClick=${() => send({ approve: true, answer: note })}>Give it<//>
          <${Button} disabled=${busy} onClick=${() => send({ approve: false, answer: note })}>Refuse<//>`}
        ${q.kind === "result" && html`<${Button} variant="primary" icon="check" disabled=${busy} onClick=${() => send({ answer: note })}>Got it<//>`}
      </div>
    </form>
    ${q.answeredBy === "agents" && html`<p class="muted small">Agents of ${q.toNames?.join(" or ")} may answer this too.</p>`}
  </article>`;
}
