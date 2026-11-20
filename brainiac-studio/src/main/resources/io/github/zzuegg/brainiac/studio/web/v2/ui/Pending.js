import { html, useState } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Button } from "/v2/ui/Button.js";
import { Icon } from "/v2/ui/Icon.js";
import { fail } from "/v2/state/toasts.js";

// What an agent waits on, and the way to answer it, the same in a chat with tools and on the
// agent's own page: a call to allow, a question to answer, or a plan to approve.

async function answer(id, project, body, done) {
  try {
    await api("/api/agents/answer", { id, ...body }, { project });
    done?.();
  } catch (e) {
    fail(e);
  }
}

export function Pending({ id, project, pending: p, done }) {
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const go = async body => {
    setBusy(true);
    await answer(id, project, body, done);
    setBusy(false);
    setNote("");
  };
  if (!p) return null;
  if (p.kind === "question") {
    return html`<div class="pending" role="group" aria-label="The agent asks you">
      <p class="pending-title"><${Icon} name="info" />It asks you</p>
      <p class="pending-text">${p.question}</p>
      ${p.options?.length > 0 && html`<div class="row-btns pending-options">${p.options.map(o =>
        html`<${Button} key=${o} disabled=${busy} onClick=${() => go({ allow: true, note: o })}>${o}<//>`)}</div>`}
      <form class="pending-reply" onSubmit=${e => { e.preventDefault(); if (note.trim()) go({ allow: true, note }); }}>
        <input class="input" value=${note} disabled=${busy} placeholder=${p.options?.length ? "Or write your own answer" : "Your answer"}
          aria-label="Your answer" onInput=${e => setNote(e.target.value)} />
        <${Button} variant="primary" type="submit" disabled=${busy || !note.trim()}>Answer<//>
        <${Button} variant="ghost" disabled=${busy} onClick=${() => go({ allow: false })}>Let it decide<//>
      </form>
    </div>`;
  }
  if (p.kind === "plan") {
    return html`<div class="pending" role="group" aria-label="The agent's plan">
      <p class="pending-title"><${Icon} name="check" />Approve the plan?</p>
      <ol class="plan">${(p.plan || []).map((s, i) => html`<li key=${i}>${s.step}</li>`)}</ol>
      <input class="input" value=${note} disabled=${busy} placeholder="Anything to add or change (optional)" aria-label="Note"
        onInput=${e => setNote(e.target.value)} />
      <div class="row-btns">
        <${Button} variant="primary" disabled=${busy} onClick=${() => go({ allow: true, note })}>Approve<//>
        <${Button} disabled=${busy} onClick=${() => go({ allow: true, note, mode: "auto" })} title="It carries the plan out without asking again">Approve, and stop asking<//>
        <${Button} variant="ghost" disabled=${busy || !note.trim()} onClick=${() => go({ allow: false, note })} title="Write what to change first">Ask for changes<//>
      </div>
    </div>`;
  }
  const args = p.arguments && Object.keys(p.arguments).length ? JSON.stringify(p.arguments, null, 2) : "";
  return html`<div class="pending" role="group" aria-label="The agent asks to call a tool">
    <p class="pending-title"><${Icon} name="warn" />Allow this?</p>
    <p class="pending-text">It ${p.acts || "calls a tool"}: <code class="mono">${p.tool}</code></p>
    ${args && html`<pre class="code pending-args"><code>${args.length > 1200 ? args.slice(0, 1200) + "\n…" : args}</code></pre>`}
    ${p.why && html`<p class="muted small">Asking because ${p.why}.</p>`}
    ${p.paths?.length > 0 && html`<ul class="muted small">${p.paths.map(x => html`<li key=${x}>${x}</li>`)}</ul>`}
    <input class="input" value=${note} disabled=${busy} placeholder="A note for the agent (optional)" aria-label="Note"
      onInput=${e => setNote(e.target.value)} />
    <div class="row-btns">
      <${Button} variant="primary" disabled=${busy} onClick=${() => go({ allow: true, note })}>Allow<//>
      <${Button} disabled=${busy} onClick=${() => go({ allow: true, note, mode: "auto" })} title="Allows this and what follows, within the project's rules">Allow, and stop asking<//>
      <${Button} variant="ghost" disabled=${busy} onClick=${() => go({ allow: false, note })}>Don't allow<//>
    </div>
  </div>`;
}

/** One line for a step an agent took, for the activity under its answer. */
export function stepLine(s) {
  const short = v => {
    const t = typeof v === "string" ? v : JSON.stringify(v ?? "");
    return t.length > 160 ? t.slice(0, 157) + "…" : t;
  };
  switch (s.kind) {
    case "call": return { icon: "sparkle", text: "Calls " + (s.tool || s.name), detail: s.arguments ? short(s.arguments) : "" };
    case "result": return { icon: s.ok === false ? "warn" : "check", text: (s.tool || "") + (s.ok === false ? " failed" : " answered"), detail: short(s.text) };
    case "decision": return { icon: s.allow ? "check" : "close", text: (s.allow ? "You allowed " : "You did not allow ") + s.tool, detail: s.note || "" };
    case "plan": return { icon: "info", text: "Plan updated", detail: (s.plan || s.steps || []).map?.(x => x.step).join(" · ") || "" };
    case "say": case "think": return { icon: "chat", text: short(s.text), detail: "" };
    case "message": return { icon: "chat", text: "Messaged agent " + s.to, detail: short(s.text) };
    case "inbox": return { icon: "chat", text: "Read messages from other agents", detail: "" };
    case "spawn": return { icon: "agent", text: "Started agent " + s.child + (s.role ? " (" + s.role + ")" : ""), detail: short(s.goal) };
    case "request": return { icon: "project", text: "Asked " + s.to, detail: short(s.text) };
    case "subproject": return { icon: "project", text: "Made the subproject " + s.name, detail: "" };
    case "error": return { icon: "warn", text: short(s.text), detail: "" };
    case "compact": return { icon: "info", text: "Made room in its memory", detail: "" };
    default: return { icon: "dot", text: s.kind, detail: short(s.text || "") };
  }
}
