import { html, useEffect, useState } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Reading } from "/v2/ui/Reading.js";
import { Text } from "/v2/ui/Text.js";
import { Button } from "/v2/ui/Button.js";
import { ItemHeader, routeItem, Gone } from "/v2/screens/ItemHeader.js";
import { fail } from "/v2/state/toasts.js";
import { when } from "/v2/lib/format.js";

// An agent run: its goal, what it is doing and its answer, in the reading column. Replying,
// approvals and the plan move here with the Agents step.

export function Agent({ tab }) {
  const item = routeItem(tab);
  const [run, setRun] = useState(null);
  const id = tab.route.value.id;
  const sig = item ? item.state + item.updated : "";

  useEffect(() => {
    let live = true;
    api("/api/agents/get", { id }, { project: tab.project }).then(r => live && setRun(r)).catch(() => live && setRun(false));
    return () => { live = false; };
  }, [id, sig]);

  if (item === false) return html`<${Gone} tab=${tab} what="agent" />`;
  if (!item) return html`<div class="loading" aria-busy="true" />`;
  const working = item.state === "running" || item.state === "waiting";
  const stop = async () => {
    try {
      await api("/api/agents/stop", { id }, { project: tab.project });
      tab.refresh();
    } catch (e) {
      fail(e);
    }
  };
  const header = html`<${ItemHeader} tab=${tab} item=${item} sub=${html`<span class=${"badge s-" + item.state}>${item.waiting ? "waits for you" : item.state}</span> ${item.role && "· " + item.role} · ${when(item.updated)}`}>
    ${working && html`<${Button} icon="stop" onClick=${stop}>Stop<//>`}
  <//>`;
  const steps = run?.steps || [];
  return html`<${Reading} header=${header}>
    <section class="block"><h2 class="block-title">Goal</h2><div class="msg-text"><${Text} text=${item.goal} /></div></section>
    ${run?.pending && html`<section class="block block-wait"><h2 class="block-title">Waiting for you</h2>
      <p>${run.pending.question || run.pending.tool || "An approval"}</p>
      <a class="btn" href="/" target="_blank" rel="noopener"><span>Answer in classic Studio</span></a></section>`}
    ${run?.answer && html`<section class="block"><h2 class="block-title">Answer</h2><div class="msg-text"><${Text} text=${run.answer} /></div></section>`}
    ${run?.error && html`<section class="block block-bad"><h2 class="block-title">Stopped</h2><p>${run.error}</p></section>`}
    ${steps.length > 0 && html`<section class="block"><h2 class="block-title">Steps</h2>
      <ol class="steps-log">${steps.map((s, i) => html`<li key=${i} class=${"log log-" + s.kind}>
        <span class="log-kind">${s.kind}</span><span class="log-text">${stepText(s)}</span></li>`)}</ol></section>`}
  <//>`;
}

function stepText(s) {
  const t = s.text || s.name || s.tool || s.note || s.question || "";
  const str = typeof t === "string" ? t : JSON.stringify(t);
  return str.length > 240 ? str.slice(0, 237) + "…" : str;
}
