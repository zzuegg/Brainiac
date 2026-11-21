import { html, useState, useEffect, useRef, cx } from "/v2/lib/html.js";
import { Layer } from "/v2/ui/Layer.js";
import { Button } from "/v2/ui/Button.js";
import { Icon } from "/v2/ui/Icon.js";

// Every "New …" is a wizard: a few steps with defaults picked, then a summary in plain words.
// Next is never greyed out; when something is missing it says what. Closing keeps the draft,
// so opening the same wizard again carries on where it was left.

const drafts = new Map();

/**
 * @param {{
 *   title: string, draftKey?: string, initial: object, close: Function, top?: boolean,
 *   steps: Array<{title: string, body: (draft, set) => any, missing?: (draft) => string|null}>,
 *   summary: (draft, goTo) => any, finish: (draft) => Promise<any>, finishLabel?: string
 * }} p
 */
export function Wizard({ title, draftKey, initial, steps, summary, finish, finishLabel = "Create", close, top }) {
  const all = [...steps, { title: "Summary", body: d => summary(d, goTo) }];
  const [draft, setDraft] = useState(() => (draftKey && drafts.get(draftKey)) || initial);
  const [at, setAt] = useState(0);
  const [problem, setProblem] = useState(null);
  const [busy, setBusy] = useState(false);
  const last = at === all.length - 1;
  const form = useRef(null);
  const first = useRef(true);
  useEffect(() => {
    if (first.current) return void (first.current = false);
    (form.current?.querySelector("input:not([type=radio]), textarea, input:checked") || form.current?.closest(".layer")?.querySelector(".layer-foot .btn-primary"))?.focus();
  }, [at]);

  const set = changes => {
    setDraft(d => {
      const n = { ...d, ...changes };
      if (draftKey) drafts.set(draftKey, n);
      return n;
    });
    setProblem(null);
  };
  function goTo(i) {
    setProblem(null);
    setAt(i);
  }
  const firstMissing = upTo => {
    for (let i = 0; i < upTo; i++) {
      const m = all[i].missing?.(draft);
      if (m) return { step: i, text: m };
    }
    return null;
  };
  const next = async () => {
    if (!last) {
      const m = all[at].missing?.(draft);
      if (m) return setProblem(m);
      return goTo(at + 1);
    }
    const m = firstMissing(all.length);
    if (m) {
      setAt(m.step);
      return setProblem(m.text);
    }
    setBusy(true);
    try {
      const result = await finish(draft);
      if (draftKey) drafts.delete(draftKey);
      close(result ?? true);
    } catch (e) {
      setProblem(e.message || String(e));
    } finally {
      setBusy(false);
    }
  };

  const aside = html`<ol class="steps">
    ${all.map((s, i) => html`<li class=${cx("step", i === at && "on", i < at && "done")}>
      <button type="button" onClick=${() => (i <= at || !firstMissing(i)) && goTo(i)}>
        <span class="step-n">${i < at ? html`<${Icon} name="check" size=${12} />` : i + 1}</span>${s.title}
      </button></li>`)}
  </ol>`;
  const footer = html`
    ${problem && html`<p class="wizard-problem" role="alert"><${Icon} name="info" />${problem}</p>`}
    <span class="grow" />
    ${at > 0 && html`<${Button} onClick=${() => goTo(at - 1)}>Back<//>`}
    <${Button} variant="primary" disabled=${busy} onClick=${next}>${last ? (busy ? "Working…" : finishLabel) : "Next"}<//>`;

  return html`<${Layer} title=${title} subtitle=${`Step ${at + 1} of ${all.length} · ${all[at].title}`} close=${close}
      wide aside=${aside} footer=${footer} top=${top}>
    <form class="wizard-step" ref=${form} onSubmit=${e => { e.preventDefault(); next(); }}>
      ${all[at].body(draft, set)}
      <button type="submit" hidden />
    </form>
  <//>`;
}

/** A labelled field for wizards and forms. */
export function Field({ label, hint, children }) {
  return html`<label class="field"><span class="field-label">${label}</span>${children}${hint && html`<span class="field-hint">${hint}</span>`}</label>`;
}

/** A row of choices, one picked, each with a line of explanation. */
export function Choices({ value, options, onChange, name }) {
  return html`<div class="choices" role="radiogroup" aria-label=${name}>
    ${options.map(o => html`<label class=${cx("choice", value === o.value && "on")}>
      <input type="radio" name=${name} value=${o.value} checked=${value === o.value} onChange=${() => onChange(o.value)} />
      <span class="choice-title">${o.icon && html`<${Icon} name=${o.icon} />`}${o.label}</span>
      ${o.text && html`<span class="choice-text">${o.text}</span>`}
    </label>`)}
  </div>`;
}

/** The summary step: what will happen, each line with a link back to its step. */
export function Summary({ lead, rows, goTo }) {
  return html`<div class="summary">
    ${lead && html`<p class="summary-lead">${lead}</p>`}
    <dl>${rows.map(r => html`<div class="summary-row"><dt>${r.label}</dt><dd>${r.value || html`<span class="muted">none</span>`}</dd>
      ${r.step != null && html`<button type="button" class="link" onClick=${() => goTo(r.step)}>Edit</button>`}</div>`)}</dl>
  </div>`;
}
