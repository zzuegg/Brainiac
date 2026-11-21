import { html, cx } from "/v2/lib/html.js";
import { toasts, dismiss, runUndo } from "/v2/state/toasts.js";
import { Icon } from "/v2/ui/Icon.js";

export function Toasts() {
  return html`<div class="toasts" role="status" aria-live="polite">
    ${toasts.value.map(t => html`<div key=${t.id} class=${cx("toast", t.tone === "error" && "toast-error")}>
      ${t.tone === "error" && html`<${Icon} name="warn" />`}
      <span class="toast-text">${t.text}</span>
      ${t.undo && html`<button type="button" class="toast-undo" onClick=${() => runUndo(t)}><${Icon} name="undo" size=${14} />Undo</button>`}
      <button type="button" class="mini" aria-label="Dismiss" onClick=${() => dismiss(t.id)}><${Icon} name="close" size=${14} /></button>
    </div>`)}
  </div>`;
}
