import { html, useEffect, useRef, cx } from "/v2/lib/html.js";
import { Button } from "/v2/ui/Button.js";
import { layers } from "/v2/ui/overlay.js";

// Dialogs and drawers share one frame: a header, one scrolling body, and a footer that stays
// at the bottom so the main button is never below the fold. Escape closes the top one.

export function LayerHost() {
  const list = layers.value;
  return html`${list.map((l, i) => html`<${l.component} key=${l.id} ...${l.props} top=${i === list.length - 1} />`)}`;
}

function useTrap(ref, close, top) {
  useEffect(() => {
    const el = ref.current;
    const first = el?.querySelector("[autofocus]") || el?.querySelector(".layer-body :is(input, select, textarea, button)")
      || el?.querySelector(".layer-foot button:last-child");
    (first || ref.current)?.focus();
  }, []);
  useEffect(() => {
    if (!top) return;
    const onKey = e => {
      if (e.key === "Escape") {
        e.preventDefault();
        close();
      } else if (e.key === "Tab") {
        const f = [...ref.current.querySelectorAll("button, [href], input, select, textarea, [tabindex]:not([tabindex='-1'])")]
          .filter(el => !el.disabled && el.offsetParent !== null);
        if (!f.length) return;
        if (e.shiftKey && document.activeElement === f[0]) { e.preventDefault(); f.at(-1).focus(); }
        else if (!e.shiftKey && document.activeElement === f.at(-1)) { e.preventDefault(); f[0].focus(); }
      }
    };
    addEventListener("keydown", onKey);
    return () => removeEventListener("keydown", onKey);
  }, [top]);
}

/**
 * A frame for a dialog (centred) or a drawer (from the right).
 * @param {{kind?: "dialog"|"drawer", title: any, close: Function, footer?: any, wide?: boolean, aside?: any}} p
 */
export function Layer({ kind = "dialog", title, subtitle, close, footer, children, wide, aside, top = true, label }) {
  const ref = useRef(null);
  useTrap(ref, () => close(), top);
  return html`<div class=${cx("scrim", kind === "drawer" && "scrim-drawer")} onPointerDown=${e => e.target === e.currentTarget && close()}>
    <section class=${cx("layer", "layer-" + kind, wide && "layer-wide")} role="dialog" aria-modal="true"
        aria-label=${label || (typeof title === "string" ? title : undefined)} tabindex="-1" ref=${ref}>
      <header class="layer-head">
        <div class="layer-titles">
          <h2 class="layer-title">${title}</h2>
          ${subtitle && html`<p class="layer-sub">${subtitle}</p>`}
        </div>
        <${Button} variant="ghost" icon="close" label="Close" data-close onClick=${() => close()} />
      </header>
      <div class="layer-main">
        ${aside && html`<aside class="layer-aside">${aside}</aside>`}
        <div class="layer-body">${children}</div>
      </div>
      ${footer && html`<footer class="layer-foot">${footer}</footer>`}
    </section>
  </div>`;
}

/** Asks before something that cannot be undone. Resolves true to go ahead. */
export function Confirm({ title, text, action = "Delete", danger = true, close, top }) {
  return html`<${Layer} title=${title} close=${() => close(false)} top=${top}
      footer=${html`<${Button} data-close onClick=${() => close(false)}>Cancel<//>
        <${Button} variant=${danger ? "danger" : "primary"} autofocus onClick=${() => close(true)}>${action}<//>`}>
    <p class="prose">${text}</p>
  <//>`;
}
