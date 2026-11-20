import { html, cx } from "/v2/lib/html.js";
import { Icon } from "/v2/ui/Icon.js";

/**
 * A button. `variant` is "primary", "ghost", "danger" or none; `icon` adds an icon before the
 * label; an icon-only button needs `label` for its name.
 */
export function Button({ variant, icon, label, children, class: cls, size, ...rest }) {
  const onlyIcon = icon && !children;
  return html`<button type="button" class=${cx("btn", variant && "btn-" + variant, onlyIcon && "btn-icon", size === "sm" && "btn-sm", cls)}
    aria-label=${onlyIcon ? label : undefined} title=${onlyIcon ? label : rest.title} ...${rest}>
    ${icon && html`<${Icon} name=${icon} />`}${children && html`<span>${children}</span>`}
  </button>`;
}

/** A keyboard shortcut, as shown in menus and tooltips. */
export const Kbd = ({ children }) => html`<kbd class="kbd">${children}</kbd>`;
