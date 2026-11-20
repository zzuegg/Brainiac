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

/**
 * A small icon for an action on an item: copy, save, fine tune, again. Its name shows as a
 * tooltip and is what a screen reader says. With `href` it is a link (a download, say).
 */
export function IconAction({ icon, label, href, download, size = 14, class: cls, ...rest }) {
  const inner = html`<${Icon} name=${icon} size=${size} />`;
  return href
    ? html`<a class=${cx("mini", cls)} href=${href} download=${download} title=${label} aria-label=${label} ...${rest}>${inner}</a>`
    : html`<button type="button" class=${cx("mini", cls)} title=${label} aria-label=${label} ...${rest}>${inner}</button>`;
}
