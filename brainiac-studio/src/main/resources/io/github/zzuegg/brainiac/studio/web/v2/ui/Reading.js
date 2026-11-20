import { html, cx } from "/v2/lib/html.js";

// One centred column for reading: the header, the text and the composer share its width, so
// on an ultrawide screen a conversation reads like a page instead of a strip in a wide frame.
// The width is a setting (narrow, normal, wide, full).

export function Reading({ header, footer, children, class: cls, scrollRef, onScroll }) {
  return html`<div class=${cx("reading", cls)}>
    ${header && html`<header class="reading-head"><div class="column">${header}</div></header>`}
    <div class="reading-scroll" ref=${scrollRef} onScroll=${onScroll}><div class="column">${children}</div></div>
    ${footer && html`<footer class="reading-foot"><div class="column">${footer}</div></footer>`}
  </div>`;
}

/** What a list or screen shows when it has nothing: what it is for, and the way to start. */
export function Empty({ icon, title, children, action }) {
  return html`<div class="empty">
    ${icon}
    <p class="empty-title">${title}</p>
    ${children && html`<p class="empty-text">${children}</p>`}
    ${action}
  </div>`;
}
