import { html, useLayoutEffect, useRef, cx } from "/v2/lib/html.js";
import { Icon } from "/v2/ui/Icon.js";
import { openMenu } from "/v2/ui/overlay.js";
import { when } from "/v2/lib/format.js";
import { keyOf } from "/v2/state/workspace.js";
import { kindOf } from "/v2/state/kinds.js";
import { open, menuFor, rename, remove, duplicate, startRename, isRenaming } from "/v2/state/actions.js";

// A list of items with the same behaviour in every place it is used: hover shows Rename and
// ⋯; right-click opens the same menu; arrow keys move, Enter opens, F2 renames in place,
// Delete deletes (with undo), Ctrl+D duplicates.

export function ItemList({ tab, items, current, compact, label, scope = "list" }) {
  const ref = useRef(null);
  const onKey = e => {
    const row = e.target.closest?.("[data-key]");
    if (!row || e.target.tagName === "INPUT") return;
    const item = items.find(i => keyOf(i) === row.dataset.key);
    if (!item) return;
    const rows = [...ref.current.querySelectorAll("[data-key]")];
    const at = rows.indexOf(row);
    const can = new Set(kindOf(item.kind).actions);
    if (e.key === "ArrowDown") rows[Math.min(rows.length - 1, at + 1)]?.focus();
    else if (e.key === "ArrowUp") rows[Math.max(0, at - 1)]?.focus();
    else if (e.key === "Enter") open(tab, item);
    else if (e.key === "F2" && can.has("rename")) startRename(tab, item, scope);
    else if ((e.key === "Delete" || (e.key === "Backspace" && e.metaKey)) && can.has("delete")) {
      (rows[at + 1] || rows[at - 1])?.focus();
      remove(tab, item);
    } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "d" && can.has("duplicate")) duplicate(tab, item);
    else if (e.key === "ContextMenu" || (e.shiftKey && e.key === "F10")) {
      const r = row.getBoundingClientRect();
      openMenu({ x: r.left + 24, y: r.bottom - 4 }, menuFor(tab, item, scope), item.title);
    } else return;
    e.preventDefault();
  };
  return html`<ul class=${cx("items", compact && "items-compact")} ref=${ref} onKeyDown=${onKey} aria-label=${label}>
    ${items.map(item => html`<${Row} key=${keyOf(item)} tab=${tab} item=${item} compact=${compact} scope=${scope}
        current=${current && current.screen === item.kind && current.id === item.id} />`)}
  </ul>`;
}

function Row({ tab, item, current, compact, scope }) {
  const k = kindOf(item.kind);
  const renaming = isRenaming(tab, item, scope);
  const busy = (item.kind === "agent" && (item.state === "running" || item.waiting || item.state === "waiting"))
    || (item.kind === "training" && ["queued", "loading", "running"].includes(item.state));
  const failed = (item.kind === "agent" || item.kind === "training") && item.state === "failed";
  const sub = subtitle(item);
  return html`<li class=${cx("item", current && "current", renaming && "renaming")} data-key=${keyOf(item)}
      tabindex="0" aria-current=${current ? "page" : undefined}
      onClick=${e => !renaming && !e.target.closest("button") && open(tab, item)}
      onContextMenu=${e => { e.preventDefault(); openMenu({ x: e.clientX, y: e.clientY }, menuFor(tab, item, scope), item.title); }}>
    <span class=${cx("item-ic", "k-" + item.kind, failed && "item-failed")} title=${failed ? "Failed" : undefined}>
      ${busy ? html`<span class=${cx("live", item.waiting || item.state === "waiting" ? "live-wait" : "live-run")} title=${item.waiting ? "Waits for you" : item.kind === "training" ? "Training" : "Working"} />`
        : html`<${Icon} name=${item.kind === "chat" && MEDIUM_ICONS[item.medium] || k.icon} />`}
    </span>
    ${renaming
      ? html`<${RenameField} value=${item.title} done=${t => rename(tab, item, t)} cancel=${() => (tab.renaming.value = null)} />`
      : html`<span class="item-text">
          <span class="item-title">${item.title}</span>
          ${!compact && sub && html`<span class="item-sub">${sub}</span>`}
        </span>
        <span class="item-when">${when(item.updated)}</span>
        <span class="item-acts">
          ${k.actions.includes("rename") && html`<button type="button" class="mini" tabindex="-1" aria-label="Rename" title="Rename (F2)"
              onClick=${() => startRename(tab, item, scope)}><${Icon} name="edit" size=${14} /></button>`}
          <button type="button" class="mini" tabindex="-1" aria-label="More actions" title="More actions"
              onClick=${e => openMenu(e.currentTarget, menuFor(tab, item, scope), item.title)}><${Icon} name="more" size=${14} /></button>
        </span>`}
  </li>`;
}

const MEDIUM_ICONS = { image: "image", speech: "speech", video: "video", transcribe: "mic", rewrite: "rewrite" };

function subtitle(item) {
  if (item.kind === "chat") return item.messages ? `${item.messages} messages` : "empty";
  if (item.kind === "agent") return item.waiting ? "waits for you" : item.state;
  if (item.kind === "graph") return `${item.nodes} blocks${item.built ? " · built" : ""}`;
  if (item.kind === "training") return item.state === "running" || item.state === "loading" ? `epoch ${item.epoch} of ${item.epochs}`
    : item.state === "finished" && item.loss != null ? `finished · loss ${(+item.loss).toPrecision(3)}` : item.state;
  return "";
}

/** Renaming in place: Enter or leaving the field keeps it, Escape cancels. */
export function RenameField({ value, done, cancel, label = "New name" }) {
  const ref = useRef(null);
  useLayoutEffect(() => {
    ref.current.focus();
    ref.current.select();
  }, []);
  const settled = useRef(false);
  const finish = ok => {
    if (settled.current || !ref.current) return;
    settled.current = true;
    ok ? done(ref.current.value) : cancel();
  };
  return html`<input class="rename" ref=${ref} defaultValue=${value} aria-label=${label} maxlength="200"
    onClick=${e => e.stopPropagation()}
    onKeyDown=${e => {
      e.stopPropagation();
      if (e.key === "Enter") finish(true);
      else if (e.key === "Escape") finish(false);
    }}
    onBlur=${() => finish(true)} />`;
}
