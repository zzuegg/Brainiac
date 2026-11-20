import { html } from "/v2/lib/html.js";
import { Icon } from "/v2/ui/Icon.js";
import { RenameField } from "/v2/ui/ItemList.js";
import { openMenu } from "/v2/ui/overlay.js";
import { kindOf } from "/v2/state/kinds.js";
import { menuFor, rename, startRename, isRenaming } from "/v2/state/actions.js";

// Every detail view opens the same way: the kind's icon, the title (double-click or F2 to
// rename) and its ⋯ menu on the left; the view's own actions on the right.

export function ItemHeader({ tab, item, children, sub }) {
  const k = kindOf(item.kind);
  const renaming = isRenaming(tab, item, "head");
  return html`<div class="item-head">
    <span class=${"item-ic k-" + item.kind}><${Icon} name=${(item.kind === "chat" && { image: "image", speech: "speech", video: "video" }[item.medium]) || k.icon} /></span>
    <div class="item-head-text">
      ${renaming
        ? html`<${RenameField} value=${item.title} done=${t => rename(tab, item, t)} cancel=${() => (tab.renaming.value = null)} />`
        : html`<h1 class="item-head-title" title="Double-click to rename" onDblClick=${() => startRename(tab, item, "head")}>${item.title}</h1>`}
      ${sub && html`<p class="item-head-sub">${sub}</p>`}
    </div>
    <button type="button" class="btn btn-ghost btn-icon" aria-label="More actions" title="More actions"
      onClick=${e => openMenu(e.currentTarget, menuFor(tab, item), item.title)}><${Icon} name="more" /></button>
    <span class="grow" />
    ${children}
  </div>`;
}

/** Finds the tab's item for its route; null while loading, false when it is gone. */
export function routeItem(tab) {
  const r = tab.route.value;
  const item = tab.items.value.find(i => i.kind === r.screen && i.id === r.id);
  if (item) return item;
  return tab.loaded.value ? false : null;
}

export function Gone({ tab, what }) {
  return html`<div class="gone">
    <p>This ${what} is not in this project any more. It may have been moved or deleted.</p>
    <button type="button" class="btn" onClick=${() => (tab.route.value = { screen: "trash" })}><span>Look in the trash</span></button>
  </div>`;
}
