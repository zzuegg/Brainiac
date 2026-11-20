import { html, useEffect, useState } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button } from "/v2/ui/Button.js";
import { Empty } from "/v2/ui/Reading.js";
import { Confirm } from "/v2/ui/Layer.js";
import { openLayer } from "/v2/ui/overlay.js";
import { kindOf } from "/v2/state/kinds.js";
import { refreshAll, openProject } from "/v2/state/workspace.js";
import { loadProjects, projectName } from "/v2/state/projects.js";
import { toast, fail } from "/v2/state/toasts.js";
import { when, until } from "/v2/lib/format.js";

// Everything deleted in the last 30 days, from every project. Restoring puts an item back in
// its project; deleting for good is the one step here that cannot be undone, so it asks.

export function Trash() {
  const [entries, setEntries] = useState(null);
  const load = () => api("/api/trash").then(r => setEntries(r.entries || [])).catch(fail);
  useEffect(() => { load(); }, []);

  const restore = async e => {
    try {
      const r = await api("/api/trash/restore", { id: e.id });
      if (e.item.kind === "project") {
        await loadProjects();
        openProject(e.item.id, { focus: false });
      }
      await Promise.all([load(), refreshAll()]);
      toast(`${kindOf(e.item.kind).label} “${r.item.title}” restored${r.item.project ? " to " + projectName(r.item.project) : ""}`);
    } catch (x) {
      fail(x);
    }
  };
  const purge = async e => {
    const ok = await openLayer(Confirm, {
      title: e ? "Delete for good?" : "Empty the trash?",
      text: e ? `“${e.item.title}” will be gone and cannot be restored.` : `All ${entries.length} items will be gone and cannot be restored.`,
      action: e ? "Delete for good" : "Empty trash",
    });
    if (!ok) return;
    try {
      const r = await api("/api/trash/empty", e ? { id: e.id } : {});
      setEntries(r.entries || []);
    } catch (x) {
      fail(x);
    }
  };

  return html`<div class="page">
    <header class="page-head">
      <div>
        <p class="eyebrow">All projects</p>
        <h1 class="page-title">Trash</h1>
        <p class="page-lead">Deleted items stay here for 30 days, then go for good.</p>
      </div>
      <div class="page-actions">${entries?.length > 0 && html`<${Button} variant="danger" icon="trash" onClick=${() => purge(null)}>Empty trash<//>`}</div>
    </header>
    ${entries == null ? html`<div class="loading" aria-busy="true" />`
      : entries.length === 0 ? html`<${Empty} icon=${html`<${Icon} name="trash" size=${28} />`} title="The trash is empty" />`
      : html`<ul class="items trash">${entries.map(e => html`<li class="item" key=${e.id}>
          <span class=${"item-ic k-" + e.item.kind}><${Icon} name=${kindOf(e.item.kind).icon} /></span>
          <span class="item-text"><span class="item-title">${e.item.title}</span>
            <span class="item-sub">${kindOf(e.item.kind).label}${e.item.project ? " in " + projectName(e.item.project) : ""} · deleted ${when(e.deleted)} · goes ${until(e.expires)}</span></span>
          <span class="row-btns">
            <${Button} size="sm" icon="restore" onClick=${() => restore(e)}>Restore<//>
            <${Button} size="sm" variant="ghost" icon="trash" label="Delete for good" onClick=${() => purge(e)} />
          </span>
        </li>`)}</ul>`}
  </div>`;
}
