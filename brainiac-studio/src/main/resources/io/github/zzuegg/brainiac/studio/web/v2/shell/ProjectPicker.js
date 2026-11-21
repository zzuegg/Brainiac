import { html, useState, useRef, useEffect } from "/v2/lib/html.js";
import { Icon } from "/v2/ui/Icon.js";
import { openLayer } from "/v2/ui/overlay.js";
import { Layer } from "/v2/ui/Layer.js";
import { projects } from "/v2/state/projects.js";
import { tabs, openProject } from "/v2/state/workspace.js";
import { newProject } from "/v2/wizards/index.js";

// Ctrl+P or the "+" after the tabs: find a project and open it in a tab, or start a new one.

export async function pickProject() {
  const r = await openLayer(Picker, {});
  if (r === "new") newProject();
  else if (r) openProject(r);
}

function Picker({ close, top }) {
  const [q, setQ] = useState("");
  const [at, setAt] = useState(0);
  const open = new Set(tabs.value.map(t => t.project));
  const list = projects.value.filter(p => !q || (p.name + " " + p.description).toLowerCase().includes(q.toLowerCase()));
  const rows = [...list.map(p => ({ id: p.id, p })), { id: "new" }];
  const listRef = useRef(null);
  useEffect(() => { listRef.current?.querySelector(".pick.on")?.scrollIntoView({ block: "nearest" }); }, [at]);
  const onKey = e => {
    if (e.key === "ArrowDown") setAt(a => Math.min(rows.length - 1, a + 1));
    else if (e.key === "ArrowUp") setAt(a => Math.max(0, a - 1));
    else if (e.key === "Enter") close(rows[at]?.id);
    else return;
    e.preventDefault();
  };
  return html`<${Layer} title="Open a project" close=${() => close()} top=${top} label="Open a project">
    <div class="search"><${Icon} name="search" /><input class="input" placeholder="Find a project" value=${q} autofocus
      onInput=${e => { setQ(e.target.value); setAt(0); }} onKeyDown=${onKey} aria-label="Find a project" /></div>
    <ul class="pick-list" ref=${listRef} role="listbox" aria-label="Projects">
      ${rows.map((r, i) => html`<li role="option" aria-selected=${i === at}><button type="button" class=${"pick" + (i === at ? " on" : "")}
          onPointerEnter=${() => setAt(i)} onClick=${() => close(r.id)}>
        ${r.p
          ? html`<${Icon} name="project" /><span class="pick-title">${r.p.name}</span>
            <span class="pick-sub">${open.has(r.id) ? "open" : r.p.description}</span>`
          : html`<${Icon} name="plus" /><span class="pick-title">New project…</span>`}
      </button></li>`)}
    </ul>
  <//>`;
}
