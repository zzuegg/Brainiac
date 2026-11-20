import { html, useEffect, useState } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";
import { Reading, Empty } from "/v2/ui/Reading.js";
import { Icon } from "/v2/ui/Icon.js";
import { ItemHeader, routeItem, Gone } from "/v2/screens/ItemHeader.js";
import { plural, when } from "/v2/lib/format.js";

// A network's outline. The canvas, with its docked agent, comes with the Builder step.

export function Graph({ tab }) {
  const item = routeItem(tab);
  const [graph, setGraph] = useState(null);
  const id = tab.route.value.id;
  useEffect(() => {
    let live = true;
    api("/api/builder/graphs", undefined, { project: tab.project })
      .then(r => live && setGraph((r.graphs || []).find(g => g.name === id) || false))
      .catch(() => live && setGraph(false));
    return () => { live = false; };
  }, [id, item && item.updated]);
  if (item === false) return html`<${Gone} tab=${tab} what="network" />`;
  if (!item) return html`<div class="loading" aria-busy="true" />`;
  const nodes = (graph && graph.nodes) || [];
  const header = html`<${ItemHeader} tab=${tab} item=${item} sub=${`${plural(nodes.length, "block")}${item.built ? " · built" : ""} · saved ${when(item.updated)}`}>
    <a class="btn" href="/" target="_blank" rel="noopener"><${Icon} name="classic" /><span>Edit in classic builder</span></a>
  <//>`;
  return html`<${Reading} header=${header}>
    ${nodes.length === 0
      ? html`<${Empty} icon=${html`<${Icon} name="graph" size=${28} />`} title="An empty network">Add blocks in the builder.<//>`
      : html`<ol class="blocks">${nodes.map((n, i) => html`<li key=${i} class="block-row">
          <span class="block-n">${i + 1}</span><span class="block-type">${n.type || n.layer || n.kind || "block"}</span>
          <span class="muted small mono">${n.id || ""}</span></li>`)}</ol>`}
  <//>`;
}
