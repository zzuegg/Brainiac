import { html, useEffect, useLayoutEffect, useRef, useState, cx } from "/v2/lib/html.js";
import { api, local } from "/v2/lib/api.js";
import { Icon } from "/v2/ui/Icon.js";
import { Button } from "/v2/ui/Button.js";
import { ModelPicker } from "/v2/ui/ModelPicker.js";
import { Thread } from "/v2/ui/AgentThread.js";
import { ItemHeader, routeItem, Gone } from "/v2/screens/ItemHeader.js";
import { fail } from "/v2/state/toasts.js";
import { navigate } from "/v2/state/workspace.js";
import { lastChatModel, rememberChatModel, loadModels, modelInfo } from "/v2/state/models.js";
import { layers, menu, openMenu } from "/v2/ui/overlay.js";
import { plural } from "/v2/lib/format.js";
import { newTraining } from "/v2/wizards/Training.js";
import {
  catalog, loadCatalog, editorFor, open, reload, setView, tryChange, addBlock, specOf, connect, disconnect, removeBlock,
  moveBlock, arrange, undo, redo, build, entryOf, entryLabel, nodeLabel, multiIn, incomplete, shapeKind, fmtShape, fmtCount,
  fmtVal, guess, parseVal, WIRING, assistantOf, assistantLive, wireTargets, unplug, toggleOutput, follow, forgetAssistant, askAssistant, saidToAssistant,
} from "/v2/state/builder.js";

// The builder: blocks on the left, the network in the middle, and on the right either the
// selected block's settings or the assistant, an agent docked beside the canvas that changes the
// network with the same rules (a change that breaks a working block is refused). What it changes
// shows on the canvas as it goes, marked, and undoes like any other change.

export function Graph({ tab }) {
  const item = routeItem(tab);
  const name = tab.route.value.id;
  const ed = editorFor(tab.project, name);
  assistantOf(ed);
  const [side, setSideState] = useState(() => local.get("builderSide", "block"));
  const setSide = s => { setSideState(s); local.set("builderSide", s); };
  const [missing, setMissing] = useState(false);

  useEffect(() => {
    loadCatalog().catch(fail);
    open(ed).catch(e => (e.status === 404 ? setMissing(true) : fail(e)));
  }, [name]);
  // a change saved elsewhere, the classic page or an agent not docked here, is read in
  useEffect(() => {
    if (item && ed.graph.value && item.updated > ed.known && !ed.timer) reload(ed, { by: assistantLive(ed) ? "by the assistant" : "elsewhere" }).catch(() => {});
  }, [item && item.updated]);
  useKeys(ed);

  if (item === false || missing) return html`<${Gone} tab=${tab} what="network" />`;
  const g = ed.graph.value;
  if (!item || !g || !catalog.value) return html`<div class="loading" aria-busy="true" />`;
  const c = ed.check.value;
  const status = !c ? "checking…" : c.ok ? "checks out · output " + fmtShape(c.outputShape) : "not finished";
  const header = html`<${ItemHeader} tab=${tab} item=${item}
      sub=${`${plural(g.nodes.length, "block")} · ${status} · ${ed.saving.value ? "saving…" : "saved"}${item.built ? " · built" : ""}`}>
    <div class="row-btns">
      <${Button} icon="undo" label="Undo (Ctrl+Z)" disabled=${!ed.past.value.length} onClick=${() => undo(ed)} />
      <${Button} icon="redo" label="Redo (Ctrl+Shift+Z)" disabled=${!ed.future.value.length} onClick=${() => redo(ed)} />
      <${Button} icon="sparkle" aria-pressed=${side === "assistant"} class=${cx(side === "assistant" && "is-on")}
        onClick=${() => setSide(side === "assistant" ? "block" : "assistant")} title="An agent that changes this network for you">
        Assistant${assistantLive(ed) && side !== "assistant" ? html`<span class="live live-run" />` : ""}<//>
      <${Button} icon="train" disabled=${!c?.ok} onClick=${() => newTraining(tab, { network: name })}
        title=${c?.ok ? "Trains this network on data" : "Every block has to check out first"}>Train…<//>
      <${Button} variant="primary" icon="build" disabled=${!c?.ok || ed.building.value} onClick=${() => build(ed)}
        title=${c?.ok ? "Builds the model and saves it" : "Every block has to check out first"}>${ed.building.value ? "Building…" : "Build"}<//>
    </div>
  <//>`;
  return html`<div class="builder-screen">
    <header class="builder-head">${header}</header>
    <div class="builder">
      <${Palette} ed=${ed} />
      <${Canvas} ed=${ed} />
      <aside class="builder-side" aria-label=${side === "assistant" ? "Assistant" : "Settings"}>
        <div class="side-tabs" role="tablist">
          <button role="tab" aria-selected=${side === "block"} onClick=${() => setSide("block")}>
            ${ed.sel.value ? nodeLabel(g.nodes.find(n => n.id === ed.sel.value) || {}) : "Network"}</button>
          <button role="tab" aria-selected=${side === "assistant"} onClick=${() => setSide("assistant")}><${Icon} name="sparkle" size=${14} />Assistant</button>
        </div>
        ${side === "assistant" ? html`<${Assistant} tab=${tab} ed=${ed} />` : html`<${Props} ed=${ed} />`}
      </aside>
    </div>
  </div>`;
}

// Undo, redo, delete and escape act on the canvas while no field has the focus.
function useKeys(ed) {
  useEffect(() => {
    const onKey = e => {
      if (e.target.closest?.("input, textarea, select, [contenteditable]") || layers.value.length || menu.value) return;
      const mod = e.ctrlKey || e.metaKey, k = e.key.toLowerCase();
      // with nothing to undo here, Ctrl+Z falls through to the page's own undo (a deleted item)
      if (mod && k === "z") { if (!(e.shiftKey ? redo : undo)(ed)) return; }
      else if (mod && k === "y") { if (!redo(ed)) return; }
      else if (!mod && (e.key === "Delete" || e.key === "Backspace") && ed.sel.value) removeBlock(ed, ed.sel.value);
      else if (e.key === "Escape" && (ed.sel.value || ed.refusal.value)) { ed.sel.value = null; ed.refusal.value = null; }
      else return;
      e.preventDefault();
      e.stopImmediatePropagation();
    };
    addEventListener("keydown", onKey, true);
    return () => removeEventListener("keydown", onKey, true);
  }, [ed]);
}

// ------------------------------------------------------------------ palette

function Palette({ ed }) {
  const [q, setQ] = useState("");
  const [openGroups, setOpen] = useState(() => new Set());
  const cat = catalog.value;
  const ql = q.toLowerCase();
  const match = e => !ql || (e.group + " " + e.name + " " + e.params.map(p => p.name).join(" ")).toLowerCase().includes(ql);
  const groups = new Map();
  cat.entries.filter(match).forEach(e => {
    const k = (e.experimental ? "x:" : "c:") + e.group;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(e);
  });
  const wiring = WIRING.filter(([op, label]) => !ql || (op + " " + label).toLowerCase().includes(ql));
  const toggle = k => setOpen(s => { const n = new Set(s); n.has(k) ? n.delete(k) : n.add(k); return n; });
  const Item = ({ k, name, sig, title }) => html`<button type="button" class="pal-item" draggable="true" title=${title || "Click to add, or drag onto the canvas"}
      onClick=${() => addBlock(ed, specOf(k))}
      onDragStart=${e => { e.dataTransfer.setData("text/x-block", k); e.dataTransfer.effectAllowed = "copy"; }}>
    <span class="pal-name">${name}</span><span class="pal-sig">${sig}</span></button>`;
  const entry = e => html`<${Item} key=${e.key} k=${e.key} name=${["new", "of"].includes(e.name) ? e.group : e.name}
    sig=${e.params.filter(p => p.kind !== "random").map(p => p.name + (p.array ? "…" : "")).join(", ")} title=${e.key} />`;
  const section = (title, exp) => {
    const gs = [...groups].filter(([k]) => k.startsWith(exp ? "x:" : "c:"));
    if (!gs.length) return null;
    return html`<div class="pal-sec">
      <p class="pal-title">${title}<span class="muted">${gs.reduce((a, [, l]) => a + l.length, 0)}</span></p>
      ${gs.map(([k, list]) => {
        if (list.length === 1) return entry(list[0]);
        const isOpen = ql || openGroups.has(k);
        return html`<div key=${k} class="pal-fold">
          <button type="button" class="pal-head" aria-expanded=${!!isOpen} onClick=${() => toggle(k)}>
            <${Icon} name=${isOpen ? "chevronDown" : "chevronRight"} size=${13} /><span>${k.slice(2)}</span><span class="muted">${list.length}</span></button>
          ${isOpen && list.map(entry)}
        </div>`;
      })}
    </div>`;
  };
  return html`<aside class="palette" aria-label="Blocks">
    <div class="search pal-search"><${Icon} name="search" size=${14} />
      <input class="input" placeholder="Search blocks" value=${q} aria-label="Search blocks" onInput=${e => setQ(e.target.value)} /></div>
    <div class="pal-scroll">
      ${wiring.length > 0 && html`<div class="pal-sec"><p class="pal-title">Wiring</p>
        ${wiring.map(([op, label, sig]) => html`<${Item} key=${op} k=${"op:" + op} name=${label} sig=${sig} />`)}</div>`}
      ${section("Layers", false)}${section("Experimental", true)}
      ${!groups.size && !wiring.length && html`<p class="muted small pal-none">Nothing matches.</p>`}
      <p class="muted small pal-none" title=${cat.skipped ? cat.skipped + " left out: they need something the builder cannot fill in, such as a tensor of weights" : ""}>
        ${cat.entries.length} blocks found in the installed modules.</p>
    </div>
  </aside>`;
}

// ------------------------------------------------------------------ canvas

const wirePath = (a, b) => {
  const k = Math.max(40, Math.abs(b.x - a.x) / 2);
  return `M${a.x},${a.y} C${a.x + k},${a.y} ${b.x - k},${b.y} ${b.x},${b.y}`;
};

function Canvas({ ed }) {
  const wrap = useRef(null), world = useRef(null), svg = useRef(null);
  const g = ed.graph.value, check = ed.check.value, notice = ed.notice.value, refusal = ed.refusal.value;
  const view = ed.view.value || { x: 0, y: 0, s: 1 };
  const shapeOf = id => check?.nodes?.[id]?.shape || null;
  const errorOf = id => check?.nodes?.[id]?.error || null;

  const point = (cx, cy) => {
    const r = world.current.getBoundingClientRect(), s = (ed.view.value || view).s;
    return { x: (cx - r.left) / s, y: (cy - r.top) / s };
  };
  const centre = el => { const r = el.getBoundingClientRect(); return point(r.left + r.width / 2, r.top + r.height / 2); };
  const draw = extra => {
    if (!svg.current || !world.current) return;
    let out = "";
    for (const n of ed.graph.value.nodes) {
      (n.inputs || []).forEach((src, i) => {
        const a = world.current.querySelector(`[data-out="${CSS.escape(src)}"]`);
        const b = world.current.querySelector(`[data-in="${CSS.escape(n.id)}"][data-slot="${i}"]`);
        if (a && b) out += `<path d="${wirePath(centre(a), centre(b))}" class="wire k-${shapeKind(shapeOf(src))}"/>`;
      });
    }
    if (extra) out += `<path d="${wirePath(extra.a, extra.b)}" class="wire drafting k-${extra.kind}"/>`;
    svg.current.innerHTML = out;
  };
  const fit = () => {
    const box = wrap.current, nodes = [...world.current.querySelectorAll(".gnode")];
    if (!box || !nodes.length) return setView(ed, { x: 0, y: 0, s: 1 });
    const x1 = Math.min(...nodes.map(n => n.offsetLeft)), y1 = Math.min(...nodes.map(n => n.offsetTop));
    const x2 = Math.max(...nodes.map(n => n.offsetLeft + n.offsetWidth)), y2 = Math.max(...nodes.map(n => n.offsetTop + n.offsetHeight));
    const s = Math.max(0.3, Math.min(1.1, (box.clientWidth - 64) / (x2 - x1 + 1), (box.clientHeight - 120) / (y2 - y1 + 1)));
    setView(ed, { s, x: 32 - x1 * s, y: 56 - y1 * s });
  };
  const shown = useRef(null);
  const [fits, setFits] = useState(null);    // while a wire is pulled: {from, ids: {id: null | why not}}
  useLayoutEffect(() => {
    if (!ed.view.value) fit();
    draw();
    // blocks added elsewhere that landed out of sight are brought into view
    if (notice && shown.current !== notice && notice.ids.length) {
      shown.current = notice;
      const box = wrap.current.getBoundingClientRect();
      const out = notice.ids.some(id => {
        const r = world.current.querySelector(`[data-node="${CSS.escape(id)}"]`)?.getBoundingClientRect();
        return r && (r.right > box.right || r.bottom > box.bottom - 40 || r.left < box.left || r.top < box.top);
      });
      if (out) fit();
    }
  });
  // the marks on blocks the assistant just changed fade after a while
  useEffect(() => {
    if (!notice) return;
    const t = setTimeout(() => { if (ed.notice.value === notice) ed.notice.value = { ...notice, ids: [] }; }, 4000);
    return () => clearTimeout(t);
  }, [notice]);

  const zoom = (f, px, py) => {
    const v = ed.view.value || view, s = Math.max(0.25, Math.min(2, v.s * f));
    const r = wrap.current.getBoundingClientRect();
    px ??= r.width / 2;
    py ??= r.height / 2;
    setView(ed, { s, x: px - (px - v.x) * s / v.s, y: py - (py - v.y) * s / v.s });
  };
  useEffect(() => {
    const el = wrap.current;
    const onWheel = e => {
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoom(e.deltaY < 0 ? 1.1 : 1 / 1.1, e.clientX - r.left, e.clientY - r.top);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const onPointerDown = e => {
    if (e.button !== 0 || e.target.closest(".canvas-tools, .canvas-status")) return;
    const nodeEl = e.target.closest(".gnode");
    if (!nodeEl) return pan(e);
    const n = ed.graph.value.nodes.find(x => x.id === nodeEl.dataset.node);
    if (!n) return;
    e.stopPropagation();
    const out = e.target.closest("[data-out]");
    if (out) return pullWire(e, n, out);
    const inSock = e.target.closest("[data-in]");
    if (inSock && inSock.classList.contains("on")) {
      disconnect(ed, n.id, +inSock.dataset.slot);    // pulling a wire off an input unplugs it
      return;
    }
    if (e.target.closest("button")) return;
    if (ed.sel.value !== n.id) {
      ed.sel.value = n.id;
      if (ed.refusal.value?.id !== n.id) ed.refusal.value = null;
    }
    dragBlock(e, n, nodeEl);
  };
  const pan = e => {
    const sx = e.clientX, sy = e.clientY, v0 = { ...(ed.view.value || view) };
    let moved = false;
    const move = ev => { moved = true; setView(ed, { ...v0, x: v0.x + ev.clientX - sx, y: v0.y + ev.clientY - sy }); };
    const up = () => {
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", up);
      if (!moved) { ed.sel.value = null; ed.refusal.value = null; }
    };
    addEventListener("pointermove", move);
    addEventListener("pointerup", up);
  };
  const dragBlock = (e, n, el) => {
    const sx = e.clientX, sy = e.clientY, s = (ed.view.value || view).s;
    let x = n.x, y = n.y, moved = false;
    const move = ev => {
      moved = true;
      x = Math.round(n.x + (ev.clientX - sx) / s);
      y = Math.round(n.y + (ev.clientY - sy) / s);
      el.style.left = x + "px";
      el.style.top = y + "px";
      draw();
    };
    const up = () => {
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", up);
      if (moved) moveBlock(ed, n.id, x, y);
    };
    addEventListener("pointermove", move);
    addEventListener("pointerup", up);
  };
  const pullWire = (e, n, out) => {
    const a = centre(out), kind = shapeKind(shapeOf(n.id));
    // every block says, while the wire is pulled, whether it would fit there and why not
    let live = true;
    setFits({ from: n.id, ids: null });
    wireTargets(ed, n.id).then(ids => { if (live) setFits({ from: n.id, ids }); });
    const move = ev => draw({ a, b: point(ev.clientX, ev.clientY), kind });
    const up = ev => {
      live = false;
      setFits(null);
      removeEventListener("pointermove", move);
      removeEventListener("pointerup", up);
      draw();
      const hit = document.elementFromPoint(ev.clientX, ev.clientY);
      const sock = hit?.closest("[data-in]"), node = hit?.closest(".gnode");
      if (sock) connect(ed, n.id, sock.dataset.in, +sock.dataset.slot);
      else if (node && node.dataset.node !== n.id) connect(ed, n.id, node.dataset.node, 99);
    };
    addEventListener("pointermove", move);
    addEventListener("pointerup", up);
  };
  const blockMenu = e => {
    const el = e.target.closest(".gnode");
    if (!el) return;
    e.preventDefault();
    const n = ed.graph.value.nodes.find(x => x.id === el.dataset.node);
    if (!n) return;
    ed.sel.value = n.id;
    openMenu({ x: e.clientX, y: e.clientY }, [
      n.op !== "input" && { label: ed.graph.value.output === n.id ? "Unmark as output" : "Make this the output", icon: "check", run: () => toggleOutput(ed, n.id) },
      n.inputs?.length > 0 && { label: "Unplug its inputs", icon: "close", run: () => unplug(ed, n.id) },
      { separator: true },
      { label: "Remove", icon: "trash", hint: "Del", danger: true, run: () => removeBlock(ed, n.id) },
    ], nodeLabel(n));
  };
  const onDrop = e => {
    const k = e.dataTransfer.getData("text/x-block");
    if (!k) return;
    e.preventDefault();
    const p = point(e.clientX, e.clientY);
    addBlock(ed, specOf(k), { x: Math.round(p.x - 90), y: Math.round(p.y - 20) });
  };

  const errs = g.nodes.filter(n => errorOf(n.id)), loose = errs.filter(n => incomplete(errorOf(n.id)));
  const changed = new Set(notice?.ids || []);
  return html`<section class=${cx("canvas", fits && "pulling")} ref=${wrap} aria-label="The network" onPointerDown=${onPointerDown} onContextMenu=${blockMenu}
      onDragOver=${e => { if (e.dataTransfer.types.includes("text/x-block")) { e.preventDefault(); e.dataTransfer.dropEffect = "copy"; } }}
      onDrop=${onDrop}>
    <div class="canvas-tools">
      <${Button} size="sm" icon="arrange" onClick=${() => arrange(ed).then(() => requestAnimationFrame(fit))} title="Lay the blocks out left to right">Arrange<//>
      <div class="seg" role="group" aria-label="Zoom">
        <button type="button" aria-label="Zoom out" onClick=${() => zoom(1 / 1.2)}>−</button>
        <button type="button" title="Fit the network in view" onClick=${fit}>${Math.round(view.s * 100)}%</button>
        <button type="button" aria-label="Zoom in" onClick=${() => zoom(1.2)}>+</button>
      </div>
    </div>
    <div class="world" ref=${world} style=${`transform:translate(${view.x}px,${view.y}px) scale(${view.s})`}>
      <svg class="wires" ref=${svg} />
      ${g.nodes.map(n => html`<${Block} key=${n.id} n=${n} sel=${n.id === ed.sel.value} out=${n.id === check?.output}
        changed=${changed.has(n.id)} shapeOf=${shapeOf} err=${errorOf(n.id)}
        fit=${!fits || fits.from === n.id ? undefined : !fits.ids ? "checking" : n.id in fits.ids ? (fits.ids[n.id] ?? true) : undefined}
        remove=${() => removeBlock(ed, n.id)} />`)}
    </div>
    ${g.nodes.length <= 1 && html`<p class="canvas-hint">Click a block on the left, or drag it here. Drag from a block's output to another's input to wire them.</p>`}
    <div class="canvas-status">
      ${!check ? html`<span class="chip">checking…</span>`
        : check.ok ? html`<span class="chip chip-ok"><${Icon} name="check" size=${12} />checks out · output ${fmtShape(check.outputShape)}</span>`
          : html`<span class="chip chip-warn">${errs.length ? `${plural(errs.length, "block")} ${loose.length === errs.length ? "not connected" : "to fix"}` : check.problem || "not finished"}</span>`}
      ${refusal && html`<span class="chip chip-bad" title=${refusal.text}>Refused: ${refusal.text.length > 90 ? refusal.text.slice(0, 90) + "…" : refusal.text}</span>`}
      ${notice && html`<span class="chip chip-accent">${notice.text}<button type="button" class="chip-btn" onClick=${() => undo(ed)}>Undo</button></span>`}
      <span class="grow" />
      <span class="muted small canvas-keys">Right-click a block for more · Del removes</span>
      <span class="legend"><i class="sock image on" />image<i class="sock vector on" />vector<i class="sock seq on" />sequence</span>
    </div>
  </section>`;
}

function Block({ n, sel, out, changed, shapeOf, err, fit, remove }) {
  const e = entryOf(n), shp = shapeOf(n.id);
  const ins = n.op === "input" ? [] : multiIn(n) ? [...(n.inputs || []), null] : [(n.inputs || [])[0] || null];
  const vals = [];
  if (n.op === "input") vals.push(["shape", fmtShape(n.shape)]);
  if (e) e.params.forEach((p, i) => { if (p.kind !== "random") vals.push([p.name, fmtVal(n.args?.[i])]); });
  Object.entries(n.options || {}).forEach(([k, a]) => vals.push([k.replace(/\(.*$/, ""), (a || []).map(fmtVal).join(", ") || "on"]));
  const fitClass = fit === true ? "fits" : fit === "checking" ? "fit-checking" : typeof fit === "string" ? "nofit" : null;
  return html`<div class=${cx("gnode", "op-" + n.op, sel && "selected", err && (incomplete(err) ? "loose" : "bad"), e?.experimental && "exp", changed && "changed", fitClass)}
      data-node=${n.id} style=${`left:${n.x}px;top:${n.y}px`} title=${typeof fit === "string" && fit !== "checking" ? "Does not fit: " + fit : undefined}>
    <div class="gnode-head"><b>${nodeLabel(n)}</b><span class="grow" />${out ? html`<span class="chip chip-ok">output</span>` : html`<span class="mono muted">${n.id}</span>`}
      <button type="button" class="gnode-x" aria-label=${"Remove " + n.id} title="Remove (Del)" onClick=${e => { e.stopPropagation(); remove(); }}><${Icon} name="close" size=${12} /></button></div>
    ${typeof fit === "string" && fit !== "checking" && html`<p class="gnode-why">${fit}</p>`}
    ${ins.map((src, i) => html`<div class="gnode-row" key=${"i" + i}>
      <span class=${cx("sock", shapeKind(shapeOf(src)), src && "on")} data-in=${n.id} data-slot=${i} title=${src ? "From " + src + ". Drag to unplug." : "Drop a wire here"} />
      <span class="gnode-nm">${src || (multiIn(n) && i ? "+" : "in")}</span><span class="gnode-shp">${src ? fmtShape(shapeOf(src)) : ""}</span></div>`)}
    ${vals.length > 0 && html`<dl class="gnode-vals">${vals.slice(0, 5).map(([k, v]) => html`<div key=${k}><dt>${k}</dt><dd>${v}</dd></div>`)}
      ${vals.length > 5 && html`<div><dt>+${vals.length - 5} more</dt><dd /></div>`}</dl>`}
    ${err && html`<p class="gnode-err">${err}</p>`}
    <div class="gnode-row gnode-out"><span class="gnode-shp">${shp ? fmtShape(shp) : ""}</span><span class="gnode-nm">out</span>
      <span class=${cx("sock", shapeKind(shp), shp && "on")} data-out=${n.id} title="Drag to another block's input" /></div>
  </div>`;
}

// ------------------------------------------------------------------ settings

function Props({ ed }) {
  const g = ed.graph.value, c = ed.check.value;
  const n = g.nodes.find(x => x.id === ed.sel.value);
  if (!n) {
    const b = ed.built.value, inShape = c?.input && c.nodes?.[c.input]?.shape;
    return html`<div class="side-body">
      <dl class="sum">
        <div><dt>blocks</dt><dd>${g.nodes.length}</dd></div>
        <div><dt>input</dt><dd>${inShape ? fmtShape(inShape) : "–"}</dd></div>
        <div><dt>output</dt><dd>${c?.outputShape ? fmtShape(c.outputShape) : "–"}</dd></div>
        <div><dt>parameters</dt><dd>${b ? fmtCount(b.parameters) : "once built"}</dd></div>
      </dl>
      ${c && !c.ok && html`<p class="flag">${c.problem || "Some blocks do not check out yet; they are marked on the canvas."}</p>`}
      ${ed.refusal.value && !ed.refusal.value.id && html`<p class="flag flag-bad">${ed.refusal.value.text}</p>`}
      ${b && html`<p class="flag flag-ok"><b>Built.</b> ${b.file ? html`Saved to <code>${b.file}</code>.` : b.note || ""}</p>
        <details class="activity"><summary>Summary</summary><pre class="code">${b.summary}</pre></details>`}
      <p class="muted small">Click a block to change its settings. Every layer and option comes from the installed modules; a change that would break a block that works is refused, with the reason.</p>
    </div>`;
  }
  const e = entryOf(n), err = c?.nodes?.[n.id]?.error, refused = ed.refusal.value?.id === n.id ? ed.refusal.value.text : null;
  const inShape = (n.inputs || []).map(i => c?.nodes?.[i]?.shape).filter(Boolean).map(fmtShape).join(" + ");
  const set = (fn) => tryChange(ed, gr => fn(gr.nodes.find(x => x.id === n.id), gr), n.id);
  return html`<div class="side-body" key=${n.id}>
    <div>
      <p class="side-kicker">${n.op === "layer" ? (e?.module || "").replace("io.github.zzuegg.", "") : "wiring"}${e?.experimental ? " · experimental" : ""}</p>
      ${e && html`<p class="mono muted side-key">${e.key.replace(/^io\.github\.zzuegg\.brainiac\./, "")}</p>`}
    </div>
    <p class="side-shape mono" title="What comes in, and what goes out">${n.op === "input" ? "gives " : (inShape || "not connected") + " → "}<b>${fmtShape(c?.nodes?.[n.id]?.shape)}</b></p>
    ${refused ? html`<p class="flag flag-bad"><b>Refused.</b> ${refused}</p>` : err && html`<p class="flag">${err}</p>`}
    ${n.op === "input" && html`<label class="field"><span>Shape of one sample <span class="muted">channels first, such as 3, 32, 32</span></span>
      <input class="input mono" value=${(n.shape || []).join(", ")}
        onChange=${ev => set(t => { t.shape = ev.target.value.split(/[\s,×x]+/).filter(Boolean).map(Number); })} /></label>`}
    ${e && e.params.map((p, i) => html`<${Field} key=${i} p=${p} v=${n.args?.[i]}
      onSet=${raw => set(t => { t.args = [...(t.args || [])]; t.args[i] = parseVal(p, raw); })} />`)}
    ${e?.options?.length > 0 && html`<${Options} n=${n} e=${e} set=${set} />`}
    <div class="row-btns">
      ${n.op !== "input" && html`<${Button} size="sm" onClick=${() => set((t, gr) => { gr.output = gr.output === n.id ? undefined : n.id; })}>
        ${g.output === n.id ? "Output (chosen)" : "Make this the output"}<//>`}
      <${Button} size="sm" variant="ghost" icon="trash" onClick=${() => removeBlock(ed, n.id)}>Remove<//>
    </div>
  </div>`;
}

function Field({ p, v, onSet }) {
  if (p.kind === "random") return null;
  if (p.kind === "boolean" && !p.array) {
    return html`<label class="check"><input type="checkbox" checked=${!!v} onChange=${e => onSet(e.target.checked)} />${p.name}</label>`;
  }
  const label = html`<span>${p.name} <span class="muted mono">${p.type}</span></span>`;
  if (p.kind === "choice" && !p.array) {
    return html`<label class="field">${label}<select class="input" value=${v} onChange=${e => onSet(e.target.value)}>
      ${p.choices.map(c => html`<option key=${c}>${c}</option>`)}</select></label>`;
  }
  return html`<label class="field">${label}<input class="input mono" spellcheck="false" value=${Array.isArray(v) ? v.join(", ") : v == null ? "" : String(v)}
    placeholder=${p.array ? (p.kind === "choice" ? "names, comma separated" : "numbers, comma separated") : ""}
    onChange=${e => onSet(e.target.value)} /></label>`;
}

function Options({ n, e, set }) {
  const [q, setQ] = useState("");
  const on = Object.keys(n.options || {});
  const shown = e.options.filter(o => !q || o.key.toLowerCase().includes(q.toLowerCase()));
  return html`<div class="optbox">
    <p class="side-kicker">Options <span class="muted">${on.length} of ${e.options.length} on</span></p>
    ${e.options.length > 6 && html`<input class="input select-sm" placeholder="Find an option" value=${q} onInput=${ev => setQ(ev.target.value)} />`}
    ${shown.map(o => {
      const isOn = on.includes(o.key);
      return html`<div key=${o.key} class=${cx("opt", isOn && "on")}>
        <label class="check"><input type="checkbox" checked=${isOn}
          onChange=${ev => set(t => { t.options = { ...(t.options || {}) }; if (ev.target.checked) t.options[o.key] = o.params.map(guess); else delete t.options[o.key]; })} />
          <span class="mono">${o.name}</span><span class="muted mono small">(${o.params.map(p => p.type).join(", ")})</span></label>
        ${isOn && o.params.map((p, i) => html`<${Field} key=${i} p=${p} v=${n.options[o.key][i]}
          onSet=${raw => set(t => { t.options = { ...t.options, [o.key]: [...t.options[o.key]] }; t.options[o.key][i] = parseVal(p, raw); })} />`)}
      </div>`;
    })}
  </div>`;
}

// ------------------------------------------------------------------ the assistant

const IDEAS = ["Add a hidden layer of 128 units before the output", "Turn this into a small CNN for 32×32 colour images",
  "Explain what this network does, block by block"];

function Assistant({ tab, ed }) {
  const a = assistantOf(ed);
  const id = a.id.value, run = a.run.value;
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [model, setModel] = useState(lastChatModel.value);
  const scroll = useRef(null), follows = useRef(true), input = useRef(null);
  const project = ed.project;
  const live = assistantLive(ed);

  useEffect(() => { if (!modelInfo.value) loadModels(); }, []);
  useEffect(() => { follows.current = true; if (id && !run) follow(ed); }, [id]);
  useLayoutEffect(() => {
    const el = scroll.current;
    if (el && follows.current) el.scrollTop = el.scrollHeight;
  });

  const send = async () => {
    const t = text.trim();
    if (!t || busy || live) return;
    setBusy(true);
    try {
      if (!run) rememberChatModel(model);
      const fresh = !run;
      await askAssistant(ed, t, model);
      follows.current = true;
      setText("");
      if (fresh) tab.refresh();
    } catch (e) {
      fail(e);
    }
    setBusy(false);
  };
  const stop = () => api("/api/agents/stop", { id }, { project }).then(() => follow(ed), fail);
  const forget = () => forgetAssistant(ed);

  return html`<div class="assistant">
    <div class="assistant-head">
      ${run ? html`<span class=${"badge s-" + run.state}>${run.pending ? "waits for you" : live ? "working" : run.state}</span>` : html`<span class="muted small">Changes this network for you</span>`}
      <span class="grow" />
      ${live && html`<${Button} size="sm" icon="stop" onClick=${stop}>Stop<//>`}
      ${run && html`<${Button} size="sm" variant="ghost" icon="open" label="Open as an agent" onClick=${() => navigate(tab, { screen: "agent", id })} />`}
      ${run && !live && html`<${Button} size="sm" variant="ghost" icon="plus" onClick=${forget} title="Start over with a new conversation">New<//>`}
    </div>
    <div class="assistant-scroll" ref=${scroll}
      onScroll=${e => { const el = e.currentTarget; follows.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40; }}>
      ${run ? html`<${Thread} run=${run} project=${project} done=${() => follow(ed)} said=${saidToAssistant} />`
        : id ? html`<div class="loading" aria-busy="true" />`
          : html`<div class="assistant-empty">
            <p>Say what to change, and it changes the network on the canvas while you watch. Every change can be undone.</p>
            <div class="ideas">${IDEAS.map(i => html`<button type="button" key=${i} class="idea" onClick=${() => { setText(i); input.current?.focus(); }}>${i}</button>`)}</div>
          </div>`}
    </div>
    <div class="composer assistant-composer">
      <div class="composer-box">
        <textarea class="composer-input" rows="2" ref=${input} value=${text} disabled=${busy || live} aria-label="Tell the assistant what to change"
          placeholder=${live ? "It is working…" : run?.pending ? "It waits for your answer above" : run ? "Reply, or ask for another change" : "What should change?"}
          onInput=${e => setText(e.target.value)}
          onKeyDown=${e => { if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); send(); } }} />
        <div class="composer-bar">
          ${!run && html`<${ModelPicker} value=${model} onChange=${setModel} />`}
          ${ed.sel.value && html`<span class="chip" title="The assistant is told which block you have selected">${ed.sel.value}</span>`}
          <span class="grow" />
          <${Button} variant="primary" onClick=${send} disabled=${busy || live || !text.trim()}>Send<//>
        </div>
      </div>
    </div>
  </div>`;
}
