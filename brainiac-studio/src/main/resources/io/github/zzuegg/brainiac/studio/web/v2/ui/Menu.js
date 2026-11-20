import { html, useEffect, useLayoutEffect, useRef, useState } from "/v2/lib/html.js";
import { Icon } from "/v2/ui/Icon.js";
import { menu, closeMenu } from "/v2/ui/overlay.js";

// The one menu every item uses, opened by its ⋯ button, a right-click or Shift+F10. Arrow keys
// move, Enter runs, Escape closes; typing a letter jumps to the entry that starts with it.

export function MenuHost() {
  const m = menu.value;
  if (!m) return null;
  return html`<${Menu} key=${m.x + "," + m.y} ...${m} />`;
}

function Menu({ x, y, entries, label }) {
  const ref = useRef(null);
  const [pos, setPos] = useState({ left: x, top: y });
  const actionable = entries.map((e, i) => (!e.separator && !e.disabled ? i : -1)).filter(i => i >= 0);
  const [at, setAt] = useState(actionable[0] ?? -1);

  useLayoutEffect(() => {
    const r = ref.current.getBoundingClientRect();
    const pad = 8;
    setPos({
      left: Math.max(pad, Math.min(x, innerWidth - r.width - pad)),
      top: y + r.height + pad > innerHeight ? Math.max(pad, y - r.height - 8) : y,
    });
    ref.current.focus();
  }, []);

  useEffect(() => {
    const away = e => { if (!ref.current?.contains(e.target)) closeMenu(false); };
    const gone = () => closeMenu(false);
    addEventListener("pointerdown", away, true);
    addEventListener("resize", gone);
    addEventListener("blur", gone);
    return () => {
      removeEventListener("pointerdown", away, true);
      removeEventListener("resize", gone);
      removeEventListener("blur", gone);
    };
  }, []);

  const run = i => {
    const e = entries[i];
    if (!e || e.separator || e.disabled) return;
    closeMenu();
    e.run?.();
  };
  const step = d => {
    if (!actionable.length) return;
    const k = actionable.indexOf(at);
    setAt(actionable[(k + d + actionable.length) % actionable.length]);
  };
  const onKey = e => {
    if (e.key === "ArrowDown") step(1);
    else if (e.key === "ArrowUp") step(-1);
    else if (e.key === "Home") setAt(actionable[0]);
    else if (e.key === "End") setAt(actionable.at(-1));
    else if (e.key === "Enter" || e.key === " ") run(at);
    else if (e.key === "Escape" || e.key === "Tab") closeMenu();
    else if (e.key.length === 1) {
      const hit = actionable.find(i => entries[i].label?.toLowerCase().startsWith(e.key.toLowerCase()));
      if (hit != null) setAt(hit);
      else return;
    } else return;
    e.preventDefault();
    e.stopPropagation();
  };

  return html`<div class="menu" role="menu" aria-label=${label} tabindex="-1" ref=${ref} onKeyDown=${onKey}
      style=${{ left: pos.left + "px", top: pos.top + "px" }} onContextMenu=${e => e.preventDefault()}>
    ${entries.map((e, i) => e.separator
      ? html`<div class="menu-sep" role="separator" />`
      : html`<div role="menuitem" class=${"menu-item" + (i === at ? " on" : "") + (e.danger ? " danger" : "")}
            aria-disabled=${e.disabled ? "true" : undefined} onPointerEnter=${() => !e.disabled && setAt(i)}
            onClick=${() => run(i)}>
          <span class="menu-ic">${e.icon && html`<${Icon} name=${e.icon} />`}</span>
          <span class="menu-label">${e.label}</span>
          ${e.hint && html`<span class="menu-hint">${e.hint}</span>`}
        </div>`)}
  </div>`;
}
