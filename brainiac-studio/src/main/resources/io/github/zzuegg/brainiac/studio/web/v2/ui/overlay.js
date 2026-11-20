import { signal } from "/v2/lib/html.js";

// What floats above the page: one menu, a stack of dialogs and drawers. Components open them
// through these functions; the hosts in the shell draw them.

/** The open menu: {x, y, entries, label, restore} or null. */
export const menu = signal(null);

/**
 * Opens a menu at a point, or under an element.
 * @param {{x:number,y:number}|Element} at
 * @param {Array<{label?:string, icon?:string, hint?:string, danger?:boolean, disabled?:boolean, run?:Function, separator?:boolean}>} entries
 */
export function openMenu(at, entries, label = "Actions") {
  let x, y;
  if (at instanceof Element) {
    const r = at.getBoundingClientRect();
    x = r.left;
    y = r.bottom + 4;
  } else {
    ({ x, y } = at);
  }
  menu.value = { x, y, entries: entries.filter(Boolean), label, restore: document.activeElement };
}

export function closeMenu(refocus = true) {
  const m = menu.value;
  menu.value = null;
  if (refocus && m?.restore?.isConnected) m.restore.focus();
}

/** Open dialogs and drawers, the last on top: {id, component, props, resolve}. */
export const layers = signal([]);
let next = 1;

/**
 * Shows a component as a layer and waits for it to close. The component gets `close(result)`.
 * @returns {Promise<any>} what it closed with, or undefined when dismissed
 */
export function openLayer(component, props = {}) {
  return new Promise(resolve => {
    const id = next++;
    const restore = document.activeElement;
    const close = result => {
      layers.value = layers.value.filter(l => l.id !== id);
      if (restore?.isConnected) restore.focus();
      resolve(result);
    };
    layers.value = [...layers.value, { id, component, props: { ...props, close } }];
  });
}
