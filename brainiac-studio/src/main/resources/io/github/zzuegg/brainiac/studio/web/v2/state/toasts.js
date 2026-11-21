import { signal } from "/v2/lib/html.js";

// Short messages at the bottom of the window. A toast with an undo also goes on the undo stack,
// so Ctrl+Z undoes the last delete, move or archive even after its toast is gone.

export const toasts = signal([]);
const undoStack = [];
let next = 1;

/**
 * Shows a message.
 * @param {string} text
 * @param {{undo?: () => Promise<any>|any, tone?: "info"|"error", timeout?: number}} [opts]
 */
export function toast(text, { undo, tone = "info", timeout } = {}) {
  const id = next++;
  const t = { id, text, tone, undo };
  toasts.value = [...toasts.value.slice(-3), t];
  if (undo) undoStack.push(t);
  setTimeout(() => dismiss(id), timeout ?? (undo ? 10_000 : tone === "error" ? 8_000 : 4_000));
  return id;
}

export function dismiss(id) {
  toasts.value = toasts.value.filter(t => t.id !== id);
}

/** Shows what went wrong; `e` is an Error or a message. */
export const fail = e => toast(e?.message || String(e), { tone: "error" });

/** Runs a toast's undo, once. */
export async function runUndo(t) {
  dismiss(t.id);
  const i = undoStack.indexOf(t);
  if (i < 0) return;
  undoStack.splice(i, 1);
  try {
    await t.undo();
  } catch (e) {
    fail(e);
  }
}

/** Undoes the most recent undoable action (Ctrl+Z). */
export function undoLast() {
  const t = undoStack.at(-1);
  if (t) runUndo(t);
  return !!t;
}
