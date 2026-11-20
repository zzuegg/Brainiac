// The one place the page imports its libraries from: Preact for components, htm for markup in
// plain template strings (no build step), and signals for state. Everything else imports this.
import { h, render, Fragment, createContext, toChildArray } from "/v2/vendor/preact.js";
import htm from "/v2/vendor/htm.js";

export { useState, useEffect, useLayoutEffect, useRef, useMemo, useCallback, useContext, useId } from "/v2/vendor/hooks.js";
export { signal, computed, effect, batch, untracked, useSignal, useComputed, useSignalEffect } from "/v2/vendor/signals.js";
export { h, render, Fragment, createContext, toChildArray };

/** Markup as a tagged template: html`<div class=${c}>${children}</div>`. */
export const html = htm.bind(h);

/** Joins class names, skipping false, null and undefined. */
export const cx = (...names) => names.filter(Boolean).join(" ");
