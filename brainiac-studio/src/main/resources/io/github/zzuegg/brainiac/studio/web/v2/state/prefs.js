import { signal, effect } from "/v2/lib/html.js";
import { local, pageState, remember } from "/v2/lib/api.js";

// Appearance choices, kept in the studio (so every browser on this PC shares them) and in the
// browser (so the page looks right before the studio's copy is read).

const saved = { ...local.get("prefs", {}), ...(pageState["v2.prefs"] || {}) };

/** "system", "light" or "dark". */
export const theme = signal(saved.theme || "system");
/** The reading column's width: "narrow", "normal", "wide" or "full". */
export const readingWidth = signal(saved.readingWidth || "normal");
export const sidebarOpen = signal(saved.sidebarOpen ?? true);
export const sidebarWidth = signal(saved.sidebarWidth || 272);

export const WIDTHS = { narrow: 640, normal: 760, wide: 960, full: 0 };

effect(() => {
  const v = { theme: theme.value, readingWidth: readingWidth.value, sidebarOpen: sidebarOpen.value, sidebarWidth: sidebarWidth.value };
  const root = document.documentElement;
  if (v.theme === "system") root.removeAttribute("data-theme");
  else root.setAttribute("data-theme", v.theme);
  const w = WIDTHS[v.readingWidth];
  root.style.setProperty("--reading", w ? w + "px" : "100%");
  local.set("prefs", v);
  remember("v2.prefs", v);
});
