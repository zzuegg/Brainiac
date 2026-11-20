import { html } from "/v2/lib/html.js";

// Line icons on a 24-unit grid, drawn with the current text color.
const PATHS = {
  plus: "M12 5v14M5 12h14",
  image: "M4 5h16v14H4zM4 16l5-5 4 4 3-3 4 4M15 9h.01",
  close: "M6 6l12 12M18 6L6 18",
  more: "M5 12h.01M12 12h.01M19 12h.01",
  chat: "M4 5h16v11H9l-5 4z",
  agent: "M12 3v3M7 8h10a2 2 0 0 1 2 2v7a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2zM9.5 13h.01M14.5 13h.01",
  graph: "M6 5a2 2 0 1 0 0 .01M18 12a2 2 0 1 0 0 .01M6 19a2 2 0 1 0 0 .01M8 6l8 5M8 18l8-5",
  project: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
  home: "M4 11l8-7 8 7M6 10v9h12v-9",
  trash: "M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13",
  settings: "M12 9a3 3 0 1 0 0 6a3 3 0 1 0 0-6zM12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9l2.1 2.1M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1",
  edit: "M4 20h4L19 9l-4-4L4 16zM14 6l4 4",
  copy: "M8 8h11v11H8zM5 16V5h11",
  move: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2zM10 13h6M13 10l3 3-3 3",
  open: "M14 4h6v6M20 4l-9 9M18 14v5H5V6h5",
  undo: "M9 14L4 9l5-5M4 9h10a6 6 0 0 1 0 12h-3",
  search: "M11 4a7 7 0 1 0 0 14a7 7 0 1 0 0-14zM20 20l-4-4",
  chevronRight: "M9 6l6 6-6 6",
  chevronDown: "M6 9l6 6 6-6",
  check: "M5 12l5 5L20 7",
  stop: "M7 7h10v10H7z",
  sidebar: "M4 5h16v14H4zM9 5v14",
  window: "M4 5h16v14H4zM4 9h16",
  info: "M12 8h.01M11 12h1v5h1M12 3a9 9 0 1 0 0 18a9 9 0 1 0 0-18z",
  warn: "M12 4l9 16H3zM12 10v4M12 17h.01",
  restore: "M3 12a9 9 0 1 0 3-6.7M3 4v5h5",
  folder: "M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z",
  dot: "M12 11a1 1 0 1 0 0 2a1 1 0 1 0 0-2z",
  classic: "M4 4h16v16H4zM4 9h16M9 9v11",
  redo: "M15 14l5-5-5-5M20 9H10a6 6 0 0 0 0 12h3",
  chevronLeft: "M15 6l-6 6 6 6",
  download: "M12 4v11M7 10l5 5 5-5M5 20h14",
  speech: "M4 10v4M8 7v10M12 4v16M16 7v10M20 10v4",
  video: "M3 6h12v12H3zM15 10l6-3v10l-6-3z",
  train: "M4 20h16M4 4v16M7 8l4 5 3-2 6 6",
  build: "M12 3l8 4.5v9L12 21l-8-4.5v-9zM12 12l8-4.5M12 12v9M12 12L4 7.5",
  arrange: "M4 5h5v5H4zM15 5h5v5h-5zM15 14h5v5h-5zM9 7.5h6M17.5 10v4",
  lock: "M7 11V8a5 5 0 0 1 10 0v3M5 11h14v10H5z",
  layers: "M12 3l9 5-9 5-9-5zM3 12.5l9 5 9-5M3 17l9 5 9-5",
  play: "M8 5.5v13l10.5-6.5z",
  pause: "M8 5v14M16 5v14",
  mic: "M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3zM5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21",
  sparkle: "M12 3l1.8 5.2L19 10l-5.2 1.8L12 17l-1.8-5.2L5 10l5.2-1.8z",
};

/** An icon by name; `label` makes it announced, else it is decoration. */
export function Icon({ name, size = 16, label, class: cls }) {
  const d = PATHS[name] || PATHS.dot;
  return html`<svg class=${"icon " + (cls || "")} width=${size} height=${size} viewBox="0 0 24 24" fill="none"
    stroke="currentColor" stroke-width=${name === "more" ? 3 : 1.8} stroke-linecap="round" stroke-linejoin="round"
    aria-hidden=${label ? undefined : "true"} role=${label ? "img" : undefined} aria-label=${label}><path d=${d} /></svg>`;
}
