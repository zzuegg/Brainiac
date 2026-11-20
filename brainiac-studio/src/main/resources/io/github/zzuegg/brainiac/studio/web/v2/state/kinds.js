// What each kind of item is called, its icon, and which of the shared actions apply to it.
// The menu, the sidebar sections and the trash all read this, so a new kind is one entry here.

export const KINDS = {
  chat: { label: "Chat", plural: "Chats", icon: "chat", actions: ["open", "rename", "duplicate", "move", "delete"] },
  agent: { label: "Agent", plural: "Agents", icon: "agent", actions: ["open", "rename", "move", "delete"] },
  graph: { label: "Network", plural: "Networks", icon: "graph", actions: ["open", "rename", "duplicate", "move", "delete"] },
  project: { label: "Project", plural: "Projects", icon: "project", actions: ["open", "rename", "delete"] },
};

/** The sections of a project's sidebar, in order. */
export const SECTIONS = ["chat", "agent", "graph"];

export const kindOf = k => KINDS[k] || { label: k, plural: k, icon: "dot", actions: ["open", "delete"] };
