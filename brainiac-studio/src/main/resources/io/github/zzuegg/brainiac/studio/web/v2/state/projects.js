import { signal, computed } from "/v2/lib/html.js";
import { api } from "/v2/lib/api.js";

// The projects, shared by every tab. A tab names its project; this says what it is called.
// Projects form a graph: a project can be part of several others ("parents"), and can never
// use more than each of them allows.

export const projects = signal([]);
export const loaded = signal(false);
export const byId = computed(() => new Map(projects.value.map(p => [p.id, p])));

export async function loadProjects() {
  const r = await api("/api/projects");
  projects.value = r.projects || [];
  loaded.value = true;
  return projects.value;
}

export const projectName = id => byId.value.get(id)?.name ?? id;

/** The projects directly part of a project. */
export const childrenOf = id => projects.value.filter(p => (p.parents || []).includes(id));

/** Every project below a project, each once. */
export function descendantsOf(id) {
  const out = new Set();
  const todo = [id];
  while (todo.length) {
    for (const c of childrenOf(todo.pop())) {
      if (!out.has(c.id) && c.id !== id) {
        out.add(c.id);
        todo.push(c.id);
      }
    }
  }
  return out;
}

/** Changes a project's own settings: sends it whole, as the studio keeps it, with the changes. */
export async function saveProject(id, changes) {
  const p = byId.value.get(id);
  if (!p) throw new Error("That project is not there any more.");
  await api("/api/projects", { project: { ...p, ...changes } });
  return loadProjects();
}
